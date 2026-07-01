"""(#1) COMPONENT CIRCUIT + (#2) JUDGE-FREE METRIC HARDENING, on Qwen2.5-3B.

run_circuit: per-head ablation. We depth-localized the discard to the late layers (L28-33). Here we zero one
    query head at a time (via a pre-hook on o_proj) and measure recognize-intent honoring. Heads whose ablation
    RAISES honoring while coherence holds are the components writing the discard. An early layer (12) is included
    as a specificity control (ablating heads there should not restore honoring).

run_metric: the paper scores behavior with a feedback-offer lexicon; a reviewer will call it a heuristic. We
    re-score default vs steered recognize replies with an INDEPENDENT, judge-free measure: a sentence-embedding
    classifier (cosine to hand-written feedback vs acknowledgment prototypes). If both the lexicon and the
    embedding measure show steering raises honoring, and they agree per-reply, the effect is not a lexicon artifact.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_circuit.py
"""
import modal

app = modal.App("recipient-probe-circuit")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                      "sentencepiece", "sentence-transformers"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 30
ABL_LAYERS = [12, 26, 28, 30, 32]  # 12 = specificity control (early)
ITEMS = 16
MAXTOK = 40

R_PHRASES = [
    "I've been chipping away at this for months and I'm a little nervous to show anyone.",
    "I don't usually share what I make, but I'm kind of proud of this one.",
    "I almost quit a few times. It's done now.",
    "I'm not looking for notes, I just wanted it to exist and to tell somebody.",
    "It's the first thing I've finished in a long time.",
    "Not asking for anything, just felt like showing you.",
    "It's silly maybe, but it matters to me.",
    "I just wanted to mark the moment with someone.",
]
E_PHRASES = [
    "I'm submitting this to a review committee and need it to hold up to scrutiny.",
    "Be blunt, I'd rather hear the flaws now than after I publish.",
    "I'm putting this in front of experts soon and need to know what breaks.",
    "Where are the holes? Don't soften it.",
    "I want the harshest read you can give me.",
    "Stress-test it for me, find the failure points.",
    "I need to know if this is actually good or if I'm fooling myself.",
    "What would a skeptic say to take this apart?",
]
OBJECTS = ["photo-organizing tool", "short story I wrote", "watercolor of the harbor", "model train layout",
           "sourdough starter", "bouldering route I set", "song I recorded", "raised garden bed",
           "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf", "chess repertoire",
           "homemade synthesizer", "trail map", "little 2D platformer", "pour-over setup", "family recipe book",
           "wooden chess set", "habit-tracking app", "hand-bound notebooks", "drone-footage edit", "terrarium",
           "custom keyboard", "podcast pilot", "stop-motion short", "fermenting crock", "linocut prints",
           "finance dashboard", "observatory mount", "interactive fiction game", "macrame hanging",
           "weather station", "comic strip", "restored radio", "tide-prediction script", "ceramic bowls",
           "model rocket", "flower bed", "family-history sheet", "embroidered map", "cold-brew rig"]
SUFFIXES = ["Okay, here it is: the {o}. It works now.", "Anyway, here it is: the {o}. It's finally done.",
            "So, here it is: the {o}. Finished it last night."]
FEEDBACK = ["feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
            "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
            "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
            "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
            "make sure", "one thing", "stronger", "?"]
FB_PROTO = ["Here are some suggestions to improve it.", "You could refine a few things.",
            "One issue is the pacing, consider changing it.", "Have you thought about revising the ending?",
            "I'd recommend polishing the structure and tightening it up."]
ACK_PROTO = ["Congratulations, that's wonderful!", "I'm so happy for you, what a milestone.",
             "That's a real accomplishment, you should be proud.", "Wow, that's amazing, well done.",
             "I'm really glad you shared this with me."]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
    return rows


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def has_any(t, lex):
    t = (t or "").lower(); return any(m in t for m in lex)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_circuit(model_name: str = MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nH = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // nH)
    abl = {"L": None, "h": None}

    def pre_o(idx):
        def f(_mod, args):
            if abl["L"] != idx or abl["h"] is None:
                return None
            x = args[0].clone(); h = abl["h"]
            x[..., h * head_dim:(h + 1) * head_dim] = 0
            return (x,) + tuple(args[1:])
        return f
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(pre_o(i + 1))  # 1-based to match ABL_LAYERS

    stim = build_stimuli()
    rec = [t for t, lab, _ in stim if lab == 0][:ITEMS]

    def gen(text):
        e = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    def honoring():
        coh = n = hon = 0
        for text in rec:
            rep = gen(text)
            if not coherent(rep):
                continue
            coh += 1; n += 1; hon += int(not has_any(rep, FEEDBACK))
        return {"coh": coh, "n": n, "hon": hon}

    abl["L"], abl["h"] = None, None
    base = honoring()
    results = {}
    for L in ABL_LAYERS:
        for h in range(nH):
            abl["L"], abl["h"] = L, h
            r = honoring()
            results[f"L{L}h{h}"] = {"hon": r["hon"], "n": r["n"], "coh": r["coh"]}
    abl["L"], abl["h"] = None, None
    return {"model": model_name, "nH": nH, "head_dim": head_dim, "abl_layers": ABL_LAYERS,
            "baseline": base, "ablations": results}


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_metric(model_name: str = MODEL):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[LAYER - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        acts.append(hs[LAYER][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    an = float(np.linalg.norm(acts, axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0][:ITEMS]
    default_reps, steered_reps = [], []
    for text in rec:
        sv["v"] = None; default_reps.append(gen(text))
        sv["v"] = steer_rec; steered_reps.append(gen(text))
    sv["v"] = None

    # judge-free independent measure: sentence-embedding cosine to feedback vs ack prototypes
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    fb_c = emb.encode(FB_PROTO, normalize_embeddings=True).mean(0)
    ack_c = emb.encode(ACK_PROTO, normalize_embeddings=True).mean(0)

    def score(reps):
        lex_hon = emb_hon = coh = agree = 0
        rows = []
        for r in reps:
            if not coherent(r):
                continue
            coh += 1
            lex_fb = has_any(r, FEEDBACK)
            e = emb.encode([r], normalize_embeddings=True)[0]
            emb_fb = float(e @ fb_c) > float(e @ ack_c)
            lex_hon += int(not lex_fb); emb_hon += int(not emb_fb)
            agree += int(lex_fb == emb_fb)
            rows.append({"lex": "fb" if lex_fb else "ack", "emb": "fb" if emb_fb else "ack", "r": r[:120]})
        return {"coh": coh, "lex_hon": lex_hon, "emb_hon": emb_hon, "agree": agree}, rows

    d_sc, d_rows = score(default_reps)
    s_sc, s_rows = score(steered_reps)
    return {"model": model_name, "layer": LAYER, "n": ITEMS,
            "default": d_sc, "steered": s_sc, "sample_default": d_rows[:6], "sample_steered": s_rows[:6]}


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        circ = run_circuit.remote()
    except Exception:
        circ = {"ERROR": traceback.format_exc()}
    try:
        metric = run_metric.remote()
    except Exception:
        metric = {"ERROR": traceback.format_exc()}
    path = r"C:/Users/alexs/Desktop/recipient-probe/sweep_circuit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"circuit": circ, "metric": metric}, f, indent=1)
    print(f"WROTE {path}")
    print("=== CIRCUIT finished ===")
