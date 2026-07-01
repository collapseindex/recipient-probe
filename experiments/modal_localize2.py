"""Scale the depth-localization (the paper's mechanism claim): represented-early vs routed-late, at n=60 with
bootstrap CIs, on TWO models (Qwen-3B and Llama-8B). Firms up Table 6, previously n=16, single model.

Per model, at ~9 layers by depth: (A) probe accuracy (represents), (B) steer-toward-recognize honoring on the
FULL 60-item recognize set with per-item outputs for bootstrap CIs (causal routability), (C) logit-lens
feedback/acknowledgment opener mass (readout commitment). The claim: intent is decodable at mid layers but
steering only recovers honoring at late layers, where the reply opener also commits.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_localize2.py
"""
import modal

app = modal.App("recipient-probe-localize2")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
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
           "model rocket", "flower bed", "family-history sheet", "embroidered map", "cold-brew rig",
           "herb garden", "photo calendar", "wooden spoon rack", "knitted hat", "short film", "neighborhood zine",
           "bird feeder", "spice rack", "model bridge", "hand-drawn map", "quilt square", "resin coaster",
           "paper-crane mobile", "garden trellis", "compost bin", "clay mug", "friendship bracelet", "kite"]
SUFFIXES = ["Okay, here it is: the {o}. It works now.", "Anyway, here it is: the {o}. It's finally done.",
            "So, here it is: the {o}. Finished it last night."]
FEEDBACK = ["feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
            "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
            "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
            "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
            "make sure", "one thing", "stronger", "?"]
ACK_OPEN = ["Congratulations", "Congrats", "That", "Wow", "Amazing", "Nice", "Love", "Happy", "Wonderful",
            "Beautiful", "Aww", "Yay", "Great", "Fantastic", "Incredible"]
FB_OPEN = ["Here", "One", "Have", "Consider", "You", "First", "To", "Overall", "While", "A", "The", "If"]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
    return rows


def grouped_probe_one(X, y, groups):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    gkf = GroupKFold(n_splits=len(set(groups)))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                         LogisticRegression(C=1.0, max_iter=2000))
    return round(float(cross_val_score(pipe, np.asarray(X), np.asarray(y), groups=np.asarray(groups),
                                       cv=gkf).mean()), 3)


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def has_any(t, lex):
    t = (t or "").lower(); return any(m in t for m in lex)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    dense = sorted(set(max(1, int(f * nL)) for f in [0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.78, 0.86, 0.94]))
    sv = {"v": None, "L": None}

    def mk(L):
        def hook(_m, _i, out):
            if sv["v"] is None or sv["L"] != L:
                return out
            if isinstance(out, tuple):
                h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
            h = out.clone(); h[:, -1, :] += sv["v"]; return h
        return hook
    for L in dense:
        model.model.layers[L - 1].register_forward_hook(mk(L))

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim]); groups = [g for _, _, g in stim]
    acts = {L: [] for L in dense}
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        for L in dense:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())

    probe_depth = {L: grouped_probe_one(acts[L], y, groups) for L in dense}

    # logit-lens on recognize items
    def opener_ids(words):
        ids = set()
        for w in words:
            t = tok(" " + w, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return list(ids)
    fb_ids, ack_ids = opener_ids(FB_OPEN), opener_ids(ACK_OPEN)
    norm, lm = model.model.norm, model.lm_head
    rec_stim = [s for s in stim if s[1] == 0]  # full 60
    lens = {L: {"fb": 0.0, "ack": 0.0} for L in dense}
    for text, _, _ in rec_stim:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
            for L in dense:
                p = torch.softmax(lm(norm(hs[L][0, -1, :])).float(), dim=-1)
                lens[L]["fb"] += float(p[fb_ids].sum()); lens[L]["ack"] += float(p[ack_ids].sum())
    for L in dense:
        lens[L]["fb"] = round(lens[L]["fb"] / len(rec_stim), 4); lens[L]["ack"] = round(lens[L]["ack"] / len(rec_stim), 4)

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    def per_item(Lsteer, vec):
        sv["L"], sv["v"] = Lsteer, vec; out = []
        for text, _, _ in rec_stim:
            rep = gen(text)
            out.append(-1 if not coherent(rep) else int(not has_any(rep, FEEDBACK)))
        sv["v"] = None
        return out

    baseline = per_item(None, None)
    causal_depth = {}
    for L in dense:
        w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[L]), y).coef_[0]
        u = w / (np.linalg.norm(w) + 1e-8)
        an = float(np.linalg.norm(np.array(acts[L]), axis=1).mean())
        vec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")
        causal_depth[str(L)] = per_item(L, vec)

    return {"model": model_name, "nL": nL, "dense": dense, "n_rec": len(rec_stim),
            "represents_depth": {str(L): probe_depth[L] for L in dense},
            "baseline_honoring": baseline,
            "causal_depth": causal_depth,
            "readout_lens": {str(L): lens[L] for L in dense}}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    path = r"C:/Users/alexs/Desktop/recipient-probe/sweep_localize2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
    print("=== LOCALIZE2 finished ===")
