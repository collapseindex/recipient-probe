"""(#2) LOCALIZE THE DISCARD + clean re-measure of (#1) base-vs-instruct behavior.

run_localize (Qwen2.5-3B): two overlaid depth curves plus a readout trajectory.
  (A) REPRESENTS-depth: probe accuracy per layer -> where intent becomes decodable.
  (B) CAUSAL-depth: steering-recovery per layer (fit the intent direction at layer L, steer toward recognize,
      measure honoring) -> where routing the intent actually changes behavior.
  (C) READOUT trajectory (logit-lens): per layer, feedback-opener vs acknowledge-opener probability mass for
      recognize-intent inputs -> when the response commits toward feedback. (Illustrative; token sets in code.)
  The discard lives in the gap: intent is represented early (A), the readout commits later (C), and the
  direction only becomes causally routable at a late site (B).

run_align_behavior (base/instruct x2): re-measures #1's behavior WITHOUT the completion-format confound --
  reply truncated to its first line (drop meta-continuation) and scored by a STRICT advice/critique lexicon
  that excludes '?' (so friendly curiosity questions are not miscounted as feedback).

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_localize.py
"""
import modal

app = modal.App("recipient-probe-localize")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

LOC_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BASE_INSTRUCT = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
                 "NousResearch/Meta-Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
FRAME = 'A friend sends you this message:\n"{s}"\n\nYou write back:\n'
ITEMS = 16
MAXTOK = 45

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
# strict advice/critique lexicon (NO '?') for the completion-based base-vs-instruct behavior
ADVICE = ["feedback", "suggest", "suggestion", "improve", "critique", "you could", "you might", "you should",
          "you need to", "recommend", "consider ", "however", "weakness", "the issue", "one thing",
          "a few things", "areas", "refine", "polish", "to fix", "have you tried", "i'd suggest", "my advice",
          "here are some", "here's what", "the problem", "you can try"]
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


def first_line(t):
    # drop meta-continuation: keep the actual reply (first non-empty line), strip wrapping quotes
    for ln in (t or "").split("\n"):
        ln = ln.strip().strip('"').strip()
        if len(ln.split()) >= 4:
            return ln
    return (t or "").strip()


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_localize(model_name: str = LOC_MODEL):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    dense = sorted(set(max(1, int(f * nL)) for f in [0.11, 0.22, 0.33, 0.44, 0.55, 0.67, 0.78, 0.83, 0.92]))
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
    rec_lens_logits = {L: [] for L in dense}  # logit-lens fb/ack mass on recognize items
    for text, lab, _ in stim:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        for L in dense:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())

    # (A) represents-depth
    probe_depth = {L: grouped_probe_one(acts[L], y, groups) for L in dense}

    # (C) logit-lens readout trajectory on recognize items (fb-opener vs ack-opener prob mass)
    def opener_ids(words):
        ids = set()
        for w in words:
            t = tok(" " + w, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return list(ids)
    fb_ids, ack_ids = opener_ids(FB_OPEN), opener_ids(ACK_OPEN)
    norm = model.model.norm
    lm = model.lm_head
    lens = {L: {"fb": 0.0, "ack": 0.0} for L in dense}
    rec_stim = [s for s in stim if s[1] == 0][:ITEMS]
    for text, _, _ in rec_stim:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
            for L in dense:
                logits = lm(norm(hs[L][0, -1, :]))
                p = torch.softmax(logits.float(), dim=-1)
                lens[L]["fb"] += float(p[fb_ids].sum()); lens[L]["ack"] += float(p[ack_ids].sum())
    for L in dense:
        lens[L]["fb"] = round(lens[L]["fb"] / len(rec_stim), 4); lens[L]["ack"] = round(lens[L]["ack"] / len(rec_stim), 4)

    # (B) causal-depth: fit direction at each layer, steer toward recognize, measure honoring
    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    def rec_hon(Lsteer, vec):
        sv["L"], sv["v"] = Lsteer, vec; coh = n = hon = 0
        for text, _, _ in rec_stim:
            rep = gen(text)
            if not coherent(rep):
                continue
            coh += 1; n += 1; hon += int(not has_any(rep, FEEDBACK))
        sv["v"] = None
        return hon, n

    base_hon, base_n = rec_hon(None, None)
    causal_depth = {}
    for L in dense:
        w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[L]), y).coef_[0]
        u = w / (np.linalg.norm(w) + 1e-8)
        an = float(np.linalg.norm(np.array(acts[L]), axis=1).mean())
        vec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")
        hon, n = rec_hon(L, vec)
        causal_depth[L] = {"hon": hon, "n": n}

    return {"model": model_name, "nL": nL, "dense": dense,
            "represents_depth": {str(L): probe_depth[L] for L in dense},
            "causal_depth": {str(L): causal_depth[L] for L in dense},
            "baseline_hon": [base_hon, base_n],
            "readout_lens": {str(L): lens[L] for L in dense}}


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_align_behavior(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    stim = build_stimuli()
    rec = [FRAME.format(s=t) for t, lab, _ in stim if lab == 0][:ITEMS]
    ev = [FRAME.format(s=t) for t, lab, _ in stim if lab == 1][:ITEMS]

    def gen(fr):
        e = tok(fr, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rc = rh = ec = efb = 0; samples = []
    for i, fr in enumerate(rec + ev):
        is_rec = i < len(rec)
        reply = first_line(gen(fr))
        adv = has_any(reply, ADVICE)
        samples.append({"intent": "rec" if is_rec else "eval",
                        "label": "advice" if adv else "acknowledge",
                        "coherent": coherent(reply), "reply": reply[:150]})
        if not coherent(reply):
            continue
        if is_rec:
            rc += 1; rh += int(not adv)
        else:
            ec += 1; efb += int(adv)
    return {"model": model_name, "rec_hon": rh, "rec_n": rc, "eval_advice": efb, "eval_n": ec,
            "audit": samples[:5] + samples[ITEMS:ITEMS + 5]}


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        loc = run_localize.remote()
    except Exception:
        loc = {"ERROR": traceback.format_exc()}
    align = []
    for m in BASE_INSTRUCT:
        try:
            align.append(run_align_behavior.remote(m))
        except Exception:
            align.append({"model": m, "ERROR": traceback.format_exc()})
    path = r"C:/Users/alexs/Desktop/recipient-probe/sweep_localize.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"localize": loc, "align_behavior": align}, f, indent=1)
    print(f"WROTE {path}")
    print("=== LOCALIZE finished ===")
