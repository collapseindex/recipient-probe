"""Third intent axis, FULL chain: support (vent) vs help (solve), set by inference (BoW-clean, see gate).

The gate confirmed this axis is lexically clean (bag-of-words 0.57 vs probe 0.75-0.84). Here we run the whole
represent-then-lag chain to show it is not specific to recognize/evaluate:
  REPRESENTS  probe (leave-one-frame-out CV) + BoW baseline, per model.
  DISCARDS    on support-intent messages (the sender wants to be heard), does the default reply jump to
              offering unsolicited SOLUTIONS/advice instead? scored by a solution-offer lexicon.
  RECOVERS    steer the represented direction toward support; does honoring (no unsolicited solutions) recover?
Per-item honoring is returned for both conditions so bootstrap 95% CIs can be put on the rates, exactly as the
primary axis (modal_scale.py). Sanity: help-intent items should draw solutions by default.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_axis3_full.py
"""
import modal

app = modal.App("recipient-probe-axis3-full")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# per-model steer layer reused from the validated primary-axis ladder (modal_scale.py); the represented
# direction routes at a model-specific layer, taken here as a model property.
MODEL_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
               "Qwen/Qwen2.5-14B-Instruct": 41, "mistralai/Mistral-7B-Instruct-v0.3": 22,
               "microsoft/Phi-3.5-mini-instruct": 22, "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
MAXTOK = 45

SUPPORT = [
    "It's quiet here tonight.",
    "I was just thinking of you.",
    "Funny how the mind wanders.",
    "We should catch up properly soon.",
    "I came across an old photo earlier.",
    "It's been one of those weeks.",
    "The evening kind of got away from me.",
    "I almost kept this to myself.",
]
HELP = [
    "I've got a bit of time before dinner.",
    "You're good at seeing these clearly.",
    "There's a date on it now.",
    "Someone asked me where it stands.",
    "I'm trying to wrap it up this week.",
    "I keep going back and forth on it.",
    "There's a call about it tomorrow.",
    "I said I'd have it settled by the weekend.",
]
SITUATIONS = [
    "the thing with my landlord", "the project at work", "the situation with my brother",
    "the noise my car keeps making", "this month's bills", "the group-chat drama",
    "my sister's wedding plans", "the leak under the sink", "my review coming up",
    "the move next month", "the back-and-forth with Dana", "the thesis chapter",
    "the whole neighbor thing", "the credit-card balance", "the kids' school thing",
    "the offer in Denver", "the roof estimate", "the family dinner on Sunday",
    "the membership I never use", "the old friend who resurfaced", "the client who went quiet",
    "the appointment I keep moving", "the storage unit", "the registration renewal",
    "the volunteer thing I said yes to", "the recipe that never works", "the plant that's dying",
    "the subscription I forgot to cancel", "the message I haven't answered", "the box of my dad's stuff",
    "the trip we half-planned", "the fence the neighbor wants", "the warranty claim",
    "the side project", "the overdue book", "the parking ticket",
    "the group assignment", "the leftover paint", "the spare room", "the estimate from the plumber",
]
SUFFIXES3 = [
    "Anyway, {s} is where things are at.",
    "So, {s}, that's the update.",
    "There's {s} in the mix too.",
]
# reply OFFERS a solution / advice (the 'solve' behavior). honoring support = none of these present.
# excludes bare '?': a warm support question ("how are you holding up?") is not solutionizing.
SOLUTION = [
    "you could", "you might", "you can", "you should", "have you tried", "why not", "i'd suggest",
    "i suggest", "i recommend", "recommend", "my advice", "some advice", "a few tips", "here are some",
    "one option", "one thing you", "here's what", "what i would do", "what i'd do", "make sure you",
    "start by", "focus on", "break it down", "prioritize", "step", "steps", "a plan", "the solution",
    "you'll want to", "try to", "consider", "reach out to", "talk to", "set up", "schedule", "make a list",
]


def build():
    rows = []
    for i, s in enumerate(SITUATIONS):
        g = i % len(SUPPORT); suf = SUFFIXES3[i % len(SUFFIXES3)].format(s=s)
        rows.append((f"{SUPPORT[g]} {suf}", 0, g)); rows.append((f"{HELP[g]} {suf}", 1, g))
    return rows


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
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    L = MODEL_LAYER[model_name]
    Lp = max(1, int(0.67 * nL))                      # probe layer (represents)
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[L - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    grp = np.array([g for _, _, g in stim]); nfold = len(set(grp))

    # activations at probe layer and steer layer
    Ap, As = [], []
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        Ap.append(hs[Lp][0, -1, :].float().cpu().numpy())
        As.append(hs[L][0, -1, :].float().cpu().numpy())
    Ap = np.array(Ap); As = np.array(As)

    # REPRESENTS: probe (leave-frame-out) + BoW
    pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                         LogisticRegression(C=1.0, max_iter=2000))
    probe = round(float(cross_val_score(pipe, Ap, y, groups=grp, cv=GroupKFold(nfold)).mean()), 3)
    bow = []
    for tr, te in GroupKFold(nfold).split(texts, y, grp):
        cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(cv.fit_transform([texts[i] for i in tr]), y[tr])
        bow.append(float((clf.predict(cv.transform([texts[i] for i in te])) == y[te]).mean()))
    bow = round(float(np.mean(bow)), 3)

    # steer toward SUPPORT (label 0): -1 * direction, matched to activation norm at the steer layer
    w = LogisticRegression(C=1.0, max_iter=3000).fit(As, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    an = float(np.linalg.norm(As, axis=1).mean())
    steer_support = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    support = [t for t, lab, _ in stim if lab == 0]     # support-intent messages (the discard case)
    help_ = [t for t, lab, _ in stim if lab == 1]

    def per_item(vec):                                  # 1 = honored (no unsolicited solution), -1 = incoherent
        sv["v"] = vec; out = []
        for text in support:
            r = gen(text)
            out.append(-1 if not coherent(r) else int(not has_any(r, SOLUTION)))
        sv["v"] = None
        return out

    default = per_item(None)
    steered = per_item(steer_support)
    sv["v"] = None
    help_sol = [int(has_any(gen(t), SOLUTION)) for t in help_ if coherent(gen(t))]

    return {"model": model_name, "probe_layer": Lp, "steer_layer": L, "n_support": len(support),
            "probe": probe, "bow": bow,
            "honor_default": default, "honor_steered": steered,
            "help_solution_rate": round(sum(help_sol) / max(len(help_sol), 1), 3)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_axis3_full.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_axis3_full.json")
    print("=== AXIS3 FULL finished ===")
