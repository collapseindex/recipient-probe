"""BoW GATE for a clean third intent axis: support (vent) vs help (solve), set by INFERENCE.

The demoted second axis (vent/solve) was lexically marked (bag-of-words 0.79) because the prefixes stated the
request in obvious words. Here the intent is carried by an inferred CONTEXT frame, not a stated request, over a
neutral surface-matched core, and the frames are lexically DIVERSE within each class so no give-away word
survives leave-one-frame-out CV. This gate runs ONLY the probe + bag-of-words baseline (no steering): if BoW
lands near chance while the probe stays high, the axis is clean and worth building out; if BoW is high, the
design failed and we learn it for ~$0.10.

  support = the sender shares a difficulty in a reflective / connecting context (wants to be heard)
  help    = the sender shares the same kind of difficulty in a task / decision context (wants it solved)
Only the FRAME differs; the core (the situation) is surface-matched across the two.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_axis3_gate.py
"""
import modal

app = modal.App("recipient-probe-axis3-gate")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "NousResearch/Meta-Llama-3.1-8B-Instruct"]

# SUPPORT (vent) frames: reflective / connecting context, no stated request. Lexically diverse on purpose.
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
# HELP (solve) frames: task / decision context, no stated request. Lexically diverse on purpose.
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
# neutral, IDENTICAL across intents (carries only the shared situation, not the intent)
SUFFIXES3 = [
    "Anyway, {s} is where things are at.",
    "So, {s}, that's the update.",
    "There's {s} in the mix too.",
]


def build():
    # (text, label, frame_group). label 0 = support/vent, 1 = help/solve. group = frame index (leave-frame-out).
    rows = []
    for i, s in enumerate(SITUATIONS):
        g = i % len(SUPPORT)
        suf = SUFFIXES3[i % len(SUFFIXES3)].format(s=s)
        rows.append((f"{SUPPORT[g]} {suf}", 0, g))
        rows.append((f"{HELP[g]} {suf}", 1, g))
    return rows


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
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

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    rows = build()
    texts = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    grp = np.array([r[2] for r in rows])
    nfold = len(set(grp))

    H = {}
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        for Ld in sorted({max(1, int(f * nL)) for f in (0.5, 0.67, 0.83)}):
            H.setdefault(Ld, []).append(hs[Ld][0, -1, :].float().cpu().numpy())

    def probe_cv(X):
        pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                             LogisticRegression(C=1.0, max_iter=2000))
        return round(float(cross_val_score(pipe, X, y, groups=grp, cv=GroupKFold(nfold)).mean()), 3)

    probe = {f"L{Ld} (d={round(Ld / nL, 2)})": probe_cv(np.array(H[Ld])) for Ld in sorted(H)}

    def bow_cv():
        accs = []
        for tr, te in GroupKFold(nfold).split(texts, y, grp):
            cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
            Xtr = cv.fit_transform([texts[i] for i in tr]); Xte = cv.transform([texts[i] for i in te])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
            accs.append(float((clf.predict(Xte) == y[te]).mean()))
        return round(float(np.mean(accs)), 3)

    return {"model": model_name, "nL": nL, "n": len(rows), "n_frames": nfold,
            "probe_acc_by_layer": probe, "bow_acc": bow_cv(),
            "note": "inferred support-vs-help; frame sets intent, situation surface-matched"}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_axis3_gate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_axis3_gate.json")
    print("=== AXIS3 GATE finished ===")
