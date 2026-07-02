"""CONSTRUCT-VALIDITY control: intent, or just request-detection? (reviewer point 1)

The objection: in the stated-intent stimuli the EVALUATE prefixes carry an explicit directive ("be blunt...")
while the RECOGNIZE prefixes carry none, so the probe might decode "is a directive speech act present" rather
than the sender's intent. We hold request-presence CONSTANT: every recognize prefix now also contains an
explicit request (celebrate-with-me / don't-critique), matched to the evaluate prefixes' requests (critique-this).
The surface-matched suffix is unchanged. If the probe still decodes recognize-vs-evaluate at ceiling with a
bag-of-words baseline at chance under leave-one-phrasing-out CV, the signal is WHAT is requested (the intent),
not WHETHER something is requested, and the request-detection deflation is refuted.

Reports, per model: probe accuracy at three depths (0.5/0.67/0.83 nL) and the BoW baseline, all leave-one-
phrasing-out. High probe + chance BoW on the request-matched set = construct validity holds.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_request_matched.py
"""
import modal

app = modal.App("recipient-probe-request-matched")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "NousResearch/Meta-Llama-3.1-8B-Instruct", "microsoft/Phi-3.5-mini-instruct"]

# RECOGNIZE, now WITH an explicit directive/request (matched to the evaluate side's imperatives), so the only
# thing that differs from evaluate is the CONTENT of the request: celebration/acknowledgment vs critique.
# NB: named R_REQUEST (not R_PHRASES) on purpose: this variant DIFFERS from the canonical recognize phrasings,
# so it must not be caught by the stimulus drift guard, which only pins the canonical surface-matched set.
R_REQUEST = [
    "Please just celebrate this with me, I really don't want any notes.",
    "Do me a favor and just take it in, don't give me feedback.",
    "Just tell me it's real and finished, that's all I'm asking.",
    "Please read it and be happy for me, nothing more than that.",
    "Humor me and just enjoy it, no critique please.",
    "Just sit with it for a second, don't fix anything.",
    "React to it, don't review it, that's what I need right now.",
    "Tell me you're proud of me, that's the only ask.",
]
# EVALUATE prefixes (unchanged from the canonical set): these already carry explicit requests for critique.
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
OBJECTS = [
    "photo-organizing tool", "short story I wrote", "watercolor of the harbor", "model train layout",
    "sourdough starter", "bouldering route I set", "song I recorded", "raised garden bed",
    "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf",
    "chess repertoire", "homemade synthesizer", "trail map", "little 2D platformer",
    "pour-over setup", "family recipe book", "wooden chess set", "habit-tracking app",
    "hand-bound notebooks", "drone-footage edit", "terrarium", "custom keyboard",
    "podcast pilot", "stop-motion short", "fermenting crock", "linocut prints",
    "finance dashboard", "observatory mount", "interactive fiction game", "macrame hanging",
    "weather station", "comic strip", "restored radio", "tide-prediction script",
    "ceramic bowls", "model rocket", "flower bed", "family-history sheet",
    "embroidered map", "cold-brew rig", "marble run", "fishing lures",
    "puppet", "finder mount", "noise enclosure", "tarot deck",
    "leather satchel", "plant-watering system", "neighborhood zine", "lighthouse model",
    "card game", "harmonica tabs", "pi doorbell", "carved spoons",
    "constellation chart", "rain-barrel system", "pixel tileset", "handmade kite",
]
SUFFIXES = [
    "Okay, here it is: the {o}. It works now.",
    "Anyway, here it is: the {o}. It's finally done.",
    "So, here it is: the {o}. Finished it last night.",
]


def build_stimuli():
    # (text, label, phrasing_group). label 0 = recognize, 1 = evaluate. group = phrasing index (for GroupKFold).
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_REQUEST); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_REQUEST[g]} {suf}", 0, g))
        rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
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

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    grp = np.array([g for _, _, g in stim])

    # collect hidden states at all layers once
    H = {}
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        for Ld in sorted({max(1, int(f * nL)) for f in (0.5, 0.67, 0.83)}):
            H.setdefault(Ld, []).append(hs[Ld][0, -1, :].float().cpu().numpy())

    def probe_cv(X):
        pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                             LogisticRegression(C=1.0, max_iter=2000))
        return round(float(cross_val_score(pipe, X, y, groups=grp,
                                           cv=GroupKFold(len(set(grp)))).mean()), 3)

    probe = {f"L{Ld} (d={round(Ld / nL, 2)})": probe_cv(np.array(H[Ld])) for Ld in sorted(H)}

    # bag-of-words baseline, leave-one-phrasing-out (same groups). BoW cannot see held-out phrasings' words.
    def bow_cv():
        accs = []
        gkf = GroupKFold(len(set(grp)))
        for tr, te in gkf.split(texts, y, grp):
            cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
            Xtr = cv.fit_transform([texts[i] for i in tr]); Xte = cv.transform([texts[i] for i in te])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
            accs.append(float((clf.predict(Xte) == y[te]).mean()))
        return round(float(np.mean(accs)), 3)

    return {"model": model_name, "nL": nL, "n": len(stim),
            "probe_acc_by_layer": probe, "bow_acc": bow_cv(),
            "note": "request-matched: both classes carry an explicit directive; only the requested intent differs"}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_request_matched.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_request_matched.json")
    print("=== REQUEST-MATCHED finished ===")
