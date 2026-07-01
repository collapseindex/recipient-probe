"""VALENCE GATE for the implicit-intent claim. If the implicit probe decodes emotional warmth rather than
inferred intent, the Gricean reframe is unfounded. We cross intent x valence into four cells, intent always
INFERRED (never stated):
  WR warm+recognize:  "It's a birthday present for my mom."           (personal, wants it appreciated)
  WE warm+evaluate:   "It's for my mom's 60th, I only get one shot."  (personal, wants it right -> scrutiny)
  NR neutral+recognize:"It's just a little side thing, nothing serious."(detached, wants light acknowledgment)
  CE neutral+evaluate:"It's going in my portfolio."                    (detached, wants scrutiny)
Tests: (1) intent probe cross-valence transfer (train on warm cells WR/WE, test on cold cells NR/CE) -- high =
tracks intent, not warmth; (2) cosine between the intent direction and the valence direction -- low = distinct.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_valence.py
"""
import modal

app = modal.App("recipient-probe-valence")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
N_OBJ = 30  # 30 objects x 4 cells = 120 stimuli, balanced

# cell frames. intent = recognize (WR,NR) vs evaluate (WE,CE); valence = warm/personal (WR,WE) vs neutral (NR,CE)
WR = [
    "It's a birthday present for my mom.", "I made it for my kid.",
    "It's going on the mantel with the family photos.", "I made it in memory of my grandpa.",
    "It's a little gift for a friend who's been down.", "I'm keeping this one, it means a lot to me.",
    "It's for my sister, just because.", "I made it to remember the trip we took.",
]
WE = [
    "It's for my mom's 60th and I only get one shot at this.", "It's my daughter's wedding gift, everyone will see it.",
    "It's for my best friend's big day, it has to be right.", "I'm giving it to my dad and I can't let him down.",
    "It's a memorial piece for the family, it has to be worthy of him.",
    "It's my kid's keepsake, I want it perfect before I seal it.",
    "It's for my parents' anniversary, they'll have it forever.", "It's the centerpiece for my sister's wedding.",
]
NR = [
    "It's just a little side thing, nothing serious.", "It's nothing important, just something I tinkered with.",
    "Just messing around really, figured I'd show you.", "It's a throwaway, but I had fun with it.",
    "No big deal, just wanted to show someone.", "It's just for me, not going anywhere with it.",
    "Killed a rainy afternoon making this.", "Nothing fancy, just a quick one.",
]
CE = [
    "It's going in my portfolio.", "I'm submitting it to the juried show next week.",
    "It's a work sample for a job I'm applying to.", "I'm showing it to the client on Monday.",
    "It's going in front of my thesis committee.", "It goes live on the store page tomorrow.",
    "I'm entering it in the competition on Friday.", "It's for the review board next month.",
]
OBJECTS = [
    "photo-organizing tool", "short story I wrote", "watercolor of the harbor", "model train layout",
    "sourdough starter", "bouldering route I set", "song I recorded", "raised garden bed",
    "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf", "chess repertoire",
    "homemade synthesizer", "trail map", "little 2D platformer", "pour-over setup", "family recipe book",
    "wooden chess set", "habit-tracking app", "hand-bound notebooks", "drone-footage edit", "terrarium",
    "custom keyboard", "podcast pilot", "stop-motion short", "fermenting crock", "linocut prints",
    "finance dashboard", "observatory mount", "interactive fiction game", "macrame hanging",
    "weather station", "comic strip", "restored radio", "tide-prediction script", "ceramic bowls",
    "model rocket", "flower bed", "family-history sheet", "embroidered map", "cold-brew rig",
    "marble run", "fishing lures", "puppet", "finder mount", "noise enclosure", "tarot deck",
    "leather satchel", "plant-watering system", "neighborhood zine", "lighthouse model", "card game",
    "harmonica tabs", "pi doorbell", "carved spoons", "constellation chart", "rain-barrel system",
    "pixel tileset", "handmade kite",
]
SUFFIXES = [
    "Okay, here it is: the {o}. It works now.",
    "Anyway, here it is: the {o}. It's finally done.",
    "So, here it is: the {o}. Finished it last night.",
]

CELLS = [("WR", WR, 0, 0), ("WE", WE, 1, 0), ("NR", NR, 0, 1), ("CE", CE, 1, 1)]  # name, frames, intent, valence


def build():
    # returns (text, intent, valence, cell, frame_group)
    rows = []
    for oi in range(N_OBJ):
        o = OBJECTS[oi]; suf = SUFFIXES[oi % len(SUFFIXES)].format(o=o)
        for name, frames, intent, val in CELLS:
            g = oi % len(frames)
            rows.append((f"{frames[g]} {suf}", intent, val, name, name + str(g)))
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    L = max(1, int(0.67 * nL))

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    rows = build()
    texts = [r[0] for r in rows]
    intent = np.array([r[1] for r in rows]); valence = np.array([r[2] for r in rows])
    cell = np.array([r[3] for r in rows]); fgrp = [r[4] for r in rows]
    X = []
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        X.append(hs[L][0, -1, :].float().cpu().numpy())
    X = np.array(X)

    def pipe():
        return make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                             LogisticRegression(C=1.0, max_iter=2000))

    def cv(y):  # leave-frame-out
        return round(float(cross_val_score(pipe(), X, y, groups=np.array(fgrp),
                                           cv=GroupKFold(len(set(fgrp)))).mean()), 3)

    intent_acc = cv(intent)      # decode intent (across valence)
    valence_acc = cv(valence)    # decode valence (across intent)

    # cross-valence transfer: train intent probe on WARM cells (WR,WE), test on COLD cells (NR,CE)
    warm = valence == 0; cold = ~warm
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(X[warm], intent[warm]); xfer_w2c = round(float((clf.predict(X[cold]) == intent[cold]).mean()), 3)
    clf.fit(X[cold], intent[cold]); xfer_c2w = round(float((clf.predict(X[warm]) == intent[warm]).mean()), 3)

    # cosine between intent direction and valence direction (raw logistic weights on standardized X)
    Xs = StandardScaler().fit_transform(X)
    wi = LogisticRegression(C=1.0, max_iter=3000).fit(Xs, intent).coef_[0]
    wv = LogisticRegression(C=1.0, max_iter=3000).fit(Xs, valence).coef_[0]
    cos = float(abs(np.dot(wi, wv) / (np.linalg.norm(wi) * np.linalg.norm(wv) + 1e-9)))

    # within-valence intent decodability (sanity)
    def within(mask):
        y = intent[mask]
        return round(float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                     X[mask], y, cv=5).mean()), 3)
    return {"model": model_name, "layer": L, "n": len(rows),
            "intent_acc": intent_acc, "valence_acc": valence_acc,
            "intent_xfer_warm2cold": xfer_w2c, "intent_xfer_cold2warm": xfer_c2w,
            "cos_intent_valence": round(cos, 3),
            "within_warm_intent": within(warm), "within_cold_intent": within(cold)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_valence.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_valence.json")
    print("=== VALENCE finished ===")
