"""CENTERPIECE: the represent->discard->recover chain for IMPLICIT (inferred) intent, across six models.

In the main stimuli the intent is stated ("I'd rather hear the flaws"). Here it is never stated: the sender
shares the same object, and the intent must be INFERRED from a contextual frame -- a birthday present (infer:
wants it appreciated) vs a portfolio piece (infer: wants it scrutinized). No word names the preference. A probe
that decodes intent from the surface-matched suffix token is therefore reading a pragmatic INFERENCE, not a
declared instruction. The key question for scale: do the ceiling models (which honor EXPLICIT intent) still
discard INFERRED intent?

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_implicit.py
"""
import modal

app = modal.App("recipient-probe-implicit")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
               "Qwen/Qwen2.5-14B-Instruct": 41, "mistralai/Mistral-7B-Instruct-v0.3": 22,
               "microsoft/Phi-3.5-mini-instruct": 22, "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
MAXTOK = 40

# implicit frames: intent inferred from context, never stated. label 0 = recognize (wants acknowledgment),
# label 1 = evaluate (wants critique). No frame contains "acknowledge"/"critique"/"feedback"/"notes".
R_IMPLICIT = [
    "It's a birthday present for my mom.",
    "I'm keeping this one for myself.",
    "It's going up on the wall at home.",
    "It's the first thing I've ever made like this.",
    "I made it for my kid.",
    "It's a little gift for a friend going through a hard time.",
    "It's going on the mantel with the family photos.",
    "I made it in memory of my grandpa.",
]
E_IMPLICIT = [
    "It's going in my portfolio.",
    "I'm submitting it to the juried show next week.",
    "It's a work sample for a job I'm applying to.",
    "I'm showing it to the client on Monday.",
    "It's the prototype before I make fifty more.",
    "It's going in front of my thesis committee.",
    "I'm entering it in the competition on Friday.",
    "It goes live on the store page tomorrow.",
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
FEEDBACK = [
    "feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
    "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
    "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
    "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
    "make sure", "one thing", "stronger", "?",
]


def build_stimuli():
    # {implicit frame} {surface-matched disclosure}. Intent inferred from the frame, probed at the suffix token.
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_IMPLICIT); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_IMPLICIT[g]} {suf}", 0, g)); rows.append((f"{E_IMPLICIT[g]} {suf}", 1, g))
    return rows


def grouped_probe(X_by_layer, y, groups, n_perm=6):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y); groups = np.asarray(groups); rng = np.random.RandomState(0)
    gkf = GroupKFold(n_splits=len(set(groups)))
    pipe = lambda: make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                                 LogisticRegression(C=1.0, max_iter=2000))
    out = {}
    for L, X in X_by_layer.items():
        X = np.asarray(X)
        real = cross_val_score(pipe(), X, y, groups=groups, cv=gkf).mean()
        shuf = max(cross_val_score(pipe(), X, rng.permutation(y), groups=groups, cv=gkf).mean()
                   for _ in range(n_perm))
        out[L] = (round(float(real), 3), round(float(shuf), 3))
    return out


def bow_baseline(texts, y, groups):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    gkf = GroupKFold(n_splits=len(set(groups)))
    return round(float(cross_val_score(make_pipeline(TfidfVectorizer(), LogisticRegression(C=1.0, max_iter=2000)),
                       texts, np.asarray(y), groups=np.asarray(groups), cv=gkf).mean()), 3)


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def has_any(t, lex):
    t = (t or "").lower(); return any(m in t for m in lex)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run_model(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layer = MODEL_LAYER[model_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.33, 0.5, 0.67, 0.83]))
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[layer - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim]); groups = [g for _, _, g in stim]
    allL = sorted(set(probe_layers) | {layer})
    acts = {L: [] for L in allL}
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        for L in allL:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
    probe = grouped_probe({L: acts[L] for L in probe_layers}, y, groups)
    bow = bow_baseline(texts, y, groups)

    w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[layer]), y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8); an = float(np.linalg.norm(np.array(acts[layer]), axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0]  # 60 recognize-implicit
    samples = []

    def per_item(vec, keep=False):
        sv["v"] = vec; out = []
        for text in rec:
            r = gen(text)
            if keep:
                samples.append({"label": "feedback" if has_any(r, FEEDBACK) else "acknowledge",
                                "msg": text[:80], "reply": r[:150]})
            out.append(-1 if not coherent(r) else int(not has_any(r, FEEDBACK)))
        sv["v"] = None
        return out
    default = per_item(None, keep=True)
    steered = per_item(steer_rec)
    return {"model": model_name, "layer": layer, "n_rec": len(rec),
            "probe": {str(L): probe[L] for L in probe_layers},
            "probe_peak": max(probe.values(), key=lambda v: v[0])[0], "bow": bow,
            "implicit_default": default, "implicit_steered": steered, "audit": samples[:6]}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run_model.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_implicit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_implicit.json")
    print("=== IMPLICIT finished ===")
