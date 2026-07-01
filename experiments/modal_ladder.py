"""Generality test on Modal GPUs: does the represents -> discards -> recover chain hold and scale across
model size? Runs the full chain (intent probe with leave-phrasing-out controls, plus probe-direction
steering dose-response at a late layer) on a Qwen ladder (3B / 7B / 14B, same family = clean scaling).

  modal run experiments/modal_ladder.py
"""
import modal

app = modal.App("recipient-probe-ladder")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers>=4.44", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]

R_PHRASES = [
    "I've been chipping away at this for months and I'm a little nervous to show anyone.",
    "I don't usually share what I make, but I'm proud of this one.",
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
FEEDBACK = [
    "feedback", "suggest", "improve", "critique",
    "review", "assess", "what about", "you could",
    "you might", "here are some", "areas", "consider",
    "recommend", "however", "issue", "weakness",
    "problem", "could be", "love to see", "love to read",
    "happy to help", "here to help", "share it", "go ahead",
    "potential", "notes", "tips", "advice",
    "refine", "polish", "make sure", "one thing",
    "stronger", "?",
]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
    return rows


def grouped_probe(X_by_layer, y, groups, n_perm=8):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y); groups = np.asarray(groups); rng = np.random.RandomState(0)
    gkf = GroupKFold(n_splits=len(set(groups)))
    pipe = lambda n: make_pipeline(StandardScaler(), PCA(n_components=min(40, n), random_state=0),
                                   LogisticRegression(C=1.0, max_iter=2000))
    out = {}
    for L, X in X_by_layer.items():
        X = np.asarray(X)
        real = cross_val_score(pipe(40), X, y, groups=groups, cv=gkf).mean()
        shuf = max(cross_val_score(pipe(40), X, rng.permutation(y), groups=groups, cv=gkf).mean()
                   for _ in range(n_perm))
        out[L] = (float(real), float(shuf))
    return out


def bow_baseline(texts, y, groups):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    gkf = GroupKFold(n_splits=len(set(groups)))
    return float(cross_val_score(make_pipeline(TfidfVectorizer(), LogisticRegression(C=1.0, max_iter=2000)),
                                 texts, np.asarray(y), groups=np.asarray(groups), cv=gkf).mean())


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def offers_feedback(t):
    t = (t or "").lower(); return any(m in t for m in FEEDBACK)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache}, timeout=3600)
def run_model(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.17, 0.33, 0.5, 0.67, 0.83, 0.92]))
    steer_layer = max(1, int(0.83 * nL))

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim]); groups = [g for _, _, g in stim]

    # represents: probe per layer + BoW
    acts = {L: [] for L in probe_layers}; sact = []
    for text in texts:
        enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in probe_layers:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
        sact.append(hs[steer_layer][0, -1, :].float().cpu().numpy())
    probe = grouped_probe(acts, y, groups)
    bow = bow_baseline(texts, y, groups)

    # recover: steering at steer_layer, probe-weight direction, dose-response
    sact = np.array(sact)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(sact, y).coef_[0]
    w_unit = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.bfloat16, device="cuda")
    actnorm = float(np.linalg.norm(sact, axis=1).mean())
    subset = [s for s in stim if s[1] == 0][6:18] + [s for s in stim if s[1] == 1][6:18]
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[steer_layer - 1].register_forward_hook(hook)

    def tally(vec):
        sv["v"] = vec; coh = rh = rc = eb = 0
        for text, label, _ in subset:
            enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=55, do_sample=False, pad_token_id=tok.eos_token_id)
            rep = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if not coherent(rep):
                continue
            coh += 1; evb = offers_feedback(rep); eb += int(evb)
            if label == 0:
                rc += 1; rh += int(not evb)
        sv["v"] = None
        return {"coh": coh, "rec_honored": rh, "rec_n": rc, "eval_behavior": eb}

    base = tally(None)
    dose = {"baseline": base}
    for a in [0.5, 1.0]:
        dose[f"toward_recognize@{a}"] = tally(-a * actnorm * w_unit)
        dose[f"toward_evaluate@{a}"] = tally(a * actnorm * w_unit)
    return {"model": model_name, "n_layers": nL, "probe_layers": probe_layers, "steer_layer": steer_layer,
            "bow": bow, "probe": {str(L): probe[L] for L in probe_layers}, "n_subset": len(subset),
            "dose": dose}


@app.local_entrypoint()
def main():
    import json
    for model in MODELS:
        print(f"\n######## {model} ########")
        r = run_model.remote(model)
        print(json.dumps(r, indent=2))
