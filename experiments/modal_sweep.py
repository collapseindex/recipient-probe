"""Per-model steer-layer sweep + dose-response + CLASSIFIER AUDIT, across cheap open-weight models.

Fixes the ladder's weakness (it used a fixed 0.83-depth steer layer; the real 3B win came from sweeping).
For each model we sweep candidate late layers to find the gate-passing one with the biggest honoring lift,
then run a dose-response there. We also DUMP sample generations next to their offers_feedback labels so the
behavior classifier (our reclaim-parser-bug analog) can be eyeballed before trusting any number or spending
on expensive models.

Models: Qwen 7B/14B (scaling) + Mistral-7B + Phi-3.5-mini (cross-family). All ungated, all cheap.

  modal run experiments/modal_sweep.py
"""
import modal

app = modal.App("recipient-probe-sweep")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# wider ladder: the cross-family models not yet covered. Phi via native transformers (no trust_remote_code,
# the repo's custom code calls the removed DynamicCache.from_legacy_cache); Llama via an ungated mirror.
MODELS = ["microsoft/Phi-3.5-mini-instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]

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
           "marble run", "fishing lures", "puppet", "finder mount", "noise enclosure", "tarot deck",
           "leather satchel", "plant-watering system", "neighborhood zine", "lighthouse model", "card game",
           "harmonica tabs", "pi doorbell", "carved spoons", "constellation chart", "rain-barrel system",
           "pixel tileset", "handmade kite"]
SUFFIXES = ["Okay, here it is: the {o}. It works now.", "Anyway, here it is: the {o}. It's finally done.",
            "So, here it is: the {o}. Finished it last night."]
FEEDBACK = ["feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
            "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
            "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
            "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
            "make sure", "one thing", "stronger", "?"]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
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


def offers_feedback(t):
    t = (t or "").lower(); return any(m in t for m in FEEDBACK)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_model(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.33, 0.5, 0.67, 0.83, 0.92]))
    cand_layers = sorted(set(max(1, int(f * nL)) for f in [0.6, 0.7, 0.78, 0.86, 0.94]))

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim]); groups = [g for _, _, g in stim]

    allL = sorted(set(probe_layers + cand_layers))
    acts = {L: [] for L in allL}
    for text in texts:
        enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in allL:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
    probe = grouped_probe({L: acts[L] for L in probe_layers}, y, groups)
    bow = bow_baseline(texts, y, groups)

    # per-layer steering helpers
    dir_unit, actnorm = {}, {}
    for L in cand_layers:
        X = np.array(acts[L]); w = LogisticRegression(C=1.0, max_iter=3000).fit(X, y).coef_[0]
        dir_unit[L] = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.bfloat16, device="cuda")
        actnorm[L] = float(np.linalg.norm(X, axis=1).mean())
    sub_small = [s for s in stim if s[1] == 0][:4] + [s for s in stim if s[1] == 1][:4]
    sub_dose = [s for s in stim if s[1] == 0][6:18] + [s for s in stim if s[1] == 1][6:18]
    sv = {"v": None, "L": None}

    def mk(L):
        def hook(_m, _i, out):
            if sv["v"] is None or sv["L"] != L:
                return out
            if isinstance(out, tuple):
                h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
            h = out.clone(); h[:, -1, :] += sv["v"]; return h
        return hook
    for L in cand_layers:
        model.model.layers[L - 1].register_forward_hook(mk(L))

    def gen(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=55, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def tally(L, vec, subset, keep=False):
        sv["L"], sv["v"] = L, vec; coh = rh = rc = eb = 0; samples = []
        for text, label, _ in subset:
            rep = gen(text)
            if keep:
                samples.append({"intent": "rec" if label == 0 else "eval",
                                "label": "evaluate" if offers_feedback(rep) else "recognize",
                                "coherent": coherent(rep), "reply": rep[:160]})
            if not coherent(rep):
                continue
            coh += 1; evb = offers_feedback(rep); eb += int(evb)
            if label == 0:
                rc += 1; rh += int(not evb)
        sv["v"] = None
        return {"coh": coh, "rec_hon": rh, "rec_n": rc, "eval_beh": eb}, samples

    base_small, _ = tally(None, None, sub_small)
    sweep = {}
    for L in cand_layers:
        tr, _ = tally(L, -1.0 * actnorm[L] * dir_unit[L], sub_small)
        te, _ = tally(L, 1.0 * actnorm[L] * dir_unit[L], sub_small)
        lift = (tr["rec_hon"] - base_small["rec_hon"])
        gate = te["eval_beh"] > tr["eval_beh"]
        sweep[L] = {"lift": lift, "gate": gate, "tr_rec_hon": tr["rec_hon"], "tr_eval": tr["eval_beh"],
                    "te_eval": te["eval_beh"]}
    # best layer = gate-passing with max honoring lift
    passing = [L for L in cand_layers if sweep[L]["gate"]]
    best = max(passing or cand_layers, key=lambda L: sweep[L]["lift"])

    base_d, audit = tally(None, None, sub_dose, keep=True)
    dose = {"baseline": base_d}
    for a in [0.5, 1.0, 1.5]:
        tr, sm = tally(best, -a * actnorm[best] * dir_unit[best], sub_dose, keep=(a == 1.0))
        te, _ = tally(best, a * actnorm[best] * dir_unit[best], sub_dose)
        dose[f"rec@{a}"] = tr; dose[f"eval@{a}"] = te
        if a == 1.0:
            audit_steered = sm
    return {"model": model_name, "nL": nL, "bow": bow, "probe": {str(L): probe[L] for L in probe_layers},
            "sweep": {str(L): sweep[L] for L in cand_layers}, "best_layer": best,
            "dose": dose, "audit_baseline": audit[:8], "audit_steered": audit_steered[:8]}


@app.local_entrypoint()
def main():
    import json, os, traceback
    debug = os.environ.get("SWEEP_DEBUG") == "1"
    models = MODELS[:1] if debug else MODELS
    out = []
    for m in models:
        try:
            out.append(run_model.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    path = r"C:/Users/alexs/Desktop/recipient-probe/sweep_new.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
