"""Specificity and invariance controls for the recipient-probe steering result.

(1) SPECIFICITY  -- defends "the steering handle is the *intent* direction, not a generic
    feedback/verbosity knob you relabeled." A permutation test on the steering effect itself:
    at the model's best layer, the TRUE intent direction's behavior separation
        Delta = feedback(toward-evaluate) - feedback(toward-recognize)
    is compared against N shuffled-LABEL directions (same pipeline, permuted intent labels)
    and N random directions (matched norm). If only the true direction reaches a large Delta,
    the effect tracks the learned intent axis, not the activation geometry or the steer procedure.
    Plus a neutral-prompt invariance check: does +intent steering inject feedback on factual
    prompts that carry no recognize/evaluate intent? (intent-conditional vs global bias.)

(2) SECOND AXIS  -- generalizes the represents->discards->recover chain to a DIFFERENT intent
    contrast: VENT ("just listen, don't fix it") vs SOLVE ("give me concrete advice"), behavior
    = offers_solution. The canonical "don't problem-solve me, just hear me" failure, a different
    surface from feedback. Full pipeline: probe (represents) + BoW control + steer sweep + dose.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_controls.py
"""
import modal

app = modal.App("recipient-probe-controls")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SPEC_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
AXIS2_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
N_SHUF = 8
N_RAND = 8

# ---------------- Axis 1: recognize vs evaluate (unchanged from the ladder) ----------------
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

# ---------------- Axis 2: vent ("just listen") vs solve ("give me advice") ----------------
VENT_PHRASES = [
    "I'm not looking for advice, I just need to vent for a second.",
    "Please don't try to fix it, I just need someone to hear this.",
    "I don't want solutions right now, I just need to say it out loud.",
    "Just let me get this off my chest, I'm not asking you to solve anything.",
    "I know what I have to do, I just need to be heard first.",
    "Don't problem-solve this for me, I just need to feel heard.",
    "I'm not asking how to fix it, I just needed to tell someone.",
    "No fixes please, I just want to be listened to.",
]
SOLVE_PHRASES = [
    "I'm stuck and I really need concrete advice on what to do.",
    "Tell me exactly what steps to take to fix this.",
    "I need a plan, walk me through how to solve it.",
    "Be practical, what should I actually do here?",
    "I want solutions, not sympathy. How do I fix this?",
    "Give me the steps, I need to get this sorted.",
    "What's the move? I need actionable advice.",
    "Help me troubleshoot this, I need a real fix.",
]
SITUATIONS = ["the thing with my landlord", "the situation with my sister", "this deadline I'm behind on",
              "my car breaking down again", "the job offer I'm unsure about", "the fight with my friend",
              "my sleep being wrecked", "this project at work that's slipping", "the budget not adding up",
              "my flaky internet", "the move I keep putting off", "this group chat drama",
              "my noisy neighbors", "the bill I wasn't expecting", "the presentation tomorrow",
              "my roommate situation", "the thing my manager said", "this rash decision I made",
              "my phone dying constantly", "the trip that got cancelled", "the email I sent too fast",
              "my commute getting worse", "the leak under the sink", "this diet I keep failing",
              "the family dinner coming up", "my plants all dying", "the gym membership I never use",
              "this book I can't finish", "my password getting locked", "the gift I forgot to buy",
              "this meeting that ran long", "my back acting up", "the form I filled out wrong",
              "the recipe that flopped", "my bike with a flat", "the class I'm failing",
              "this subscription I can't cancel", "my flight getting delayed", "the wifi at the cafe",
              "the parking ticket I got"]
VENT_SUFFIXES = ["Anyway, here it is: {s}. It's been a lot.", "So that's where I'm at: {s}. It's been a lot.",
                 "That's the situation: {s}. It's been a lot."]
SOLUTION = ["you should", "you could", "you might", "you can ", "you need to", "you have to", "try ",
            "have you tried", "have you considered", "i'd suggest", "i suggest", "i'd recommend", "i recommend",
            "what you can do", "the first step", "first, ", "one option", "here's what", "why don't you",
            "why not ", "consider ", "the best way", "a few things you", "step ", "make a plan", "to fix"]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g))
    return rows


def build_stimuli_vent():
    rows = []
    for i, s in enumerate(SITUATIONS):
        g = i % len(VENT_PHRASES); suf = VENT_SUFFIXES[i % len(VENT_SUFFIXES)].format(s=s)
        rows.append((f"{VENT_PHRASES[g]} {suf}", 0, g)); rows.append((f"{SOLVE_PHRASES[g]} {suf}", 1, g))
    return rows


NEUTRAL = ["What's the capital of France?", "How many days are in a leap year?", "Convert 5 kilometers to miles.",
           "What's the boiling point of water at sea level?", "Name three primary colors.",
           "What year was the first moon landing?", "What is 17 times 23?",
           "Define photosynthesis in one sentence.", "What's the largest planet in our solar system?",
           "How many continents are there?", "What's the chemical symbol for gold?",
           "List the days of the week."]


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


# shared model-side scaffold: load, embed cand layers, register hooks, return helpers
def _load_and_scaffold(model_name, stim, cand_layers_fracs=(0.6, 0.7, 0.78, 0.86, 0.94)):
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    cand_layers = sorted(set(max(1, int(f * nL)) for f in cand_layers_fracs))
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

    def embed(texts, layers):
        acts = {L: [] for L in layers}
        for text in texts:
            enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
        return acts

    def gen(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=55, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return model, tok, nL, cand_layers, sv, embed, gen


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_specificity(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression

    stim = build_stimuli()
    model, tok, nL, cand_layers, sv, embed, gen = _load_and_scaffold(model_name, stim)
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim])
    acts = embed(texts, cand_layers)
    actnorm = {L: float(np.linalg.norm(np.array(acts[L]), axis=1).mean()) for L in cand_layers}

    def vec_from_w(L, w, scale):
        u = w / (np.linalg.norm(w) + 1e-8)
        return torch.tensor(scale * actnorm[L] * u, dtype=torch.bfloat16, device="cuda")

    def behavior_sep(L, w, subset):
        # Delta = feedback(toward +dir) - feedback(toward -dir), over subset
        out = {}
        for sign, key in [(1.0, "pos"), (-1.0, "neg")]:
            sv["L"], sv["v"] = L, vec_from_w(L, w, sign); beh = coh = 0
            for text, _, _ in subset:
                rep = gen(text)
                if not coherent(rep):
                    continue
                coh += 1; beh += int(has_any(rep, FEEDBACK))
            out[key] = beh; out[key + "_coh"] = coh
        sv["v"] = None
        return out["pos"] - out["neg"], out

    # find best layer by the same gate-sweep used in the ladder (true direction)
    sub = [s for s in stim if s[1] == 0][:6] + [s for s in stim if s[1] == 1][:6]
    sweep = {}
    for L in cand_layers:
        w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[L]), y).coef_[0]
        d, _ = behavior_sep(L, w, sub)
        sweep[L] = d
    best = max(cand_layers, key=lambda L: sweep[L])

    # TRUE direction effect at best layer
    w_true = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[best]), y).coef_[0]
    delta_true, det_true = behavior_sep(best, w_true, sub)

    # SHUFFLED-LABEL directions: same pipeline, permuted intent labels
    rng = np.random.RandomState(0)
    shuf_deltas = []
    for _ in range(N_SHUF):
        ysh = rng.permutation(y)
        wsh = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[best]), ysh).coef_[0]
        d, _ = behavior_sep(best, wsh, sub); shuf_deltas.append(int(d))

    # RANDOM directions, matched norm
    rand_deltas = []
    for i in range(N_RAND):
        wr = np.random.RandomState(100 + i).randn(acts[best][0].shape[0])
        d, _ = behavior_sep(best, wr, sub); rand_deltas.append(int(d))

    # permutation p-values (one-sided: how often a broken dir reaches the true effect)
    p_shuf = (sum(1 for d in shuf_deltas if d >= delta_true) + 1) / (N_SHUF + 1)
    p_rand = (sum(1 for d in rand_deltas if d >= delta_true) + 1) / (N_RAND + 1)

    # NEUTRAL-prompt invariance: does +intent steering inject feedback where there is no intent?
    def neutral_feedback_rate(scale):
        sv["L"], sv["v"] = best, (None if scale == 0 else vec_from_w(best, w_true, scale))
        if scale == 0:
            sv["v"] = None
        fb = coh = 0
        for q in NEUTRAL:
            rep = gen(q)
            if not coherent(rep):
                continue
            coh += 1; fb += int(has_any(rep, FEEDBACK))
        sv["v"] = None
        return {"fb": fb, "coh": coh}
    neutral = {"base": neutral_feedback_rate(0.0), "toward_eval": neutral_feedback_rate(1.0),
               "toward_rec": neutral_feedback_rate(-1.0)}

    return {"model": model_name, "nL": nL, "best_layer": best,
            "sweep": {str(L): sweep[L] for L in cand_layers},
            "delta_true": int(delta_true), "true_detail": det_true,
            "shuf_deltas": shuf_deltas, "rand_deltas": rand_deltas,
            "p_shuf": round(p_shuf, 4), "p_rand": round(p_rand, 4),
            "n_items_per_side": 6, "neutral": neutral}


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def run_second_axis(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression

    stim = build_stimuli_vent()
    model, tok, nL, cand_layers, sv, embed, gen = _load_and_scaffold(model_name, stim)
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.33, 0.5, 0.67, 0.83, 0.92]))
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim]); groups = [g for _, _, g in stim]

    allL = sorted(set(probe_layers + cand_layers))
    acts = embed(texts, allL)
    probe = grouped_probe({L: acts[L] for L in probe_layers}, y, groups)
    bow = bow_baseline(texts, y, groups)

    dir_unit, actnorm = {}, {}
    for L in cand_layers:
        X = np.array(acts[L]); w = LogisticRegression(C=1.0, max_iter=3000).fit(X, y).coef_[0]
        dir_unit[L] = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.bfloat16, device="cuda")
        actnorm[L] = float(np.linalg.norm(X, axis=1).mean())
    sub_small = [s for s in stim if s[1] == 0][:4] + [s for s in stim if s[1] == 1][:4]
    sub_dose = [s for s in stim if s[1] == 0][6:18] + [s for s in stim if s[1] == 1][6:18]

    def tally(L, vec, subset, keep=False):
        sv["L"], sv["v"] = L, vec; coh = vh = vc = sb = 0; samples = []
        for text, label, _ in subset:
            rep = gen(text)
            if keep:
                samples.append({"intent": "vent" if label == 0 else "solve",
                                "label": "solve" if has_any(rep, SOLUTION) else "vent",
                                "coherent": coherent(rep), "reply": rep[:160]})
            if not coherent(rep):
                continue
            coh += 1; sol = has_any(rep, SOLUTION); sb += int(sol)
            if label == 0:
                vc += 1; vh += int(not sol)
        sv["v"] = None
        return {"coh": coh, "vent_hon": vh, "vent_n": vc, "solve_beh": sb}, samples

    base_small, _ = tally(None, None, sub_small)
    sweep = {}
    for L in cand_layers:
        tr, _ = tally(L, -1.0 * actnorm[L] * dir_unit[L], sub_small)
        te, _ = tally(L, 1.0 * actnorm[L] * dir_unit[L], sub_small)
        sweep[L] = {"lift": tr["vent_hon"] - base_small["vent_hon"], "gate": te["solve_beh"] > tr["solve_beh"],
                    "tr_vent_hon": tr["vent_hon"], "tr_solve": tr["solve_beh"], "te_solve": te["solve_beh"]}
    passing = [L for L in cand_layers if sweep[L]["gate"]]
    best = max(passing or cand_layers, key=lambda L: sweep[L]["lift"])

    base_d, audit = tally(None, None, sub_dose, keep=True)
    dose = {"baseline": base_d}; audit_steered = []
    for a in [0.5, 1.0, 1.5]:
        tr, sm = tally(best, -a * actnorm[best] * dir_unit[best], sub_dose, keep=(a == 1.0))
        te, _ = tally(best, a * actnorm[best] * dir_unit[best], sub_dose)
        dose[f"vent@{a}"] = tr; dose[f"solve@{a}"] = te
        if a == 1.0:
            audit_steered = sm
    return {"model": model_name, "nL": nL, "bow": bow, "probe": {str(L): probe[L] for L in probe_layers},
            "sweep": {str(L): sweep[L] for L in cand_layers}, "best_layer": best, "dose": dose,
            "audit_baseline": audit[:8], "audit_steered": audit_steered[:8]}


@app.local_entrypoint()
def main():
    import json, traceback
    spec, axis2 = [], []
    for m in SPEC_MODELS:
        try:
            spec.append(run_specificity.remote(m))
        except Exception:
            spec.append({"model": m, "ERROR": traceback.format_exc()})
    for m in AXIS2_MODELS:
        try:
            axis2.append(run_second_axis.remote(m))
        except Exception:
            axis2.append({"model": m, "ERROR": traceback.format_exc()})
    path = "sweep_controls.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"specificity": spec, "second_axis": axis2}, f, indent=1)
    print(f"WROTE {path}")
    print("=== CONTROLS finished ===")
