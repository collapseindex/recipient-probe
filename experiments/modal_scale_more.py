"""Scale the last two n=12-24 sub-experiments to n=60 with bootstrap CIs, so every table carries the
headline's statistical weight.

run_dose  (Qwen-3B): the steering dose-response on the FULL 60 recognize items, coefficients 0/0.5/1.0,
    per-item honoring returned for bootstrap CIs.
run_axis2 (Qwen-3B, Llama-8B): the second intent axis (vent vs solve) on 60 situations, per-item vent honoring
    under default and steered-toward-vent, for bootstrap CIs.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_scale_more.py
"""
import modal

app = modal.App("recipient-probe-scale-more")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

DOSE_MODEL, DOSE_LAYER = "Qwen/Qwen2.5-3B-Instruct", 30
AXIS2 = [("Qwen/Qwen2.5-3B-Instruct", 30), ("NousResearch/Meta-Llama-3.1-8B-Instruct", 24)]
MAXTOK = 40  # match the headline behavioral runs (modal_scale/localize2/nested) for cross-table consistency

# --- canonical axis-1 stimuli (mirror of src/recipient_probe/stimuli.py; guarded by tests) ---
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
# --- axis 2: vent vs solve, expanded to 60 situations ---
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
SITUATIONS = [
    "the thing with my landlord", "the situation with my sister", "this deadline I'm behind on",
    "my car breaking down again", "the job offer I'm unsure about", "the fight with my friend",
    "my sleep being wrecked", "this project at work that's slipping", "the budget not adding up",
    "my flaky internet", "the move I keep putting off", "this group chat drama", "my noisy neighbors",
    "the bill I wasn't expecting", "the presentation tomorrow", "my roommate situation",
    "the thing my manager said", "this rash decision I made", "my phone dying constantly",
    "the trip that got cancelled", "the email I sent too fast", "my commute getting worse",
    "the leak under the sink", "this diet I keep failing", "the family dinner coming up",
    "my plants all dying", "the gym membership I never use", "this book I can't finish",
    "my password getting locked", "the gift I forgot to buy", "this meeting that ran long",
    "my back acting up", "the form I filled out wrong", "the recipe that flopped", "my bike with a flat",
    "the class I'm failing", "this subscription I can't cancel", "my flight getting delayed",
    "the wifi at the cafe", "the parking ticket I got", "my overdue library books", "the interview I bombed",
    "this cough that won't quit", "my cluttered garage", "the neighbor's dog barking", "my overflowing inbox",
    "the printer that keeps jamming", "this stain on the carpet", "my thermostat acting up",
    "the group project slacker", "my dwindling savings", "the app that keeps crashing", "this headache all week",
    "my squeaky door", "the delayed package", "my messy closet", "the meeting that got moved",
    "this typo in my report", "my dying houseplant", "the traffic on my street",
]
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


def _load(model_name, layer):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
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

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)
    return model, tok, sv, enc, gen


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run_dose(model_name: str = DOSE_MODEL, layer: int = DOSE_LAYER):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    model, tok, sv, enc, gen = _load(model_name, layer)
    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        acts.append(hs[layer][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8); an = float(np.linalg.norm(acts, axis=1).mean())
    rec = [t for t, lab, _ in stim if lab == 0]

    def per_item(c):
        sv["v"] = None if c == 0 else torch.tensor(-c * an * u, dtype=torch.bfloat16, device="cuda")
        out = []
        for text in rec:
            r = gen(text); out.append(-1 if not coherent(r) else int(not has_any(r, FEEDBACK)))
        sv["v"] = None
        return out
    return {"model": model_name, "layer": layer, "n": len(rec),
            "dose": {str(c): per_item(c) for c in [0.0, 0.5, 1.0]}}


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run_axis2(model_name: str, layer: int):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    model, tok, sv, enc, gen = _load(model_name, layer)
    probe_layers = sorted(set(max(1, int(f * model.config.num_hidden_layers)) for f in [0.33, 0.5, 0.67, 0.83]))
    stim = build_stimuli_vent()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim]); groups = [g for _, _, g in stim]
    acts = {L: [] for L in set(probe_layers) | {layer}}
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        for L in acts:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
    probe = max(grouped_probe_one(acts[L], y, groups) for L in probe_layers)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(np.array(acts[layer]), y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8); an = float(np.linalg.norm(np.array(acts[layer]), axis=1).mean())
    vent = [t for t, lab, _ in stim if lab == 0]  # 60

    def per_item(steer):
        sv["v"] = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda") if steer else None
        out = []
        for text in vent:
            r = gen(text); out.append(-1 if not coherent(r) else int(not has_any(r, SOLUTION)))
        sv["v"] = None
        return out
    return {"model": model_name, "layer": layer, "n": len(vent), "probe": probe,
            "vent_default": per_item(False), "vent_steered": per_item(True)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = {}
    try:
        out["dose"] = run_dose.remote()
    except Exception:
        out["dose"] = {"ERROR": traceback.format_exc()}
    out["axis2"] = []
    for m, L in AXIS2:
        try:
            out["axis2"].append(run_axis2.remote(m, L))
        except Exception:
            out["axis2"].append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_scale_more.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_scale_more.json")
    print("=== SCALE-MORE finished ===")
