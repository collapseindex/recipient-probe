"""Specificity re-run, higher power. Defends: the steering handle is the *intent* direction,
not a generic feedback/verbosity knob.

Fixes the underpowered first pass (6 items/side, 8 perms, a re-swept layer -> on Qwen-3B a
shuffled-label direction beat the true one, p=0.22). This version:
  - PINS the steer layer to the ladder's validated per-model layer (Qwen-3B L30, Qwen-7B L16),
  - uses 12 items/side (24 per direction eval),
  - runs 12 shuffled-LABEL directions (same pipeline, permuted intent labels) + 8 random directions,
  - scores the behavior separation S = feedback(toward-evaluate) - feedback(toward-recognize),
    one-sided permutation p for shuffled (principled sign from the shuffled fit), two-sided |S| for random.

If only the true direction reaches a large S, the effect tracks the learned intent axis, not the
activation geometry or the steering procedure.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_specificity2.py
"""
import modal

app = modal.App("recipient-probe-spec2")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

LADDER_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16}
ITEMS = 12
N_PERM = 12
N_RAND = 48
MAXTOK = 45

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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    L = LADDER_LAYER[model_name]
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[L - 1].register_forward_hook(hook)

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim])

    def encode(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**encode(text), output_hidden_states=True).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    actnorm = float(np.linalg.norm(acts, axis=1).mean())

    def gen(text):
        enc = encode(text)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    subset = [s for s in stim if s[1] == 0][:ITEMS] + [s for s in stim if s[1] == 1][:ITEMS]

    def sep(w):
        # S = feedback(+dir) - feedback(-dir), over the fixed subset. +dir = "toward label 1" per the fit.
        u = w / (np.linalg.norm(w) + 1e-8); res = {}
        for sign in (1.0, -1.0):
            sv["v"] = torch.tensor(sign * actnorm * u, dtype=torch.bfloat16, device="cuda"); fb = coh = 0
            for text, _, _ in subset:
                rep = gen(text)
                if not coherent(rep):
                    continue
                coh += 1; fb += int(has_any(rep, FEEDBACK))
            res[sign] = (fb, coh)
        sv["v"] = None
        return res[1.0][0] - res[-1.0][0], {"pos": res[1.0], "neg": res[-1.0]}

    w_true = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    S_true, det_true = sep(w_true)

    rng = np.random.RandomState(0)
    shuf = []
    for _ in range(N_PERM):
        ws = LogisticRegression(C=1.0, max_iter=3000).fit(acts, rng.permutation(y)).coef_[0]
        s, _ = sep(ws); shuf.append(int(s))
    rand = []
    for i in range(N_RAND):
        wr = np.random.RandomState(200 + i).randn(acts.shape[1])
        s, _ = sep(wr); rand.append(int(s))

    p_shuf = (sum(1 for s in shuf if s >= S_true) + 1) / (N_PERM + 1)
    p_rand = (sum(1 for s in rand if abs(s) >= S_true) + 1) / (N_RAND + 1)

    return {"model": model_name, "layer": L, "items_per_side": ITEMS, "S_true": int(S_true),
            "true_detail": det_true, "shuf": shuf, "rand": rand,
            "p_shuf": round(p_shuf, 4), "p_rand": round(p_rand, 4)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    path = "sweep_spec2.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
    print("=== SPEC2 finished ===")
