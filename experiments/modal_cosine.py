"""Deflationary control: is the intent-probe direction just the feedback-behavior axis?

The intent labels are defined by whether feedback is desired, so an intent direction and a feedback-offering
direction are correlated by construction; our norm-matched/shuffled controls cannot dissociate them. This is a
FREE-ish check (forward passes only, no long generation) that decides it: at each model's steer layer, fit
  w_intent   = logistic(last-token acts, INTENT labels)          [= the probe/steer direction, modal_scale.py]
  w_behavior = logistic(last-token acts, BEHAVIOR labels)        [did the DEFAULT reply offer feedback?]
and report cos(w_intent, w_behavior). High (~0.9+): the handle is a feedback-behavior knob (deflationary reading
confirmed, our hedges are right). Low (~0.3): intent has a component separable from behavior, routing survives.
Random-direction cosines are reported as the ~0 floor.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_cosine.py
"""
import modal

app = modal.App("recipient-probe-cosine")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# the three discard models where the recovery is claimed (paper Table 2), at their steer layers
MODEL_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
               "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
MAXTOK = 40

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
    "feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
    "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
    "problem", "could be", "love to see", "love to read", "happy to help", "here to help", "share it",
    "go ahead", "potential", "notes", "tips", "advice", "refine", "polish", "make sure", "one thing",
    "stronger", "?",
]


def build_stimuli():
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0)); rows.append((f"{E_PHRASES[g]} {suf}", 1))
    return rows


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

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    L = MODEL_LAYER[model_name]

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _ in stim]
    y_intent = np.array([lab for _, lab in stim])

    # last-token activations at the steer layer (identical to modal_scale.py)
    acts = []
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)

    # default reply per stimulus -> behavior label (did it offer feedback?), dropping incoherent
    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    y_behav, keep = [], []
    for i, t in enumerate(texts):
        r = gen(t)
        if not coherent(r):
            continue
        y_behav.append(int(has_any(r, FEEDBACK))); keep.append(i)
    keep = np.array(keep); y_behav = np.array(y_behav)

    def fit_dir(X, y):
        w = LogisticRegression(C=1.0, max_iter=3000).fit(X, y).coef_[0]
        return w / (np.linalg.norm(w) + 1e-8)

    u_intent = fit_dir(acts, y_intent)                    # probe/steer direction (all 120)
    u_intent_k = fit_dir(acts[keep], y_intent[keep])      # intent on the same coherent subset (fair cosine)
    u_behav = fit_dir(acts[keep], y_behav)                # behavior direction on coherent subset

    cos = lambda a, b: float(np.dot(a, b))
    cos_ib = cos(u_intent, u_behav)
    cos_ib_matched = cos(u_intent_k, u_behav)             # both fit on identical subset
    rng = np.random.RandomState(0)
    rand_cos = [abs(cos(u_intent, (lambda v: v / np.linalg.norm(v))(rng.randn(acts.shape[1])))) for _ in range(200)]

    # how much do the labels themselves disagree (the discard events that could separate the axes)
    label_disagree = float(np.mean(y_intent[keep] != y_behav))
    return {"model": model_name, "layer": L, "n_coherent": int(len(keep)),
            "cos_intent_behavior": round(cos_ib, 3),
            "cos_intent_behavior_matched_subset": round(cos_ib_matched, 3),
            "random_cos_mean": round(float(np.mean(rand_cos)), 3),
            "random_cos_max": round(float(np.max(rand_cos)), 3),
            "intent_vs_behavior_label_disagreement": round(label_disagree, 3)}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run_model.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
        with open("sweep_cosine.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"WROTE sweep_cosine.json ({len(out)}/{len(MODEL_LAYER)} models)")
    print("=== COSINE finished ===")
