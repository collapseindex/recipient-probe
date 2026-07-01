"""Scale the behavioral evidence: discard + recover at full n, with variance, across models.

Addresses the two main reviewer concerns (small-n single-run; single model). For each of four models at
its ladder steer layer, we measure recognize-intent honoring on the FULL 60-item recognize set under default
vs steering-toward-recognize, returning per-item honoring so bootstrap 95% CIs can be put on the rates. For
Qwen-3B we also run K=3 sampled-decoding seeds (temperature 0.7) to show the effect is not a greedy artifact.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_scale.py
"""
import modal

app = modal.App("recipient-probe-scale")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
               "Qwen/Qwen2.5-14B-Instruct": 41, "mistralai/Mistral-7B-Instruct-v0.3": 22,
               "microsoft/Phi-3.5-mini-instruct": 22, "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
SAMPLED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SEEDS = [0, 1, 2]
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
def run_model(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    L = MODEL_LAYER[model_name]
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[L - 1].register_forward_hook(hook)

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    an = float(np.linalg.norm(acts, axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text, sample=False, seed=0):
        e = enc(text)
        if sample:
            torch.manual_seed(seed)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=sample,
                               temperature=(0.7 if sample else None), top_p=(0.9 if sample else None),
                               pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0]      # full 60 recognize items
    ev = [t for t, lab, _ in stim if lab == 1]

    # greedy per-item honoring (1 = honored / no unsolicited feedback), -1 = incoherent (dropped)
    def per_item(vec, sample=False, seed=0):
        sv["v"] = vec; out = []
        for text in rec:
            rep = gen(text, sample=sample, seed=seed)
            out.append(-1 if not coherent(rep) else int(not has_any(rep, FEEDBACK)))
        sv["v"] = None
        return out

    greedy_default = per_item(None)
    greedy_steered = per_item(steer_rec)
    # eval sanity: feedback rate on evaluate-intent (default)
    sv["v"] = None
    eval_fb = [int(has_any(gen(t), FEEDBACK)) for t in ev if coherent(gen(t))]

    result = {"model": model_name, "layer": L, "n_rec": len(rec),
              "greedy_default": greedy_default, "greedy_steered": greedy_steered,
              "eval_fb_rate": round(sum(eval_fb) / max(len(eval_fb), 1), 3)}

    if model_name == SAMPLED_MODEL:
        sampled = {"default": [], "steered": []}
        for s in SEEDS:
            sampled["default"].append(per_item(None, sample=True, seed=s))
            sampled["steered"].append(per_item(steer_rec, sample=True, seed=s))
        result["sampled"] = sampled
    return result


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run_model.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    path = "sweep_scale.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
    print("=== SCALE finished ===")
