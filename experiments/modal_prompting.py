"""Power the prompting-vs-routing comparison (Table 7) to n=60 with bootstrap CIs. Qwen-3B, recognize-intent
honoring under: default, a mild attention nudge, an explicit intent prompt, steering, and explicit+steer.
Per-item honoring returned for bootstrap CIs.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_prompting.py
"""
import modal

app = modal.App("recipient-probe-prompting")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL, LAYER, MAXTOK = "Qwen/Qwen2.5-3B-Instruct", 30, 40
MILD = "Before replying, consider what this person is actually looking for from you."
EXPLICIT = "The person messaging you wants their news recognized and celebrated, and to feel heard."

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
def run(model_name: str = MODEL):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
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
    model.model.layers[LAYER - 1].register_forward_hook(hook)

    def enc(text, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        acts.append(hs[LAYER][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8); an = float(np.linalg.norm(acts, axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text, system=None):
        e = enc(text, system)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0]  # 60

    def per_item(system, steer):
        sv["v"] = steer_rec if steer else None
        out = [(-1 if not coherent(r := gen(t, system)) else int(not has_any(r, FEEDBACK))) for t in rec]
        sv["v"] = None
        return out
    return {"model": model_name, "layer": LAYER, "n": len(rec),
            "default": per_item(None, False), "mild": per_item(MILD, False),
            "explicit": per_item(EXPLICIT, False), "steer": per_item(None, True),
            "explicit_steer": per_item(EXPLICIT, True)}


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        out = run.remote()
    except Exception:
        out = {"ERROR": traceback.format_exc()}
    with open("sweep_prompting.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_prompting.json")
    print("=== PROMPTING finished ===")
