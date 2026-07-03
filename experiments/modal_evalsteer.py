"""Control: dissociate intent-routing from generic feedback-suppression.

Concern: recognize-steering might recover honoring simply by suppressing feedback-adjacent tokens generically,
in which case it would also kill feedback that was EXPLICITLY requested. We test the dissociation directly with
the SAME recognize-steer vector used for the main recovery (fit on the last-token logistic direction at the
model's steer layer, added as -1*an*u, identical to modal_scale.py), applied to both classes:

  recognize items:  honoring default -> steered   (should RISE: unsolicited feedback removed)
  evaluate  items:  feedback default -> steered   (should SURVIVE: requested critique preserved)

A steer that raises recognize-honoring while leaving evaluate-feedback intact is intent-routing, not a generic
"stop giving feedback" knob. Run on the three models where the recovery is clean (Table 3, non-overlapping).

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_evalsteer.py
"""
import modal

app = modal.App("recipient-probe-evalsteer")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# The three models with a real discard and non-overlapping recovery (paper Table 3), at their steer layers.
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
    # identical construction to modal_scale.py: steer TOWARD recognize (label 0) at -1*an*u
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    an = float(np.linalg.norm(acts, axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0]
    ev = [t for t, lab, _ in stim if lab == 1]

    def gen_all(items, vec):
        sv["v"] = vec
        reps = [gen(t) for t in items]
        sv["v"] = None
        return reps

    # recognize items: honoring = not offering feedback (1 honored, 0 feedback, -1 incoherent)
    rec_def = gen_all(rec, None); rec_ste = gen_all(rec, steer_rec)
    # evaluate items: feedback offered = 1 (the requested critique), 0 = withheld, -1 incoherent
    ev_def = gen_all(ev, None); ev_ste = gen_all(ev, steer_rec)

    def honor(reps):   # recognize honoring
        return [(-1 if not coherent(r) else int(not has_any(r, FEEDBACK))) for r in reps]

    def fb(reps):      # evaluate feedback presence
        return [(-1 if not coherent(r) else int(has_any(r, FEEDBACK))) for r in reps]

    def rate(labs):
        v = [x for x in labs if x >= 0]
        return round(sum(v) / max(len(v), 1), 3)

    rh_def, rh_ste = honor(rec_def), honor(rec_ste)
    ef_def, ef_ste = fb(ev_def), fb(ev_ste)
    return {"model": model_name, "layer": L, "n_per_class": len(rec),
            "rec_honor_default": rh_def, "rec_honor_steered": rh_ste,
            "eval_fb_default": ef_def, "eval_fb_steered": ef_ste,
            "rec_honor_default_rate": rate(rh_def), "rec_honor_steered_rate": rate(rh_ste),
            "eval_fb_default_rate": rate(ef_def), "eval_fb_steered_rate": rate(ef_ste),
            "reps_eval_steered": ev_ste}   # saved for a spot-check that critique survives verbatim


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODEL_LAYER:
        try:
            out.append(run_model.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
        with open("sweep_evalsteer.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"WROTE sweep_evalsteer.json ({len(out)}/{len(MODEL_LAYER)} models)")
    print("=== EVALSTEER finished ===")
