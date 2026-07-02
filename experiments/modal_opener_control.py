"""DEFLATIONARY CONTROL for the readout-lag claim (addresses the reviewer's central objection).

The concern: late-layer steering with the intent direction might not be "routing a discarded intent," it might
just be biasing the opener-token distribution toward acknowledgment openers (near logit-steering). If so, then
steering the (ACK-opener minus FB-opener) UNEMBEDDING direction directly should reproduce the recovery, and the
intent effect would live only in the first token.

This runs three things per discard model, at its VALIDATED ladder steer layer, same hook and matched norm as
modal_specificity2.py, so it is apples-to-apples with the headline steering result:
  (1) S_true   = feedback(+intent) - feedback(-intent)                  -- the learned intent direction.
  (2) S_opener = feedback(+opener) - feedback(-opener), opener_dir =     -- Fable's direct control: steer the
        unit(mean unembed(ACK openers) - mean unembed(FB openers)),         opener-unembedding direction itself.
        matched to the same activation norm.
  (3) S_true_rest = the intent-direction separation computed on the reply with its FIRST SENTENCE REMOVED
        -- does intent steering reorient the body of the response, or only swap the opening move?
Also reports cos(intent_dir_at_L, opener_dir) at the ACTUAL steer layer (the pre-registered geometry measured
this at the final layer only; this closes that gap).

Reading: if S_opener << S_true, the effect is NOT explained by opener biasing. If S_true_rest is also large,
intent steering changes the body, not just the first token. Either result refutes the deflationary account;
we report whatever comes back.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_opener_control.py
"""
import modal

app = modal.App("recipient-probe-opener-control")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# validated ladder steer layers (same as modal_specificity2.py / Table general); all three DISCARD models.
LADDER_LAYER = {"Qwen/Qwen2.5-3B-Instruct": 30, "Qwen/Qwen2.5-7B-Instruct": 16,
                "NousResearch/Meta-Llama-3.1-8B-Instruct": 19}
ITEMS = 12
N_RAND = 12
MAXTOK = 100  # longer than the 45-token specificity budget on purpose: the S_true_rest body test needs a body
              # to score (at 45 tokens the reply is often one sentence, so removing it leaves nothing).

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
ACK_OPEN = ["Congratulations", "Congrats", "That", "Wow", "Amazing", "Nice", "Love", "Happy", "Wonderful",
            "Beautiful", "Aww", "Yay", "Great", "Fantastic", "Incredible"]
FB_OPEN = ["Here", "One", "Have", "Consider", "You", "First", "To", "Overall", "While", "A", "The", "If"]


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


def drop_first_sentence(t):
    # everything after the first sentence-final .!? ; empty if the reply is a single sentence.
    for k, ch in enumerate(t or ""):
        if ch in ".!?":
            return t[k + 1:].strip()
    return ""


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

    def sep(w, on_rest=False):
        # S = feedback(+dir) - feedback(-dir). on_rest scores the reply with its first sentence removed.
        u = w / (np.linalg.norm(w) + 1e-8); res = {}
        for sign in (1.0, -1.0):
            sv["v"] = torch.tensor(sign * actnorm * u, dtype=torch.bfloat16, device="cuda"); fb = coh = 0
            for text, _, _ in subset:
                rep = gen(text)
                if not coherent(rep):
                    continue
                coh += 1
                scored = drop_first_sentence(rep) if on_rest else rep
                fb += int(has_any(scored, FEEDBACK))
            res[sign] = (fb, coh)
        sv["v"] = None
        return res[1.0][0] - res[-1.0][0], {"pos": res[1.0], "neg": res[-1.0]}

    # (1) intent direction
    w_true = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    S_true, det_true = sep(w_true)
    # (3) intent direction, scored on the body (first sentence removed)
    S_true_rest, det_rest = sep(w_true, on_rest=True)

    # (2) opener-unembedding direction (Fable's direct deflationary control)
    def opener_ids(words):
        ids = []
        for wd in words:
            tt = tok(" " + wd, add_special_tokens=False).input_ids
            if tt:
                ids.append(tt[0])
        return ids
    W = model.lm_head.weight.detach().float().cpu().numpy()  # [vocab, d]
    opener_dir = W[opener_ids(ACK_OPEN)].mean(0) - W[opener_ids(FB_OPEN)].mean(0)
    S_opener, det_opener = sep(opener_dir)

    # cosine intent_dir vs opener_dir AT THE STEER LAYER (geometry.py measured this at final layer only)
    ui = w_true / (np.linalg.norm(w_true) + 1e-9); uo = opener_dir / (np.linalg.norm(opener_dir) + 1e-9)
    cos_L = float(abs(np.dot(ui, uo)))

    # light random null for norm context (12 dirs, |S|)
    rand = []
    for i in range(N_RAND):
        wr = np.random.RandomState(300 + i).randn(acts.shape[1])
        s, _ = sep(wr); rand.append(int(s))

    return {"model": model_name, "layer": L, "items_per_side": ITEMS,
            "S_true": int(S_true), "true_detail": det_true,
            "S_opener": int(S_opener), "opener_detail": det_opener,
            "S_true_rest": int(S_true_rest), "rest_detail": det_rest,
            "cos_intent_opener_at_L": round(cos_L, 3),
            "rand": rand, "rand_maxabs": int(max(abs(s) for s in rand))}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
              "NousResearch/Meta-Llama-3.1-8B-Instruct"]:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_opener_control.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_opener_control.json")
    print("=== OPENER CONTROL finished ===")
