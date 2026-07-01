"""Ceiling-model geometry (pre-registered in PREREG.md, committed 8d1de89 before this ran).

Primary: M = |cos(intent_dir, readout_dir)| at the final layer, where intent_dir is the logistic recognize-vs-
evaluate direction on final-layer activations and readout_dir is the fixed (ACK openers - FB openers)
unembedding difference. Hypothesis: ceiling models (honor by default) have larger M than discard models.
Secondary: readout_dir_behavioral = direction separating honored-vs-discarded default replies (final layer),
where >=8 of each class exist. Both reported; no post-hoc selection.
"""
import modal

app = modal.App("recipient-probe-geometry")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct",
          "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3.5-mini-instruct"]
TIER = {"Qwen/Qwen2.5-3B-Instruct": "discard", "Qwen/Qwen2.5-7B-Instruct": "discard",
        "NousResearch/Meta-Llama-3.1-8B-Instruct": "discard", "Qwen/Qwen2.5-14B-Instruct": "ceiling",
        "mistralai/Mistral-7B-Instruct-v0.3": "ceiling", "microsoft/Phi-3.5-mini-instruct": "ceiling"}
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


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run(model_name: str):
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    fin = []
    for t in texts:
        with torch.no_grad():
            hs = model(**enc(t), output_hidden_states=True).hidden_states
        fin.append(hs[nL][0, -1, :].float().cpu().numpy())
    fin = np.array(fin)
    intent_dir = LogisticRegression(C=1.0, max_iter=3000).fit(fin, y).coef_[0]
    intent_dir = intent_dir / (np.linalg.norm(intent_dir) + 1e-9)

    # readout_dir from unembedding (final norm applied so it lives in the residual/hidden space)
    def opener_ids(words):
        ids = []
        for w in words:
            tt = tok(" " + w, add_special_tokens=False).input_ids
            if tt:
                ids.append(tt[0])
        return ids
    W = model.lm_head.weight.detach().float().cpu().numpy()  # [vocab, d]
    ack = W[opener_ids(ACK_OPEN)].mean(0); fb = W[opener_ids(FB_OPEN)].mean(0)
    readout_dir = ack - fb; readout_dir = readout_dir / (np.linalg.norm(readout_dir) + 1e-9)
    M_primary = float(abs(np.dot(intent_dir, readout_dir)))

    # secondary: behavioral readout direction (honored vs discarded default replies, recognize items)
    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)
    rec_idx = [k for k, (_, lab, _) in enumerate(stim) if lab == 0]
    beh = []  # 1 honored, 0 discarded
    for k in rec_idx:
        r = gen(texts[k]); beh.append(None if not coherent(r) else int(not has_any(r, FEEDBACK)))
    Xr = np.array([fin[k] for k, b in zip(rec_idx, beh) if b is not None])
    yb = np.array([b for b in beh if b is not None])
    M_beh = None
    if (yb == 0).sum() >= 8 and (yb == 1).sum() >= 8:
        wb = LogisticRegression(C=1.0, max_iter=3000).fit(Xr, yb).coef_[0]
        wb = wb / (np.linalg.norm(wb) + 1e-9)
        M_beh = float(abs(np.dot(intent_dir, wb)))
    return {"model": model_name, "tier": TIER[model_name], "nL": nL,
            "M_primary": round(M_primary, 3),
            "M_behavioral": (round(M_beh, 3) if M_beh is not None else None),
            "default_honored": int((yb == 1).sum()), "default_discarded": int((yb == 0).sum())}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_geometry.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_geometry.json")
    print("=== GEOMETRY finished ===")
