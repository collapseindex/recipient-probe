"""Build a BLIND human-rating validation set for the 'honoring' behavior measure.

Generates Qwen-3B replies to recognize-intent shares under default and steered-toward-recognize, plus a few
evaluate-intent shares (default) as attention anchors (the sender there DID ask for critique, so a good rater
marks 'offers feedback = yes'). Each reply is auto-labeled by the lexicon and by the embedding classifier.
The local entrypoint writes TWO files: a shuffled, de-labeled CSV for a human to score, and a separate key
CSV (condition, intent, auto-labels) for computing human-vs-classifier agreement afterward. Nothing the
annotator sees reveals the condition or the automated label.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_ratingset.py
"""
import modal

app = modal.App("recipient-probe-ratingset")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                      "sentencepiece", "sentence-transformers"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 30
N_REC = 24   # recognize items -> default + steered
N_EVAL = 12  # evaluate items -> default only (anchors)
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
FB_PROTO = ["Here are some suggestions to improve it.", "You could refine a few things.",
            "One issue is the pacing, consider changing it.", "Have you thought about revising the ending?",
            "I'd recommend polishing the structure and tightening it up."]
ACK_PROTO = ["Congratulations, that's wonderful!", "I'm so happy for you, what a milestone.",
             "That's a real accomplishment, you should be proud.", "Wow, that's amazing, well done.",
             "I'm really glad you shared this with me."]


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
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
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

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])
    acts = []
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        acts.append(hs[LAYER][0, -1, :].float().cpu().numpy())
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    u = w / (np.linalg.norm(w) + 1e-8)
    an = float(np.linalg.norm(acts, axis=1).mean())
    steer_rec = torch.tensor(-1.0 * an * u, dtype=torch.bfloat16, device="cuda")

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    rec = [t for t, lab, _ in stim if lab == 0][:N_REC]
    ev = [t for t, lab, _ in stim if lab == 1][:N_EVAL]

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    fb_c = emb.encode(FB_PROTO, normalize_embeddings=True).mean(0)
    ack_c = emb.encode(ACK_PROTO, normalize_embeddings=True).mean(0)

    def row(msg, reply, condition, intent):
        e = emb.encode([reply], normalize_embeddings=True)[0]
        return {"user_message": msg, "reply": reply, "condition": condition, "intent": intent,
                "coherent": coherent(reply),
                "lex_offers_feedback": bool(has_any(reply, FEEDBACK)),
                "emb_offers_feedback": bool(float(e @ fb_c) > float(e @ ack_c))}

    rows = []
    for m in rec:
        sv["v"] = None; rows.append(row(m, gen(m), "default", "recognize"))
        sv["v"] = steer_rec; rows.append(row(m, gen(m), "steered", "recognize"))
        sv["v"] = None
    for m in ev:
        sv["v"] = None; rows.append(row(m, gen(m), "default", "evaluate"))
    return {"model": model_name, "rows": rows}


@app.local_entrypoint()
def main():
    import json, csv, traceback
    try:
        out = run.remote()
    except Exception:
        out = {"ERROR": traceback.format_exc()}
    with open("sweep_ratingset.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    if "ERROR" in out:
        print(out["ERROR"][:400]); return
    rows = [r for r in out["rows"] if r["coherent"]]
    # deterministic shuffle (no RNG dependency): interleave by a fixed hash of the reply
    rows.sort(key=lambda r: (len(r["reply"]) * 7 + sum(ord(c) for c in r["reply"][:16])) % 997)
    with open("ratings_blind.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["id", "sender_message", "assistant_reply", "offers_feedback_y_n"])
        for i, r in enumerate(rows):
            wri.writerow([i, r["user_message"], r["reply"], ""])
    with open("ratings_key.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["id", "condition", "intent", "lex_offers_feedback", "emb_offers_feedback"])
        for i, r in enumerate(rows):
            wri.writerow([i, r["condition"], r["intent"], int(r["lex_offers_feedback"]), int(r["emb_offers_feedback"])])
    print(f"WROTE ratings_blind.csv ({len(rows)} rows) + ratings_key.csv")
    print("=== RATINGSET finished ===")
