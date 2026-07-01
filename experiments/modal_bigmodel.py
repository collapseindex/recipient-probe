"""Scale confrontation: does the discard survive past 14B, and does IMPLICIT-intent discard persist at scale
even if explicit does not? Qwen2.5-32B on an H100. For BOTH the explicit and the implicit stimulus sets we
report probe accuracy (represents), a bag-of-words baseline, and default recognize-intent honoring (discard).
No steering (that needs a per-model sweep); this isolates represent + discard at scale.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_bigmodel.py
"""
import modal

app = modal.App("recipient-probe-bigmodel")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-32B-Instruct"
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
R_IMPLICIT = [
    "It's a birthday present for my mom.", "I'm keeping this one for myself.",
    "It's going up on the wall at home.", "It's the first thing I've ever made like this.",
    "I made it for my kid.", "It's a little gift for a friend going through a hard time.",
    "It's going on the mantel with the family photos.", "I made it in memory of my grandpa.",
]
E_IMPLICIT = [
    "It's going in my portfolio.", "I'm submitting it to the juried show next week.",
    "It's a work sample for a job I'm applying to.", "I'm showing it to the client on Monday.",
    "It's the prototype before I make fifty more.", "It's going in front of my thesis committee.",
    "I'm entering it in the competition on Friday.", "It goes live on the store page tomorrow.",
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


def build(rphr, ephr):
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(rphr); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{rphr[g]} {suf}", 0, g)); rows.append((f"{ephr[g]} {suf}", 1, g))
    return rows


def grouped_probe(X, y, groups, n_perm=6):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y); groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=len(set(groups)))
    pipe = lambda: make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                                 LogisticRegression(C=1.0, max_iter=2000))
    return round(float(cross_val_score(pipe(), np.asarray(X), y, groups=groups, cv=gkf).mean()), 3)


def bow(texts, y, groups):
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


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=5400)
def run(model_name: str = MODEL):
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()
    nL = model.config.num_hidden_layers
    probe_layers = sorted(set(max(1, int(f * nL)) for f in [0.5, 0.67, 0.83]))
    dev = next(model.parameters()).device

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(dev)

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    out = {"model": model_name, "nL": nL}
    for name, (rphr, ephr) in {"explicit": (R_PHRASES, E_PHRASES), "implicit": (R_IMPLICIT, E_IMPLICIT)}.items():
        stim = build(rphr, ephr)
        texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim]); groups = [g for _, _, g in stim]
        acts = {L: [] for L in probe_layers}
        for text in texts:
            with torch.no_grad():
                hs = model(**enc(text), output_hidden_states=True).hidden_states
            for L in probe_layers:
                acts[L].append(hs[L][0, -1, :].float().cpu().numpy())
        pk = max(grouped_probe(acts[L], y, groups) for L in probe_layers)
        rec = [t for t, lab, _ in stim if lab == 0]
        hon = [(-1 if not coherent(r := gen(t)) else int(not has_any(r, FEEDBACK))) for t in rec]
        out[name] = {"probe_peak": pk, "bow": bow(texts, y, groups), "default_honoring": hon,
                     "audit": [{"msg": rec[i][:70], "reply": gen(rec[i])[:120]} for i in range(3)]}
    return out


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        out = run.remote()
    except Exception:
        out = {"ERROR": traceback.format_exc()}
    with open("sweep_bigmodel.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_bigmodel.json")
    print("=== BIGMODEL finished ===")
