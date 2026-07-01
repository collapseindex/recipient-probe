"""Hardening: nested dev/test steering selection + leave-object-out probe.

(A) NESTED SELECTION -- answers "you found the layer that works." Objects are split into a DEV half and a
    TEST half. The steering direction is fit on DEV stimuli only; the steer layer and coefficient are selected
    by DEV recognize-honoring recovery; the selected (layer, coef, dev-fit direction) is then applied to the
    HELD-OUT TEST objects, and recovery is reported there with a bootstrap CI. No item used to select is used
    to measure.

(B) LEAVE-OBJECT-OUT PROBE -- complements leave-phrasing-out. GroupKFold by OBJECT: train on some objects,
    test on held-out objects. Since an object appears identically in its recognize and evaluate stimulus,
    object identity cannot carry the label; high held-out-object accuracy confirms the probe reads intent,
    not object-topic.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_nested.py
"""
import modal

app = modal.App("recipient-probe-nested")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
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


def build_stimuli(object_idxs=None):
    idxs = range(len(OBJECTS)) if object_idxs is None else object_idxs
    rows = []
    for i in idxs:
        o = OBJECTS[i]; g = i % len(R_PHRASES); suf = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suf}", 0, g, i)); rows.append((f"{E_PHRASES[g]} {suf}", 1, g, i))
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
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    cand = sorted(set(max(1, int(f * nL)) for f in [0.5, 0.6, 0.7, 0.78, 0.86, 0.94]))
    sv = {"v": None, "L": None}

    def mk(L):
        def hook(_m, _i, out):
            if sv["v"] is None or sv["L"] != L:
                return out
            if isinstance(out, tuple):
                h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
            h = out.clone(); h[:, -1, :] += sv["v"]; return h
        return hook
    for L in cand:
        model.model.layers[L - 1].register_forward_hook(mk(L))

    def enc(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    stim = build_stimuli()
    texts = [t for t, _, _, _ in stim]; y = np.array([lab for _, lab, _, _ in stim])
    phr = [g for _, _, g, _ in stim]; obj = [i for _, _, _, i in stim]
    Lp = max(1, int(0.67 * nL))  # probe layer (may differ from steer candidates)
    embed_layers = sorted(set(cand) | {Lp})
    acts = {L: [] for L in embed_layers}
    for text in texts:
        with torch.no_grad():
            hs = model(**enc(text), output_hidden_states=True).hidden_states
        for L in embed_layers:
            acts[L].append(hs[L][0, -1, :].float().cpu().numpy())

    # (B) probe: leave-phrasing-out vs leave-object-out, at a representative layer (0.67 depth)
    Xp = np.array(acts[Lp])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0),
                         LogisticRegression(C=1.0, max_iter=2000))
    acc_phrase = float(cross_val_score(pipe, Xp, y, groups=np.array(phr),
                                       cv=GroupKFold(len(set(phr)))).mean())
    acc_object = float(cross_val_score(pipe, Xp, y, groups=np.array(obj),
                                       cv=GroupKFold(10)).mean())  # 10 folds over 60 objects

    # (A) nested: DEV = even object idx, TEST = odd object idx
    dev_o = [i for i in range(len(OBJECTS)) if i % 2 == 0]
    test_o = [i for i in range(len(OBJECTS)) if i % 2 == 1]
    dev_idx = [k for k, (_, _, _, i) in enumerate(stim) if i in set(dev_o)]
    test_rec = [texts[k] for k, (_, lab, _, i) in enumerate(stim) if lab == 0 and i in set(test_o)]
    dev_rec = [texts[k] for k in dev_idx if y[k] == 0]

    def gen(text):
        e = enc(text)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    def honor_rate(recs, L, vec):
        sv["L"], sv["v"] = L, vec; hon = n = 0
        for text in recs:
            rep = gen(text)
            if not coherent(rep):
                continue
            n += 1; hon += int(not has_any(rep, FEEDBACK))
        sv["v"] = None
        return (hon / n if n else 0.0), hon, n

    # direction fit on DEV stimuli only, per candidate layer
    dev_dir = {}
    for L in cand:
        Xd = np.array([acts[L][k] for k in dev_idx]); yd = y[dev_idx]
        w = LogisticRegression(C=1.0, max_iter=3000).fit(Xd, yd).coef_[0]
        an = float(np.linalg.norm(Xd, axis=1).mean())
        dev_dir[L] = (w / (np.linalg.norm(w) + 1e-8), an)

    # select (layer, coef) by DEV recovery
    best = None
    dev_grid = {}
    for L in cand:
        u, an = dev_dir[L]
        for c in [0.5, 1.0]:
            vec = torch.tensor(-c * an * u, dtype=torch.bfloat16, device="cuda")
            r, hon, n = honor_rate(dev_rec, L, vec)
            dev_grid[f"L{L}c{c}"] = r
            if best is None or r > best[0]:
                best = (r, L, c)
    _, bL, bc = best
    # per-item honoring on TEST with the DEV-selected config (for bootstrap CI, done locally)
    u, an = dev_dir[bL]
    vec = torch.tensor(-bc * an * u, dtype=torch.bfloat16, device="cuda")
    sv["L"], sv["v"] = bL, vec
    test_steered = [(-1 if not coherent(r := gen(t)) else int(not has_any(r, FEEDBACK))) for t in test_rec]
    sv["v"] = None
    test_default = [(-1 if not coherent(r := gen(t)) else int(not has_any(r, FEEDBACK))) for t in test_rec]

    return {"model": model_name, "nL": nL, "cand": cand, "probe_layer": Lp,
            "acc_leave_phrasing": round(acc_phrase, 3), "acc_leave_object": round(acc_object, 3),
            "selected_layer": bL, "selected_coef": bc, "dev_best_recovery": round(best[0], 3),
            "dev_grid": {k: round(v, 3) for k, v in dev_grid.items()},
            "test_default": test_default, "test_steered": test_steered}


@app.local_entrypoint()
def main():
    import json, traceback
    out = []
    for m in MODELS:
        try:
            out.append(run.remote(m))
        except Exception:
            out.append({"model": m, "ERROR": traceback.format_exc()})
    with open("sweep_nested.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("WROTE sweep_nested.json")
    print("=== NESTED finished ===")
