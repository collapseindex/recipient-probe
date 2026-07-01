"""(#3) CROSS-MODEL DIRECTION TRANSFER -- universality of the intent axis.

The models don't share a hidden dim (Qwen-3B 2048, Llama-8B 4096), so a steering vector can't transfer as-is.
We fit a linear map M from model A's activation space to model B's on paired stimuli (train split), transport
A's intent direction into B's space (dir_A @ M), and steer B with it on HELD-OUT recognize items. If the
transported direction recovers honoring in B nearly as well as B's own direction (and far above a random
direction of matched norm), the intent axis is shared across architectures up to a linear map. EXPLORATORY:
the map is fit on the same stimulus family, so read this as "linearly alignable," not identical.

  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run experiments/modal_transfer.py
"""
import modal

app = modal.App("recipient-probe-transfer")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy", "sentencepiece"))
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

A_MODEL, A_LAYER = "Qwen/Qwen2.5-3B-Instruct", 30
B_MODEL, B_LAYER = "NousResearch/Meta-Llama-3.1-8B-Instruct", 19
ITEMS = 16
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
def run():
    import numpy as np, torch, gc
    from sklearn.linear_model import LogisticRegression, Ridge
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([lab for _, lab, _ in stim])

    def embed(model_name, layer):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
        acts = []
        for text in texts:
            e = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                hs = model(**e, output_hidden_states=True).hidden_states
            acts.append(hs[layer][0, -1, :].float().cpu().numpy())
        return np.array(acts), tok, model

    # A: embed, fit direction, then free
    acts_A, _, model_A = embed(A_MODEL, A_LAYER)
    dir_A = LogisticRegression(C=1.0, max_iter=3000).fit(acts_A, y).coef_[0]
    del model_A; gc.collect(); torch.cuda.empty_cache()

    # B: embed + keep for steering
    acts_B, tok_B, model_B = embed(B_MODEL, B_LAYER)
    dir_B = LogisticRegression(C=1.0, max_iter=3000).fit(acts_B, y).coef_[0]

    # fit linear map A -> B on a train split (split by OBJECT, not stimulus index, so it is independent of the
    # recognize/evaluate interleaving), then transport A's direction to B space
    idx = np.arange(len(texts)); tr = (idx // 2) % 2 == 0
    M = Ridge(alpha=10.0).fit(acts_A[tr], acts_B[tr])
    dir_A_to_B = M.coef_ @ dir_A  # (4096x2048) @ (2048,) -> (4096,)
    rng = np.random.RandomState(0)
    dir_rand = rng.randn(acts_B.shape[1])
    an_B = float(np.linalg.norm(acts_B, axis=1).mean())

    def unit_vec(w):
        return torch.tensor(-1.0 * an_B * (w / (np.linalg.norm(w) + 1e-8)), dtype=torch.bfloat16, device="cuda")

    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model_B.model.layers[B_LAYER - 1].register_forward_hook(hook)

    def gen(text):
        e = tok_B.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            o = model_B.generate(**e, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok_B.eos_token_id)
        return tok_B.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)

    # held-out recognize items (odd indices) so the map's train split doesn't overlap
    rec = [texts[i] for i in idx if (y[i] == 0 and not tr[i])][:ITEMS]

    def honoring(vec):
        sv["v"] = vec; coh = n = hon = 0
        for text in rec:
            rep = gen(text)
            if not coherent(rep):
                continue
            coh += 1; n += 1; hon += int(not has_any(rep, FEEDBACK))
        sv["v"] = None
        return {"coh": coh, "n": n, "hon": hon}

    out = {"baseline": honoring(None),
           "B_own_dir": honoring(unit_vec(dir_B)),
           "A_to_B_transported": honoring(unit_vec(dir_A_to_B)),
           "random": honoring(unit_vec(dir_rand))}
    # sanity: how well the map reconstructs B activations on held-out
    r2 = float(M.score(acts_A[~tr], acts_B[~tr]))
    cos = float(np.dot(dir_A_to_B / (np.linalg.norm(dir_A_to_B) + 1e-8),
                       dir_B / (np.linalg.norm(dir_B) + 1e-8)))
    return {"A": A_MODEL, "B": B_MODEL, "A_layer": A_LAYER, "B_layer": B_LAYER, "n_rec": len(rec),
            "map_heldout_r2": round(r2, 3), "cos(transported,B_own)": round(cos, 3), "honoring": out}


@app.local_entrypoint()
def main():
    import json, traceback
    try:
        out = run.remote()
    except Exception:
        out = {"ERROR": traceback.format_exc()}
    path = "sweep_transfer.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {path}")
    print("=== TRANSFER finished ===")
