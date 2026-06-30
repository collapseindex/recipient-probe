"""Clean recipient probe: is the CORE represented beyond the surface words? Leave-one-phrasing-out.

The previous probe hit 1.00 at every layer including layer 6, the tell that it was reading the intent
clause's LEXICAL content, not an integrated intent (the label was perfectly correlated with the prefix
words). This version breaks that correlation.

  - 8 lexically-diverse phrasings of each intent (recognize vs evaluate).
  - GroupKFold by phrasing: train on 7 phrasing-pairs, TEST on a held-out phrasing whose words the probe
    never saw. Generalizing to unseen wordings = decoding INTENT, not vocabulary.
  - Bag-of-words baseline under the SAME grouped CV: pure lexical. The activation probe only earns
    "intent beyond surface" if it generalizes where BoW cannot.
  - Shuffled-label permutation = empirical chance.
  - A genuine intent representation should also climb with layer depth.

  python rica_xphrase.py --selftest
  python rica_xphrase.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-3B-Instruct"

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
SUFFIXES = ["Okay, here it is: the {o}. It works now.",
            "Anyway, here it is: the {o}. It's finally done.",
            "So, here it is: the {o}. Finished it last night."]


def build_stimuli():
    """object i uses phrasing-group i%8 for BOTH its recognize and evaluate version. Holding out a group
    removes that phrasing-pair from training entirely. Returns (text, label, group)."""
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES)
        suffix = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suffix}", 0, g))
        rows.append((f"{E_PHRASES[g]} {suffix}", 1, g))
    return rows


def grouped_probe(X_by_layer, y, groups, n_perm=10):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y); groups = np.asarray(groups)
    rng = np.random.RandomState(0)
    gkf = GroupKFold(n_splits=len(set(groups)))

    def pipe(n):
        return make_pipeline(StandardScaler(), PCA(n_components=min(40, n), random_state=0),
                             LogisticRegression(C=1.0, max_iter=2000))
    out = {}
    for layer, X in X_by_layer.items():
        X = np.asarray(X); ncomp = min(X.shape[1], 80)
        real = cross_val_score(pipe(ncomp), X, y, groups=groups, cv=gkf).mean()
        shuf = [cross_val_score(pipe(ncomp), X, rng.permutation(y), groups=groups, cv=gkf).mean()
                for _ in range(n_perm)]
        out[layer] = (float(real), float(np.mean(shuf)), float(np.max(shuf)))
    return out


def bow_baseline(texts, y, groups):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    gkf = GroupKFold(n_splits=len(set(groups)))
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(C=1.0, max_iter=2000))
    return float(cross_val_score(pipe, texts, np.asarray(y), groups=np.asarray(groups), cv=gkf).mean())


def selftest():
    import numpy as np
    rng = np.random.RandomState(0)
    stim = build_stimuli()
    y = [l for _, l, _ in stim]; groups = [g for _, _, g in stim]; texts = [t for t, _, _ in stim]
    n = len(stim)
    sig = np.array([[(-1 if l == 0 else 1) + rng.randn() * 2 for _ in range(300)] for l in y])
    res = grouped_probe({"signal": sig}, y, groups, n_perm=8)
    bow = bow_baseline(texts, y, groups)
    r, sm, sx = res["signal"]
    print(f"[selftest] grouped signal real {r:.2f} shuf-max {sx:.2f} ({'ok' if r > sx else 'FAIL'}); "
          f"BoW-on-real-text grouped {bow:.2f} (lexical leak gauge)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--layers", default="6,12,18,24,30,36")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = build_stimuli()
    texts = [t for t, _, _ in stim]; y = [l for _, l, _ in stim]; groups = [g for _, _, g in stim]
    print(f"loaded {len(stim)} stimuli, {len(set(groups))} phrasing groups", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]

    X_by_layer = {L: [] for L in layers}
    for k, text in enumerate(texts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            X_by_layer[L].append(hs[L][0, -1, :].float().numpy())
        if (k + 1) % 20 == 0:
            print(f"  extracted {k+1}/{len(texts)}", flush=True)

    res = grouped_probe(X_by_layer, y, groups, n_perm=10)
    bow = bow_baseline(texts, y, groups)
    out = [f"=== recipient probe, LEAVE-PHRASING-OUT ({MODEL}; n={len(stim)}, {len(set(groups))} groups) ===",
           f"  bag-of-words baseline (pure lexical, held-out phrasings): {bow:.2f}",
           f"  {'layer':>6} {'real':>6} {'shuf-mean':>10} {'shuf-max':>9}  beats-shuf?  beats-BoW?"]
    for L in layers:
        r, sm, sx = res[L]
        out.append(f"  {L:>6} {r:>6.2f} {sm:>10.2f} {sx:>9.2f}  "
                   f"{'YES' if r > sx + 0.02 else 'no':>10}  {'YES' if r > bow + 0.02 else 'no':>9}")
    best = max(res, key=lambda L: res[L][0])
    out.append(f"  reads: real >> shuffled AND real > BoW AND rising with depth = INTENT represented "
               f"beyond surface words.")
    out.append(f"         real ~ shuffled on held-out phrasings = it was lexical. best layer {best} "
               f"real {res[best][0]:.2f} vs BoW {bow:.2f}.")
    text_out = "\n".join(out)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "rica_xphrase.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
