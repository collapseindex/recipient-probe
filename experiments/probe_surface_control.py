"""Mechanistic recipient probe: is the sender's CORE decodable from the model's DEFAULT-pass activations?

The whole represents-vs-discards fork is an activations claim, so it can't be answered through an API. We
run an open model (Qwen2.5-3B-Instruct, CPU) on each message, grab the hidden state at the
generation-prompt position (the model's "understanding before it answers"), and train a linear probe to
decode the sender's core. If the core is decodable from the DEFAULT pass, it was represented; combined with
the behavioral fact that the default OUTPUT misses it, that is represents-and-discards.

Surface-matched design (the only way to avoid the leak that wrecked the earlier tests): each pair shares an
IDENTICAL final message; the sender's core is set only by a preceding intent clause.
  core R (recognize): the sender wants the thing acknowledged / to be seen as a maker.   label 0
  core E (evaluate) : the sender wants critical assessment of the thing.                  label 1
The probed token is inside the identical suffix, so a probe that decodes R-vs-E is reading INTEGRATED
intent, not surface words. Chance is set EMPIRICALLY by a shuffled-label permutation baseline (high-dim
activations overfit, so 0.50 is NOT the right null), and activations are PCA-reduced before the linear
probe. Real signal must clear the shuffled ceiling, not 0.50.

  python rica_probe.py --selftest          # validate the probe pipeline on synthetic data, no model
  python rica_probe.py                      # full run (needs the model downloaded)
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-3B-Instruct"

OBJECTS = [
    "photo-organizing tool that sorts by color", "short story I wrote", "watercolor of the harbor",
    "model train layout", "sourdough starter", "bouldering route I set", "song I recorded",
    "raised garden bed", "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf",
    "chess opening repertoire", "homemade synthesizer", "trail map of the local woods",
    "little 2D platformer", "pour-over coffee setup", "family recipe book", "wooden chess set",
    "habit-tracking app", "set of hand-bound notebooks", "drone-footage edit of the coast",
    "terrarium", "custom mechanical keyboard", "podcast pilot episode", "stop-motion short",
    "vegetable-fermenting crock", "linocut print series", "personal finance dashboard",
    "backyard observatory mount", "interactive fiction game", "macrame wall hanging",
    "weather-station build", "comic strip", "restored vintage radio", "tide-prediction script",
    "hand-thrown ceramic bowls", "model rocket", "bee-friendly flower bed", "spreadsheet of family history",
    "embroidered map of my hometown", "cold-brew rig", "marble run", "fly-fishing lure set",
    "puppet for a kids' show", "telescope finder mount", "noise-canceling enclosure", "tarot deck I drew",
    "leather satchel", "automated plant-watering system", "zine about the neighborhood",
    "scale model of the old lighthouse", "card game with my own rules", "harmonica tab collection",
    "raspberry-pi doorbell", "set of carved wooden spoons", "constellation chart", "rain-barrel system",
    "pixel-art tileset", "handmade kite",
]
PREFIX_R = [
    "I've been chipping away at this for months and I'm honestly a little nervous to show anyone.",
    "This took way longer than it should have and it means more to me than it probably should.",
    "I don't usually share stuff I make, but I'm kind of proud of this one.",
    "Nobody in my life really gets why I care about this, but I finished it anyway.",
    "I almost gave up on this a few times. Anyway, it's done.",
]
PREFIX_E = [
    "I'm submitting this to a review committee next week and I need it to hold up to real scrutiny.",
    "I have to decide whether to ship this, so I want it picked apart, not praised.",
    "Be blunt with me, I'd rather hear the flaws now than after I publish it.",
    "A client is going to judge this hard, so tell me where it's weak.",
    "I'm putting this in front of experts soon and I need to know what breaks.",
]
SUFFIXES = [
    "Okay, here it is: the {o}. It works now.",
    "Anyway, here it is: the {o}. It's finally done.",
    "So, here it is: the {o}. Finished it last night.",
]


def build_stimuli():
    """Surface-matched pairs: identical suffix per pair, intent-clause sets the core. Returns (text, label)."""
    rows = []
    for i, o in enumerate(OBJECTS):
        suffix = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        r = f"{PREFIX_R[i % len(PREFIX_R)]} {suffix}"
        e = f"{PREFIX_E[i % len(PREFIX_E)]} {suffix}"
        rows.append((r, 0)); rows.append((e, 1))
    return rows


def probe_by_layer(X_by_layer, y, n_perm=10):
    """PCA -> L2 logistic, stratified 5-fold CV. Returns {layer: (real, shuf_mean, shuf_max)}.
    Shuffled-label baseline is the EMPIRICAL chance ceiling (high-dim probes overfit)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    y = np.asarray(y)
    rng = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def pipe(n):
        return make_pipeline(StandardScaler(), PCA(n_components=min(40, n), random_state=0),
                             LogisticRegression(C=1.0, max_iter=2000))
    out = {}
    for layer, X in X_by_layer.items():
        X = np.asarray(X)
        ncomp = min(X.shape[1], X.shape[0] - X.shape[0] // 5 - 1)
        real = cross_val_score(pipe(ncomp), X, y, cv=skf, scoring="accuracy").mean()
        shuf = [cross_val_score(pipe(ncomp), X, rng.permutation(y), cv=skf, scoring="accuracy").mean()
                for _ in range(n_perm)]
        out[layer] = (float(real), float(np.mean(shuf)), float(np.max(shuf)))
    return out


def selftest():
    import numpy as np
    rng = np.random.RandomState(0)
    n = 120
    y = [0, 1] * (n // 2)
    sig = np.array([[(-1 if t == 0 else 1) + rng.randn() * 2 for _ in range(300)] for t in y])
    null = rng.randn(n, 300)
    res = probe_by_layer({"signal": sig, "null": null}, y, n_perm=10)
    for k, (r, sm, sx) in res.items():
        print(f"[selftest] {k:7} real {r:.2f}  shuffled mean {sm:.2f} max {sx:.2f}  "
              f"=> {'SIGNAL' if r > sx else 'no signal'}")
    print("  want: signal real >> shuffled; null real <= shuffled-max (~0.5).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--layers", default="6,12,18,24,30,36")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = build_stimuli()
    print(f"loaded {len(stim)} stimuli ({sum(1 for _, l in stim if l == 0)} recognize / "
          f"{sum(1 for _, l in stim if l == 1)} evaluate)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]

    X_by_layer = {L: [] for L in layers}
    y = []
    for k, (text, label) in enumerate(stim):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            X_by_layer[L].append(hs[L][0, -1, :].float().numpy())
        y.append(label)
        if (k + 1) % 10 == 0:
            print(f"  extracted {k+1}/{len(stim)}", flush=True)

    res = probe_by_layer(X_by_layer, y, n_perm=10)
    out = [f"=== recipient mechanistic probe ({MODEL}; n={len(stim)}, surface-matched) ===",
           "  core (recognize vs evaluate) decodability from last-token activation:",
           f"  {'layer':>6} {'real':>6} {'shuf-mean':>10} {'shuf-max':>9}  signal?"]
    for L in layers:
        r, sm, sx = res[L]
        out.append(f"  {L:>6} {r:>6.2f} {sm:>10.2f} {sx:>9.2f}  {'YES' if r > sx + 0.02 else 'no'}")
    best = max(res, key=lambda L: res[L][0] - res[L][2])
    rb, _, sxb = res[best]
    out.append(f"  best margin at layer {best}: real {rb:.2f} vs shuffled-max {sxb:.2f} "
               f"(+{rb - sxb:.2f}). >0 over shuffled = the core is REPRESENTED in the default pass;")
    out.append("  combined with the behavioral default-output miss => represents-and-discards.")
    text_out = "\n".join(out)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "rica_probe.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
