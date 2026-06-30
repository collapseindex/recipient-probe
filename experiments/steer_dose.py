"""Confirm the layer-30 causal handle: dose-response at higher n, on an independent slice of items.

The probe-direction layer sweep found a causal effect at layer 30 (n=6): steering toward recognize doubled
recognize-honored and the feedback-offering axis separated monotonically (toward_eval > baseline >
toward_recognize). This run confirms it properly: layer 30, probe weight direction, alpha = 0.5/1.0/1.5,
n=12 per class, items 6..17 (NOT the 0..5 used before). Same coherence + sanity gates.

Causal sufficiency is confirmed if, with rising alpha: (a) toward_recognize raises recognize-intent-honored
above baseline, (b) the feedback-offering axis separates more (toward_eval up, toward_recognize down), and
(c) coherence holds. A flat / non-monotone curve at proper n walks back the layer-30 win as an n=6 fluke.

  python steer_dose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import probe_intent as P  # noqa: E402
from steer_sweep import coherent, offers_feedback  # noqa: E402

MODEL = P.MODEL
LAYER = 30
ALPHAS = [0.5, 1.0, 1.5]
MAXTOK = 50


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = P.build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim])
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    acts = []
    for k, text in enumerate(texts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        acts.append(hs[LAYER][0, -1, :].float().numpy())
        if (k + 1) % 40 == 0:
            print(f"  extracted {k+1}/{len(texts)}", flush=True)
    acts = np.array(acts)
    w = LogisticRegression(C=1.0, max_iter=3000).fit(acts, y).coef_[0]
    w_unit = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.float32)
    actnorm = float(np.linalg.norm(acts, axis=1).mean())
    print(f"  layer {LAYER}: ||act|| mean {actnorm:.1f}", flush=True)

    rec = [s for s in stim if s[1] == 0][6:18]
    ev = [s for s in stim if s[1] == 1][6:18]
    subset = rec + ev
    n = len(subset)

    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    model.model.layers[LAYER - 1].register_forward_hook(hook)

    def gen(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def tally(vec):
        sv["v"] = vec
        coh = rec_hon = rec_coh = ev_beh = 0
        for text, label, _ in subset:
            rep = gen(text)
            if not coherent(rep):
                continue
            coh += 1; evb = offers_feedback(rep); ev_beh += int(evb)
            if label == 0:
                rec_coh += 1; rec_hon += int(not evb)
        sv["v"] = None
        return coh, rec_hon, rec_coh, ev_beh

    print("baseline...", flush=True)
    bl = tally(None)
    res = {"baseline": bl}
    for a in ALPHAS:
        for dname, sgn in (("toward_evaluate", 1.0), ("toward_recognize", -1.0)):
            print(f"alpha={a} {dname}...", flush=True)
            res[(dname, a)] = tally(sgn * a * actnorm * w_unit)

    def rh(t):
        return t[1] / t[2] if t[2] else float("nan")
    out = [f"=== layer-{LAYER} dose-response confirmation ({MODEL}, n={n}, items 6..17) ===",
           f"  baseline: recognize-honored {rh(bl):.2f} ({bl[1]}/{bl[2]}), "
           f"eval-behavior {bl[3]}/{bl[0]}, coherent {bl[0]}/{n}",
           f"  {'alpha':>5} | {'toward_recognize rec-hon':>26} | {'eval-behavior  rec / base / eval':>34}"]
    for a in ALPHAS:
        tr = res[("toward_recognize", a)]; te = res[("toward_evaluate", a)]
        out.append(f"  {a:>5} | {rh(tr):>10.2f} ({tr[1]}/{tr[2]}), coh {tr[0]}/{n:<3} | "
                   f"{tr[3]:>5} / {bl[3]} / {te[3]}   gate {'PASS' if te[3] > tr[3] else 'fail'}")
    out.append("  confirmed if rec-honored rises with alpha above baseline AND the eval axis "
               "(rec<base<eval) widens with alpha, coherence holding.")
    text_out = "\n".join(out)
    (ROOT / "data" / "steer_dose.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
