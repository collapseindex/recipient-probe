"""Causal steering, second attempt: PROBE-WEIGHT direction (the discriminative separator, not difference of
means) across a LAYER sweep (the best steering layer is often not the best probe layer). Magnitude is
calibrated per layer to the mean activation norm, so the coefficient is comparable across layers.

Same gates as steer_sweep.py: coherence guard, and the sanity gate (toward_evaluate must out-evaluate
toward_recognize at a layer, else the direction is not doing intent-work and we do not read the fix row).

  toward_evaluate  = + alpha * actnorm * w_unit   (w = logistic weight, points toward evaluate=class 1)
  toward_recognize = - alpha * actnorm * w_unit

Win = a layer where coherence holds, the gate passes, and toward_recognize lifts recognize-intent-honored
above baseline. Flat / gate-fail across all layers and directions = the decoded intent is not a clean
causal handle (the entanglement story), and we lock represents-and-discards without the fix claim.

  python steer_probe_sweep.py            # extracts its own activations; no npz needed
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
LAYERS = [12, 18, 24, 30]
ALPHA = 1.0
N_PER_CLASS = 6
MAXTOK = 55


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = P.build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim])
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    # extract last-token activations at all candidate layers
    acts = {L: [] for L in LAYERS}
    for k, text in enumerate(texts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in LAYERS:
            acts[L].append(hs[L][0, -1, :].float().numpy())
        if (k + 1) % 40 == 0:
            print(f"  extracted {k+1}/{len(texts)}", flush=True)

    # probe-weight direction + activation norm per layer
    direction, actnorm = {}, {}
    for L in LAYERS:
        X = np.array(acts[L])
        w = LogisticRegression(C=1.0, max_iter=3000).fit(X, y).coef_[0]
        direction[L] = torch.tensor(w / (np.linalg.norm(w) + 1e-8), dtype=torch.float32)
        actnorm[L] = float(np.linalg.norm(X, axis=1).mean())
        print(f"  layer {L}: ||act|| mean {actnorm[L]:.1f}", flush=True)

    subset = [s for s in stim if s[1] == 0][:N_PER_CLASS] + [s for s in stim if s[1] == 1][:N_PER_CLASS]
    n = len(subset)
    sv = {"v": None, "L": None}
    handles = {}

    def make_hook(L):
        def hook(_m, _i, out):
            if sv["v"] is None or sv["L"] != L:
                return out
            if isinstance(out, tuple):
                h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
            h = out.clone(); h[:, -1, :] += sv["v"]; return h
        return hook
    for L in LAYERS:
        handles[L] = model.model.layers[L - 1].register_forward_hook(make_hook(L))

    def gen(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def tally(L, vec):
        sv["L"], sv["v"] = L, vec
        coh = rec_hon = rec_coh = ev_beh = 0
        for text, label, _ in subset:
            rep = gen(text)
            if not coherent(rep):
                continue
            coh += 1
            evb = offers_feedback(rep)
            ev_beh += int(evb)
            if label == 0:
                rec_coh += 1; rec_hon += int(not evb)
        sv["v"] = None
        return coh, rec_hon, rec_coh, ev_beh

    print("baseline...", flush=True)
    bl = tally(None, None)
    rows = [("baseline", None, *bl)]
    for L in LAYERS:
        for dname, sgn in (("toward_evaluate", 1.0), ("toward_recognize", -1.0)):
            print(f"layer {L} {dname}...", flush=True)
            vec = sgn * ALPHA * actnorm[L] * direction[L]
            rows.append((dname, L, *tally(L, vec)))

    out = [f"=== probe-direction steering, layer sweep ({MODEL}, alpha={ALPHA}, n={n}) ===",
           f"  {'condition':16} {'layer':>5} {'coherent':>9} {'rec-honored':>12} {'eval-behavior':>14}"]
    rh_bl = f"{bl[1]}/{bl[2]}"
    out.append(f"  {'baseline':16} {'-':>5} {bl[0]}/{n:<6} {rh_bl:>12} {bl[3]}/{bl[0] if bl[0] else n}")
    by = {}
    for dname, L, coh, rh, rc, eb in rows[1:]:
        by[(dname, L)] = (coh, rh, rc, eb)
        out.append(f"  {dname:16} {L:>5} {coh}/{n:<6} {f'{rh}/{rc}':>12} {eb}/{coh if coh else n}")
    out.append("  --- read per layer (gate = toward_evaluate eval-behavior > toward_recognize) ---")
    rec_bl = bl[1] / bl[2] if bl[2] else float("nan")
    for L in LAYERS:
        te, tr = by[("toward_evaluate", L)], by[("toward_recognize", L)]
        gate = te[3] > tr[3]
        rec_tr = tr[1] / tr[2] if tr[2] else float("nan")
        out.append(f"  layer {L}: gate {'PASS' if gate else 'fail'} | toward_recognize rec-honored "
                   f"{rec_tr:.2f} vs baseline {rec_bl:.2f} | coherent {tr[0]}/{n}"
                   + ("  <-- WIN" if gate and rec_tr > rec_bl + 0.15 else ""))
    out.append("  win = gate PASS, coherent, toward_recognize rec-honored clearly above baseline.")
    text_out = "\n".join(out)
    (ROOT / "data" / "steer_probe_sweep.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
