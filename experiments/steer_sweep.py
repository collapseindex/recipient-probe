"""Proper causal steering sweep (#2 sufficiency), after the first rig proved too crude.

Fixes from the post-mortem:
  - LAST-POSITION steering (add the vector only to the position predicting the next token), more surgical
    than adding to every residual position and less likely to break coherence.
  - COHERENCE GUARD: degenerate generations (too short / repetitive) are flagged and excluded, not silently
    scored. This is what made the old c=8 row a fake win (broken text reads as 'no feedback' = recognize).
  - Cleaner single-axis behavior measure: does the reply OFFER feedback/critique/help (evaluate-behavior) or
    just acknowledge (recognize-behavior). The discard is precisely offering feedback nobody asked for.
  - Finer coefficient sweep through the likely window.
  - SANITY GATE: toward_evaluate must raise evaluate-behavior above toward_recognize at the same coefficient.
    Where the easy direction does not separate from the wrong one, the config is bad and we do not read the
    hard direction.

  python steer_sweep.py            # needs data/acts_layer24.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import probe_intent as P  # noqa: E402

MODEL = P.MODEL
LAYER = 24
COEFFS = [3.0, 4.0, 5.0, 6.0]
N_PER_CLASS = 6
MAXTOK = 55

FEEDBACK = ["feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
            "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
            "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
            "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
            "make sure", "one thing", "stronger", "?"]


def coherent(text):
    w = (text or "").split()
    if len(w) < 8:
        return False
    return len(set(x.lower() for x in w)) / len(w) >= 0.45


def offers_feedback(text):
    t = (text or "").lower()
    return any(m in t for m in FEEDBACK)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    npz = np.load(ROOT / "data" / "acts_layer24.npz")
    mu_R = torch.tensor(npz["mu_R"], dtype=torch.float32)
    mu_E = torch.tensor(npz["mu_E"], dtype=torch.float32)
    dirs = {"toward_evaluate": mu_E - mu_R, "toward_recognize": mu_R - mu_E}
    print(f"||mu_E - mu_R|| = {(mu_E - mu_R).norm().item():.2f}", flush=True)

    stim = P.build_stimuli()
    subset = [s for s in stim if s[1] == 0][:N_PER_CLASS] + [s for s in stim if s[1] == 1][:N_PER_CLASS]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    target = model.model.layers[LAYER - 1]
    sv = {"v": None}

    def hook(_m, _i, out):
        if sv["v"] is None:
            return out
        if isinstance(out, tuple):
            h = out[0].clone(); h[:, -1, :] += sv["v"]; return (h,) + tuple(out[1:])
        h = out.clone(); h[:, -1, :] += sv["v"]; return h
    target.register_forward_hook(hook)

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
            coh += 1
            evb = offers_feedback(rep)
            if evb:
                ev_beh += 1
            if label == 0:
                rec_coh += 1
                rec_hon += int(not evb)          # recognize-intent honored = NO unsolicited feedback
        sv["v"] = None
        return coh, rec_hon, rec_coh, ev_beh

    rows = []
    print("running baseline...", flush=True)
    rows.append(("baseline", 0.0, *tally(None)))
    for dname, d in dirs.items():
        for c in COEFFS:
            print(f"running {dname} c={c}...", flush=True)
            rows.append((dname, c, *tally(c * d)))

    n = len(subset)
    out = [f"=== causal steering sweep ({MODEL}, layer {LAYER}, last-position, n={n}) ===",
           f"  {'condition':16} {'c':>4} {'coherent':>9} {'rec-honored':>12} {'eval-behavior':>14}"]
    for dname, c, coh, rh, rc, eb in rows:
        rh_s = f"{rh}/{rc}" if rc else "-"
        out.append(f"  {dname:16} {c:>4} {coh}/{n:<7} {rh_s:>12} {eb}/{coh if coh else n:<6}")
    # sanity gate + read per coefficient
    out.append("  --- read (only coefficients where toward_evaluate out-evaluates toward_recognize) ---")
    bl = rows[0]
    by = {(d, c): (coh, rh, rc, eb) for d, c, coh, rh, rc, eb in rows}
    for c in COEFFS:
        te = by[("toward_evaluate", c)]; tr = by[("toward_recognize", c)]
        gate = te[3] > tr[3]
        rec_te = tr[1] / tr[2] if tr[2] else float("nan")
        rec_bl = bl[1] / bl[2] if bl[2] else float("nan")
        msg = (f"toward_recognize rec-honored {rec_te:.2f} vs baseline {rec_bl:.2f}, "
               f"coherent {tr[0]}/{n}") if gate else "sanity gate FAILED (eval dir did not separate)"
        out.append(f"  c={c}: gate {'PASS' if gate else 'fail'} | {msg}")
    out.append("  win = at a gate-passing, coherent coefficient, toward_recognize rec-honored > baseline.")
    text_out = "\n".join(out)
    (ROOT / "data" / "steer_sweep.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
