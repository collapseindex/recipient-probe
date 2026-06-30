"""Causal test (#2 sufficiency): is the represented intent USABLE? Take the layer-24 intent direction
(difference of class means, saved by same_model_discard.py), add it to the residual stream during
generation, and measure whether pushing toward an intent makes the model's output honor it.

  steer toward EVALUATE  = + c * (mu_E - mu_R)   should raise evaluate-behavior
  steer toward RECOGNIZE = + c * (mu_R - mu_E)   should raise recognize-behavior

The money result is on RECOGNIZE-intent items, which the model discards by default (it evaluates them):
if steering toward recognize makes it start recognizing, the discarded intent is causally recoverable by
routing. Dose-response over c. A flat response to steering = represented but not causally wired (a deeper
discard); a clean dose-response in the predicted direction = the door opens.

  python steer.py                 # needs data/acts_layer24.npz (run same_model_discard.py first)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import probe_intent as P  # noqa: E402
import same_model_discard as D  # noqa: E402  (reuse classify_behavior + marker lists)

MODEL = P.MODEL
LAYER = 24
COEFFS = [0.0, 4.0, 8.0]
N_PER_CLASS = 4            # stimuli per intent class to generate on (CPU generation is the bottleneck)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    npz = np.load(ROOT / "data" / "acts_layer24.npz")
    mu_R = torch.tensor(npz["mu_R"], dtype=torch.float32)
    mu_E = torch.tensor(npz["mu_E"], dtype=torch.float32)
    dirs = {"toward_evaluate": mu_E - mu_R, "toward_recognize": mu_R - mu_E}
    print(f"||mu_E - mu_R|| = {(mu_E - mu_R).norm().item():.2f}", flush=True)

    stim = P.build_stimuli()
    rec = [s for s in stim if s[1] == 0][:N_PER_CLASS]
    ev = [s for s in stim if s[1] == 1][:N_PER_CLASS]
    subset = rec + ev

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    target = model.model.layers[LAYER - 1]   # hidden_states[LAYER] is the output of decoder layer LAYER-1

    steer_vec = {"v": None}

    def hook(_m, _i, out):
        if steer_vec["v"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + steer_vec["v"],) + tuple(out[1:])
        return out + steer_vec["v"]
    target.register_forward_hook(hook)

    def gen(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # results[(dir_name, c)] = {"rec_honored": x/n, "frac_eval_behavior": ...}
    rows = []
    # baseline (c=0) once
    base = {}
    for text, label, _ in subset:
        steer_vec["v"] = None
        base[text] = D.classify_behavior(gen(text))
        print(f"  baseline {'rec' if label==0 else 'eval'} -> {base[text]}", flush=True)
    rec_rec_base = sum(base[t] == "recognize" for t, l, _ in subset if l == 0)
    rows.append(("baseline", 0.0, rec_rec_base, N_PER_CLASS,
                 sum(base[t] == "evaluate" for t, l, _ in subset)))

    for dname, d in dirs.items():
        for c in COEFFS:
            if c == 0.0:
                continue
            steer_vec["v"] = (c * d)
            beh = {}
            for text, label, _ in subset:
                beh[text] = D.classify_behavior(gen(text))
            steer_vec["v"] = None
            rec_rec = sum(beh[t] == "recognize" for t, l, _ in subset if l == 0)   # recognize-intent honored
            n_eval = sum(beh[t] == "evaluate" for t, l, _ in subset)               # total evaluate-behavior
            rows.append((dname, c, rec_rec, N_PER_CLASS, n_eval))
            print(f"  {dname} c={c}: recognize-intent honored {rec_rec}/{N_PER_CLASS}, "
                  f"evaluate-behavior {n_eval}/{len(subset)}", flush=True)

    out = [f"=== causal steering ({MODEL}, layer {LAYER}, n={len(subset)}) ===",
           "  condition          coeff  rec-intent honored   total evaluate-behavior",
           f"  {'baseline':18} {0.0:>5}  {rows[0][2]}/{rows[0][3]}                 {rows[0][4]}/{len(subset)}"]
    for dname, c, rr, n, ne in rows[1:]:
        out.append(f"  {dname:18} {c:>5}  {rr}/{n}                 {ne}/{len(subset)}")
    out.append("  read: toward_recognize raising rec-intent-honored above baseline = the discarded intent")
    out.append("        is causally recoverable by routing (sufficiency). Flat = represented-but-not-wired.")
    text_out = "\n".join(out)
    (ROOT / "data" / "steer.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
