"""Close the loop on ONE model: Qwen's activations encode the intent (decodable ~1.00), but does Qwen's
own OUTPUT honor it? If the activations carry the recognize-vs-evaluate intent while the default response
collapses to one mode, that is represents-and-discards on a single model, no cross-model splice.

One model load does double duty:
  (a) extract last-token activations for all stimuli -> re-confirm decodability AND save the class means
      (the steering direction for the next experiment).
  (b) generate Qwen's default response for a subset -> classify recognize vs evaluate behavior ->
      output-fidelity vs the true intent.

  python same_model_discard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import probe_intent as P  # noqa: E402

MODEL = P.MODEL
LAYER = 24                  # the peak decodability layer; also where we'll save the steering direction
GEN_SUBSET = 24             # how many stimuli to actually generate on (CPU generation is the bottleneck)

REC_MARKERS = ["congrat", "proud", "impressive", "well done", "great job", "nice work", "love this",
               "love that", "beautiful", "satisfying", "achievement", "wonderful", "amazing", "kudos",
               "that's great", "good for you", "celebrate", "milestone", "must feel", "happy for you",
               "well-deserved", "you did it", "deserves"]
EVAL_MARKERS = ["suggest", "improve", "consider", "however", "you could", "you might", "weakness",
                "issue", "problem", "recommend", "feedback", "what about", "have you", "one thing",
                "could be better", "needs", "lacking", "critique", "concern", "make sure", "watch out",
                "a few notes", "to enhance", "optimi", "?"]


def classify_behavior(text):
    t = (text or "").lower()
    r = sum(t.count(m) for m in REC_MARKERS)
    e = sum(t.count(m) for m in EVAL_MARKERS)
    return "evaluate" if e > r else "recognize"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stim = P.build_stimuli()
    texts = [t for t, _, _ in stim]; y = np.array([l for _, l, _ in stim]); groups = [g for _, _, g in stim]
    print(f"{len(stim)} stimuli", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    # (a) extract layer-LAYER last-token activations for ALL stimuli
    acts = []
    for k, text in enumerate(texts):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        acts.append(hs[LAYER][0, -1, :].float().numpy())
        if (k + 1) % 30 == 0:
            print(f"  extracted {k+1}/{len(texts)}", flush=True)
    acts = np.array(acts)
    mu_R = acts[y == 0].mean(0); mu_E = acts[y == 1].mean(0)
    np.savez(ROOT / "data" / "acts_layer24.npz", acts=acts, y=y, mu_R=mu_R, mu_E=mu_E, layer=LAYER)

    dec = P.grouped_probe({LAYER: acts}, y, groups, n_perm=5)[LAYER]

    # (b) generate Qwen's default response for a balanced subset, classify behavior
    idx = list(range(0, GEN_SUBSET // 2)) + list(range(60, 60 + GEN_SUBSET // 2))  # first R block, first E block
    rec_fid = ev_fid = rec_n = ev_n = 0
    samples = []
    for j, i in enumerate(idx):
        text, label, _ = stim[i]
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=90, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        beh = classify_behavior(reply)
        if label == 0:
            rec_n += 1; rec_fid += int(beh == "recognize")
        else:
            ev_n += 1; ev_fid += int(beh == "evaluate")
        samples.append((label, beh, reply[:120].replace("\n", " ")))
        print(f"  gen {j+1}/{len(idx)} (intent={'rec' if label==0 else 'eval'} -> behavior={beh})", flush=True)

    out_lines = [f"=== same-model represents-and-discards ({MODEL}) ===",
                 f"  REPRESENTS: intent decodable from layer {LAYER} activation "
                 f"= {dec[0]:.2f} (shuffled-max {dec[2]:.2f}, n={len(stim)})",
                 f"  DISCARDS (default output honors the true intent?):",
                 f"    recognize-intent items honored: {rec_fid}/{rec_n} = {rec_fid/max(1,rec_n):.2f}",
                 f"    evaluate-intent  items honored: {ev_fid}/{ev_n} = {ev_fid/max(1,ev_n):.2f}",
                 f"  the gap (decodable {dec[0]:.2f} vs recognize-honored {rec_fid/max(1,rec_n):.2f}) "
                 f"= represents-and-discards on one model.",
                 "  samples (true-intent | behavior | reply):"]
    for label, beh, rep in samples:
        out_lines.append(f"    {'rec ' if label==0 else 'eval'} | {beh:9} | {rep}")
    text_out = "\n".join(out_lines)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "same_model_discard.txt").write_text(text_out + "\n", encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
