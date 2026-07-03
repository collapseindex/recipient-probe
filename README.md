# recipient-probe

**Language models represent a sender's communicative intent in their hidden states more reliably than they act on it.**

Code, stimuli, and results for the paper *"They Infer What You Meant: Models Represent Communicative Intent More
Reliably Than They Act On It."* (arXiv link forthcoming.)

When you share something with a language model, it often answers the *surface* of your message rather than what
you were *doing* by sending it: share a finished project and it critiques the code; share a raw late-night line
and it runs a wellness check. The question is whether the model is **blind** to your intent (never forms it) or
**holds it and doesn't reach for it** (forms it, discards it at output). Those have opposite fixes. This repo
shows it is the second: the intent is represented, and the failure is one of readout.

## The result

- **Represented.** A linear probe decodes the sender's intent (does the sender want a thing *recognized* or
  *evaluated*) from the model's default-pass hidden states at **1.00** under leave-one-phrasing-out
  cross-validation, with a bag-of-words baseline at chance (**0.48**), across **six models and four families**
  (Qwen2.5-3B/7B/14B, Mistral-7B, Phi-3.5-mini, Llama-3.1-8B) and in the **base checkpoints**. It also decodes
  intent that is only *pragmatically inferred*, never stated.
- **Discarded at readout.** On three of the six models the default output nonetheless misses the intent,
  offering unsolicited feedback on a recognize-intent; the other three already honor it at baseline.
- **Recoverable.** Where the gap is open, steering the residual stream along the discriminative intent direction
  at a searched later layer recovers the intended behavior with a clean dose-response, as well as an explicit
  instruction does and with no prompt at all. (A difference-of-means direction at the peak-probe layer does not
  steer; the discriminative direction at a later layer does.)
- **Represented before routed.** Depth sweeps place the intent decodable several layers before steering there
  becomes effective: the discard is a routing gap, not a failure to represent.
- **It routes intent, not a feedback knob.** The steered direction is near-orthogonal to a direction fit on
  reply behavior alone (cosine 0.09–0.13 vs a random floor of 0.01–0.09), so it routes a represented intent
  rather than generically suppressing feedback.

Nulls are reported alongside the confirmations: an inconclusive pre-registered geometry test, a failed
specificity control on a third intent axis, and an exploratory-only cross-model transport. See `results/`.

## Install and run

```bash
pip install -e .

# core mechanistic chain (CPU-only, downloads Qwen2.5-3B-Instruct, no API key):
python experiments/probe_intent.py --selftest   # validate the pipeline first (no model)
python experiments/probe_intent.py               # represents: leave-phrasing-out probe + BoW + permutation
python experiments/same_model_discard.py         # discards: intent decodable but default output misses it
python experiments/steer_probe_sweep.py          # find the causal layer
python experiments/steer_dose.py                 # recover: dose-response
pytest                                            # pipeline + no-lexical-leak guards

# behavioral elicitation (needs ANTHROPIC_API_KEY / OPENROUTER_API_KEY in a .env file):
python experiments/behavioral_elicitation.py --model sonnet

# extended experiments run on a single A100 via Modal (modal run experiments/modal_*.py):
#   modal_sweep      six-model probe ladder
#   modal_scale      discard/recover at n=60 with bootstrap CIs
#   modal_specificity2, modal_opener_control, modal_request_matched, modal_valence   controls
#   modal_localize2  depth localization
```

## Layout

```
experiments/probe_intent.py            the clean mechanistic probe (leave-phrasing-out, BoW + permutation)
experiments/probe_surface_control.py   the deliberately-leaked version, kept to show why the control matters
experiments/same_model_discard.py      the represents-and-discards demonstration on one model
experiments/steer_*.py                 finding the causal layer and the dose-response
experiments/modal_*.py                 the GPU experiments (six models, controls, localization)
src/recipient_probe/                   stimuli construction and minimal Anthropic/OpenRouter clients
results/                               the recorded outputs, including the honest nulls
tests/                                 pipeline + lexical-leak guards
scripts/                               human-annotation scoring
PREREG.md                              pre-registration
```

## Citation

```bibtex
@misc{kwon2026recipientprobe,
  title  = {They Infer What You Meant: Models Represent Communicative Intent
            More Reliably Than They Act On It},
  author = {Kwon, Alex},
  year   = {2026},
  note   = {arXiv preprint (identifier forthcoming)}
}
```

## License

Apache 2.0 (see [LICENSE](LICENSE)).
