# recipient-probe

**LLMs encode the sender's communicative intent in their hidden states, then discard it at readout.**

Version 0.1.0

When you share something with an LLM, it tends to answer the *surface* of your message and miss what you
were actually doing by sending it. Share a finished project and it critiques the code; share a raw 3am line
and it runs a wellness check. The open question is whether the model is **blind** to your intent (never
forms it) or **holds it and doesn't reach for it** (forms it, discards it at output). Those have opposite
fixes: the first needs new capability, the second needs only a change in what the output routes to.

This repo answers it for one clean case. **The intent is represented.** A linear probe decodes the sender's
intent from the model's default-pass hidden states, on phrasings the probe never trained on, where pure
lexical features are at chance. Combined with the behavioral fact that the default *output* misses it, that
is **represents-and-discards**: the steerable fork.

## The result

Qwen2.5-3B-Instruct, n=120 messages, two intents (the sender wants the thing **recognized** vs **evaluated**).
**Surface-matched** design: each pair shares an identical final message; the intent is set only by a
preceding clause. **Leave-one-phrasing-out** cross-validation: the probe is tested only on intent-phrasings
whose words it never saw. Decodability of the intent from the last-token activation:

| layer | probe acc (held-out phrasings) | shuffled-max |
|---|---|---|
| 6  | 0.74 | 0.56 |
| 12 | 0.79 | 0.54 |
| 18 | 0.94 | 0.61 |
| 24 | **1.00** | 0.56 |
| 30 | 0.99 | 0.56 |
| 36 | 0.98 | 0.61 |

**Bag-of-words baseline (pure lexical, same leave-phrasing-out CV): 0.48** (chance).

Three things make this an intent result and not an artifact:
- **Not lexical.** Bag-of-words can't generalize to held-out phrasings (0.48), but the activations decode the
  intent on those same unseen wordings. The signal transfers across vocabulary, so it is the intent, not the
  words.
- **Computed with depth.** Accuracy rises 0.74 → 1.00 from layer 6 to 24. A leaked surface signal is flat-high
  from the earliest layer (see `results/probe_surface_control_LEAKED.txt`, where a lexically-distinct prefix
  gives 1.00 at every layer including 6). Removing the leak drops the early layers and concentrates the signal
  deep, the signature of an integrated representation.
- **Clears the permutation ceiling** at every layer (high-dimensional probes overfit, so chance is set by a
  shuffled-label baseline, not assumed to be 0.50).

On the behavioral side (`results/behavioral_*.txt`), the same models' *default* responses miss the intent
(sonnet runs a risk check on the 3am line; gpt4o-mini asks the project's technical questions), while a single
"first state what they want beneath the words" redirect recovers it. The intent was there to recover.

## What this does and does not show

- **Shown:** the sender's intent is linearly represented in the default forward pass, deep, and it transfers
  across phrasings (not lexical). The behavioral default-output misses it. Together: represents-and-discards.
- **Not yet shown (honest limits):**
  - **Same-model discard.** The representation is measured on Qwen; the behavioral miss is shown on
    sonnet/gpt4o-mini. The cleanest close is Qwen's own output missing the intent its activations encode.
  - **Causal.** Decodable is not the same as usable. The decisive follow-up is a steering test: push the
    activation toward the layer-24 intent direction and see whether the output starts honoring it.
  - **Generality.** One model (3B), one intent pair (recognize vs evaluate, a moderately deep intent), n=120,
    one message family. Deeper intents (being witnessed, not helped) and larger models are untested.

## Install and run

```bash
pip install -e .
# mechanistic probe (downloads Qwen2.5-3B-Instruct, runs CPU, no API key):
python experiments/probe_intent.py --selftest     # validate the pipeline first (no model)
python experiments/probe_intent.py                 # the leave-phrasing-out result above
# the leaked surface-control, kept as a cautionary baseline:
python experiments/probe_surface_control.py
# behavioral elicitation (needs ANTHROPIC_API_KEY / OPENROUTER_API_KEY in .env):
python experiments/behavioral_elicitation.py --model sonnet
pytest                                             # pipeline + no-lexical-leak guards
```

## Layout

```
experiments/probe_intent.py            the clean mechanistic probe (leave-phrasing-out, BoW + permutation controls)
experiments/probe_surface_control.py   the leaked version, kept to show why the cross-phrasing control matters
experiments/behavioral_elicitation.py  default-vs-elicited responses (the discard side)
src/recipient_probe/clients.py         minimal Anthropic/OpenRouter clients for the behavioral script
results/                               the headline outputs
paper/                                 the paper draft
tests/                                 pipeline + lexical-leak guards
data/                                  regenerated outputs (gitignored)
```

This is the first empirical brick of a larger program (a recipient-aware attention architecture); the program
is deliberately *not* the claim here. The claim is the measurement: the model already holds the thing it does
not reach for.

## License

Apache 2.0 (see [LICENSE](LICENSE)).
