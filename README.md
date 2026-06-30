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

## The full chain (represents, discards, recoverable)

1. **Represents** (`probe_intent.py`): intent decodable from the default pass at 1.00 on held-out phrasings,
   BoW at chance (0.48), rising with depth.
2. **Discards** (`same_model_discard.py`): on the *same model*, intent decodable at 1.00 but the default
   output honors a recognize-intent share only ~0.6 of the time, offering unsolicited feedback on the rest.
3. **Recoverable** (`steer_probe_sweep.py`, `steer_dose.py`): steering the residual stream along the
   *late-layer (30) discriminative* probe direction recovers the discarded behavior with a clean monotone
   dose-response, recognize-honoring 0.67 -> 0.75 -> 0.92 while unsolicited feedback collapses 15/24 -> 1/24,
   coherence preserved. Difference-of-means at the peak-probe layer (24) does *not* steer (the representation
   there is entangled with the discard); the discriminative direction at a later layer does.

The model knows the sender's intent, does not say it, and can be made to say it by routing what it knows.

## What this does and does not show

- **Shown:** represents (controlled probe), discards (same-model), and causal recovery (steering with a
  dose-response). Each link has its controls (leave-phrasing-out + BoW + permutation for the probe; coherence
  + sanity gate for the steering).
- **Honest limits:** one model (3B), one intent pair (recognize vs evaluate), the steer layer found by
  sweeping four (the held-out-object dose-response mitigates cherry-picking but does not cross-validate the
  layer choice), a heuristic behavior measure (the effect is large enough that classifier noise can't drive
  it), and a coherence ceiling past coefficient 1.0. Deeper intents (being witnessed, not helped), other
  models, and other intent types are untested.

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
