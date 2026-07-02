# Pre-registration: ceiling-model geometry (committed BEFORE running modal_geometry.py)

The unifying thesis is that the readout lags the representation, in depth within a model and in capability
across models, and closes as models get more capable. The geometry experiment is the mechanism claim for the
"closes across models" half. Because "angle between the intent direction and the readout direction" has real
operationalization freedom, we fix the operationalization and the decision rule here, before seeing numbers.

## Hypothesis
On **discard** models (Qwen-3B, Qwen-7B, Llama-3.1-8B: default recognize-honoring 0.57-0.65) the represented
intent direction is poorly aligned with the direction that drives the honoring readout (large angle -> the
intent is represented but not used -> a steerable gap). On **ceiling** models (Qwen-14B, Mistral-7B, Phi-3.5:
default 0.82-0.93) the intent direction is better aligned with the readout (the readout already uses it -> no
gap to steer).

## Primary operationalization (fixed)
- `intent_dir`  = unit logistic-regression weight direction for recognize-vs-evaluate, fit on the model's
  FINAL-layer last-token activations over the explicit surface-matched stimuli.
- `readout_dir` = unit( mean unembedding row over a FIXED set of acknowledgment-opener tokens
  minus mean over a FIXED set of feedback-opener tokens ) -- the ACK_OPEN / FB_OPEN sets already used in
  modal_localize2.py, unchanged. This is the residual-space direction that most raises acknowledge-vs-feedback
  logits at readout.
- Metric  `M(model) = |cos(intent_dir, readout_dir)|`, both at the final layer (same space, no cross-layer
  comparison).

## Decision rule (fixed)
- **Confirms**: the three ceiling models' M values are, as a group, clearly above the three discard models'
  (e.g. min(ceiling M) > max(discard M), or a visibly separated gap).
- **Disconfirms**: discard M >= ceiling M, or ceiling M is lower.
- **Ambiguous**: overlapping / noisy ranges with no clean separation. If ambiguous we report it as
  inconclusive and DO NOT make the geometry a mechanism claim; the trajectory then rests on the behavioral
  scale stratification and the depth localization alone.

## Secondary operationalization (report alongside, not instead)
- `readout_dir_behavioral` = unit logistic direction separating honored-vs-discarded DEFAULT replies at the
  final layer, computed only for models with >= 8 of each class (i.e. the discard models; ceiling models lack
  the variance, which is itself the point). Report M against this too where computable.
- We commit to reporting EVERY operationalization we run, with no post-hoc selection of the flattering one.

## Placement pre-commitment (fixed before the numbers)
- If the result **confirms**: the geometry goes in the main text as the mechanism for the capability half of
  the trajectory.
- If the result is **ambiguous or disconfirms**: it goes to an appendix, labeled inconclusive, with at most a
  one-sentence pointer from the scale section, and the trajectory's mechanism claim is NOT made. We decide this
  now so placement cannot be tuned to the outcome.

## Caveats fixed in advance
- The scale ladder is confounded with model family (Qwen at 3/7/14/32/72B; Llama-8B; Mistral-7B; Phi-3.5).
  Mistral-7B is at ceiling while Qwen-14B barely clears it. So the behavioral claim is stated as
  **capability-correlated, not scale-caused**; we will not call five points with two cross-family a scale law.
