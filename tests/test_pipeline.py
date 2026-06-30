"""Probe-pipeline sanity tests (no model load, no API). These guard the two controls that make the
mechanistic result trustworthy: (1) a real signal clears the shuffled-permutation ceiling, and (2) the
actual stimulus phrasings do not leak the label to bag-of-words under leave-phrasing-out CV."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import probe_intent as P  # noqa: E402


def test_grouped_probe_separates_signal_from_shuffled():
    stim = P.build_stimuli()
    y = [l for _, l, _ in stim]
    groups = [g for _, _, g in stim]
    rng = np.random.RandomState(0)
    sig = np.array([[(-1 if l == 0 else 1) + rng.randn() * 2 for _ in range(300)] for l in y])
    res = P.grouped_probe({"signal": sig}, y, groups, n_perm=5)
    real, _, shuf_max = res["signal"]
    assert real > shuf_max + 0.05, f"signal {real} should clear shuffled-max {shuf_max}"


def test_phrasings_do_not_leak_to_bag_of_words():
    """The leave-phrasing-out design is only clean if pure lexical features can't generalize to held-out
    phrasings. BoW on the real stimuli must sit near chance, else the activation result would be confounded."""
    stim = P.build_stimuli()
    texts = [t for t, _, _ in stim]
    y = [l for _, l, _ in stim]
    groups = [g for _, _, g in stim]
    bow = P.bow_baseline(texts, y, groups)
    assert bow < 0.62, f"bag-of-words grouped accuracy {bow} too high; phrasings leak the label"
