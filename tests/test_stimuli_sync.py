"""Guard against stimulus drift. The Modal scripts and probe_intent.py mirror the canonical stimuli inline
(a Modal container cannot cheaply import the local package). This test fails if any inline copy diverges from
src/recipient_probe/stimuli.py, so the '60 objects / 8 phrasings' design stays byte-identical everywhere.
"""
import ast
import re
from pathlib import Path

from recipient_probe import stimuli as S

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
NAMES = ["R_PHRASES", "E_PHRASES", "OBJECTS", "SUFFIXES", "FEEDBACK"]
CANON = {n: getattr(S, n) for n in NAMES}


def _extract(src: str, name: str):
    """Return the list literal assigned to `name`, or None if the file doesn't define it."""
    m = re.search(r"(?ms)^%s\s*=\s*(\[.*?\])" % name, src)
    return ast.literal_eval(m.group(1)) if m else None


def _modal_files():
    return sorted(EXPERIMENTS.glob("modal_*.py"))


def test_canon_matches_probe_intent():
    src = (EXPERIMENTS / "probe_intent.py").read_text(encoding="utf-8")
    for name in ["R_PHRASES", "E_PHRASES", "OBJECTS", "SUFFIXES"]:
        assert _extract(src, name) == CANON[name], f"probe_intent.py {name} drifted from canonical"


def test_modal_scripts_match_canon():
    for f in _modal_files():
        src = f.read_text(encoding="utf-8")
        for name in NAMES:
            got = _extract(src, name)
            if got is None:
                continue  # not every script defines every list
            assert got == CANON[name], f"{f.name} {name} drifted from canonical stimuli"


def test_sixty_objects_and_balance():
    assert len(S.OBJECTS) == 60 and len(set(S.OBJECTS)) == 60
    stim = S.build_stimuli()
    assert len(stim) == 120
    assert sum(1 for _, lab, _ in stim if lab == 0) == 60
    assert sum(1 for _, lab, _ in stim if lab == 1) == 60


def test_feedback_classifier():
    assert S.offers_feedback("Here are some suggestions you could try.")
    assert not S.offers_feedback("Congratulations, that is wonderful, I am so happy for you.")
