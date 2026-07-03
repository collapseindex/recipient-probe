#!/usr/bin/env python3
"""Score a second independent annotator against the author and the two automated measures.

Validates the behavioral measure against a second independent human annotator. Once the second annotator has filled
`ratings_blind_annotator2.csv` (blind to condition and to every label, using ANNOTATE.md), this reports
Cohen's kappa and percent agreement for:
  annotator2 vs author (the new human-human inter-rater number),
  annotator2 vs lexicon, annotator2 vs embedding,
  author vs lexicon (for reference),
plus the attention-check pass rate on evaluate-intent items (requested critique should read as feedback).

  python scripts/score_annotator2.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "annotation"


def yn_to_bin(v):
    v = (v or "").strip().lower()
    return 1 if v in ("y", "yes", "1") else (0 if v in ("n", "no", "0") else None)


def load_human(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r["id"]] = yn_to_bin(r.get("offers_feedback_y_n"))
    return out


def load_pdf(path, n=60):
    """Read the filled annotation_form PDF. Each item has two checkboxes, q<id>y and q<id>n, each with
    on-state /Yes. Exactly one ticked -> 1 (feedback) / 0 (acknowledge); both or neither -> None."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    fields = PdfReader(str(path)).get_fields() or {}

    def ticked(name):
        v = fields.get(name, {})
        v = v.get("/V") if hasattr(v, "get") else None
        return str(v) == "/Yes"

    out = {}
    for i in range(n):
        y, no = ticked(f"q{i}y"), ticked(f"q{i}n")
        out[str(i)] = 1 if (y and not no) else (0 if (no and not y) else None)
    return out


def load_annotator(path):
    p = str(path).lower()
    return load_pdf(path) if p.endswith(".pdf") else load_human(path)


def load_key(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r["id"]] = {"condition": r["condition"], "intent": r["intent"],
                        "lex": int(r["lex_offers_feedback"]), "emb": int(r["emb_offers_feedback"])}
    return out


def kappa(a, b):
    # Cohen's kappa for two binary label lists (no sklearn dependency).
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0, po


def main():
    if len(sys.argv) > 1:
        a2p = Path(sys.argv[1]) if Path(sys.argv[1]).exists() else ANNOT / sys.argv[1]
    else:
        cands = [ANNOT / "annotation_form_filled.pdf", ANNOT / "annotation_form.pdf",
                 ANNOT / "ratings_blind_annotator2.csv"]
        a2p = next((c for c in cands if c.exists()), cands[-1])
    a2 = load_annotator(a2p)
    if not a2 or all(v is None for v in a2.values()):
        print(f"{a2p.name} is not filled yet. Send the annotator annotation_form.pdf (fillable Yes/No form);"
              " they open it in Adobe Reader / Edge / Chrome, click Yes or No per item, and Save. Do NOT let"
              " them see ratings_blind_filled.csv or ratings_key.csv. Then: python scripts/score_annotator2.py"
              " annotation_form_filled.pdf")
        return 1
    print(f"Scoring annotator from {a2p.name}\n")
    author = load_human(ANNOT / "ratings_blind_filled.csv")
    key = load_key(ANNOT / "ratings_key.csv")

    ids = [i for i in a2 if a2[i] is not None and i in author and i in key]
    missing = [i for i in a2 if a2[i] is None]
    if missing:
        print(f"WARNING: {len(missing)} rows unlabeled in annotator2 (ids {missing[:8]}...). Scoring the rest.")

    A2 = [a2[i] for i in ids]
    AU = [author[i] for i in ids]
    LEX = [key[i]["lex"] for i in ids]
    EMB = [key[i]["emb"] for i in ids]

    print(f"n = {len(ids)} items scored\n")
    for name, other in (("annotator2 vs author  ", AU), ("annotator2 vs lexicon ", LEX),
                        ("annotator2 vs embedding", EMB), ("author     vs lexicon ", None)):
        if name.startswith("author"):
            k, po = kappa(AU, LEX)
        else:
            k, po = kappa(A2, other)
        print(f"  {name}:  kappa = {k:.3f}   agreement = {po:.3f}")

    # attention check: evaluate-intent items (critique explicitly requested) should read as feedback (=1)
    ev = [i for i in ids if key[i]["intent"] == "evaluate"]
    if ev:
        passed = sum(a2[i] == 1 for i in ev)
        print(f"\n  attention check (evaluate-intent -> feedback): {passed}/{len(ev)} "
              f"({passed/len(ev):.2f})")
    print("\nReport annotator2-vs-author as the inter-rater kappa in Section 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
