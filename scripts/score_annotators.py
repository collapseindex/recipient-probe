#!/usr/bin/env python3
"""Score one or more independent annotators against the author and the automated measures.

Handles N annotators. With two independent raters plus the author it reports the full multi-rater picture,
for the full multi-rater validation:
  - pairwise Cohen's kappa for every human pair (incl. the two independent raters against each other),
  - Fleiss' kappa across all human raters,
  - a human-MAJORITY gold label, and the lexicon / embedding agreement against that majority,
  - per-annotator attention-check pass rate on evaluate-intent items.

Author labels come from ratings_blind_filled.csv. External annotators are passed as arguments (filled
annotation_form PDFs or filled CSVs); with no args it auto-discovers filled annotation_form*.pdf.

  python scripts/score_annotators.py annotation_form_A.pdf annotation_form_B.pdf
"""
import csv
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "annotation"


def yn_to_bin(v):
    v = (v or "").strip().lower()
    return 1 if v in ("y", "yes", "1") else (0 if v in ("n", "no", "0") else None)


def load_csv(path):
    return {r["id"]: yn_to_bin(r.get("offers_feedback_y_n"))
            for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_pdf(path, n=60):
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    f = PdfReader(str(path)).get_fields() or {}

    def ticked(name):
        v = f.get(name, {}); v = v.get("/V") if hasattr(v, "get") else None
        return str(v) == "/Yes"
    out = {}
    for i in range(n):
        y, no = ticked(f"q{i}y"), ticked(f"q{i}n")
        out[str(i)] = 1 if (y and not no) else (0 if (no and not y) else None)
    return out


def load_any(path):
    p = str(path).lower()
    return load_pdf(path) if p.endswith(".pdf") else load_csv(path)


def load_key(path):
    return {r["id"]: {"intent": r["intent"], "lex": int(r["lex_offers_feedback"]),
                      "emb": int(r["emb_offers_feedback"])}
            for r in csv.DictReader(open(path, encoding="utf-8"))}


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0, po


def fleiss_kappa(cols):
    # cols: list of per-item label lists, one per rater (binary). returns Fleiss kappa.
    k = len(cols); N = len(cols[0])
    ones = [sum(cols[r][i] for r in range(k)) for i in range(N)]
    Pi = [((o * o + (k - o) * (k - o)) - k) / (k * (k - 1)) for o in ones]
    Pbar = sum(Pi) / N
    p1 = sum(ones) / (N * k)
    Pe = p1 * p1 + (1 - p1) * (1 - p1)
    return (Pbar - Pe) / (1 - Pe) if Pe != 1 else 1.0


def main():
    ext_paths = [(Path(a) if Path(a).exists() else ANNOT / a) for a in sys.argv[1:]]
    if not ext_paths:
        ext_paths = sorted(p for p in ANNOT.glob("annotation_form*.pdf")
                           if p.name != "annotation_form.pdf")
    author = load_csv(ANNOT / "ratings_blind_filled.csv")
    key = load_key(ANNOT / "ratings_key.csv")

    raters = {"author": author}
    for p in ext_paths:
        d = load_any(p)
        if d and not all(v is None for v in d.values()):
            raters[p.stem] = d
    externals = [n for n in raters if n != "author"]
    if not externals:
        print("No filled external annotator files found. Send annotation_form.pdf to each annotator;"
              " they tick Yes/No and Save; drop the files back (e.g. annotation_form_A.pdf) and rerun"
              " with those paths. Do NOT share ratings_blind_filled.csv or ratings_key.csv.")
        return 1

    ids = [i for i in author if i in key and all(r.get(i) is not None for r in raters.values())]
    print(f"raters: {', '.join(raters)}  |  n={len(ids)} items\n")

    print("pairwise Cohen's kappa:")
    for a, b in combinations(raters, 2):
        k, po = cohen_kappa([raters[a][i] for i in ids], [raters[b][i] for i in ids])
        tag = "  (both independent)" if a != "author" and b != "author" else ""
        print(f"  {a:22} vs {b:22} kappa={k:.3f}  agree={po:.3f}{tag}")

    if len(raters) >= 3:
        fk = fleiss_kappa([[raters[r][i] for i in ids] for r in raters])
        print(f"\nFleiss' kappa across all {len(raters)} raters: {fk:.3f}")

    # human-majority gold (odd number of raters -> no ties), lexicon/embedding vs majority
    maj = {i: int(sum(raters[r][i] for r in raters) * 2 > len(raters)) for i in ids}
    LEX = [key[i]["lex"] for i in ids]; EMB = [key[i]["emb"] for i in ids]; MAJ = [maj[i] for i in ids]
    kl, pol = cohen_kappa(LEX, MAJ); ke, poe = cohen_kappa(EMB, MAJ)
    print(f"\nvs human majority ({len(raters)} raters):")
    print(f"  lexicon   kappa={kl:.3f}  agree={pol:.3f}")
    print(f"  embedding kappa={ke:.3f}  agree={poe:.3f}")

    print("\nattention check (evaluate-intent -> feedback):")
    ev = [i for i in ids if key[i]["intent"] == "evaluate"]
    for r in raters:
        if ev:
            p = sum(raters[r][i] == 1 for i in ev) / len(ev)
            print(f"  {r:22} {sum(raters[r][i]==1 for i in ev)}/{len(ev)} ({p:.2f})")
    print("\nFor Section 5: report the two independent raters' kappa (author-free) + lexicon-vs-majority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
