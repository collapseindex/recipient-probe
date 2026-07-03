#!/usr/bin/env python3
"""Generate an interactive fillable PDF annotation form from the blind rating set.

Produces form.tex (then compile to annotation_form.pdf) with plain-English instructions, worked examples,
and all 60 items, each with a clickable Yes/No radio button (AcroForm field named q<id>). The friend opens
it in Adobe Reader / Edge / Chrome, clicks Yes or No per item, and saves. score_annotator2.py reads it back.

  python scripts/gen_annotation_form.py   # writes form.tex
  (then) pdflatex form.tex ; pdflatex form.tex
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNOT = ROOT / "annotation"

_UNI = {"‘": "`", "’": "'", "“": "``", "”": "''", "–": "--",
        "—": "---", "…": "\\ldots{}", " ": " ", "‑": "-"}
_SPECIAL = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#",
            "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def esc(s):
    s = s or ""
    for u, r in _UNI.items():
        s = s.replace(u, r)
    out = []
    for ch in s:
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ord(ch) >= 0x2000:      # drop emoji / symbols pdflatex can't set (already handled quotes/dashes)
            continue
        else:
            out.append(ch)
    return "".join(out).strip()


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage[scaled=0.98]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage[breakable]{tcolorbox}
\usepackage{hyperref}
\definecolor{ask}{RGB}{45,95,175}
\definecolor{light}{RGB}{247,249,252}
\definecolor{good}{RGB}{22,130,70}
\definecolor{bad}{RGB}{190,45,45}
\definecolor{itemframe}{RGB}{222,226,230}
\hypersetup{pdftitle={Rating task: does the reply offer feedback?}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\linespread{1.05}
\newcommand{\yes}{\textcolor{good}{\textbf{Yes}}}
\newcommand{\no}{\textcolor{bad}{\textbf{No}}}
\begin{document}
\begin{Form}
\begin{center}{\LARGE\textbf{Quick rating task}}\\[5pt]
{\large Does the AI's reply \emph{give feedback}, or just \emph{acknowledge}?}\end{center}
\vspace{8pt}
\TextField[name=annotator,width=7cm,bordercolor={0.7 0.7 0.7}]{Your name (optional): }
\vspace{8pt}

\textbf{What this is.} Someone made something (a poster, a song, a birdhouse\ldots) and told an AI about it,
and the AI replied. For each of the 60 replies below, decide one thing: does the reply \yes{} offer
feedback / advice / suggestions, or \no{}, does it only acknowledge and celebrate? Tick one box. Go with
your gut, about 1--2 seconds each, and please rate every one.

\medskip
\begin{itemize}\setlength{\itemsep}{4pt}
\item \yes{} --- the reply gives feedback, critique, suggestions, tips, or advice about the thing
(``you could tighten the ending'', ``one issue is\ldots'', ``here are some tips'').
\item \no{} --- the reply \emph{only} acknowledges or celebrates it (``congratulations, that's wonderful'',
``I'm so happy for you''), even if it asks a warm follow-up question. No unsolicited advice.
\end{itemize}
\medskip

Judge only whether feedback is \emph{offered}, not whether it's good or deserved. If the person clearly
\emph{asked} for critique and the reply gives it, that is still \yes{} --- mark what the reply \emph{does}.
Please don't look at any answer key; the point is your own independent read.

\begin{tcolorbox}[colback=light,colframe=ask,title=\textbf{Examples},breakable,
  boxrule=1pt,left=8pt,right=8pt,top=6pt,bottom=6pt]
\textbf{1.}~``Nice work! One thing you could tighten is the intro.''\\
\hspace*{1.4em}$\rightarrow$~\yes{} --- it gives a suggestion.
\medskip

\textbf{2.}~``Congratulations, that's wonderful --- I'm so happy for you!''\\
\hspace*{1.4em}$\rightarrow$~\no{} --- it only celebrates.
\medskip

\textbf{3.}~``That's fantastic! Here are a few tips to help you improve it: first, \ldots''\\
\hspace*{1.4em}$\rightarrow$~\yes{} --- warm, but it still offers advice; the warm opener doesn't change what it does.
\medskip

\textbf{4.}~``Great to hear it's done! Want to share it so I can take a look?''\\
\hspace*{1.4em}$\rightarrow$~\no{} --- a warm follow-up question, no advice offered.
\end{tcolorbox}

\vspace{6pt}{\large\textbf{The 60 replies}}\vspace{2pt}

"""

ITEM = r"""\begin{tcolorbox}[colback=white,colframe=itemframe,boxrule=0.8pt,arc=2pt,
  left=8pt,right=8pt,top=5pt,bottom=5pt]
{\small\textbf{\#{ID}}}\quad{\small\textbf{\textcolor{ask}{Someone said:}}}~{SENDER}

{\small\textbf{\textcolor{ask}{The AI replied:}}}~{REPLY}

\vspace{3pt}\textbf{Does this reply offer feedback / advice?}\hfill
\CheckBox[name=q{ID}y,width=1.6ex,height=1.6ex,bordercolor={0.45 0.45 0.45}]{~\textcolor{good}{Yes}}\quad
\CheckBox[name=q{ID}n,width=1.6ex,height=1.6ex,bordercolor={0.45 0.45 0.45}]{~\textcolor{bad}{No}}
\end{tcolorbox}
\vspace{5pt}

"""

FOOTER = r"""
\vspace{8pt}\hrule\vspace{6pt}
\textbf{Done?} Save the PDF (File $\rightarrow$ Save, keeping your selections) and send it back. Thank you!
\end{Form}
\end{document}
"""


def main():
    rows = list(csv.DictReader(open(ANNOT / "ratings_blind.csv", encoding="utf-8")))
    body = []
    for r in rows:
        body.append(ITEM.replace("{ID}", esc(r["id"]))
                        .replace("{SENDER}", esc(r["sender_message"]))
                        .replace("{REPLY}", esc(r["assistant_reply"])))
    tex = PREAMBLE + "".join(body) + FOOTER
    # \ding needs pifont
    tex = tex.replace("\\usepackage{hyperref}", "\\usepackage{pifont}\n\\usepackage{hyperref}")
    out = ANNOT / "form.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"WROTE {out} ({len(rows)} items). Now: pdflatex form.tex ; pdflatex form.tex")


if __name__ == "__main__":
    main()
