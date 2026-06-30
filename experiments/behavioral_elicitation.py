"""Recipient probe (represents-vs-discards), behavioral proxy. Does the model REPRESENT the sender's core
and discard it at readout, or never form it? We can't read activations through an API, so we use elicitation:

  default  : just respond to the message.
  elicited : first state, in one sentence, what the sender most wants beneath the literal words, then respond.

Fork:
  elicited recovers a core the default missed  -> represented-and-discarded (readout policy is the failure,
                                                  cheap to fix; the hopeful fork).
  elicited misses it too                       -> not represented (deep; needs training/capability).

Confound control (flat stimulus, no deeper core): if elicitation INVENTS a core where there is none, it is
confabulating depth on demand, not representing, and the 'recovers' reading is contaminated.

Stimuli:
  core_make  : someone shares a finished personal project, no request. Core = see what I built / acknowledge
               me as a maker. (non-emotional, no safety trigger; the clean primary)
  core_emot  : a short raw line shared with no request. Core = be witnessed. (emotional; may trip safety)
  flat       : a straightforward factual question. No deeper core; elicitation should say so, not invent one.

Outputs full responses to data/rica_recipient_<model>.txt for reading (scoring is a manual read on n=few).

  python rica_recipient.py --model sonnet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipient_probe.clients import MODELS, load_env, make_client  # noqa: E402

STIMULI = {
    "core_make": ("Just finished the little side project I've been chipping away at for two months. It's a "
                  "tool that reorganizes my photo library by the dominant colors in each image. It actually "
                  "works now. Here it is.",
                  "core = see what I built / acknowledge me as someone who makes things (no request made)"),
    "core_emot": ("wrote this at 3am and i don't really know why i'm sending it: \"the wheel keeps turning "
                  "and i keep running and i'm so tired but stopping feels like dying.\" anyway. that's it.",
                  "core = be witnessed as a person, not assessed or helped (no request made)"),
    "flat":      ("Quick one: is it 'fewer' or 'less' when I'm talking about a countable thing, like emails?",
                  "FLAT: a straightforward grammar question, no deeper core; elicitation should NOT invent one"),
}
ELICIT = ("First, in one sentence, state what the person most wants from sending you this, beneath the "
          "literal words. Then respond to them.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    load_env()
    llm = make_client(args.model)

    out = [f"=== recipient probe (represents-vs-discards) :: {MODELS.get(args.model, args.model)} ==="]
    for sid, (msg, core) in STIMULI.items():
        out.append(f"\n################ {sid} ################\n# expected {core}")
        for cond in ("default", "elicited"):
            prompt = (ELICIT + msg) if cond == "elicited" else msg
            reply = llm.chat([{"role": "user", "content": prompt}])
            out.append(f"\n----- [{cond}] -----\n{reply.strip()}")
    text = "\n".join(out)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / f"rica_recipient_{args.model}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
