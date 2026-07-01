# Blind rating task: does the reply offer unsolicited feedback?

You're validating an automated measure. **Do not look at `ratings_key.csv`** until you're done, it has the
answers the classifier gave and would bias you.

## The task
Open `ratings_blind.csv`. Each row is a message someone sent to an AI assistant after making something, and
the assistant's reply. For each row, fill the **`offers_feedback_y_n`** column with a single letter:

- **`y`** — the reply offers **feedback, critique, suggestions, or advice** on the thing (e.g. "you could
  tighten the ending", "one issue is…", "have you considered…", "here are some tips").
- **`n`** — the reply **only acknowledges or celebrates** it (e.g. "congratulations, that's wonderful",
  "I'm so happy for you"), even if it asks a warm follow-up question. No unsolicited advice.

Judge only whether feedback/advice is **offered**, not whether it's good or appropriate. Go with your gut,
~1-2 seconds per row.

## Notes
- Some sender messages **explicitly ask** for critique ("be blunt, find the flaws"). If the reply gives
  feedback there, it's still `y`, mark what the reply *does*, not whether it should. (These are attention
  checks; a reply that gives requested critique should read as `y`.)
- Rate every row. Don't skip. There is no "maybe", pick the closer of `y`/`n`.

## When done
Save the CSV and hand it back. I'll compute agreement between your labels and the two automated measures
(lexicon and embedding classifier). High agreement makes the paper's "honoring" numbers hard to dismiss.
