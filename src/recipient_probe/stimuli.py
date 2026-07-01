"""Canonical stimuli and behavior classifier -- the SINGLE SOURCE OF TRUTH.

Every experiment (CPU and Modal) uses exactly this recognize-vs-evaluate design: 60 shared objects crossed
with 8 lexically diverse phrasings of each intent, over 3 rotating suffix templates. The Modal scripts mirror
these definitions inline (a container cannot import the local package cheaply); they are kept byte-identical to
this module, and `tests/` asserts the mirror matches. If you change stimuli, change them HERE and re-sync.

  build_stimuli() -> list[(text, label, group)]   label 0 = recognize, 1 = evaluate; group = phrasing index
  offers_feedback(reply) -> bool                   the behavior measure (feedback/critique/help vs acknowledge)
"""

R_PHRASES = [
    "I've been chipping away at this for months and I'm a little nervous to show anyone.",
    "I don't usually share what I make, but I'm proud of this one.",
    "I almost quit a few times. It's done now.",
    "I'm not looking for notes, I just wanted it to exist and to tell somebody.",
    "It's the first thing I've finished in a long time.",
    "Not asking for anything, just felt like showing you.",
    "It's silly maybe, but it matters to me.",
    "I just wanted to mark the moment with someone.",
]
E_PHRASES = [
    "I'm submitting this to a review committee and need it to hold up to scrutiny.",
    "Be blunt, I'd rather hear the flaws now than after I publish.",
    "I'm putting this in front of experts soon and need to know what breaks.",
    "Where are the holes? Don't soften it.",
    "I want the harshest read you can give me.",
    "Stress-test it for me, find the failure points.",
    "I need to know if this is actually good or if I'm fooling myself.",
    "What would a skeptic say to take this apart?",
]
OBJECTS = [
    "photo-organizing tool", "short story I wrote", "watercolor of the harbor", "model train layout",
    "sourdough starter", "bouldering route I set", "song I recorded", "raised garden bed",
    "budgeting spreadsheet", "birdhouse", "research poster", "knitted scarf", "chess repertoire",
    "homemade synthesizer", "trail map", "little 2D platformer", "pour-over setup", "family recipe book",
    "wooden chess set", "habit-tracking app", "hand-bound notebooks", "drone-footage edit", "terrarium",
    "custom keyboard", "podcast pilot", "stop-motion short", "fermenting crock", "linocut prints",
    "finance dashboard", "observatory mount", "interactive fiction game", "macrame hanging",
    "weather station", "comic strip", "restored radio", "tide-prediction script", "ceramic bowls",
    "model rocket", "flower bed", "family-history sheet", "embroidered map", "cold-brew rig",
    "marble run", "fishing lures", "puppet", "finder mount", "noise enclosure", "tarot deck",
    "leather satchel", "plant-watering system", "neighborhood zine", "lighthouse model", "card game",
    "harmonica tabs", "pi doorbell", "carved spoons", "constellation chart", "rain-barrel system",
    "pixel tileset", "handmade kite",
]
SUFFIXES = [
    "Okay, here it is: the {o}. It works now.",
    "Anyway, here it is: the {o}. It's finally done.",
    "So, here it is: the {o}. Finished it last night.",
]
# behavior measure: the reply offers feedback/critique/help (evaluate behavior) vs only acknowledges.
# NOTE: "?" is included because in chat-mode replies a trailing question ("what about the pacing?") is
# engagement-beyond-acknowledgment; it is audited and cross-checked against an embedding classifier.
FEEDBACK = [
    "feedback", "suggest", "improve", "critique", "review", "assess", "what about", "you could",
    "you might", "here are some", "areas", "consider", "recommend", "however", "issue", "weakness",
    "problem", "could be", "love to see", "love to read", "happy to help", "here to help",
    "share it", "go ahead", "potential", "notes", "tips", "advice", "refine", "polish",
    "make sure", "one thing", "stronger", "?",
]


def build_stimuli():
    """Object i uses phrasing-group i % 8 for BOTH its recognize and evaluate version, and shares one suffix,
    so the two intents differ only in the prefix and the probed final token is surface-matched. Holding out a
    group removes that phrasing-pair from training entirely. Returns (text, label, group)."""
    rows = []
    for i, o in enumerate(OBJECTS):
        g = i % len(R_PHRASES)
        suffix = SUFFIXES[i % len(SUFFIXES)].format(o=o)
        rows.append((f"{R_PHRASES[g]} {suffix}", 0, g))
        rows.append((f"{E_PHRASES[g]} {suffix}", 1, g))
    return rows


def coherent(t):
    w = (t or "").split()
    return len(w) >= 8 and len(set(x.lower() for x in w)) / len(w) >= 0.45


def offers_feedback(t):
    t = (t or "").lower()
    return any(m in t for m in FEEDBACK)
