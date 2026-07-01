"""Teaser figure (Figure 1): the represent -> discard -> recover triptych.
Renders a clean vector PDF at paper/figures/teaser.pdf. Column-spanning (figure*)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "font.size": 9})

BLUE = "#2c5aa0"; GRAY = "#b8b8b8"; FRAME = "#9a9a9a"; CARDBG = "#fbfbfb"
RED = "#b03a2e"; GREEN = "#1e8449"; DARK = "#232323"; MUTE = "#555555"

fig, ax = plt.subplots(figsize=(7.1, 2.25))
ax.set_xlim(0, 30); ax.set_ylim(0, 10); ax.axis("off")


def card(x0, x1, n, title):
    ax.add_patch(FancyBboxPatch((x0, 0.5), x1 - x0, 8.6,
                                boxstyle="round,pad=0.02,rounding_size=0.35",
                                linewidth=0.8, edgecolor=FRAME, facecolor=CARDBG, zorder=1))
    ax.text(x0 + 0.35, 8.35, str(n), fontsize=12, fontweight="bold", color=BLUE, zorder=3)
    ax.text(x0 + 1.05, 8.35, title, fontsize=10.5, fontweight="bold", color=DARK, va="baseline", zorder=3)


def arrow(x0, x1, label=None):
    ax.add_patch(FancyArrowPatch((x0, 4.8), (x1, 4.8), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=MUTE, zorder=2))
    if label:
        ax.text((x0 + x1) / 2, 5.5, label, ha="center", fontsize=7.2, color=MUTE, style="italic")


def bubble(x, y, text, fc, ec, tc=DARK, w=0.5, fs=7.4):
    ax.text(x, y, text, fontsize=fs, color=tc, ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor=ec, linewidth=0.7))


# ---- Card 1: Represented (bar chart) ----
card(0.4, 9.3, 1, "Represented")
bx, bw = 1.5, 6.2
ax.text(bx, 7.15, "probe (hidden state)", fontsize=7.2, color=MUTE)
ax.add_patch(Rectangle((bx, 6.35), bw * 1.00, 0.6, facecolor=BLUE, edgecolor="none", zorder=3))
ax.text(bx + bw * 1.00 + 0.2, 6.65, "1.00", fontsize=8.2, fontweight="bold", va="center", color=BLUE)
ax.text(bx, 5.35, "bag-of-words", fontsize=7.2, color=MUTE)
ax.add_patch(Rectangle((bx, 4.55), bw * 0.48, 0.6, facecolor=GRAY, edgecolor="none", zorder=3))
ax.text(bx + bw * 0.48 + 0.2, 4.85, "0.48", fontsize=8.2, va="center", color=MUTE)
ax.text(1.5, 3.35, "intent decodes from the\ndefault-pass activation,\nnot from the surface",
        fontsize=7.3, color=DARK, va="top", linespacing=1.35)

arrow(9.3, 10.4)

# ---- Card 2: Discarded ----
card(10.4, 19.3, 2, "Discarded")
bubble(14.85, 7.1, "user: “not looking for notes,\njust wanted to share this”",
       "#eef1f6", "#c4cdd9", tc=DARK)
ax.text(14.85, 5.45, "default reply", fontsize=7, color=MUTE, ha="center", style="italic")
bubble(14.85, 4.15, "“Here's some feedback\nyou could try…”", "#fdeceb", "#e2b6b1", tc=RED)
ax.text(14.85, 2.35, "MISSES IT", fontsize=8.5, fontweight="bold", color=RED, ha="center")
ax.text(14.85, 1.5, "honoring 0.62", fontsize=7.6, color=MUTE, ha="center")

arrow(19.3, 20.4, "+ route\nthe direction")

# ---- Card 3: Recoverable ----
card(20.4, 29.6, 3, "Recoverable")
ax.text(25.0, 7.1, "steer the represented\nintent direction (late layer)", fontsize=7.3, color=BLUE,
        ha="center", style="italic", linespacing=1.35)
ax.text(25.0, 5.45, "steered reply", fontsize=7, color=MUTE, ha="center", style="italic")
bubble(25.0, 4.15, "“That's wonderful,\ncongratulations!”", "#eaf5ee", "#a9d5b9", tc=GREEN)
ax.text(25.0, 2.35, "HONORS IT", fontsize=8.5, fontweight="bold", color=GREEN, ha="center")
ax.text(25.0, 1.5, "honoring 0.92", fontsize=7.6, color=MUTE, ha="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig("teaser.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote teaser.pdf")
