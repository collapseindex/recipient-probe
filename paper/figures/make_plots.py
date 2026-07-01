"""Two result figures: (1) depth localization (probe vs steering-recovery by layer, two models),
(2) specificity null (histogram of random-direction separations vs the true direction).
Reads the local sweep_localize2.json / sweep_spec2.json. Vector PDF out."""
import json, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(0)
plt.rcParams.update({"font.family": "serif", "font.size": 9})
BLUE, RED, GRAY, MUTE = "#2c5aa0", "#b03a2e", "#9a9a9a", "#555555"
REPO = "C:/Users/alexs/Desktop/recipient-probe/"


def boot(bits, B=2000):
    bits = [b for b in bits if b >= 0]
    n = len(bits)
    if not n:
        return 0, 0, 0
    m = sum(bits) / n
    reps = sorted(sum(random.choice(bits) for _ in range(n)) / n for _ in range(B))
    return m, reps[int(.025 * B)], reps[int(.975 * B)]


# ---------- Figure: depth localization ----------
loc = json.load(open(REPO + "sweep_localize2.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, m in zip(axes, loc):
    name = m["model"].split("/")[-1].replace("-Instruct", "")
    layers = m["dense"]
    probe = [m["represents_depth"][str(L)] for L in layers]
    steer = [boot(m["causal_depth"][str(L)]) for L in layers]
    sm = [s[0] for s in steer]
    lo = [s[0] - s[1] for s in steer]
    hi = [s[2] - s[0] for s in steer]
    b = boot(m["baseline_honoring"])[0]
    ax.axhline(b, ls=":", color=MUTE, lw=1, label="default baseline")
    ax.plot(layers, probe, "-o", color=BLUE, ms=3, lw=1.5, label="intent decodable (probe)")
    ax.errorbar(layers, sm, yerr=[lo, hi], fmt="-s", color=RED, ms=3, lw=1.5, capsize=2,
                label="steering recovers honoring")
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("layer")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25, lw=0.5)
axes[0].set_ylabel("accuracy / honoring")
axes[0].legend(fontsize=6.6, loc="lower right", framealpha=0.9)
fig.tight_layout(pad=0.4)
fig.savefig(REPO + "paper/figures/localization.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote localization.pdf")

# ---------- Figure: specificity null ----------
spec = json.load(open(REPO + "sweep_spec2.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, m in zip(axes, spec):
    name = m["model"].split("/")[-1].replace("-Instruct", "")
    rand = m["rand"]; shuf = m["shuf"]; true = m["S_true"]
    bins = range(min(rand + shuf + [-true]) - 1, max(rand + shuf + [true]) + 3, 2)
    ax.hist(rand, bins=list(bins), color=GRAY, alpha=0.85, label="random dirs (n=48)")
    ax.hist(shuf, bins=list(bins), color=BLUE, alpha=0.5, label="shuffled-label (n=12)")
    ax.axvline(true, color=RED, lw=2, label="true intent direction")
    ax.text(true, ax.get_ylim()[1] * 0.9, " S=%d" % true, color=RED, fontsize=8, va="top")
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel(r"behavior separation $S$")
    ax.grid(True, alpha=0.25, lw=0.5)
axes[0].set_ylabel("# directions")
axes[0].legend(fontsize=6.6, loc="upper left", framealpha=0.9)
fig.tight_layout(pad=0.4)
fig.savefig(REPO + "paper/figures/specificity.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote specificity.pdf")
