"""Generate all figures for the workshop paper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Data from experiments ──
steps = [0, 5, 10, 15, 20, 25, 30]

standard_rl = [0.10, 0.13, 0.16, 0.17, 0.20, 0.18, 0.23]
agent_seg = [0.14, 0.12, 0.12, 0.13, 0.14, 0.16, 0.16]
agent_noseg = [0.14, 0.19, 0.15, 0.18, 0.19, 0.22, 0.31]
agent_random = [0.14, 0.11, 0.11, 0.14, 0.14, 0.08, 0.08]

# Tool call rates
tc_agent_seg = [0.17, 0.12, 0.09, 0.12, 0.11, 0.02, 0.03]
tc_agent_noseg = [0.17, 0.04, 0.01, 0.02, 0.01, 0.01, 0.00]
tc_agent_random = [0.17, 0.13, 0.08, 0.13, 0.11, 0.08, 0.11]

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }
)

colors = {
    "std": "#2196F3",
    "seg": "#E91E63",
    "noseg": "#4CAF50",
    "random": "#FF9800",
    "zero": "#9E9E9E",
}

# ════════════════════════════════════════════════════
# Figure 1: Learning curves (accuracy)
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.axhline(
    y=0.085,
    color=colors["zero"],
    linestyle="--",
    linewidth=1,
    label="Zero-shot",
    alpha=0.7,
)
ax.plot(steps, standard_rl, "s-", color=colors["std"], label="Standard RL (no tools)")
ax.plot(
    steps, agent_noseg, "D-", color=colors["noseg"], label="Agent RL (no segment mask)"
)
ax.plot(steps, agent_seg, "o-", color=colors["seg"], label="Agent RL (segment-aware)")
ax.plot(
    steps, agent_random, "^-", color=colors["random"], label="Agent RL (random mask)"
)

ax.set_xlabel("Training Step")
ax.set_ylabel("GSM8K Accuracy")
ax.set_xlim(-1, 31)
ax.set_ylim(0.0, 0.35)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title("(a) Accuracy over Training Steps")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_learning_curves.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "fig_learning_curves.png"), bbox_inches="tight", dpi=300)
print("Saved fig_learning_curves.pdf/png")
plt.close()

# ════════════════════════════════════════════════════
# Figure 2: Tool call rate over training
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.2))

ax.plot(
    steps, tc_agent_seg, "o-", color=colors["seg"], label="Agent RL (segment-aware)"
)
ax.plot(
    steps,
    tc_agent_noseg,
    "D-",
    color=colors["noseg"],
    label="Agent RL (no segment mask)",
)
ax.plot(
    steps, tc_agent_random, "^-", color=colors["random"], label="Agent RL (random mask)"
)

ax.set_xlabel("Training Step")
ax.set_ylabel("Tool Call Rate")
ax.set_xlim(-1, 31)
ax.set_ylim(-0.01, 0.22)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title("(b) Tool Call Rate over Training Steps")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tool_rate.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "fig_tool_rate.png"), bbox_inches="tight", dpi=300)
print("Saved fig_tool_rate.pdf/png")
plt.close()

# ════════════════════════════════════════════════════
# Figure 3: Bar chart – final accuracy comparison
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 3.2))

methods = [
    "Zero-shot",
    "Agent RL\n(random)",
    "Agent RL\n(TraceRL+seg)",
    "Standard RL\n(no tools)",
    "Agent RL\n(TraceRL, no seg)",
]
accs = [0.085, 0.120, 0.140, 0.180, 0.200]
bar_colors = [
    colors["zero"],
    colors["random"],
    colors["seg"],
    colors["std"],
    colors["noseg"],
]

bars = ax.bar(
    range(len(methods)),
    accs,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    width=0.7,
)
for bar, val in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.1%}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("GSM8K Accuracy")
ax.set_ylim(0, 0.27)
ax.grid(True, axis="y", alpha=0.3)
ax.set_title("(c) Final Accuracy Comparison (200 test samples)")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_bar_comparison.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT, "fig_bar_comparison.png"), bbox_inches="tight", dpi=300)
print("Saved fig_bar_comparison.pdf/png")
plt.close()

print("\nAll figures generated in:", OUT)
