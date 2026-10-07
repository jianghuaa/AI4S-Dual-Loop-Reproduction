"""Summary visualization: CV comparison + method comparison + gap analysis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

# ============ DATA ============
# CV R² for 5 models × 2 systems
cv_models = ["XGBoost", "CatBoost", "RandomForest", "TabPFN", "SVR"]
cv_ternary = [0.860, 0.842, 0.771, 0.503, -0.245]
cv_quat    = [0.594, 0.615, 0.592, 0.208, -0.349]

# Method comparison (train best / grid / BO)
# 三元
tern_methods = ["Train\nbest", "Grid\n(XGB)", "BO\n(XGB,fixed)", "BO\n(XGB,orig)", "BO\n(TabPFN)"]
tern_a45 = [0.822, 0.819, 0.759, 0.819, 0.594]
tern_a135 = [0.956, 0.950, 0.950, 0.717, 0.604]

# 四元
quat_methods = ["Train\nbest", "Grid\n(XGB)", "Grid\n(CatBoost)", "Grid\n(RF)", "BO\n(XGB)"]
quat_a45 = [0.810, 0.806, 0.771, 0.775, 0.721]
quat_a135 = [0.940, 0.935, 0.896, 0.880, 0.703]

# ============ PLOT ============
fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

# ---------- (a) CV R² ----------
ax = axes[0, 0]
x = np.arange(len(cv_models))
w = 0.35
b1 = ax.bar(x - w/2, cv_ternary, w, label="Ternary (54 samples)",
            color="#4C72B0", edgecolor="k")
b2 = ax.bar(x + w/2, cv_quat, w, label="Quaternary (54 samples)",
            color="#DD8452", edgecolor="k")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(cv_models, fontsize=10)
ax.set_ylabel("5-fold CV R²", fontsize=11)
ax.set_title("(a) Surrogate model comparison", fontsize=12, fontweight="bold")
ax.set_ylim(-0.5, 1.05)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---------- (b) Ternary method comparison ----------
ax = axes[0, 1]
x = np.arange(len(tern_methods))
w = 0.35
b1 = ax.bar(x - w/2, tern_a45, w, label="angle=45°", color="#4C72B0", edgecolor="k")
b2 = ax.bar(x + w/2, tern_a135, w, label="angle=135°", color="#DD8452", edgecolor="k")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
# 训练集最优参考线
ax.axhline(0.822, color="#4C72B0", linewidth=1, linestyle=":", alpha=0.6)
ax.axhline(0.956, color="#DD8452", linewidth=1, linestyle=":", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(tern_methods, fontsize=9)
ax.set_ylabel("Max G", fontsize=11)
ax.set_title("(b) Ternary: method comparison", fontsize=12, fontweight="bold")
ax.set_ylim(0.5, 1.05)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---------- (c) Quaternary method comparison ----------
ax = axes[1, 0]
x = np.arange(len(quat_methods))
w = 0.35
b1 = ax.bar(x - w/2, quat_a45, w, label="angle=45°", color="#4C72B0", edgecolor="k")
b2 = ax.bar(x + w/2, quat_a135, w, label="angle=135°", color="#DD8452", edgecolor="k")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
ax.axhline(0.810, color="#4C72B0", linewidth=1, linestyle=":", alpha=0.6)
ax.axhline(0.940, color="#DD8452", linewidth=1, linestyle=":", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(quat_methods, fontsize=9)
ax.set_ylabel("Max G", fontsize=11)
ax.set_title("(c) Quaternary: method comparison", fontsize=12, fontweight="bold")
ax.set_ylim(0.5, 1.05)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---------- (d) Gap to train best: Grid vs BO ----------
ax = axes[1, 1]
# 与训练集最优的差距（绝对值，越小越好）
gaps = {
    "Ternary\nGrid (XGB)": [0.822 - 0.819, 0.956 - 0.950],
    "Ternary\nBO (XGB, fixed)": [0.822 - 0.759, 0.956 - 0.950],
    "Quaternary\nGrid (XGB)": [0.810 - 0.806, 0.940 - 0.935],
    "Quaternary\nBO (XGB)": [0.810 - 0.721, 0.940 - 0.703],
}
labels = list(gaps.keys())
gap_a45 = [gaps[k][0] for k in labels]
gap_a135 = [gaps[k][1] for k in labels]

x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, gap_a45, w, label="angle=45°", color="#4C72B0", edgecolor="k")
b2 = ax.bar(x + w/2, gap_a135, w, label="angle=135°", color="#DD8452", edgecolor="k")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("|Train best − Method|", fontsize=11)
ax.set_title("(d) Gap to train best (smaller = better)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.28)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.savefig(rf"{BASE}\figs\summary_all.png", dpi=200, bbox_inches="tight")
print("saved: figs/summary_all.png")