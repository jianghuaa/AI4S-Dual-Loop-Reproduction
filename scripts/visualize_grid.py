import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter


rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 11

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
d = json.load(open(rf"{BASE}\data\pl_cpl_4.json", encoding="utf-8"))

glum = np.abs(np.array([x["glum"] for x in d])).reshape(-1, 1)
ff = np.abs(np.array([x["ff"] for x in d])).reshape(-1, 1)
gs = MinMaxScaler().fit_transform(glum).ravel()
fs = MinMaxScaler().fit_transform(ff).ravel()
G = 0.5 * fs + 0.5 * gs

th = np.array([x["thickness"] for x in d])
st = np.array([x["stretching"] for x in d])
an = np.array([x["angle"] for x in d])

model = joblib.load(rf"{BASE}\models\4\best_xgboost_model.pkl")
scaler = joblib.load(rf"{BASE}\models\4\scaler.pkl")

thg = np.linspace(30, 80, 200)
stg = np.linspace(10, 150, 200)
TH, ST = np.meshgrid(thg, stg)

# 预测两个切片
G_pred = {}
for angle in [45, 135]:
    Xg = np.c_[TH.ravel(), ST.ravel(), np.full(TH.size, angle)]
    G_pred[angle] = model.predict(scaler.transform(Xg)).reshape(TH.shape)

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

# ---- (a) angle=45 ----
ax = axes[0, 0]
im = ax.contourf(TH, ST, G_pred[45], levels=25, cmap="viridis", vmin=0, vmax=1)
m = an == 45
ax.scatter(th[m], st[m], c="white", s=60, edgecolor="k", linewidth=0.8,
           alpha=0.85, zorder=3, label="train pts")
i_grid = int(G_pred[45].argmax())
ax.scatter(TH.ravel()[i_grid], ST.ravel()[i_grid], marker="*", s=600,
           facecolor="gold", edgecolor="red", linewidth=2, zorder=4,
           label=f"grid max G={G_pred[45].ravel()[i_grid]:.3f}")
ax.set_xlabel("Thickness (μm)")
ax.set_ylabel("Stretching (%)")
ax.set_title("(a) XGBoost prediction: angle=45°", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
plt.colorbar(im, ax=ax, label="Predicted G", fraction=0.046)

# ---- (b) angle=135 ----
ax = axes[0, 1]
im = ax.contourf(TH, ST, G_pred[135], levels=25, cmap="viridis", vmin=0, vmax=1)
m = an == 135
ax.scatter(th[m], st[m], c="white", s=60, edgecolor="k", linewidth=0.8,
           alpha=0.85, zorder=3, label="train pts")
i_grid = int(G_pred[135].argmax())
ax.scatter(TH.ravel()[i_grid], ST.ravel()[i_grid], marker="*", s=600,
           facecolor="gold", edgecolor="red", linewidth=2, zorder=4,
           label=f"grid max G={G_pred[135].ravel()[i_grid]:.3f}")
ax.set_xlabel("Thickness (μm)")
ax.set_ylabel("Stretching (%)")
ax.set_title("(b) XGBoost prediction: angle=135°", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
plt.colorbar(im, ax=ax, label="Predicted G", fraction=0.046)

# ---- (c) difference ----
ax = axes[1, 0]
delta = G_pred[135] - G_pred[45]
im = ax.contourf(TH, ST, delta, levels=25, cmap="coolwarm",
                 vmin=-0.3, vmax=0.3)
ax.scatter(th, st, c="white", s=20, edgecolor="k", linewidth=0.5, alpha=0.4)
ax.scatter(30, 25, marker="*", s=700, facecolor="gold", edgecolor="red",
           linewidth=2.5, zorder=4, label="train best (30, 25)")
ax.set_xlabel("Thickness (μm)")
ax.set_ylabel("Stretching (%)")
ax.set_title("(c) ΔG = G(135°) − G(45°)", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
plt.colorbar(im, ax=ax, label="ΔG", fraction=0.046)

# ---- (d) method comparison bar ----
ax = axes[1, 1]
methods = ["Train\nbest", "Grid\nsearch", "BO\nfixed", "BO\noriginal"]
a45 = [0.8219, 0.8189, 0.7587, 0.8189]
a135 = [0.9557, 0.9496, 0.9496, 0.7172]
x = np.arange(len(methods))
w = 0.35
ax.bar(x - w/2, a45, w, label="angle=45°", color="#4C72B0", edgecolor="k")
ax.bar(x + w/2, a135, w, label="angle=135°", color="#DD8452", edgecolor="k")
for i, (v1, v2) in enumerate(zip(a45, a135)):
    ax.text(i - w/2, v1 + 0.015, f"{v1:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, v2 + 0.015, f"{v2:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Max G")
ax.set_ylim(0, 1.15)
ax.set_title("(d) Method comparison", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.savefig(rf"{BASE}\figs\grid_search_overview_v2.png", dpi=200, bbox_inches="tight")
print("saved: figs/grid_search_overview_v2.png")