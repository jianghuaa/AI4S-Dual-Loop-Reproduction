import json
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

d = json.load(open(rf"{BASE}\data\pl_cpl_3.json", encoding="utf-8"))

glum = np.abs(np.array([x["glum"] for x in d])).reshape(-1, 1)
ff = np.abs(np.array([x["ff"] for x in d])).reshape(-1, 1)
gs = MinMaxScaler().fit_transform(glum).ravel()
fs = MinMaxScaler().fit_transform(ff).ravel()
G = 0.5 * fs + 0.5 * gs

th = np.array([x["thickness"] for x in d])
st = np.array([x["stretching"] for x in d])
an = np.array([x["angle"] for x in d])

print("=== training set coverage ===")
print(f"thickness : min={th.min()} max={th.max()} unique={sorted(set(th.tolist()))}")
print(f"stretching: min={st.min()} max={st.max()} unique={sorted(set(st.tolist()))}")
print(f"angle     : unique={sorted(set(an.tolist()))}")
print()

print("=== per-angle G_max ===")
for a in sorted(set(an.tolist())):
    m = an == a
    i = int(np.where(m)[0][np.argmax(G[m])])
    print(f"angle={a}: n={int(m.sum())} G_max={G[i]:.4f} "
          f"@ th={d[i]['thickness']}, st={d[i]['stretching']}")
print()

print(f"n(thickness<=25) = {int((th <= 25).sum())}")
print(f"n(thickness==20) = {int((th == 20).sum())}")
print()

print("=== grid search on XGBoost ===")
model = joblib.load(rf"{BASE}\models\3\best_xgboost_model.pkl")
scaler = joblib.load(rf"{BASE}\models\3\scaler.pkl")

thg = np.linspace(20, 80, 121)
stg = np.linspace(10, 150, 141)
TH, ST = np.meshgrid(thg, stg)
TH_f, ST_f = TH.ravel(), ST.ravel()

for angle in [45, 135]:
    X = np.c_[TH_f, ST_f, np.full(TH_f.size, angle)]
    Gp = model.predict(scaler.transform(X))
    i = int(Gp.argmax())
    print(f"angle={angle}: full grid max={Gp[i]:.4f} "
          f"@ th={TH_f[i]:.1f}, st={ST_f[i]:.1f}")

    mask = ((TH_f >= th.min()) & (TH_f <= th.max())
            & (ST_f >= st.min()) & (ST_f <= st.max()))
    Gp_in = Gp.copy()
    Gp_in[~mask] = -np.inf
    i2 = int(Gp_in.argmax())
    print(f"          in-range max={Gp_in[i2]:.4f} "
          f"@ th={TH_f[i2]:.1f}, st={ST_f[i2]:.1f}")