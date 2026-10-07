"""Generate Top-10 tables for PPT: ternary, quaternary, and combined."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

def build_table(json_path, label):
    d = json.load(open(json_path, encoding="utf-8"))
    glum = np.abs(np.array([x["glum"] for x in d])).reshape(-1, 1)
    ff   = np.abs(np.array([x["ff"]   for x in d])).reshape(-1, 1)
    gs = MinMaxScaler().fit_transform(glum).ravel()
    fs = MinMaxScaler().fit_transform(ff).ravel()
    G  = 0.5 * fs + 0.5 * gs

    rows = []
    for i, x in enumerate(d):
        rows.append({
            "体系": label,
            "angle(°)": x["angle"],
            "厚度(μm)": x["thickness"],
            "拉伸(%)":  x["stretching"],
            "|g_lum|":  round(abs(x["glum"]), 3),
            "CPL积分":  round(abs(x["ff"]), 1),
            "G":        round(G[i], 4),
        })
    df = pd.DataFrame(rows).sort_values("G", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df

df3 = build_table(rf"{BASE}\data\pl_cpl_3.json", "三元")
df4 = build_table(rf"{BASE}\data\pl_cpl_4.json", "四元")

print("=" * 80)
print("三元 Top-10")
print("=" * 80)
print(df3.head(10).to_string(index=False))
print()
print("=" * 80)
print("四元 Top-10")
print("=" * 80)
print(df4.head(10).to_string(index=False))

# 合并保存到 Excel
out = rf"{BASE}\figs\top10_tables.xlsx"
with pd.ExcelWriter(out) as w:
    df3.head(10).to_excel(w, sheet_name="三元Top10", index=False)
    df4.head(10).to_excel(w, sheet_name="四元Top10", index=False)
print(f"\nExcel saved: {out}")