#!/usr/bin/env python3
"""Permutation feature importance for the saved severity models (cam), to show
WHICH acoustic features drive the discharge / suction level reading.

Output: run/fig_importance.png  (+ appends 'feature_importance' to results.json)
Run: python3 run/importance.py
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "features_allch.csv"))
m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]

panels = [("valveOut", "discharge"), ("valveIn", "suction")]
fig, axarr = plt.subplots(1, 2, figsize=(13, 6))
imp_out = {}
for j, (target, axis) in enumerate(panels):
    b = joblib.load(os.path.join(HERE, "models", f"{target}_severity_cam.joblib"))
    d = m1[(m1.dev_type == "cam") & (m1[target].isin(b["levels"]))].sample(
        n=min(2500, len(m1)), random_state=0)
    cols = b["features"]
    X = b["scaler"].transform(d[cols].values)
    y = d[target].map(b["rankmap"]).values
    r = permutation_importance(b["model"], X, y, n_repeats=5,
                               random_state=0, scoring="neg_mean_absolute_error")
    order = np.argsort(r.importances_mean)[::-1][:12]
    names = [cols[i] for i in order]
    vals = r.importances_mean[order]
    imp_out[axis] = {cols[i]: float(r.importances_mean[i]) for i in order}
    ax = axarr[j]
    ax.barh(range(len(names))[::-1], vals)
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"{axis} severity (cam): top features\n(permutation MAE drop)")
    ax.set_xlabel("importance")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_importance.png"), dpi=110)
print("wrote fig_importance.png")

res = json.load(open(os.path.join(HERE, "results.json")))
res["feature_importance_cam_severity"] = imp_out
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
print("appended feature_importance to results.json")
for axis, d in imp_out.items():
    print(axis, "->", list(d.keys())[:6])
