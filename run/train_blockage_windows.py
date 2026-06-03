#!/usr/bin/env python3
"""Retrain M1 blockage severity on 8 s windows (matches the live listener), with
leakage-honest CV. Saves window models for the listener and reports:
  * window-level within-1 / exact (one 8 s window, one sensor)
  * single-sensor clip-level (one sensor, mean prediction over its windows - what
    rolling smoothing converges to). Held out so the 8 sibling mics never split.
Two CV groupings: config (vin,vout) and leave-one-campaign-out.

Saves models/<valveOut|valveIn>_severity_win_<mic|cam>.joblib
Output: run/blockage_windows_results.json
Run: python3 run/train_blockage_windows.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
META = ["folder", "session", "device", "dev_type", "win", "noise", "noise_cat",
        "valveIn", "valveOut"]
DISCHARGE = [1, 2, 3, 4, 5, 8, 11]
SUCTION = [1, 2, 3, 4, 5, 8]


def reg():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         min_samples_leaf=40, random_state=0)


def metrics(true_rank, pred_cont, maxr):
    pr = np.clip(np.rint(pred_cont), 0, maxr).astype(int)
    return {"within1": float(np.mean(np.abs(pr - true_rank) <= 1)),
            "exact": float(np.mean(pr == true_rank)),
            "MAE_steps": float(np.mean(np.abs(pr - true_rank))),
            "spearman": float(spearmanr(true_rank, pred_cont).correlation)}


def cv(d, cols, target, levels, group_col):
    rankmap = {v: i for i, v in enumerate(levels)}
    d = d[d[target].isin(levels)].copy()
    d["rank"] = d[target].map(rankmap)
    X, y = d[cols].values, d["rank"].values
    groups = d[group_col].astype(str).values if group_col != "config" else \
        (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).values
    ng = len(np.unique(groups))
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = reg().fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    win = metrics(y, oof, len(levels) - 1)
    win["n"] = int(len(d))
    # single-sensor clip-level: mean prediction per (session, device)
    g = pd.DataFrame({"k": (d.session + "/" + d.device).values,
                      "r": y, "p": oof}).groupby("k").agg(r=("r", "first"), p=("p", "mean"))
    clip = metrics(g.r.values, g.p.values, len(levels) - 1)
    clip["n_clips"] = int(len(g))
    return {"window": win, "clip_single_sensor": clip}


def main():
    df = pd.read_csv(os.path.join(HERE, "features_blockage_windows.csv"))
    cols = [c for c in df.columns if c not in META]
    res = {"window_s": 8.0}
    for axis, target, levels in [("discharge", "valveOut", DISCHARGE),
                                 ("suction", "valveIn", SUCTION)]:
        res[axis] = {}
        for dev in ["mic", "cam"]:
            d = df[df.dev_type == dev]
            res[axis][dev] = {
                "config_grouped": cv(d, cols, target, levels, "config"),
                "leave_campaign_out": cv(d, cols, target, levels, "folder"),
            }
            # production window model on all windows of this device type
            rankmap = {v: i for i, v in enumerate(levels)}
            dd = d[d[target].isin(levels)]
            sc = StandardScaler().fit(dd[cols].values)
            m = reg().fit(sc.transform(dd[cols].values), dd[target].map(rankmap).values)
            joblib.dump({"scaler": sc, "model": m, "features": cols, "kind": "severity",
                         "levels": levels, "rankmap": rankmap, "target": target,
                         "dev_type": dev, "window_s": 8.0},
                        os.path.join(MODELS, f"{target}_severity_win_{dev}.joblib"))
            cg = res[axis][dev]["config_grouped"]
            lc = res[axis][dev]["leave_campaign_out"]
            print(f"{axis:9s} {dev}: config-grp win w1={cg['window']['within1']:.2f}/"
                  f"ex={cg['window']['exact']:.2f} clip w1={cg['clip_single_sensor']['within1']:.2f}/"
                  f"ex={cg['clip_single_sensor']['exact']:.2f} | campaign-out clip w1="
                  f"{lc['clip_single_sensor']['within1']:.2f}/ex={lc['clip_single_sensor']['exact']:.2f}")
    json.dump(res, open(os.path.join(HERE, "blockage_windows_results.json"), "w"), indent=2)
    print("wrote blockage_windows_results.json")


if __name__ == "__main__":
    main()
