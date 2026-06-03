#!/usr/bin/env python3
"""Train and test a LIVE 'which machine is running' detector on 8 s windows.

The rig has M1 (large pump, ALWAYS on -> cannot be isolated from sound), plus
switchable M2 (2nd large pump), M3 (exhaust fan), M4 (exhaust fan). 5_25 is a
clean 2x2x2 factorial of M2/M3/M4, so we learn each machine's presence.

We train per device type, session-grouped CV (windows from one recording never
split across train/test). Output is multi-label: independent on/off for M2, M3,
M4 (the deployable form - tells you which of the three is running, including
combinations). We also report a single 8-way combo classifier for reference.

Honest expectations from prior work: M3 (loud fan, 4-8 kHz airflow) is easy; M2
(pump, 250-500 Hz tonal) moderate; M4 (near-silent fan) hardest. M1 is never a
target (always on).

Saves models/pump_<M2|M3|M4>_<mic|cam>.joblib.
Output: run/which_pump_results.json, run/fig_which_pump.png
Run: python3 run/which_pump.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
META = ["session", "device", "dev_type", "win", "M2", "M3", "M4",
        "aeration", "valveIn", "valveOut"]


def clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          min_samples_leaf=30, random_state=0)


def binary_cv(d, cols, target):
    X = d[cols].values
    y = d[target].values
    groups = d.session.values
    gkf = GroupKFold(n_splits=5)
    proba = np.zeros(len(d)); pred = np.zeros(len(d))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), y[tr])
        proba[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        pred[te] = (proba[te] >= 0.5).astype(int)
    return {"f1": float(f1_score(y, pred)), "acc": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "n": int(len(d)), "n_on": int(y.sum())}


def combo_cv(d, cols):
    """8-way (M2,M3,M4) combo single-label accuracy, for reference."""
    X = d[cols].values
    y = (d.M2.astype(str) + d.M3.astype(str) + d.M4.astype(str)).values
    groups = d.session.values
    gkf = GroupKFold(n_splits=5)
    pred = np.empty(len(d), dtype=object)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return {"combo_acc": float(accuracy_score(y, pred)),
            "chance": 1.0 / len(np.unique(y)), "n_classes": int(len(np.unique(y)))}


def main():
    df = pd.read_csv(os.path.join(HERE, "features_5_25_windows.csv"))
    # exclude the anomalous aeration-ON windows so machine presence isn't confounded
    df = df[df.aeration == 0]
    cols = [c for c in df.columns if c not in META]
    res = {"window_s": 8.0, "note": ("M1 always on (not a target). Multi-label "
            "independent presence of M2/M3/M4; trained on 5_25 (aeration OFF).")}

    for dev_type in ["mic", "cam"]:
        sub = df[df.dev_type == dev_type]
        r = {m: binary_cv(sub, cols, m) for m in ("M2", "M3", "M4")}
        r["combo_8way"] = combo_cv(sub, cols)
        res[dev_type] = r
        print(f"[{dev_type}] M2 F1={r['M2']['f1']:.2f} M3 F1={r['M3']['f1']:.2f} "
              f"M4 F1={r['M4']['f1']:.2f} | 8-way combo acc={r['combo_8way']['combo_acc']:.2f} "
              f"(chance {r['combo_8way']['chance']:.2f})")
        # production per-machine models (all 5_25 aeration-off windows)
        for m in ("M2", "M3", "M4"):
            sc = StandardScaler().fit(sub[cols].values)
            mdl = clf().fit(sc.transform(sub[cols].values), sub[m].values)
            joblib.dump({"scaler": sc, "model": mdl, "features": cols,
                         "kind": "pump_presence", "machine": m, "dev_type": dev_type,
                         "window_s": 8.0,
                         "label": {"M2": "2nd large pump", "M3": "exhaust fan (loud)",
                                   "M4": "exhaust fan (near-silent)"}[m]},
                        os.path.join(MODELS, f"pump_{m}_{dev_type}.joblib"))

    json.dump(res, open(os.path.join(HERE, "which_pump_results.json"), "w"), indent=2)
    print("wrote which_pump_results.json")

    # figure: per-machine F1 by device type
    fig, ax = plt.subplots(figsize=(8, 4.5))
    machines = ["M2", "M3", "M4"]
    labels = ["M2\n(2nd pump)", "M3\n(loud fan)", "M4\n(silent fan)"]
    x = np.arange(3); w = 0.35
    for k, dev_type in enumerate(["mic", "cam"]):
        f1s = [res[dev_type][m]["f1"] for m in machines]
        ax.bar(x + (k - 0.5) * w, f1s, w, label=dev_type)
        for i, v in enumerate(f1s):
            ax.text(x[i] + (k - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 (session-grouped CV)")
    ax.set_title("Which machine is running? (live 8 s windows, 5_25)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_which_pump.png"), dpi=110)
    print("wrote fig_which_pump.png")


if __name__ == "__main__":
    main()
