#!/usr/bin/env python3
"""Train and rigorously test a LIVE aeration on/off detector (per device type),
on 8 s windows.

Two evaluations, the second decisive:
  (1) session-grouped CV over ALL 5_25 windows - overall on/off accuracy. But
      aeration-ON only exists at valve config vin1/vout1, so a model could cheat
      by recognising that config instead of aeration.
  (2) MATCHED-config CV - restrict to vin1/vout1 windows only and toggle aeration
      with the valve held fixed (aux M2/M3/M4 still vary on both sides). This
      isolates the aeration signal from the valve confound. We run it both WITH
      and WITHOUT the absolute-level feature (rms_db) to show the detector does
      not merely key on a recording-gain artifact.

Saves the production live model (trained on matched-config windows, gain-invariant
features) to models/aeration_live_<mic|cam>.joblib.

Output: run/aeration_results.json, run/fig_aeration.png
Run: python3 run/aeration.py
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)
META = ["session", "device", "dev_type", "win", "M2", "M3", "M4",
        "aeration", "valveIn", "valveOut"]


def cols_of(df, drop_level):
    c = [x for x in df.columns if x not in META]
    return [x for x in c if x != "rms_db"] if drop_level else c


def clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          min_samples_leaf=30, random_state=0)


def grouped_cv(d, cols):
    X = d[cols].values
    y = d.aeration.values
    groups = d.session.values
    n = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n))
    proba = np.zeros(len(d)); pred = np.zeros(len(d))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), y[tr])
        proba[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        pred[te] = (proba[te] >= 0.5).astype(int)
    out = {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
           "n": int(len(d)), "n_on": int(y.sum()), "n_sessions": int(n)}
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    except ValueError:
        out["roc_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def main():
    df = pd.read_csv(os.path.join(HERE, "features_5_25_windows.csv"))
    res = {"window_s": 8.0, "hop_s": 4.0,
           "note": ("Aeration ON exists only at valve config vin1/vout1 in 5_25. "
                    "Matched-config CV (vin1/vout1 only) is the honest test; it "
                    "isolates aeration from the valve confound.")}

    for dev_type in ["mic", "cam"]:
        sub = df[df.dev_type == dev_type]
        matched = sub[(sub.valveIn == 1) & (sub.valveOut == 1)]
        r = {
            "all_5_25_grouped": grouped_cv(sub, cols_of(sub, drop_level=False)),
            "matched_vin1vout1_with_level": grouped_cv(matched, cols_of(matched, False)),
            "matched_vin1vout1_gain_invariant": grouped_cv(matched, cols_of(matched, True)),
            "n_matched": int(len(matched)),
            "n_matched_on": int((matched.aeration == 1).sum()),
            "n_matched_off": int((matched.aeration == 0).sum()),
        }
        res[dev_type] = r
        print(f"[{dev_type}] all-5_25 F1={r['all_5_25_grouped']['f1']:.3f} "
              f"AUC={r['all_5_25_grouped']['roc_auc']:.3f} | "
              f"MATCHED gain-invariant F1={r['matched_vin1vout1_gain_invariant']['f1']:.3f} "
              f"AUC={r['matched_vin1vout1_gain_invariant']['roc_auc']:.3f} "
              f"acc={r['matched_vin1vout1_gain_invariant']['acc']:.3f}")

        # production model: matched-config, gain-invariant features, all windows
        cols = cols_of(matched, drop_level=True)
        sc = StandardScaler().fit(matched[cols].values)
        m = clf().fit(sc.transform(matched[cols].values), matched.aeration.values)
        joblib.dump({"scaler": sc, "model": m, "features": cols,
                     "kind": "aeration_onoff", "dev_type": dev_type,
                     "window_s": 8.0, "trained_on": "5_25 vin1/vout1 windows",
                     "caveat": ("aeration only recorded at one valve config; "
                                "validity when M1 is throttled is unverified")},
                    os.path.join(MODELS, f"aeration_live_{dev_type}.joblib"))

    json.dump(res, open(os.path.join(HERE, "aeration_results.json"), "w"), indent=2)
    print("wrote aeration_results.json")

    # figure: matched-config confusion + AUC bars
    fig, axarr = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, dev_type in enumerate(["mic", "cam"]):
        r = res[dev_type]["matched_vin1vout1_gain_invariant"]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax = axarr[j]
        ax.imshow(cm, cmap="Blues")
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["off", "ON"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["off", "ON"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{dev_type}: matched vin1/vout1, gain-invariant\n"
                     f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f} acc={r['acc']:.3f}")
    fig.suptitle("Live aeration on/off (8 s windows) - decisive matched-config test")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_aeration.png"), dpi=110)
    print("wrote fig_aeration.png")


if __name__ == "__main__":
    main()
