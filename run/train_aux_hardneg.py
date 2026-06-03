#!/usr/bin/env python3
"""Harden the aeration + which-machine detectors with HARD NEGATIVES.

The first versions were trained only within 5_25, so live they false-alarm on
operating conditions they never saw — the broad listener test showed aeration
firing on ~19 % of non-aeration mic clips (mostly throttled-M1 blockage clips).

Fix (same lesson as the old detect_m3 work): add the M1-only valve-sweep windows
— which are definitively aeration-OFF and all-aux-OFF, across every blockage level,
noise type and campaign — as negatives. This forces each detector onto its own
machine's signature rather than "anything that isn't a quiet 5_25 baseline".

Held-out by session; reports recall + false-alarm, split out on the M1-only
blockage negatives specifically (the ones that used to trip it). Overwrites
models/aeration_live_<dev>.joblib and models/pump_<M2|M3|M4>_<dev>.joblib.

Output: run/aux_hardneg_results.json
Run: python3 run/train_aux_hardneg.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
FEATS = None  # set after load


def clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          min_samples_leaf=30, random_state=0)


def load():
    a = pd.read_csv(os.path.join(HERE, "features_5_25_windows.csv"))
    a["src"] = "5_25"
    b = pd.read_csv(os.path.join(HERE, "features_blockage_windows.csv"))
    for c in ("M2", "M3", "M4", "aeration"):
        b[c] = 0
    b["src"] = "m1_block"
    global FEATS
    META = {"folder", "session", "device", "dev_type", "win", "noise", "noise_cat",
            "M2", "M3", "M4", "aeration", "valveIn", "valveOut", "src"}
    FEATS = [c for c in a.columns if c not in META]
    keep = FEATS + ["session", "dev_type", "M2", "M3", "M4", "aeration",
                    "valveIn", "valveOut", "src"]
    return pd.concat([a[keep], b[keep]], ignore_index=True)


def sample_m1_negatives(df, dev, n=8000, seed=0):
    """Stratified sample of M1-only blockage windows (this dev) across (vin,vout)."""
    m = df[(df.src == "m1_block") & (df.dev_type == dev)]
    rng = np.random.RandomState(seed)
    idx = []
    for _, g in m.groupby(["valveIn", "valveOut"]):
        take = min(len(g), max(1, n // max(1, m.groupby(['valveIn', 'valveOut']).ngroups)))
        idx += list(rng.choice(g.index, size=take, replace=False))
    return df.loc[idx]


def cv_eval(pos, neg, cols, drop_level=False):
    use = [c for c in cols if not (drop_level and c == "rms_db")]
    d = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    X, y = d[use].values, d.y.values
    groups = d.session.values
    proba = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), y[tr])
        proba[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (proba >= 0.5).astype(int)
    d["pred"] = pred
    recall = float(pred[y == 1].mean())
    fa_all = float(pred[y == 0].mean())
    fa_m1 = float(d[(d.y == 0) & (d.src == "m1_block")].pred.mean())
    fa_525 = float(d[(d.y == 0) & (d.src == "5_25")].pred.mean()) if \
        ((d.y == 0) & (d.src == "5_25")).any() else None
    return {"recall": recall, "false_alarm": fa_all, "fa_on_m1_blockage": fa_m1,
            "fa_on_5_25_off": fa_525, "f1": float(f1_score(y, pred)),
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def fit_save(pos, neg, cols, name, extra, drop_level=False):
    use = [c for c in cols if not (drop_level and c == "rms_db")]
    d = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    sc = StandardScaler().fit(d[use].values)
    m = clf().fit(sc.transform(d[use].values), d.y.values)
    bundle = {"scaler": sc, "model": m, "features": use, "window_s": 8.0,
              "hard_negatives": "M1-only blockage windows", **extra}
    joblib.dump(bundle, os.path.join(MODELS, name))


def main():
    df = load()
    res = {"note": "hard-negative retrain: M1-only blockage windows added as negatives"}
    for dev in ["mic", "cam"]:
        res[dev] = {}
        dd = df[df.dev_type == dev]
        m1neg = sample_m1_negatives(df, dev)

        # aeration: pos=aeration-on; neg=5_25 aeration-off + M1 blockage; gain-invariant
        pos = dd[(dd.src == "5_25") & (dd.aeration == 1)]
        neg = pd.concat([dd[(dd.src == "5_25") & (dd.aeration == 0)], m1neg])
        res[dev]["aeration"] = cv_eval(pos, neg, FEATS, drop_level=True)
        fit_save(pos, neg, FEATS, f"aeration_live_{dev}.joblib",
                 {"kind": "aeration_onoff", "dev_type": dev,
                  "caveat": "aeration only recorded at vin1/vout1"}, drop_level=True)

        # machines: pos/neg within aeration-off; + M1 blockage as all-off negatives
        for m in ["M2", "M3", "M4"]:
            base = dd[(dd.src == "5_25") & (dd.aeration == 0)]
            pos = base[base[m] == 1]
            neg = pd.concat([base[base[m] == 0], m1neg])
            res[dev][m] = cv_eval(pos, neg, FEATS)
            fit_save(pos, neg, FEATS, f"pump_{m}_{dev}.joblib",
                     {"kind": "pump_presence", "machine": m, "dev_type": dev,
                      "label": {"M2": "2nd large pump", "M3": "exhaust fan (loud)",
                                "M4": "exhaust fan (near-silent)"}[m]})
        r = res[dev]
        print(f"[{dev}] aeration: recall={r['aeration']['recall']:.2f} "
              f"FA_all={r['aeration']['false_alarm']:.3f} "
              f"FA_on_blockage={r['aeration']['fa_on_m1_blockage']:.3f}")
        for m in ["M2", "M3", "M4"]:
            print(f"       {m}: recall={r[m]['recall']:.2f} FA_all={r[m]['false_alarm']:.3f} "
                  f"FA_on_blockage={r[m]['fa_on_m1_blockage']:.3f}")
    json.dump(res, open(os.path.join(HERE, "aux_hardneg_results.json"), "w"), indent=2)
    print("wrote aux_hardneg_results.json; models overwritten")


if __name__ == "__main__":
    main()
