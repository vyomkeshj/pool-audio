#!/usr/bin/env python3
"""Honest held-out test of the LIVE detectors: hold out whole recording sessions,
train on the rest, then 'stream' each unseen session window-by-window and score:
  * window-level accuracy  (verdict from a single 8 s window)
  * clip-level accuracy     (verdict after listening to the whole session, i.e.
                             mean probability over its windows -> what a live
                             monitor reports after a few seconds of smoothing)

This is the number that matters for live deployment: on a recording it has never
heard, listening for a few windows, how often is the on/off / which-machine
verdict correct?

Output: run/live_test_results.json  (+ console table)
Run: python3 run/test_live.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
META = ["session", "device", "dev_type", "win", "M2", "M3", "M4",
        "aeration", "valveIn", "valveOut"]


def clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          min_samples_leaf=30, random_state=0)


def held_out_stream(d, target, cols, drop_level=False):
    """GroupKFold by session; collect per-window prob on held-out sessions, then
    aggregate to clip level. Returns window-acc and clip-acc/f1."""
    if drop_level:
        cols = [c for c in cols if c != "rms_db"]
    X = d[cols].values
    y = d[target].values
    sess = d.session.values
    gkf = GroupKFold(n_splits=5)
    win_prob = np.zeros(len(d))
    for tr, te in gkf.split(X, y, sess):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), y[tr])
        win_prob[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    win_pred = (win_prob >= 0.5).astype(int)
    win_acc = float(accuracy_score(y, win_pred))
    win_f1 = float(f1_score(y, win_pred, zero_division=0))

    # SINGLE-SENSOR clip level (the honest live number): one device, mean prob over
    # its ~14 time windows. Each recording has 8 sibling mics; we do NOT pool them.
    dev = d.device.values
    s1 = pd.DataFrame({"k": [f"{a}/{b}" for a, b in zip(sess, dev)], "y": y, "p": win_prob})
    g1 = s1.groupby("k").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    p1 = (g1.p >= 0.5).astype(int)

    # multi-sensor FUSED clip (8 channels of a recording averaged) - an upgrade if
    # you deploy several mics; reported separately so it isn't mistaken for 1 mic.
    sf = pd.DataFrame({"k": sess, "y": y, "p": win_prob})
    gf = sf.groupby("k").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    pf = (gf.p >= 0.5).astype(int)

    return {
        "window_acc": win_acc, "window_f1": win_f1,
        "clip_acc_single_sensor": float(accuracy_score(g1.y, p1)),
        "clip_f1_single_sensor": float(f1_score(g1.y, p1, zero_division=0)),
        "clip_acc_fused_8ch": float(accuracy_score(gf.y, pf)),
        "clip_f1_fused_8ch": float(f1_score(gf.y, pf, zero_division=0)),
        "n_windows": int(len(d)), "n_single_sensor_clips": int(len(g1)),
        "n_recordings": int(len(gf)), "n_recordings_on": int((gf.y == 1).sum()),
    }


def main():
    df = pd.read_csv(os.path.join(HERE, "features_5_25_windows.csv"))
    cols = [c for c in df.columns if c not in META]
    out = {"window_s": 8.0, "smoothing": "clip = mean prob over a session's windows"}

    print(f"{'task':28s} {'sensor':6s} {'win-acc':>8s} "
          f"{'1mic-clip':>10s} {'8ch-fused':>10s}  n_rec")
    print("-" * 78)

    def show(name, dt, r):
        print(f"{name:28s} {dt:6s} {r['window_acc']:8.3f} "
              f"{r['clip_acc_single_sensor']:10.3f} {r['clip_acc_fused_8ch']:10.3f}  "
              f"{r['n_recordings']} ({r['n_recordings_on']} on)")

    # --- aeration: matched config (vin1/vout1), gain-invariant ---
    out["aeration_matched_gain_invariant"] = {}
    for dt in ["mic", "cam"]:
        d = df[(df.dev_type == dt) & (df.valveIn == 1) & (df.valveOut == 1)]
        r = held_out_stream(d, "aeration", cols, drop_level=True)
        out["aeration_matched_gain_invariant"][dt] = r
        show("aeration on/off (matched)", dt, r)

    # --- which machine: M2/M3/M4 presence (aeration-off windows) ---
    out["machine_presence"] = {}
    dff = df[df.aeration == 0]
    for m in ["M2", "M3", "M4"]:
        out["machine_presence"][m] = {}
        for dt in ["mic", "cam"]:
            d = dff[dff.dev_type == dt]
            r = held_out_stream(d, m, cols)
            out["machine_presence"][m][dt] = r
            name = {"M2": "M2 2nd-pump", "M3": "M3 loud-fan", "M4": "M4 silent-fan"}[m]
            show(name + " present?", dt, r)

    json.dump(out, open(os.path.join(HERE, "live_test_results.json"), "w"), indent=2)
    print("\nwrote live_test_results.json")
    print("\nHeld out by recording session, so the 8 sibling mics (and 8 cams) of one")
    print("recording never split across train/test. '1mic-clip' = ONE sensor, mean prob")
    print("over its ~14 time windows (the honest single-mic live number). '8ch-fused' =")
    print("all 8 same-type channels of a recording averaged (an upgrade if you deploy")
    print("several sensors). Window-acc = a single 8 s window from a single sensor.")


if __name__ == "__main__":
    main()
