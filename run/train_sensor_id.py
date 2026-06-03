#!/usr/bin/env python3
"""Train a tiny sensor-family auto-detector (mic vs camera-mic) on 8 s windows so
the listener can pick the right model family itself — no manual selector.

The two families are acoustically very distinct (pump tone ~270 Hz on mics vs a
~720 Hz structural tone on the AGC camera mics), so this is near-perfect. Trained
gain-invariant (rms_db dropped) so loopback volume changes don't fool it.

Saves models/sensor_id.joblib.  Run: python3 run/train_sensor_id.py
"""
import os, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
META = {"folder", "session", "device", "dev_type", "win", "noise", "noise_cat",
        "M2", "M3", "M4", "aeration", "valveIn", "valveOut"}


def main():
    a = pd.read_csv(os.path.join(HERE, "features_5_25_windows.csv"))
    b = pd.read_csv(os.path.join(HERE, "features_blockage_windows.csv"))
    feats = [c for c in a.columns if c not in META]
    cols = [c for c in feats if c != "rms_db"]          # gain-invariant
    df = pd.concat([a[feats + ["dev_type", "session"]],
                    b[feats + ["dev_type", "session"]]], ignore_index=True)
    X = df[cols].values
    y = (df.dev_type == "cam").astype(int).values
    g = df.session.values

    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    acc = accuracy_score(y, oof)
    print(f"sensor-id window CV acc={acc:.4f}  (1=cam) n={len(df)}")

    sc = StandardScaler().fit(X)
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(sc.transform(X), y)
    joblib.dump({"scaler": sc, "model": m, "features": cols,
                 "classes": ["mic", "cam"], "window_cv_acc": float(acc)},
                os.path.join(HERE, "models", "sensor_id.joblib"))
    print("saved models/sensor_id.joblib")


if __name__ == "__main__":
    main()
