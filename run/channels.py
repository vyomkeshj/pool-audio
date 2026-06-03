#!/usr/bin/env python3
"""Per-CHANNEL acoustic signatures and model quality — quantify how the 8 mics
(and 8 camera-mics) differ from one another. This is the "channel placement
dominates" story made concrete.

Computes, per device (mic1..8, cam1..8):
  * mean Welch spectrum over M1-only clean clips (the channel's characteristic shape)
  * mean of key features (level rms_db, pump-tone freq, 4-8 kHz band, centroid)
  * blockage-reading quality: config-grouped CV severity (within-1 + Spearman) for
    discharge and suction — i.e. how well THAT channel reads the level.

Outputs: run/channels.json (features + quality) and run/channels_psd.npz (spectra).
Run: python3 run/channels.py
"""
import os, sys, json, random, warnings
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR = 44100
NPER = 8192
MAXF = 12000
PER_CH_PSD = 22
rng = random.Random(0)
DISCHARGE = [1, 2, 3, 4, 5, 8, 11]
SUCTION = [1, 2, 3, 4, 5, 8]
DEVICES = [f"mic{i}" for i in range(1, 9)] + [f"cam{i}" for i in range(1, 9)]
META = ["folder", "file", "device", "dev_type", "session", "noise", "noise_cat",
        "M2", "M3", "M4", "aeration", "valveIn", "valveOut"]


def mean_psd(paths):
    acc = None; fref = None; n = 0
    for p in paths[:PER_CH_PSD]:
        try:
            x, sr = sf.read(p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                x = signal.resample(x, int(len(x) * SR / sr))
            f, pp = signal.welch(x - x.mean(), SR, nperseg=NPER, noverlap=NPER // 2)
        except Exception:
            continue
        fref = f if fref is None else fref
        acc = pp if acc is None else acc + pp
        n += 1
    return (fref, acc / n) if n else (None, None)


def reg():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         min_samples_leaf=15, random_state=0)


def channel_quality(d, target, levels, cols):
    rankmap = {v: i for i, v in enumerate(levels)}
    d = d[d[target].isin(levels)].copy()
    if d[target].nunique() < 3 or len(d) < 40:
        return None
    d["rank"] = d[target].map(rankmap)
    X, y = d[cols].values, d["rank"].values
    groups = (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).values
    ng = len(np.unique(groups))
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(min(5, ng)).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = reg().fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    pr = np.clip(np.rint(oof), 0, len(levels) - 1).astype(int)
    return {"within1": float(np.mean(np.abs(pr - y) <= 1)),
            "exact": float(np.mean(pr == y)),
            "spearman": float(spearmanr(y, oof).correlation), "n": int(len(d))}


def main():
    idx = json.load(open(os.path.join(HERE, "measurement_index.json")))
    # build device -> list of M1-only clean file paths
    paths = {d: [] for d in DEVICES}
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M3"] or p["M4"] or p["aeration"] or p["noise"] != "N":
            continue
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                if dev in paths:
                    paths[dev].append(os.path.join(ROOT, s["disk_path"], fn))
    for d in paths:
        rng.shuffle(paths[d])

    psd = {}
    for d in DEVICES:
        f, p = mean_psd(paths[d])
        if p is not None:
            psd["freq"] = f
            psd[d] = p
    # trim
    keep = psd["freq"] <= MAXF
    out_psd = {"freq": psd["freq"][keep]}
    for d in DEVICES:
        if d in psd:
            out_psd[d] = psd[d][keep]
    np.savez(os.path.join(HERE, "channels_psd.npz"), **out_psd)
    print("wrote channels_psd.npz")

    # per-channel features + quality from features_allch.csv
    df = pd.read_csv(os.path.join(HERE, "features_allch.csv"))
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]
    cols = [c for c in df.columns if c not in META]
    res = {}
    for d in DEVICES:
        sub = m1[m1.device == d]
        if len(sub) == 0:
            continue
        feats = {k: float(sub[k].mean()) for k in
                 ("rms_db", "tone_freq", "centroid", "band_4000_8000", "band_0_50")}
        qd = channel_quality(sub, "valveOut", DISCHARGE, cols)
        qs = channel_quality(sub, "valveIn", SUCTION, cols)
        res[d] = {"features": feats, "discharge": qd, "suction": qs}
        dd = qd["spearman"] if qd else float("nan")
        print(f"  {d}: rms={feats['rms_db']:6.1f} tone={feats['tone_freq']:6.0f} "
              f"discharge ρ={dd:.2f}" + (f" within1={qd['within1']:.2f}" if qd else ""))
    json.dump(res, open(os.path.join(HERE, "channels.json"), "w"), indent=2)
    print("wrote channels.json")


if __name__ == "__main__":
    main()
