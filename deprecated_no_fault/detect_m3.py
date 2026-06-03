#!/usr/bin/env python3
"""Detect the M3 broken-fan signature in ANY audio clip.

The fault has two fingerprints (from characterize_m3.py + fault_confirm.py):
  1. a broadband 3-9 kHz "haystack" (damaged-impeller turbulence), and
  2. amplitude modulation at the ~21.3 Hz shaft line + 2x (mass imbalance).
We build gain-invariant features around both, train on the 5_25 full factorial
(ground-truth M3 on/off), and validate that it is M3-SPECIFIC (does not fire on
healthy M2/M4) and NOISE-ROBUST (low false-alarm on noisy M1-only clips).

Train + evaluate:   python3 detect_m3.py
Score a clip:       python3 detect_m3.py path/to/file.wav
"""
import os
import sys
import json
import numpy as np
from scipy import signal
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
DEC = 10
SHAFT = 21.3                       # Hz, from characterize_m3.py
MODEL_PATH = os.path.join(ROOT, "m3_detector.joblib")


# ----------------------------------------------------------- feature extraction
def detect_features(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    x = x - x.mean()
    f, p = signal.welch(x, SR, nperseg=8192, noverlap=4096)
    p = p + 1e-20

    def E(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].sum())

    def band_flat(lo, hi):
        m = (f >= lo) & (f < hi)
        pp = p[m]
        return float(np.exp(np.mean(np.log(pp))) / np.mean(pp))

    hay = E(3000, 9000)
    # gain-invariant spectral-shape features (ratios in dB)
    feat = {
        "hay_over_low": 10 * np.log10(hay / (E(300, 3000) + 1e-20)),
        "hay_over_high": 10 * np.log10(hay / (E(9000, 22050) + 1e-20)),
        "hay_over_tot": 10 * np.log10(hay / (p.sum())),
        "flat_hay": band_flat(3000, 9000),
        "peak_prom_hay": 10 * np.log10(
            p[(f >= 4000) & (f < 7500)].max() /
            (np.median(p[(f >= 1000) & (f < 12000)]) + 1e-20)),
        "centroid": float((f * (p / p.sum())).sum()),
    }

    # imbalance modulation: envelope spectrum of the haystack band
    sos = signal.butter(6, (3000, 9000), "bp", fs=SR, output="sos")
    env = np.abs(signal.hilbert(signal.decimate(signal.sosfilt(sos, x), DEC,
                                                ftype="fir")))
    env = env - env.mean()
    fe, pe = signal.welch(env, SR / DEC, nperseg=16384, noverlap=8192)
    base = np.median(pe[(fe >= 8) & (fe <= 120)]) + 1e-20

    def modpeak(fc, bw=1.2):
        m = (fe >= fc - bw) & (fe <= fc + bw)
        return 10 * np.log10(pe[m].max() / base)
    feat["mod_1x"] = float(modpeak(SHAFT))
    feat["mod_2x"] = float(modpeak(2 * SHAFT))
    feat["mod_3x"] = float(modpeak(3 * SHAFT))
    return feat


FEAT_ORDER = ["hay_over_low", "hay_over_high", "hay_over_tot", "flat_hay",
              "peak_prom_hay", "centroid", "mod_1x", "mod_2x", "mod_3x"]


# ----------------------------------------------------------------- scoring mode
def score_clip(path):
    import joblib
    bundle = joblib.load(MODEL_PATH)
    feat = detect_features(path)
    x = np.array([[feat[k] for k in FEAT_ORDER]])
    xs = bundle["scaler"].transform(x)
    prob = float(bundle["clf"].predict_proba(xs)[0, 1])
    thr = bundle["threshold"]
    return prob, prob >= thr, feat


# -------------------------------------------------------------- train + evaluate
def _detect_worker(rec):
    path, m3, m2, m4, aer, ts, fo, noise = rec
    try:
        feat = detect_features(path)
    except Exception:
        return None
    if fo == "testbedmotor5_25wav":
        role = "core"
    elif fo in HARDNEG_TRAIN:
        role = "hardneg_train"
    else:
        role = "holdout"
    feat.update({"M3": m3, "M2": m2, "M4": m4, "aeration": aer,
                 "timestamp": ts, "folder": fo,
                 "noise_cat": noise[0] if noise != "N" else "N", "role": role})
    return feat


# noisy M1-only campaigns: 2 used as TRAIN hard-negatives, 2 fully HELD OUT.
HARDNEG_TRAIN = {"Testbedmotor5_14", "Testbedmotor5_15"}
HARDNEG_HOLDOUT = {"testbed_motor_audio", "Testbedmotor"}


def build_dataset():
    """5_25 (M3 on/off) = the labelled core. Noisy M1-only clips (all M3 off)
    are split by CAMPAIGN: 5_14/5_15 become TRAIN hard-negatives (teach the model
    that environmental/blockage broadband != M3's fault), 5_07 campaigns are a
    fully held-out cross-campaign false-alarm test."""
    from concurrent.futures import ProcessPoolExecutor
    import collections
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    core, hn_train, hn_holdout = [], collections.defaultdict(list), collections.defaultdict(list)
    for sig, e in idx["measurements"].items():
        p = e["params"]
        for s in e["sessions"]:
            fo = s["folder"]
            ncat = p["noise"][0] if p["noise"] != "N" else "N"
            for dev, fn in s["devices"].items():
                rec = (os.path.join(ROOT, fo, fn), p["M3"], p["M2"], p["M4"],
                       p["aeration"], s["timestamp"], fo, p["noise"])
                if fo == "testbedmotor5_25wav":
                    core.append(rec)
                elif p["M3"] == 0 and p["M2"] == 0 and p["M4"] == 0:
                    (hn_train if fo in HARDNEG_TRAIN else hn_holdout)[(fo, ncat)].append(rec)

    def sample(pool, per_cell):
        out = []
        for k, v in pool.items():
            step = max(1, len(v) // per_cell)
            out += v[::step]
        return out
    hn_tr = sample(hn_train, 70)       # ~ balanced over (campaign, noise)
    hn_ho = sample(hn_holdout, 70)
    allrecs = core + hn_tr + hn_ho
    print(f"extracting detector features: {len(core)} 5_25 core + "
          f"{len(hn_tr)} train hard-neg + {len(hn_ho)} held-out cross-campaign neg ...")
    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(_detect_worker, allrecs, chunksize=8)):
            if r:
                rows.append(r)
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(allrecs)}")
    return rows


def main_train():
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(build_dataset())
    df.to_csv(os.path.join(ROOT, "detect_features.csv"), index=False)
    # training pool = 5_25 core (labelled by M3) + 5_14/5_15 noisy hard-negatives
    tr = df[df.role.isin(["core", "hardneg_train"])].reset_index(drop=True)
    tr = tr.assign(label=(tr.role == "core") * tr.M3)          # only M3-on are positive
    ho = df[df.role == "holdout"].reset_index(drop=True)       # cross-campaign, all M3 off

    X = tr[FEAT_ORDER].to_numpy(float)
    y = tr["label"].to_numpy(int)
    groups = tr["timestamp"].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=600, random_state=0,
                                 class_weight="balanced", n_jobs=-1)
    # grouped CV (a recording's 16 channels stay together) -> honest OOF probs
    proba = cross_val_predict(clf, Xs, y, cv=GroupKFold(5), groups=groups,
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    # pick threshold for a LOW false-alarm rate (<=5%) on ALL CV negatives
    neg_prob = proba[y == 0]
    best_thr = float(np.quantile(neg_prob, 0.95))
    pred = (proba >= best_thr).astype(int)
    acc = accuracy_score(y, pred)
    tr = tr.assign(cv_prob=proba, cv_pred=pred)
    recall = float(tr[tr.label == 1].cv_pred.mean())
    print(f"\ngrouped-CV: ROC-AUC={auc:.3f}  recall@5%FA={recall:.3f}  thr={best_thr:.2f}")

    # M3-SPECIFICITY: FP among M3-off, split by healthy aux activity / noisy negs
    core = tr[tr.role == "core"]
    cneg = core[core.M3 == 0]
    spec = {
        "recall_M3on": round(recall, 3),
        "FP_M3off_healthy_M2orM4_ON": round(float(cneg[(cneg.M2 == 1) | (cneg.M4 == 1)].cv_pred.mean()), 3),
        "FP_M3off_all_aux_off": round(float(cneg[(cneg.M2 == 0) & (cneg.M4 == 0)].cv_pred.mean()), 3),
        "FP_noisy_hardneg_train(CV)": round(float(tr[tr.role == "hardneg_train"].cv_pred.mean()), 3),
    }
    print("specificity:", spec)

    clf.fit(Xs, y)
    joblib.dump({"clf": clf, "scaler": scaler, "threshold": best_thr,
                 "features": FEAT_ORDER, "shaft_hz": SHAFT}, MODEL_PATH)

    # CROSS-CAMPAIGN false-alarm on fully held-out noisy M1-only (all M3 off)
    Xf = scaler.transform(ho[FEAT_ORDER].to_numpy(float))
    ho = ho.assign(prob=clf.predict_proba(Xf)[:, 1])
    ho = ho.assign(pred=(ho.prob >= best_thr).astype(int))
    fa_by_noise = {c: round(float(ho[ho.noise_cat == c].pred.mean()), 3)
                   for c in sorted(ho.noise_cat.unique())}
    print("cross-campaign false-alarm by noise:", fa_by_noise,
          "| overall", round(float(ho.pred.mean()), 3))

    result = {
        "train": "5_25 (M3 on/off) + 5_14/5_15 noisy M1-only hard-negatives; "
                 "grouped-CV by recording; threshold set to 5% FA on CV negatives",
        "holdout": "testbed_motor_audio + Testbedmotor (5_07) noisy M1-only, all M3 off",
        "roc_auc": round(float(auc), 3), "recall_at_5pct_FA": round(recall, 3),
        "threshold": round(best_thr, 3), "specificity": spec,
        "cross_campaign_false_alarm": fa_by_noise,
        "overall_holdout_FP_rate": round(float(ho.pred.mean()), 3),
        "feature_importance": {k: round(float(v), 3) for k, v in
                               sorted(zip(FEAT_ORDER, clf.feature_importances_),
                                      key=lambda t: -t[1])},
    }
    json.dump(result, open(os.path.join(ROOT, "detect_m3_result.json"), "w"),
              indent=2)

    # figure: ROC + score distributions + FP-by-noise
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate")
    ax[0].set_title(f"M3 detector ROC (AUC={auc:.3f})"); ax[0].grid(alpha=0.3)

    ax[1].hist(proba[(y == 0) & (tr.role == "core").values], bins=25, alpha=0.55,
               label="M3 off (5_25)", density=True)
    ax[1].hist(proba[(tr.role == "hardneg_train").values], bins=25, alpha=0.55,
               label="noisy M1-only (train)", density=True, color="green")
    ax[1].hist(proba[y == 1], bins=25, alpha=0.6, label="M3 on", density=True, color="red")
    ax[1].axvline(best_thr, color="k", ls="--", label=f"thr ({recall:.0%} recall)")
    ax[1].set_xlabel("P(M3 broken-fan present)"); ax[1].set_title("OOF score distributions")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    cats = list(fa_by_noise.keys())
    ax[2].bar(cats, [fa_by_noise[c] for c in cats], color="green", alpha=0.7)
    ax[2].set_ylim(0, max(0.15, max(fa_by_noise.values()) * 1.3 + 0.01))
    ax[2].set_ylabel("false-alarm rate")
    ax[2].set_xlabel("noise (held-out 5_07 campaigns, M3 off)")
    ax[2].set_title("Cross-campaign false alarms\n(target ~5%; M3 absent here)")
    ax[2].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_detector.png"), dpi=120)
    plt.close()
    print("wrote m3_detector.joblib, detect_m3_result.json, detect_features.csv, "
          "fig_detector.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prob, flag, feat = score_clip(sys.argv[1])
        print(f"P(M3 broken-fan signature present) = {prob:.3f}  "
              f"=> {'DETECTED' if flag else 'not detected'}")
        print(f"  haystack 3-9kHz/low = {feat['hay_over_low']:+.1f} dB, "
              f"imbalance 1x@{SHAFT}Hz = {feat['mod_1x']:+.1f} dB")
    else:
        main_train()
