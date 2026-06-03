#!/usr/bin/env python3
"""Acoustic analysis of pump states from the clean (noise=N) recordings.

Extracts diagnostic features from one mic channel per session, characterises
each pump state, and runs a session-independent baseline classifier.

Outputs: features_clean.csv, fig_state_psd.png, fig_feature_box.png,
         fig_state_scatter.png, fig_confusion.png, analysis_summary.json
"""
import os
import json
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
BANDS = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 4000), (4000, 8000), (8000, 16000), (16000, 22050)]
DEVICE_PREF = ["mic1", "mic2", "cam1", "cam2"]

STATE_ORDER = ["normal", "suction_blockage", "discharge_blockage",
               "aerating", "multi_pump"]


def pick_device(devices):
    for d in DEVICE_PREF:
        if d in devices:
            return d, devices[d]
    k = sorted(devices)[0]
    return k, devices[k]


def extract(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = signal.resample(x, n)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2)) + 1e-12

    # Welch PSD
    f, p = signal.welch(x, SR, nperseg=8192, noverlap=4096)
    p = p + 1e-20
    pn = p / p.sum()

    # spectral shape
    centroid = float((f * pn).sum())
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * pn).sum()))
    cumsum = np.cumsum(pn)
    rolloff = float(f[np.searchsorted(cumsum, 0.85)])
    flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))  # Wiener entropy
    crest = float(p.max() / np.mean(p))

    # band energies (log, relative to total power)
    bandfeat = {}
    tot = p.sum()
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        bandfeat[f"band_{lo}_{hi}"] = float(10 * np.log10(p[m].sum() / tot + 1e-12))

    # tonal peak below 2 kHz (pump rotation / blade-pass)
    lowmask = f < 2000
    fl, pl = f[lowmask], p[lowmask]
    pk = int(np.argmax(pl))
    tone_freq = float(fl[pk])
    tone_prom = float(10 * np.log10(pl[pk] / (np.median(pl) + 1e-20)))

    # amplitude-modulation / burstiness (cavitation, aeration -> bubbles)
    env = np.abs(signal.hilbert(signal.decimate(x, 10, ftype="fir")))
    env = env - env.mean()
    fe, pe = signal.welch(env, SR / 10, nperseg=4096)
    modmask = (fe >= 5) & (fe <= 150)
    mod_energy = float(np.log10(pe[modmask].sum() + 1e-20))
    mod_peak = float(fe[modmask][np.argmax(pe[modmask])])
    zcr = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2)

    feat = {
        "rms_db": float(20 * np.log10(rms)),
        "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
        "flatness": flatness, "crest": float(10 * np.log10(crest)),
        "zcr": zcr, "tone_freq": tone_freq, "tone_prom": tone_prom,
        "mod_energy": mod_energy, "mod_peak": mod_peak,
    }
    feat.update(bandfeat)
    return feat, f, p


def main():
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    rows = []
    psd_by_state = {s: [] for s in STATE_ORDER}
    psd_f = None

    sessions = []
    for sig, e in idx["measurements"].items():
        if e["params"]["noise"] != "N":
            continue
        state = e["state"]["nominal_state"]
        for s in e["sessions"]:
            dev, fn = pick_device(s["devices"])
            sessions.append((state, s["folder"], fn, dev, e["params"]))

    print(f"extracting features from {len(sessions)} clean sessions "
          f"(1 channel each)...")
    for i, (state, folder, fn, dev, params) in enumerate(sessions):
        path = os.path.join(ROOT, folder, fn)
        try:
            feat, f, p = extract(path)
        except Exception as ex:
            print(f"  skip {fn}: {ex}")
            continue
        if psd_f is None:
            psd_f = f
        if state in psd_by_state:
            psd_by_state[state].append(p)
        row = {"state": state, "folder": folder, "device": dev,
               "valveIn": params["valveIn"], "valveOut": params["valveOut"],
               "aeration": params["aeration"], "file": fn}
        row.update(feat)
        rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(sessions)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "features_clean.csv"), index=False)
    print("wrote features_clean.csv", df.shape)

    feat_cols = [c for c in df.columns if c not in
                 ("state", "folder", "device", "valveIn", "valveOut",
                  "aeration", "file")]

    # ---- Fig 1: mean PSD per state ----
    plt.figure(figsize=(10, 6))
    for s in STATE_ORDER:
        if psd_by_state[s]:
            mp = np.mean(np.vstack(psd_by_state[s]), axis=0)
            plt.semilogx(psd_f, 10 * np.log10(mp + 1e-20), label=f"{s} (n={len(psd_by_state[s])})")
    plt.xlim(20, 22050); plt.xlabel("Hz"); plt.ylabel("PSD (dB)")
    plt.title("Mean power spectrum per pump state (clean recordings)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_state_psd.png"), dpi=110)
    plt.close()

    # ---- Fig 2: key feature boxplots ----
    key = ["rms_db", "centroid", "rolloff", "flatness", "tone_freq",
           "tone_prom", "mod_energy", "band_4000_8000"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, k in zip(axes.flat, key):
        data = [df[df.state == s][k].values for s in STATE_ORDER if (df.state == s).any()]
        labs = [s for s in STATE_ORDER if (df.state == s).any()]
        ax.boxplot(data, labels=range(len(labs)))
        ax.set_title(k); ax.grid(alpha=0.3)
    fig.legend([f"{i}={s}" for i, s in enumerate(STATE_ORDER)],
               loc="lower center", ncol=5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(ROOT, "fig_feature_box.png"), dpi=110)
    plt.close()

    # ---- classifier: standardize, LDA scatter + RandomForest grouped CV ----
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X = df[feat_cols].to_numpy(dtype=float)
    y = np.asarray(df["state"], dtype=object)
    Xs = StandardScaler().fit_transform(X)

    # LDA 2D projection
    present = [s for s in STATE_ORDER if (y == s).any()]
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z = lda.fit_transform(Xs, y)
    plt.figure(figsize=(9, 7))
    for s in present:
        m = y == s
        plt.scatter(Z[m, 0], Z[m, 1], label=f"{s} (n={m.sum()})", alpha=0.7, s=30)
    plt.xlabel("LD1"); plt.ylabel("LD2")
    plt.title("LDA projection of pump states (clean, session-level)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_state_scatter.png"), dpi=110)
    plt.close()

    # RandomForest with stratified 5-fold CV (each sample = independent session)
    clf = RandomForestClassifier(n_estimators=400, random_state=0,
                                 class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    yp = cross_val_predict(clf, Xs, y, cv=cv)
    acc = accuracy_score(y, yp)
    print(f"\n5-fold CV accuracy (RandomForest): {acc:.3f}")
    print(classification_report(y, yp, labels=present, zero_division=0))

    cm = confusion_matrix(y, yp, labels=present)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(present)), present, rotation=45, ha="right")
    plt.yticks(range(len(present)), present)
    for i in range(len(present)):
        for j in range(len(present)):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("true"); plt.xlabel("predicted")
    plt.title(f"Confusion matrix (5-fold CV, acc={acc:.2f})")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_confusion.png"), dpi=110)
    plt.close()

    # feature importance
    clf.fit(Xs, y)
    imp = sorted(zip(feat_cols, clf.feature_importances_),
                 key=lambda t: -t[1])

    summary = {
        "n_sessions_analyzed": len(df),
        "channel": "one mic/cam per session (mic1 preferred)",
        "states": {s: int((y == s).sum()) for s in present},
        "cv_accuracy": round(float(acc), 4),
        "top_features": [{"feature": f, "importance": round(float(v), 4)}
                         for f, v in imp[:12]],
        "per_state_means": {
            s: {k: round(float(df[df.state == s][k].mean()), 3)
                for k in key}
            for s in present
        },
    }
    json.dump(summary, open(os.path.join(ROOT, "analysis_summary.json"), "w"),
              indent=2)
    print("\nTop features:", [f for f, _ in imp[:8]])
    print("wrote analysis_summary.json + 4 figures")


if __name__ == "__main__":
    main()
