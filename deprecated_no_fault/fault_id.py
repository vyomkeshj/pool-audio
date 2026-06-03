#!/usr/bin/env python3
"""Identify a damaged auxiliary pump (broken fan) from audio.

The 5_25 campaign is a full factorial over M2/M3/M4 on/off (M1 always on,
valveIn=1). By matched differencing we isolate the acoustic increment each
pump adds when switched on. A healthy pump adds mostly a clean tonal component;
a broken fan adds broadband noise + rotating-imbalance amplitude modulation +
impulsiveness. We score each pump on those fault signatures.

Run: python3 fault_id.py
"""
import os
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import signal
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
# log-spaced PSD bins for difference spectra
PSD_EDGES = np.logspace(np.log10(20), np.log10(22050), 81)
PSD_CTR = np.sqrt(PSD_EDGES[:-1] * PSD_EDGES[1:])


def fault_features(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2)) + 1e-12

    f, p = signal.welch(x, SR, nperseg=8192, noverlap=4096)
    p = p + 1e-20
    # binned log-PSD (dB)
    idx = np.clip(np.digitize(f, PSD_EDGES) - 1, 0, len(PSD_CTR) - 1)
    psd_bin = np.zeros(len(PSD_CTR))
    cnt = np.zeros(len(PSD_CTR))
    np.add.at(psd_bin, idx, p)
    np.add.at(cnt, idx, 1)
    psd_bin = 10 * np.log10(psd_bin / np.maximum(cnt, 1) + 1e-20)

    def bandE(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].sum())
    tot = p.sum()
    flat = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    spec_kurt = float(kurtosis(p))

    # amplitude envelope (modulation) -> rotating-imbalance fingerprint
    xd = signal.decimate(x, 10, ftype="fir")
    env = np.abs(signal.hilbert(xd))
    env = env - env.mean()
    fe, pe = signal.welch(env, SR / 10, nperseg=4096)
    mod = (fe >= 5) & (fe <= 200)
    mod_energy = float(pe[mod].sum())
    mod_crest = float(pe[mod].max() / (pe[mod].mean() + 1e-20))   # peaky modulation = imbalance
    mod_peak = float(fe[mod][np.argmax(pe[mod])])

    # impulsiveness (broken-blade knock/rub)
    t_kurt = float(kurtosis(x))
    crest = float(np.max(np.abs(x)) / rms)
    # high-band impulsiveness: kurtosis of >4 kHz envelope
    sos = signal.butter(4, 4000, "hp", fs=SR, output="sos")
    hp = signal.sosfilt(sos, x)
    hp_env = np.abs(signal.hilbert(signal.decimate(hp, 10, ftype="fir")))
    hp_kurt = float(kurtosis(hp_env))

    feat = {
        "rms_db": float(20 * np.log10(rms)),
        "bb_2_16k_db": float(10 * np.log10(bandE(2000, 16000) / tot + 1e-12)),
        "hf_8_22k_db": float(10 * np.log10(bandE(8000, 22050) / tot + 1e-12)),
        "flatness": flat, "spec_kurt": spec_kurt,
        "mod_energy": float(np.log10(mod_energy + 1e-20)),
        "mod_crest": mod_crest, "mod_peak": mod_peak,
        "time_kurt": t_kurt, "crest": crest, "hp_kurt": hp_kurt,
    }
    return feat, psd_bin


def worker(task):
    folder, fn, params, dev = task
    try:
        feat, psd = fault_features(os.path.join(ROOT, folder, fn))
    except Exception:
        return None
    row = {"file": fn, "device": dev,
           "devtype": "mic" if dev.startswith("mic") else "cam",
           "M2": params["M2"], "M3": params["M3"], "M4": params["M4"],
           "aeration": params["aeration"], "valveOut": params["valveOut"]}
    row.update(feat)
    row.update({f"psd{i}": v for i, v in enumerate(psd)})
    return row


def main():
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    tasks = []
    for sig, e in idx["measurements"].items():
        for s in e["sessions"]:
            if s["folder"] != "testbedmotor5_25wav":
                continue
            for dev, fn in s["devices"].items():
                tasks.append((s["folder"], fn, e["params"], dev))
    print(f"extracting fault features from {len(tasks)} 5_25 recordings...")

    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(worker, tasks, chunksize=8)):
            if r:
                rows.append(r)
            if (i + 1) % 256 == 0:
                print(f"  {i+1}/{len(tasks)}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "fault_features_525.csv"), index=False)
    print("saved fault_features_525.csv", df.shape)

    PUMPS = ["M2", "M3", "M4"]
    psd_cols = [c for c in df.columns if c.startswith("psd")]
    FAULT = ["rms_db", "bb_2_16k_db", "hf_8_22k_db", "mod_energy",
             "mod_crest", "time_kurt", "crest", "hp_kurt", "spec_kurt", "flatness"]

    # ---- marginal contrast: mean(pump on) - mean(pump off) (balanced factorial)
    print("\nMarginal acoustic increment when each pump is switched ON")
    print("(positive = pump adds this; fault tells = louder, broadband, modulated, impulsive)\n")
    contrast = {}
    for pmp in PUMPS:
        on = df[df[pmp] == 1]
        off = df[df[pmp] == 0]
        d = {k: float(on[k].mean() - off[k].mean()) for k in FAULT}
        contrast[pmp] = d
    cdf = pd.DataFrame(contrast).T
    pd.set_option("display.width", 160, "display.float_format", lambda v: f"{v:7.3f}")
    print(cdf.to_string())

    # ---- difference PSD per pump ----
    diff_psd = {}
    for pmp in PUMPS:
        on = df[df[pmp] == 1][psd_cols].mean().to_numpy()
        off = df[df[pmp] == 0][psd_cols].mean().to_numpy()
        diff_psd[pmp] = on - off

    plt.figure(figsize=(11, 6))
    for pmp in PUMPS:
        plt.semilogx(PSD_CTR, diff_psd[pmp], label=f"{pmp} on−off", lw=1.8)
    plt.axhline(0, color="k", lw=0.6)
    plt.xlim(20, 22050); plt.xlabel("Hz")
    plt.ylabel("Δ PSD when pump ON (dB)")
    plt.title("Acoustic increment of each auxiliary pump (5_25 full factorial)\n"
              "broad elevated hump + high-freq lift = broken-fan signature")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_pump_diff_psd.png"), dpi=120)
    plt.close()

    # ---- fault-score bar: z-score each contrast feature across pumps, sum the
    #      "bad" directions (louder/broadband/modulated/impulsive) ----
    bad_dir = {"rms_db": 1, "bb_2_16k_db": 1, "hf_8_22k_db": 1, "mod_energy": 1,
               "mod_crest": 1, "time_kurt": 1, "crest": 1, "hp_kurt": 1,
               "spec_kurt": 1, "flatness": -1}  # flat spectrum -> broadband, lower flatness number
    Z = cdf.copy()
    for k in FAULT:
        col = Z[k].to_numpy()
        s = col.std() + 1e-9
        Z[k] = (col - col.mean()) / s * bad_dir[k]
    score = Z.sum(axis=1)
    print("\nFault score per pump (higher = more broken-fan-like increment):")
    print(score.sort_values(ascending=False).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(score.index, score.values,
              color=["#d62728" if v == score.max() else "#1f77b4" for v in score.values])
    ax[0].set_ylabel("fault score (Σ z-scored fault increments)")
    ax[0].set_title("Damaged-pump score")
    ax[0].grid(axis="y", alpha=0.3)
    # key raw increments
    show = ["rms_db", "bb_2_16k_db", "mod_crest", "time_kurt", "hp_kurt"]
    x = np.arange(len(show)); w = 0.25
    for j, pmp in enumerate(PUMPS):
        ax[1].bar(x + (j - 1) * w, [contrast[pmp][k] for k in show], w, label=pmp)
    ax[1].set_xticks(x); ax[1].set_xticklabels(show, rotation=20, fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("Key fault-feature increments (on − off)")
    ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_pump_faultscore.png"), dpi=120)
    plt.close()

    out = {
        "method": "matched on/off contrast on 5_25 full factorial (M2/M3/M4)",
        "n_recordings": int(len(df)),
        "marginal_increment_on_minus_off": {p: contrast[p] for p in PUMPS},
        "fault_score": {p: round(float(score[p]), 3) for p in PUMPS},
        "most_likely_damaged": str(score.idxmax()),
    }
    json.dump(out, open(os.path.join(ROOT, "fault_id_result.json"), "w"), indent=2)
    print(f"\n==> most broken-fan-like pump: {score.idxmax()}")
    print("wrote fault_id_result.json, fig_pump_diff_psd.png, fig_pump_faultscore.png")


if __name__ == "__main__":
    main()
