#!/usr/bin/env python3
"""Analyse WHAT acoustically distinguishes each operating class -- the physics
behind the models. For each task: which features drive it (RF importance) and how
the key feature trends across levels. Best-placed channels (cam for discharge,
mic1/cam1 for suction). Healthy equipment; nothing is broken.

Outputs: class_drivers.json, fig_class_differences.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT = ["rms_db", "centroid", "bandwidth", "rolloff", "flatness", "crest", "zcr",
        "tone_freq", "tone_prom", "mod_energy", "mod_peak", "band_0_100",
        "band_100_250", "band_250_500", "band_500_1000", "band_1000_2000",
        "band_2000_4000", "band_4000_8000", "band_8000_16000", "band_16000_22050"]
OUT = {}


def importances(X, y, regress=False):
    m = (RandomForestRegressor if regress else RandomForestClassifier)(
        300, random_state=0, n_jobs=-1)
    Xs = StandardScaler().fit_transform(X)
    m.fit(Xs, y)
    return sorted(zip(FEAT, m.feature_importances_), key=lambda t: -t[1])


def main():
    df = pd.read_csv(os.path.join(ROOT, "features_allch.csv"))
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]
    d25 = df[df.folder == "testbedmotor5_25wav"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- discharge: drivers + spectral trend across levels (cameras) ----
    cam = m1[m1.dev_type == "cam"]
    imp = importances(cam[FEAT].to_numpy(float), cam.valveOut.to_numpy(), regress=True)
    OUT["discharge_severity_drivers"] = [(f, round(float(v), 3)) for f, v in imp[:6]]
    print("discharge severity top drivers:", [f for f, _ in imp[:6]])
    bands = ["band_0_100", "band_100_250", "band_250_500", "band_500_1000",
             "band_1000_2000", "band_2000_4000", "band_4000_8000", "band_8000_16000"]
    bhz = [50, 175, 375, 750, 1500, 3000, 6000, 12000]
    ax = axes[0, 0]
    for lv in [1, 2, 3, 4, 5, 8, 11]:
        v = cam[cam.valveOut == lv][bands].mean().to_numpy()
        ax.plot(bhz, v, marker="o", label=f"vout {lv}")
    ax.set_xscale("log"); ax.set_xlabel("band centre (Hz)"); ax.set_ylabel("rel. band energy (dB)")
    ax.set_title("Discharge blockage: spectrum vs level (cameras)\nrising level reshapes the band balance")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # trend of the single top driver vs level
    ax = axes[0, 1]
    top = imp[0][0]
    means = [cam[cam.valveOut == lv][top].mean() for lv in [1, 2, 3, 4, 5, 8, 11]]
    ax.plot([1, 2, 3, 4, 5, 8, 11], means, "o-", color="tab:red")
    ax.set_xlabel("valveOut (discharge level)"); ax.set_ylabel(top)
    ax.set_title(f"Top discharge driver '{top}' vs level (cameras)"); ax.grid(alpha=0.3)

    # ---- suction: drivers + trend (best mic1) ----
    sub = m1[m1.device == "mic1"]
    imp = importances(sub[FEAT].to_numpy(float), sub.valveIn.to_numpy(), regress=True)
    OUT["suction_severity_drivers"] = [(f, round(float(v), 3)) for f, v in imp[:6]]
    print("suction severity top drivers:", [f for f, _ in imp[:6]])
    ax = axes[1, 0]
    for lv in [1, 2, 3, 4]:
        v = sub[sub.valveIn == lv][bands].mean().to_numpy()
        ax.plot(bhz, v, marker="o", label=f"vin {lv}")
    ax.set_xscale("log"); ax.set_xlabel("band centre (Hz)"); ax.set_ylabel("rel. band energy (dB)")
    ax.set_title("Suction blockage: spectrum vs level (mic1)\nsubtler shifts -> harder to read"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- motor presence drivers (what frequency IDs each motor) ----
    ax = axes[1, 1]
    motor_drv = {}
    for m in ["M2", "M3", "M4"]:
        imp = importances(d25[FEAT].to_numpy(float), d25[m].to_numpy())
        motor_drv[m] = [(f, round(float(v), 3)) for f, v in imp[:5]]
        print(f"{m} presence top drivers:", [f for f, _ in imp[:4]])
    OUT["motor_presence_drivers"] = motor_drv
    # bar of band-delta on/off per motor (reuse signature idea on cam+mic pooled is fine for viz)
    x = np.arange(len(bands))
    w = 0.25
    for i, m in enumerate(["M2", "M3", "M4"]):
        on = d25[d25[m] == 1][bands].mean().to_numpy()
        off = d25[d25[m] == 0][bands].mean().to_numpy()
        ax.bar(x + (i - 1) * w, on - off, w, label=m)
    ax.set_xticks(x); ax.set_xticklabels([str(h) for h in bhz], rotation=45, fontsize=8)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Δ band energy on−off (dB)")
    ax.set_title("Which band identifies each motor\nM3 fan = 4–8 kHz; M2 pump = low/mid; M4 ≈ flat")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_class_differences.png"), dpi=110)
    plt.close()
    json.dump(OUT, open(os.path.join(ROOT, "class_drivers.json"), "w"), indent=2)
    print("\nwrote class_drivers.json + fig_class_differences.png")


if __name__ == "__main__":
    main()
