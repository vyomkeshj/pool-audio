#!/usr/bin/env python3
"""Characterize the M3 broken-fan fault: recover its shaft-rotation frequency
and blade-pass / harmonic structure by demodulating the haystack band.

Rotating imbalance from a broken fan amplitude-modulates the broadband noise
at the shaft frequency (1x) and its harmonics. We band-pass the 3-9 kHz
haystack (where M3 dominates), take the Hilbert envelope, and look at the
envelope spectrum DIFFERENCE (M3 on - M3 off) to isolate M3's modulation.
We also look for sidebands in the high-resolution spectrum.

Isolation: M2=0, M4=0, aeration=0 -> M3 toggles against an all-aux-off baseline.

Run: python3 characterize_m3.py
"""
import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
DEC = 10
EFS = SR / DEC               # envelope sample rate
HAY = (3000, 9000)


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    return x - x.mean()


def env_spectrum(x):
    sos = signal.butter(6, HAY, "bp", fs=SR, output="sos")
    b = signal.sosfilt(sos, x)
    env = np.abs(signal.hilbert(signal.decimate(b, DEC, ftype="fir")))
    env = env - env.mean()
    f, p = signal.welch(env, EFS, nperseg=16384, noverlap=8192)
    return f, p


def hires_spectrum(x):
    f, p = signal.welch(x, SR, nperseg=65536, noverlap=32768)
    return f, p


def worker(task):
    state, path = task
    try:
        x = load(path)
        fe, pe = env_spectrum(x)
        ff, pf = hires_spectrum(x)
        return state, pe, pf, fe, ff
    except Exception:
        return None


def main():
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    tasks = []
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M4"] or p["aeration"]:
            continue
        for s in e["sessions"]:
            if s["folder"] != "testbedmotor5_25wav":
                continue
            st = "on" if p["M3"] == 1 else "off"
            for dev, fn in s["devices"].items():
                tasks.append((st, os.path.join(ROOT, s["folder"], fn)))
    print(f"demodulating {len(tasks)} recordings (M2=M4=aer=0; M3 on vs off)...")

    on_e, off_e, on_f, off_f = [], [], [], []
    fe = ff = None
    with ProcessPoolExecutor() as ex:
        for r in ex.map(worker, tasks, chunksize=8):
            if r is None:
                continue
            st, pe, pf, fe, ff = r
            (on_e if st == "on" else off_e).append(pe)
            (on_f if st == "on" else off_f).append(pf)
    print(f"  on={len(on_e)}  off={len(off_e)}")

    on_e = np.mean(on_e, 0); off_e = np.mean(off_e, 0)
    on_f = np.mean(on_f, 0); off_f = np.mean(off_f, 0)
    diff_e = on_e - off_e               # M3's modulation
    # high-res spectral difference (dB) around haystack
    diff_f_db = 10 * np.log10(on_f + 1e-20) - 10 * np.log10(off_f + 1e-20)

    # ---- find shaft frequency from envelope-spectrum difference ----
    band = (fe >= 6) & (fe <= 250)
    fb, db = fe[band], diff_e[band]
    db = np.maximum(db, 0)
    # Harmonic Product Spectrum to robustly find fundamental
    f0_grid = np.arange(8, 60, 0.1)
    hps = []
    for f0 in f0_grid:
        s = 0.0
        for h in range(1, 7):
            s += np.interp(f0 * h, fb, db)
        hps.append(s)
    hps = np.array(hps)
    f0 = float(f0_grid[np.argmax(hps)])

    # collect harmonic peaks near multiples of f0
    peaks, props = signal.find_peaks(db, prominence=db.max() * 0.05)
    pk = sorted(([float(fb[i]), float(db[i])] for i in peaks),
                key=lambda t: -t[1])[:8]
    harmonics = []
    for h in range(1, 7):
        target = f0 * h
        j = np.argmin(np.abs(fb - target))
        if abs(fb[j] - target) < 2.5:
            harmonics.append({"order": h, "freq_hz": round(float(fb[j]), 2),
                              "amp": round(float(db[j]), 4)})

    # ---- look for blade-pass: peak in haystack spectrum & sideband spacing ----
    hmask = (ff >= 2000) & (ff <= 9000)
    hay_peak = float(ff[hmask][np.argmax(diff_f_db[hmask])])

    rpm = 60 * f0
    result = {
        "isolation": "M2=0,M4=0,aeration=0; M3 on vs off; 3-9 kHz envelope demod",
        "n_on": len(tasks) // 2,
        "shaft_freq_hz": round(f0, 2),
        "shaft_rpm": round(rpm, 0),
        "harmonics_detected": harmonics,
        "top_envelope_peaks_hz": [round(p[0], 2) for p in pk],
        "haystack_center_hz": round(hay_peak, 0),
        "interpretation": (
            "Dominant 1x rotational line with a 2x harmonic = mass imbalance from "
            "a broken/missing fan blade (once-per-rev heavy spot). If f0 does not "
            "match a clean 50 Hz sub-multiple, the impeller likely runs at its own "
            "(geared/independent) speed rather than line-locked."),
    }
    json.dump(result, open(os.path.join(ROOT, "m3_fault_signature.json"), "w"),
              indent=2)
    print(json.dumps(result, indent=2))

    # ---- figures ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    m = (fe >= 0) & (fe <= 250)
    ax[0].plot(fe[m], off_e[m] / off_e[m].max(), label="M3 off (baseline)", alpha=0.7)
    ax[0].plot(fe[m], on_e[m] / off_e[m].max(), label="M3 on", alpha=0.9)
    for h in harmonics:
        ax[0].axvline(h["freq_hz"], color="r", ls="--", lw=0.8, alpha=0.6)
    ax[0].axvline(f0, color="r", lw=1.5, alpha=0.8,
                  label=f"shaft f0={f0:.1f} Hz ({rpm:.0f} rpm)")
    ax[0].set_xlim(0, 250); ax[0].set_xlabel("modulation frequency (Hz)")
    ax[0].set_ylabel("envelope PSD (norm.)")
    ax[0].set_title("M3 haystack (3-9 kHz) envelope spectrum — imbalance modulation")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(fe[m], diff_e[m], color="purple")
    for h in harmonics:
        ax[1].axvline(h["freq_hz"], color="r", ls="--", lw=0.8, alpha=0.6)
        ax[1].annotate(f"{h['order']}x", (h["freq_hz"], diff_e[m].max() * 0.9),
                       color="r", fontsize=9, ha="center")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlim(0, 250); ax[1].set_xlabel("modulation frequency (Hz)")
    ax[1].set_ylabel("Δ envelope PSD (on − off)")
    ax[1].set_title("Isolated M3 modulation (difference) — shaft harmonics")
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_m3_envelope.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(11, 5))
    m2 = (ff >= 1000) & (ff <= 10000)
    plt.plot(ff[m2], diff_f_db[m2], color="darkorange")
    plt.axvline(hay_peak, color="r", lw=1, label=f"haystack ~{hay_peak:.0f} Hz")
    plt.axhline(0, color="k", lw=0.6)
    plt.xlabel("Hz"); plt.ylabel("Δ PSD on−off (dB)")
    plt.title("M3 high-resolution spectral increment (haystack + fine structure)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_m3_sidebands.png"), dpi=120)
    plt.close()
    print("wrote m3_fault_signature.json, fig_m3_envelope.png, fig_m3_sidebands.png")


if __name__ == "__main__":
    main()
