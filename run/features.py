#!/usr/bin/env python3
"""Enhanced, gain-invariant acoustic feature extraction for every channel of
every file (mic1-8 + cam1-8) across all 7 campaigns.

Builds on the proven 20-feature vector from the original analyze.extract(), and
adds robustness/severity-focused features that earlier work flagged as the weak
spot (suction-side reading):

  * finer low-frequency band resolution (0-350 Hz, where the suction signature
    and the migrating pump tone live)
  * harmonic structure of the dominant pump tone (f0, 2f0/f0, 3f0/f0 ratios) -
    a gain-invariant tonal descriptor
  * coarse band ratios (low/mid/high) and extra spectral percentiles

Everything except `rms_db` is gain-invariant (relative to total power or a
ratio), so models do not lean on absolute level / channel gain. `rms_db` is kept
as one explicit (optional) level feature so its contribution can be ablated.

Output: run/features_allch.csv  (one row per channel-file)
Run:    python3 run/features.py
"""
import os
import json
import numpy as np
import soundfile as sf
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(HERE, "measurement_index.json")
OUT = os.path.join(HERE, "features_allch.csv")
SR = 44100

# octave-ish bands (proven set) + finer low bands for the suction signature
BANDS = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 350), (350, 500),
         (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
         (8000, 16000), (16000, 22050)]


def extract(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    return extract_array(x)


def extract_array(x):
    """Compute the feature vector from a mono float array already at SR.
    Identical features to extract(path); used for live/windowed inference."""
    x = np.asarray(x, dtype="float32")
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2)) + 1e-12

    nperseg = min(8192, len(x))
    f, p = signal.welch(x, SR, nperseg=nperseg, noverlap=nperseg // 2)
    p = p + 1e-20
    tot = p.sum()
    pn = p / tot

    centroid = float((f * pn).sum())
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * pn).sum()))
    cumsum = np.cumsum(pn)
    rolloff = float(f[np.searchsorted(cumsum, 0.85)])
    rolloff95 = float(f[np.searchsorted(cumsum, 0.95)])
    median_f = float(f[np.searchsorted(cumsum, 0.50)])
    flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    crest = float(p.max() / np.mean(p))

    feat = {
        "rms_db": float(20 * np.log10(rms)),
        "centroid": centroid, "bandwidth": bandwidth,
        "rolloff": rolloff, "rolloff95": rolloff95, "median_f": median_f,
        "flatness": flatness, "crest": float(10 * np.log10(crest)),
    }

    # band energies, log, relative to total power (gain-invariant)
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        feat[f"band_{lo}_{hi}"] = float(10 * np.log10(p[m].sum() / tot + 1e-12))

    # coarse band ratios (dB) - gain-invariant shape descriptors
    def be(lo, hi):
        return p[(f >= lo) & (f < hi)].sum() + 1e-20
    low = be(0, 500); mid = be(500, 2000); high = be(2000, 8000); vhigh = be(8000, 22050)
    feat["ratio_low_mid"] = float(10 * np.log10(low / mid))
    feat["ratio_mid_high"] = float(10 * np.log10(mid / high))
    feat["ratio_high_vhigh"] = float(10 * np.log10(high / vhigh))

    # dominant pump tone below 2 kHz + harmonic structure (gain-invariant)
    lowmask = f < 2000
    fl, pl = f[lowmask], p[lowmask]
    pk = int(np.argmax(pl))
    f0 = float(fl[pk])
    tone_prom = float(10 * np.log10(pl[pk] / (np.median(pl) + 1e-20)))
    df = f[1] - f[0]

    def amp_at(freq):
        if freq <= 0 or freq >= f[-1]:
            return 1e-20
        i = int(round(freq / df))
        i0, i1 = max(0, i - 2), min(len(p), i + 3)
        return p[i0:i1].max()
    a1 = amp_at(f0)
    feat["tone_freq"] = f0
    feat["tone_prom"] = tone_prom
    feat["harm2_ratio"] = float(10 * np.log10(amp_at(2 * f0) / (a1 + 1e-20)))
    feat["harm3_ratio"] = float(10 * np.log10(amp_at(3 * f0) / (a1 + 1e-20)))

    # amplitude-modulation / burstiness (cavitation, flow fluctuation)
    env = np.abs(signal.hilbert(signal.decimate(x, 10, ftype="fir")))
    env = env - env.mean()
    fe, pe = signal.welch(env, SR / 10, nperseg=4096)
    modmask = (fe >= 5) & (fe <= 150)
    feat["mod_energy"] = float(np.log10(pe[modmask].sum() + 1e-20))
    feat["mod_peak"] = float(fe[modmask][np.argmax(pe[modmask])])
    feat["zcr"] = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2)
    return feat


def worker(task):
    disk, fn, dev, ts, params = task
    try:
        feat = extract(os.path.join(ROOT, disk, fn))
    except Exception:
        return None
    noise = params["noise"]
    row = {
        "folder": params["_folder"], "file": fn, "device": dev,
        "dev_type": "mic" if dev.startswith("mic") else "cam",
        "session": f"{params['_folder']}/{ts}",
        "noise": noise, "noise_cat": "N" if noise == "N" else noise[0],
        "M2": params["M2"], "M3": params["M3"], "M4": params["M4"],
        "aeration": params["aeration"],
        "valveIn": params["valveIn"], "valveOut": params["valveOut"],
    }
    row.update(feat)
    return row


def main():
    idx = json.load(open(INDEX))
    tasks = []
    for sig, e in idx["measurements"].items():
        pr = dict(e["params"])
        for s in e["sessions"]:
            pr2 = dict(pr); pr2["_folder"] = s["folder"]
            for dev, fn in s["devices"].items():
                tasks.append((s["disk_path"], fn, dev, s["timestamp"], pr2))
    print(f"extracting {len(tasks)} channel-files...")

    import pandas as pd
    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(worker, tasks, chunksize=16)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape)
    print("dev_type:\n", df.dev_type.value_counts().to_string())
    print("valveIn:\n", df.valveIn.value_counts().sort_index().to_string())
    print("valveOut:\n", df.valveOut.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
