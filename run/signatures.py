#!/usr/bin/env python3
"""Precompute mean power spectra per operating condition so the report app can
show — fast and interactively — HOW the acoustic signature changes with:
  * suction blockage level (valveIn, discharge held open)
  * discharge blockage level (valveOut, suction held open)
  * aeration on vs off (matched valve config)
  * each auxiliary machine M2/M3/M4 on vs off
  * aeration while a motor runs

Averages the Welch PSD over a sample of clean (noise=N) clips per group, per
device family (mic vs cam — they must not be pooled). Saves run/signatures.npz.

Run: python3 run/signatures.py
"""
import os, sys, json, random
import numpy as np
import soundfile as sf
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SR = 44100
NPER = 8192
MAXF = 12000
PER_GROUP = 14
rng = random.Random(0)


def welch_psd(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    x = x - x.mean()
    f, p = signal.welch(x, SR, nperseg=NPER, noverlap=NPER // 2)
    return f, p


def mean_psd(files):
    files = files[:PER_GROUP]
    acc = None; fref = None; n = 0
    for fn in files:
        try:
            f, p = welch_psd(fn)
        except Exception:
            continue
        if fref is None:
            fref = f
        acc = p if acc is None else acc + p
        n += 1
    if n == 0:
        return None, None
    return fref, acc / n


def collect():
    idx = json.load(open(os.path.join(HERE, "measurement_index.json")))
    # flat list of (params, dev_type, abspath)
    items = []
    for sig, e in idx["measurements"].items():
        p = e["params"]
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                items.append((p, "mic" if dev.startswith("mic") else "cam",
                              os.path.join(ROOT, s["disk_path"], fn)))
    return items


def filt(items, dt, **conds):
    out = []
    for p, d, path in items:
        if d != dt:
            continue
        ok = all(p[k] == v for k, v in conds.items())
        if ok:
            out.append(path)
    rng.shuffle(out)
    return out


def main():
    items = collect()
    data = {"freq_max": MAXF}
    for dt in ["mic", "cam"]:
        # M1-only, clean
        def m1(**c):
            return filt(items, dt, M2=0, M3=0, M4=0, aeration=0, noise="N", **c)

        # suction sweep (discharge open vout=1)
        suc = {}
        for lv in [1, 2, 3, 4, 5]:
            f, p = mean_psd(m1(valveIn=lv, valveOut=1))
            if p is not None:
                suc[lv] = p; data[f"freq"] = f
        data[f"{dt}_suction"] = suc

        # discharge sweep (suction open vin=1)
        dis = {}
        for lv in [1, 2, 3, 4, 5, 8, 11]:
            f, p = mean_psd(m1(valveIn=1, valveOut=lv))
            if p is not None:
                dis[lv] = p
        data[f"{dt}_discharge"] = dis

        # aeration on vs off (matched vin1/vout1, aux off, clean)
        f, off = mean_psd(filt(items, dt, M2=0, M3=0, M4=0, aeration=0,
                               valveIn=1, valveOut=1, noise="N"))
        f, on = mean_psd(filt(items, dt, M2=0, M3=0, M4=0, aeration=1,
                              valveIn=1, valveOut=1, noise="N"))
        data[f"{dt}_aer_off"] = off
        data[f"{dt}_aer_on"] = on

        # each machine on vs off (5_25, aeration off, vin1/vout1; on=that machine=1)
        for m in ["M2", "M3", "M4"]:
            base = dict(aeration=0, valveIn=1, valveOut=1, noise="N")
            others = [x for x in ["M2", "M3", "M4"] if x != m]
            off_c = {**base, m: 0, **{o: 0 for o in others}}
            on_c = {**base, m: 1, **{o: 0 for o in others}}
            f, off = mean_psd(filt(items, dt, **off_c))
            f, on = mean_psd(filt(items, dt, **on_c))
            data[f"{dt}_{m}_off"] = off
            data[f"{dt}_{m}_on"] = on
        print(f"[{dt}] suction levels={list(suc)} discharge levels={list(dis)} "
              f"aer_on={'y' if data[f'{dt}_aer_on'] is not None else 'n'}")

    # trim to MAXF and save
    f = data["freq"]
    keep = f <= MAXF
    out = {"freq": f[keep]}
    for k, v in data.items():
        if k in ("freq", "freq_max"):
            continue
        if isinstance(v, dict):
            for lv, p in v.items():
                out[f"{k}_{lv}"] = p[keep]
        elif v is not None:
            out[k] = v[keep]
    np.savez(os.path.join(HERE, "signatures.npz"), **out)
    print("wrote signatures.npz with", len(out), "arrays")


if __name__ == "__main__":
    main()
