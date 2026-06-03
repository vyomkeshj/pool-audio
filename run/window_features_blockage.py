#!/usr/bin/env python3
"""Window the M1-only valve-sweep data into 8 s windows so the blockage severity
models match the live listener's window length (the 60 s models lose accuracy on
mics at 8 s - see test_window_blockage.py).

Output: run/features_blockage_windows.csv  (M1-only, 8 s win / 8 s hop)
Run:    python3 run/window_features_blockage.py
"""
import os, sys, json
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_array, SR
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(HERE, "measurement_index.json")
OUT = os.path.join(HERE, "features_blockage_windows.csv")
WIN_S, HOP_S = 8.0, 8.0


def worker(task):
    disk, fn, dev, ts, folder, p = task
    try:
        x, sr = sf.read(os.path.join(ROOT, disk, fn), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = signal.resample(x, int(len(x) * SR / sr))
    except Exception:
        return []
    w, h = int(WIN_S * SR), int(HOP_S * SR)
    rows = []
    for wi, st in enumerate(range(0, max(1, len(x) - w + 1), h)):
        seg = x[st:st + w]
        if len(seg) < w * 0.9:
            break
        try:
            feat = extract_array(seg)
        except Exception:
            continue
        row = {"folder": folder, "session": f"{folder}/{ts}", "device": dev,
               "dev_type": "mic" if dev.startswith("mic") else "cam", "win": wi,
               "noise": p["noise"], "noise_cat": "N" if p["noise"] == "N" else p["noise"][0],
               "valveIn": p["valveIn"], "valveOut": p["valveOut"]}
        row.update(feat)
        rows.append(row)
    return rows


def main():
    idx = json.load(open(INDEX))
    tasks = []
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M3"] or p["M4"] or p["aeration"]:
            continue
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                tasks.append((s["disk_path"], fn, dev, s["timestamp"], s["folder"], p))
    print(f"windowing {len(tasks)} M1-only channel-files ({WIN_S}s/{HOP_S}s)...")
    rows = []
    with ProcessPoolExecutor() as ex:
        for i, rs in enumerate(ex.map(worker, tasks, chunksize=16)):
            rows.extend(rs)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)} files, {len(rows)} windows")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape)
    print("valveIn windows:", df.valveIn.value_counts().sort_index().to_dict())
    print("valveOut windows:", df.valveOut.value_counts().sort_index().to_dict())
    print("dev_type:", df.dev_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
