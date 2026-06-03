#!/usr/bin/env python3
"""Extract SHORT-WINDOW features from the 5_25 campaign (the only campaign that
toggles aeration and the auxiliary machines M2/M3/M4).

A live detector listens to a few seconds, not a full 60 s clip, so we train on
~8 s windows (4 s hop) to match inference. Each channel-file becomes ~14 window
rows; CV is grouped by recording session so windows from one recording never
straddle train/test.

Output: run/features_5_25_windows.csv
Run:    python3 run/window_features.py
"""
import os, json
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_array, SR

ROOT = os.path.dirname(HERE)
INDEX = os.path.join(HERE, "measurement_index.json")
OUT = os.path.join(HERE, "features_5_25_windows.csv")
WIN_S = 8.0
HOP_S = 4.0


def worker(task):
    disk, fn, dev, ts, params = task
    try:
        x, sr = sf.read(os.path.join(ROOT, disk, fn), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = signal.resample(x, int(len(x) * SR / sr))
    except Exception:
        return []
    w = int(WIN_S * SR); h = int(HOP_S * SR)
    rows = []
    for wi, start in enumerate(range(0, max(1, len(x) - w + 1), h)):
        seg = x[start:start + w]
        if len(seg) < w * 0.9:
            break
        try:
            feat = extract_array(seg)
        except Exception:
            continue
        row = {"session": f"{params['_folder']}/{ts}", "device": dev,
               "dev_type": "mic" if dev.startswith("mic") else "cam",
               "win": wi, "M2": params["M2"], "M3": params["M3"],
               "M4": params["M4"], "aeration": params["aeration"],
               "valveIn": params["valveIn"], "valveOut": params["valveOut"]}
        row.update(feat)
        rows.append(row)
    return rows


def main():
    idx = json.load(open(INDEX))
    tasks = []
    for sig, e in idx["measurements"].items():
        if "testbedmotor5_25" not in e["folders"]:
            continue
        pr = dict(e["params"])
        for s in e["sessions"]:
            pr2 = dict(pr); pr2["_folder"] = s["folder"]
            for dev, fn in s["devices"].items():
                tasks.append((s["disk_path"], fn, dev, s["timestamp"], pr2))
    print(f"windowing {len(tasks)} channel-files ({WIN_S}s win / {HOP_S}s hop)...")

    rows = []
    with ProcessPoolExecutor() as ex:
        for i, rs in enumerate(ex.map(worker, tasks, chunksize=16)):
            rows.extend(rs)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} files, {len(rows)} windows")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape)
    print("aeration windows:", df.aeration.value_counts().to_dict())
    print("by dev_type:", df.dev_type.value_counts().to_dict())
    print("M2/M3/M4 on-windows:",
          {m: int((df[m] == 1).sum()) for m in ("M2", "M3", "M4")})


if __name__ == "__main__":
    main()
