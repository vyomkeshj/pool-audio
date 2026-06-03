#!/usr/bin/env python3
"""Extract diagnostic features for EVERY session (one channel each), clean +
noisy, and cache to features_all.csv. Reuses analyze.extract for identical
features. Parallel across cores.

Run: python3 extract_features.py
"""
import os
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from analyze import extract, pick_device, ROOT


def worker(task):
    state, folder, fn, dev, params = task
    try:
        feat, _, _ = extract(os.path.join(ROOT, folder, fn))
    except Exception as ex:
        return None
    noise = params["noise"]
    row = {
        "state": state, "folder": folder, "device": dev, "file": fn,
        "noise": noise, "noise_cat": ("N" if noise == "N" else noise[0]),
        "M2": params["M2"], "M3": params["M3"], "M4": params["M4"],
        "aeration": params["aeration"],
        "valveIn": params["valveIn"], "valveOut": params["valveOut"],
        "suction_blockage": int(params["valveIn"] > 1),
        "discharge_blockage": int(params["valveOut"] > 1),
        "multi_pump": int(any(params[m] for m in ("M2", "M3", "M4"))),
        "aerating": int(params["aeration"] >= 1),
    }
    row.update(feat)
    return row


def main():
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    tasks = []
    for sig, e in idx["measurements"].items():
        state = e["state"]["nominal_state"]
        for s in e["sessions"]:
            dev, fn = pick_device(s["devices"])
            tasks.append((state, s["folder"], fn, dev, e["params"]))
    print(f"extracting {len(tasks)} sessions across cores...")

    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(worker, tasks, chunksize=8)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "features_all.csv"), index=False)
    print("wrote features_all.csv", df.shape)
    print("by noise_cat:\n", df.noise_cat.value_counts().to_string())
    print("by state:\n", df.state.value_counts().to_string())


if __name__ == "__main__":
    main()
