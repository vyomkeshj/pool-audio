#!/usr/bin/env python3
"""Extract the 20-feature vector for EVERY channel of EVERY file (all 16 devices
per session), clean + noisy, cached to features_allch.csv.

This is the full per-file feature table (~11,984 rows) used for training the
detection/identification models. Each row carries its device (mic1-8 / cam1-8)
and session id so cross-validation can be grouped by session (no channel leak)
and device effects can be studied. Reuses analyze.extract for identical features.

Run: python3 extract_allch.py
"""
import os, json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from analyze import extract, ROOT


def worker(task):
    folder, fn, dev, timestamp, params = task
    try:
        feat, _, _ = extract(os.path.join(ROOT, folder, fn))
    except Exception:
        return None
    noise = params["noise"]
    row = {
        "folder": folder, "file": fn, "device": dev,
        "dev_type": "mic" if dev.startswith("mic") else "cam",
        "session": f"{folder}/{timestamp}",
        "noise": noise, "noise_cat": ("N" if noise == "N" else noise[0]),
        "M1": params["M1"], "M2": params["M2"], "M3": params["M3"],
        "M4": params["M4"], "aeration": params["aeration"],
        "valveIn": params["valveIn"], "valveOut": params["valveOut"],
    }
    row.update(feat)
    return row


def main():
    idx = json.load(open(os.path.join(ROOT, "measurement_audio_index.json")))
    tasks = []
    for sig, e in idx["measurements"].items():
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                tasks.append((s["folder"], fn, dev, s["timestamp"], e["params"]))
    print(f"extracting {len(tasks)} channel-files across cores...")

    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(worker, tasks, chunksize=16)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "features_allch.csv"), index=False)
    print("wrote features_allch.csv", df.shape)
    print("by dev_type:\n", df.dev_type.value_counts().to_string())
    print("by noise_cat:\n", df.noise_cat.value_counts().to_string())


if __name__ == "__main__":
    main()
