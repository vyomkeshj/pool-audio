#!/usr/bin/env python3
"""Does the 60 s-trained blockage severity model still read levels correctly from
the SHORT windows a live listener uses? Probe several window lengths against the
data so we can choose the live listening window (and decide whether to retrain).

For a stratified sample of M1-only clips (all vin/vout levels, mic+cam), extract
features from the first W seconds and from the full clip, run the saved
valveOut/valveIn severity models, and report within-1 / exact vs ground truth at
each W. (These are the deployed models; this measures window-length robustness,
not generalization - that was established in train.py.)

Output: run/window_blockage_results.json
Run: python3 run/test_window_blockage.py
"""
import os, sys, json, random
import numpy as np
import soundfile as sf
import joblib
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_array, SR
ROOT = os.path.dirname(HERE)
WINDOWS = [8, 16, 30, 60]
PER_CELL = 6  # clips per (dev_type, level) cell


def load_files():
    idx = json.load(open(os.path.join(HERE, "measurement_index.json")))
    items = []
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M3"] or p["M4"] or p["aeration"]:
            continue
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                items.append((s["disk_path"], fn, dev,
                              "mic" if dev.startswith("mic") else "cam",
                              p["valveIn"], p["valveOut"]))
    return items


def stratified(items, key_idx, levels, dev):
    rng = random.Random(0)
    out = []
    for lv in levels:
        pool = [it for it in items if it[3] == dev and it[key_idx] == lv]
        rng.shuffle(pool)
        out += pool[:PER_CELL]
    return out


def predict_level(bundle, feat):
    X = np.array([[feat[c] for c in bundle["features"]]])
    cont = float(bundle["model"].predict(bundle["scaler"].transform(X))[0])
    levels = bundle["levels"]
    rank = int(np.clip(round(cont), 0, len(levels) - 1))
    return rank


def main():
    items = load_files()
    res = {"windows_s": WINDOWS, "per_cell": PER_CELL}
    axes = [("discharge", "valveOut", [1, 2, 3, 4, 5, 8, 11], 5),
            ("suction", "valveIn", [1, 2, 3, 4, 5, 8], 4)]
    for axis, target, levels, kidx in axes:
        res[axis] = {}
        rankmap = {v: i for i, v in enumerate(levels)}
        for dev in ["mic", "cam"]:
            bundle = joblib.load(os.path.join(HERE, "models", f"{target}_severity_{dev}.joblib"))
            sample = stratified(items, kidx, levels, dev)
            # cache decoded audio once per file
            per_w = {w: {"within1": [], "exact": []} for w in WINDOWS}
            for disk, fn, d, dt, vin, vout in sample:
                try:
                    x, sr = sf.read(os.path.join(ROOT, disk, fn), dtype="float32", always_2d=False)
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    if sr != SR:
                        x = signal.resample(x, int(len(x) * SR / sr))
                except Exception:
                    continue
                true_rank = rankmap[vout if axis == "discharge" else vin]
                for w in WINDOWS:
                    seg = x[:int(w * SR)] if w * SR < len(x) else x
                    pr = predict_level(bundle, extract_array(seg))
                    per_w[w]["within1"].append(abs(pr - true_rank) <= 1)
                    per_w[w]["exact"].append(pr == true_rank)
            res[axis][dev] = {str(w): {"within1": float(np.mean(per_w[w]["within1"])),
                                       "exact": float(np.mean(per_w[w]["exact"])),
                                       "n": len(per_w[w]["within1"])} for w in WINDOWS}
            line = "  ".join(f"{w}s:w1={res[axis][dev][str(w)]['within1']:.2f}/"
                             f"ex={res[axis][dev][str(w)]['exact']:.2f}" for w in WINDOWS)
            print(f"{axis:9s} {dev}: {line}")
    json.dump(res, open(os.path.join(HERE, "window_blockage_results.json"), "w"), indent=2)
    print("wrote window_blockage_results.json")


if __name__ == "__main__":
    main()
