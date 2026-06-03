#!/usr/bin/env python3
"""Read M1 blockage levels (and an aeration verdict) from a single audio clip.

Usage:
    python3 run/predict.py CLIP.wav [--sensor mic|cam] [--json]

The two sensor families behave differently (mics vs camera mics), so pass the
sensor type the clip came from; default is 'mic'. If a filename contains 'cam'
or 'mic' it is auto-detected.

Outputs the predicted discharge-blockage level (valveOut in {1,2,3,4,5,8,11})
and suction-blockage level (valveIn in {1,2,3,4,5,8}), where 1 = fully open /
no restriction and higher = more restricted, plus the nearest-level confidence
(distance in ordinal steps). Aeration is reported as on/off ONLY with a caveat:
the training aeration data is anomalous (single valve config, ~34 dB level
collapse) so an aeration *level* is not recoverable.
"""
import os, sys, json, argparse
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract  # noqa: E402

MODELS = os.path.join(HERE, "models")


def load(target, kind, dev):
    return joblib.load(os.path.join(MODELS, f"{target}_{kind}_{dev}.joblib"))


def severity_predict(bundle, feat):
    X = np.array([[feat[c] for c in bundle["features"]]])
    cont = float(bundle["model"].predict(bundle["scaler"].transform(X))[0])
    levels = bundle["levels"]
    rank = int(np.clip(round(cont), 0, len(levels) - 1))
    level = levels[rank]
    # ordinal distance to the snapped level = a simple confidence proxy
    dist = abs(cont - rank)
    return level, cont, dist


def presence_predict(bundle, feat):
    X = np.array([[feat[c] for c in bundle["features"]]])
    Xs = bundle["scaler"].transform(X)
    if hasattr(bundle["model"], "predict_proba"):
        return float(bundle["model"].predict_proba(Xs)[0, 1])
    return float(bundle["model"].predict(Xs)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--sensor", choices=["mic", "cam"], default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    dev = a.sensor
    if dev is None:
        base = os.path.basename(a.wav).lower()
        dev = "cam" if "cam" in base else "mic"

    feat = extract(a.wav)

    dis_lvl, dis_cont, dis_dist = severity_predict(load("valveOut", "severity", dev), feat)
    suc_lvl, suc_cont, suc_dist = severity_predict(load("valveIn", "severity", dev), feat)
    aer_p = presence_predict(load("aeration", "presence", dev), feat)

    out = {
        "file": os.path.basename(a.wav),
        "sensor_assumed": dev,
        "discharge_blockage": {
            "level": dis_lvl, "scale": [1, 2, 3, 4, 5, 8, 11],
            "restricted": bool(dis_lvl > 1),
            "ordinal_estimate": round(dis_cont, 2), "snap_distance": round(dis_dist, 2),
        },
        "suction_blockage": {
            "level": suc_lvl, "scale": [1, 2, 3, 4, 5, 8],
            "restricted": bool(suc_lvl > 1),
            "ordinal_estimate": round(suc_cont, 2), "snap_distance": round(suc_dist, 2),
        },
        "aeration": {
            "p_on": round(aer_p, 3), "on": bool(aer_p >= 0.5),
            "caveat": ("aeration training data is anomalous (single valve config, "
                       "~34 dB level collapse) -> only on/off, no level; treat as "
                       "low-confidence."),
        },
    }

    if a.json:
        print(json.dumps(out, indent=2))
        return
    print(f"\n  Clip:   {out['file']}   (sensor: {dev})")
    print(f"  {'-'*52}")
    d = out["discharge_blockage"]
    print(f"  Discharge blockage : level {d['level']:>2}  "
          f"({'RESTRICTED' if d['restricted'] else 'open'})   "
          f"[ordinal {d['ordinal_estimate']}, ±{d['snap_distance']}]")
    s = out["suction_blockage"]
    print(f"  Suction blockage   : level {s['level']:>2}  "
          f"({'RESTRICTED' if s['restricted'] else 'open'})   "
          f"[ordinal {s['ordinal_estimate']}, ±{s['snap_distance']}]")
    aer = out["aeration"]
    print(f"  Aeration           : {'ON' if aer['on'] else 'off'}  "
          f"(p={aer['p_on']})  [low-confidence, see caveat]")
    print()


if __name__ == "__main__":
    main()
