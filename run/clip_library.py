#!/usr/bin/env python3
"""Index-backed clip picker for the desktop app: find a recording matching the
requested operating condition (blockage levels / aeration / which machines /
sensor / noise) and return its on-disk path + ground-truth labels."""
import os, json, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class ClipLibrary:
    def __init__(self):
        idx = json.load(open(os.path.join(HERE, "measurement_index.json")))
        self.clips = []   # list of dicts with params + path + device
        for sig, e in idx["measurements"].items():
            p = e["params"]
            for s in e["sessions"]:
                for dev, fn in s["devices"].items():
                    self.clips.append({
                        "path": os.path.join(ROOT, s["disk_path"], fn),
                        "file": fn, "device": dev,
                        "dev_type": "mic" if dev.startswith("mic") else "cam",
                        "valveIn": p["valveIn"], "valveOut": p["valveOut"],
                        "aeration": p["aeration"], "M2": p["M2"], "M3": p["M3"],
                        "M4": p["M4"], "noise": p["noise"],
                        "noise_cat": "N" if p["noise"] == "N" else p["noise"][0],
                    })
        self.rng = random.Random()

    def options(self, field):
        return sorted({c[field] for c in self.clips})

    def find(self, dev_type=None, valveIn=None, valveOut=None, aeration=None,
             M2=None, M3=None, M4=None, noise_cat=None, device=None, pick="random"):
        cand = self.clips
        def f(c, k, v):
            return v is None or c[k] == v
        cand = [c for c in cand if
                f(c, "dev_type", dev_type) and f(c, "valveIn", valveIn) and
                f(c, "valveOut", valveOut) and f(c, "aeration", aeration) and
                f(c, "M2", M2) and f(c, "M3", M3) and f(c, "M4", M4) and
                f(c, "noise_cat", noise_cat) and f(c, "device", device)]
        if not cand:
            return None
        return self.rng.choice(cand) if pick == "random" else cand[0]

    @staticmethod
    def describe(c):
        machines = "M1" + "".join(f"+{m}" for m in ("M2", "M3", "M4") if c[m])
        return (f"suction(valveIn)=L{c['valveIn']}  discharge(valveOut)=L{c['valveOut']}  "
                f"aeration={'ON' if c['aeration'] else 'off'}  running={machines}  "
                f"noise={c['noise']}  [{c['device']}]")


if __name__ == "__main__":
    lib = ClipLibrary()
    print("clips:", len(lib.clips))
    print("valveIn opts:", lib.options("valveIn"))
    print("valveOut opts:", lib.options("valveOut"))
    c = lib.find(dev_type="cam", valveIn=5, valveOut=2, aeration=0)
    print("example:", ClipLibrary.describe(c) if c else "none")
