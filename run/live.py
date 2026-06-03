#!/usr/bin/env python3
"""LIVE acoustic monitor: listens to audio in a sliding window and continuously
reports (a) whether AERATION is on/off and (b) WHICH machines (M2/M3/M4) are
running. Uses the 8 s-window models trained by aeration.py / which_pump.py.

Two input sources:
  --mic            capture from the default microphone live (via `arecord`)
  --file CLIP.wav  stream a wav file through the same pipeline ("simulated live";
                   add --realtime to play it at wall-clock speed)

Options:
  --sensor mic|cam   which sensor family's models to use (default: mic for --mic,
                     auto-detected from the filename for --file)
  --task aeration|pump|all   what to report (default: all)
  --seconds N        stop after N seconds (default: whole file / Ctrl-C for mic)
  --window 8 --hop 2 window / update period in seconds
  --smooth 3         majority/mean smoothing over the last N windows

Example:
  python3 run/live.py --file SOMECLIP.wav
  python3 run/live.py --mic --sensor mic --seconds 30
"""
import os, sys, json, argparse, subprocess, collections, time
import numpy as np
import soundfile as sf
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_array, SR  # noqa: E402
MODELS = os.path.join(HERE, "models")


def load(name):
    p = os.path.join(MODELS, name)
    return joblib.load(p) if os.path.exists(p) else None


def predict_proba(bundle, feat):
    X = np.array([[feat[c] for c in bundle["features"]]])
    return float(bundle["model"].predict_proba(bundle["scaler"].transform(X))[0, 1])


class Monitor:
    """Holds the loaded models and renders one verdict per window."""
    def __init__(self, sensor, task, smooth):
        self.sensor, self.task = sensor, task
        self.aer = load(f"aeration_live_{sensor}.joblib") if task in ("aeration", "all") else None
        self.pumps = {}
        if task in ("pump", "all"):
            for m in ("M2", "M3", "M4"):
                self.pumps[m] = load(f"pump_{m}_{sensor}.joblib")
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=smooth))

    def update(self, seg, t):
        feat = extract_array(seg)
        parts = [f"t={t:5.1f}s"]
        if self.aer is not None:
            p = predict_proba(self.aer, feat)
            self.hist["aer"].append(p)
            ps = float(np.mean(self.hist["aer"]))
            parts.append(f"aeration: {'ON ' if ps >= 0.5 else 'off'} (p={ps:.2f})")
        if self.pumps:
            seg_str = []
            for m, b in self.pumps.items():
                if b is None:
                    continue
                p = predict_proba(b, feat)
                self.hist[m].append(p)
                ps = float(np.mean(self.hist[m]))
                tag = {"M2": "M2pump", "M3": "M3fan", "M4": "M4fan"}[m]
                seg_str.append(f"{tag}:{'ON ' if ps >= 0.5 else 'off'}({ps:.2f})")
            parts.append("running[+M1]: " + " ".join(seg_str))
        return "  |  ".join(parts)


def stream_file(path, win, hop, limit, realtime):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from scipy import signal
        x = signal.resample(x, int(len(x) * SR / sr))
    w, h = int(win * SR), int(hop * SR)
    for start in range(0, len(x) - w + 1, h):
        t = (start + w) / SR
        if limit and t > limit:
            break
        if realtime:
            time.sleep(hop)
        yield x[start:start + w], t


def stream_mic(win, hop, limit):
    w, h = int(win * SR), int(hop * SR)
    cmd = ["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", str(SR), "-t", "raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = np.zeros(0, dtype=np.float32)
    bytes_per = 2
    t0 = time.time()
    try:
        # fill first window
        while len(buf) < w:
            raw = proc.stdout.read(h * bytes_per)
            if not raw:
                return
            buf = np.concatenate([buf, np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0])
        yield buf[-w:].copy(), time.time() - t0
        while True:
            if limit and (time.time() - t0) > limit:
                break
            raw = proc.stdout.read(h * bytes_per)
            if not raw:
                break
            chunk = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
            buf = np.concatenate([buf, chunk])[-w:]
            yield buf.copy(), time.time() - t0
    finally:
        proc.terminate()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mic", action="store_true")
    g.add_argument("--file")
    ap.add_argument("--sensor", choices=["mic", "cam"], default=None)
    ap.add_argument("--task", choices=["aeration", "pump", "all"], default="all")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()

    sensor = a.sensor
    if sensor is None:
        sensor = "cam" if (a.file and "cam" in os.path.basename(a.file).lower()) else "mic"

    mon = Monitor(sensor, a.task, a.smooth)
    src = ("mic" if a.mic else os.path.basename(a.file))
    print(f"# live monitor  source={src}  sensor-models={sensor}  task={a.task}  "
          f"win={a.window}s hop={a.hop}s smooth={a.smooth}")
    print("# (verdicts are smoothed over the last %d windows)\n" % a.smooth)

    gen = (stream_mic(a.window, a.hop, a.seconds) if a.mic
           else stream_file(a.file, a.window, a.hop, a.seconds, a.realtime))
    try:
        for seg, t in gen:
            print("  " + mon.update(seg, t), flush=True)
    except KeyboardInterrupt:
        print("\n# stopped")


if __name__ == "__main__":
    main()
