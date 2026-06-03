#!/usr/bin/env python3
"""Independent acoustic LISTENER: given audio (a live capture buffer or a clip),
report what the rig is doing —
  * which machines are running (M2 2nd-pump / M3 loud-fan / M4 silent-fan; M1 is
    always on),
  * aeration on/off,
  * M1 discharge-blockage level (valveOut) and suction-blockage level (valveIn).

Dual analysis window (chosen from the honest CV): aeration + which-machine read
from the LAST 8 s (their models are trained on 8 s windows); blockage reads from
the WHOLE current buffer up to 60 s (the severity models need a long window —
8 s loses too much, esp. on mics). So in live use, machine/aeration verdicts are
immediate and blockage firms up as the buffer fills toward ~60 s.

Capture sources (helpers): system-audio loopback (`parec` on the sink monitor),
microphone (`arecord`), or a file. Loopback feeds the exact played PCM, so the
listener is genuinely independent of the player yet sees training-quality audio.

This module is imported by the GUI and by test_listener.py. It is also runnable:
  python3 run/listener.py --file CLIP.wav [--sensor mic|cam]
  python3 run/listener.py --loopback [--sensor mic] [--seconds 60]
  python3 run/listener.py --mic [--sensor mic] [--seconds 60]
"""
import os, sys, json, argparse, subprocess, collections, time
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import extract_array, SR  # noqa: E402
MODELS = os.path.join(HERE, "models")

WIN_SHORT_S = 8.0     # aeration / which-machine
BUF_MAX_S = 60.0      # blockage reads from up to this much buffer
BLOCK_MIN_S = 12.0    # below this, blockage flagged "warming up"
BLOCK_GOOD_S = 45.0   # at/above this, blockage flagged reliable
MACHINE_ON = 0.60     # confidence to call an aux machine "on" (margin vs loopback noise)


def _load(name):
    p = os.path.join(MODELS, name)
    return joblib.load(p) if os.path.exists(p) else None


class Listener:
    def __init__(self, sensor="auto"):
        # load BOTH sensor families up front; the family used per analysis is
        # auto-detected from the audio (the listener is independent — it is not
        # told whether it's hearing a microphone or a camera mic).
        self.fam = {s: {"aer": _load(f"aeration_live_{s}.joblib"),
                        "pumps": {m: _load(f"pump_{m}_{s}.joblib") for m in ("M2", "M3", "M4")},
                        "block": {"discharge": _load(f"valveOut_severity_{s}.joblib"),
                                  "suction": _load(f"valveIn_severity_{s}.joblib")}}
                    for s in ("mic", "cam")}
        self.sensor_id = _load("sensor_id.joblib")
        self.set_sensor(sensor)

    def set_sensor(self, sensor):
        # "auto" => detect per analysis; "mic"/"cam" => force that family
        self.sensor_mode = sensor or "auto"

    def _detect_sensor(self, feats):
        """Majority vote of the mic/cam classifier over the buffer's windows."""
        if self.sensor_id is None:
            return "mic"
        cols = self.sensor_id["features"]
        X = np.array([[f[c] for c in cols] for f in feats])
        pred = self.sensor_id["model"].predict(self.sensor_id["scaler"].transform(X))
        return "cam" if pred.mean() >= 0.5 else "mic"

    @staticmethod
    def _p1(bundle, feat):
        X = np.array([[feat[c] for c in bundle["features"]]])
        return float(bundle["model"].predict_proba(bundle["scaler"].transform(X))[0, 1])

    @staticmethod
    def _level(bundle, feat):
        X = np.array([[feat[c] for c in bundle["features"]]])
        cont = float(bundle["model"].predict(bundle["scaler"].transform(X))[0])
        levels = bundle["levels"]
        rank = int(np.clip(round(cont), 0, len(levels) - 1))
        return levels[rank], cont, abs(cont - rank)

    def analyze(self, x):
        """x: mono float array at SR (any length >= ~1 s). Returns a dict of
        verdicts. Aeration/which-machine are averaged over ALL 8 s windows in the
        buffer (so a live verdict smooths and firms up as the buffer fills);
        blockage uses the whole buffer up to 60 s."""
        x = np.asarray(x, dtype="float32")
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        rms_db = 20 * np.log10(rms)
        out = {"buffer_s": round(len(x) / SR, 1),
               "input_rms_db": round(rms_db, 1), "silent": bool(rms_db < -70)}
        if out["silent"]:
            # nothing meaningful to classify — don't emit garbage verdicts
            out["sensor"] = "—"; out["sensor_auto"] = (self.sensor_mode == "auto")
            out["aeration"] = {"on": False, "p": 0.0}
            out["machines"] = {m: {"on": False, "p": 0.0, "label": m}
                               for m in ("M2", "M3", "M4")}
            out["running"] = ["M1"]
            out["blockage"] = {}
            return out

        # features for every 8 s window in the buffer (4 s hop), for aer/machines
        w, h = int(WIN_SHORT_S * SR), int(WIN_SHORT_S * SR / 2)
        feats = []
        if len(x) >= w:
            for st in range(0, len(x) - w + 1, h):
                feats.append(extract_array(x[st:st + w]))
        else:
            feats.append(extract_array(x))
        out["n_windows"] = len(feats)

        # pick the sensor family: auto-detect from the audio, or forced
        sensor = self._detect_sensor(feats) if self.sensor_mode == "auto" else self.sensor_mode
        out["sensor"] = sensor
        out["sensor_auto"] = (self.sensor_mode == "auto")
        models = self.fam[sensor]

        if models["aer"] is not None:
            p = float(np.mean([self._p1(models["aer"], f) for f in feats]))
            out["aeration"] = {"on": bool(p >= 0.5), "p": round(p, 3)}
        machines = {}
        for m, b in models["pumps"].items():
            if b is None:
                continue
            p = float(np.mean([self._p1(b, f) for f in feats]))
            machines[m] = {"on": bool(p >= MACHINE_ON), "p": round(p, 3),
                           "label": b.get("label", m)}
        out["machines"] = machines
        out["running"] = ["M1"] + [m for m, v in machines.items() if v["on"]]

        long = x[-int(BUF_MAX_S * SR):]
        flong = extract_array(long)
        block = {}
        secs = len(long) / SR
        aer_on = out.get("aeration", {}).get("on", False)
        aux_on = len(out["running"]) > 1   # any M2/M3/M4 detected
        # The blockage severity models were trained on M1-ONLY audio. Running an
        # auxiliary machine (M2/M3/M4) or aeration changes the acoustic baseline,
        # so blockage is only trustworthy in the M1-only regime — flag otherwise.
        rel = ("uncertain_aux" if (aer_on or aux_on) else
               "reliable" if secs >= BLOCK_GOOD_S else
               "warming_up" if secs < BLOCK_MIN_S else "partial")
        for axis, b in models["block"].items():
            if b is None:
                continue
            lvl, cont, dist = self._level(b, flong)
            block[axis] = {"level": int(lvl), "restricted": bool(lvl > 1),
                           "ordinal": round(cont, 2), "snap_dist": round(dist, 2),
                           "scale": b["levels"], "reliability": rel}
        out["blockage"] = block
        return out

    def render(self, r):
        parts = [f"buf={r['buffer_s']:4.0f}s", f"in={r['input_rms_db']:6.1f}dB"]
        if r.get("silent"):
            return "  |  ".join(parts) + "  |  (silent — no audio on this input)"
        sens = r.get("sensor", "?")
        parts.append(f"sensor:{sens}{'(auto)' if r.get('sensor_auto') else ''}")
        if "aeration" in r:
            a = r["aeration"]
            parts.append(f"aeration:{'ON ' if a['on'] else 'off'}({a['p']:.2f})")
        m = r["machines"]
        ms = " ".join(f"{ {'M2':'M2pump','M3':'M3fan','M4':'M4fan'}[k] }:"
                      f"{'ON' if v['on'] else 'off'}({v['p']:.2f})" for k, v in m.items())
        parts.append("running[+M1]: " + ms)
        b = r.get("blockage", {})
        if "discharge" in b:
            d, s = b["discharge"], b["suction"]
            tag = {"reliable": "", "partial": "~", "warming_up": "?",
                   "uncertain_aux": " (aux?)"}[d["reliability"]]
            parts.append(f"discharge:L{d['level']}{tag} suction:L{s['level']}{tag}")
        return "  |  ".join(parts)


# ---------------------------------------------------------------- capture helpers
def default_monitor():
    try:
        info = subprocess.run(["pactl", "info"], capture_output=True, text=True).stdout
        sink = next(l.split(":", 1)[1].strip() for l in info.splitlines()
                    if l.startswith("Default Sink"))
        return sink + ".monitor"
    except Exception:
        return None


def stream_capture(source, win, hop, limit, monitor=None, flush_event=None):
    """Yield (buffer_so_far, t) growing to 60 s then rolling. source in
    {'mic','loopback'}. Reads s16le mono 44100 via PipeWire `parec` (mic = default
    source; loopback = the default sink's .monitor).

    The buffer is FLUSHED when (a) a short silence gap is seen — playback stopped /
    a new clip is starting — or (b) flush_event is set (the player started a new
    clip). This stops consecutive clips from blending in the rolling buffer, which
    would otherwise corrupt the verdict for up to a minute after switching clips.
    Silence is checked on fine 0.5 s sub-reads so a brief gap isn't masked by a
    coarse hop."""
    bufmax = int(BUF_MAX_S * SR)
    sub = max(1, int(0.5 * SR))                      # fine read granularity
    base = ["parec", "--format=s16le", "--rate", str(SR), "--channels=1"]
    cmd = base if source == "mic" else base + ["-d", monitor or default_monitor()]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = np.zeros(0, dtype=np.float32)
    t0 = time.time()
    last_yield = 0.0
    SILENT = 10 ** (-65 / 20)
    try:
        while True:
            if limit and (time.time() - t0) > limit:
                break
            raw = proc.stdout.read(sub * 2)
            if not raw:
                break
            chunk = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
            now = time.time() - t0
            if flush_event is not None and flush_event.is_set():
                buf = np.zeros(0, dtype=np.float32); flush_event.clear()
            if np.sqrt(np.mean(chunk ** 2)) < SILENT:        # gap -> start fresh
                buf = np.zeros(0, dtype=np.float32)
                if now - last_yield >= hop:
                    last_yield = now
                    yield chunk.copy(), now              # UI shows "silent"
                continue
            buf = np.concatenate([buf, chunk])[-bufmax:]
            if len(buf) >= int(win * SR) and now - last_yield >= hop:
                last_yield = now
                yield buf.copy(), now
    finally:
        proc.terminate()


def stream_file(path, win, hop, limit):
    import soundfile as sf
    from scipy import signal
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = signal.resample(x, int(len(x) * SR / sr))
    bufmax = int(BUF_MAX_S * SR); h = int(hop * SR); w = int(win * SR)
    for end in range(w, len(x) + 1, h):
        t = end / SR
        if limit and t > limit:
            break
        yield x[max(0, end - bufmax):end].copy(), t


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mic", action="store_true")
    g.add_argument("--loopback", action="store_true")
    g.add_argument("--file")
    ap.add_argument("--sensor", choices=["auto", "mic", "cam"], default="auto",
                    help="model family; 'auto' (default) detects it from the audio")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--hop", type=float, default=2.0)
    a = ap.parse_args()
    lis = Listener(a.sensor)
    src = "mic" if a.mic else ("loopback" if a.loopback else os.path.basename(a.file))
    print(f"# listener  source={src}  sensor={a.sensor}  hop={a.hop}s")
    if a.file:
        gen = stream_file(a.file, WIN_SHORT_S, a.hop, a.seconds)
    else:
        gen = stream_capture("mic" if a.mic else "loopback", WIN_SHORT_S, a.hop, a.seconds)
    try:
        for buf, t in gen:
            print("  " + lis.render(lis.analyze(buf)), flush=True)
    except KeyboardInterrupt:
        print("\n# stopped")


if __name__ == "__main__":
    main()
