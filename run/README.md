# `run/` — Blockage-level reader and live aeration / which-machine detectors

> Live detectors (added 2026-06-03). `run/live.py` listens to audio in a sliding
> 8 s window — from the real microphone (`--mic`, via `arecord`) or by streaming a
> wav (`--file`) — and continuously reports aeration on/off and which machines
> (M2/M3/M4) are running. Held-out streaming test (whole recording sessions held
> out, so the 8 sibling mics of one recording never leak across the split):
> aeration on/off = 0.99 single-mic / 1.00 camera clip-level accuracy; machine
> presence (single mic) M2 0.87/0.96, M3 0.96/0.97, M4 0.72/0.88 (mic/cam) —
> averaging the 8 channels lifts these to 0.98–1.00. See the Live detectors
> section below.
>
> This corrects the earlier assessment that aeration is not answerable: the on/off
> question is answerable. At the matched valve config (vin1/vout1), toggling only
> aeration produces a strong gain-invariant spectral signature (mic: +4.5 dB
> 50–100 Hz, d=1.75; cam: tone shift −640 Hz, harm3 d=2.9), so the detector keys on
> real aeration acoustics, not the recording-gain artifact. Caveat unchanged:
> aeration was only ever recorded at one valve config, so behaviour when M1 is
> throttled is unverified (re-record with the throttle swept to close this).

---

# Blockage-level reader (re-run with the two new campaigns)

Rebuilt the whole pipeline after two new recording folders were added
(`testbedmotor5_19wav`, `testbedmotor5_20wav` — the ones the protocol referenced
but were previously missing on disk). They are M1-only valve sweeps that extend
the suction axis to new levels `valveIn ∈ {5, 8}` (it previously stopped at 4),
which is the axis the previous round identified as the weak spot.

Goal: a model that takes a ~60 s clip and outputs the discharge blockage level
(`valveOut`) and suction blockage level (`valveIn`) accurately, robustly under
environmental noise. (`1` = fully open, higher = more restricted.)

## Summary of results (leakage-honest)

Per device type (mics vs camera-mics modeled separately — they are two sensor
families). Evaluated three ways, strictest last:

| axis | sensor | config-grouped¹ within-1 / exact | leave-campaign-out² within-1 / exact / MAE | clean→noisy³ |
|---|---|---|---|---|
| Discharge (`valveOut` 1-11) | mic | 1.00 / 1.00 | 1.00 / 1.00 / 0.00 steps | within-1 = 1.00 on every noise A–E |
| Discharge | cam | 1.00 / 1.00 | 1.00 / 1.00 / 0.00 steps | within-1 = 1.00 on every noise A–E |
| Suction (`valveIn` 1-8) | mic | 1.00 / 1.00 | 1.00 / 0.65 / 0.35 steps | within-1 = 1.00 on every noise A–E |
| Suction | cam | 1.00 / 1.00 | 1.00 / 0.65 / 0.35 steps | within-1 = 1.00 on every noise A–E |

¹ config-grouped CV — GroupKFold where a whole physical valve config `(vin,vout)`
(and its noisy/repeat twins) is held out, so a clip is never scored from a sibling
recording of its own config. This measures "read a calibrated level on an unseen
config".

² leave-one-campaign-out — train on some recording days, test on an entirely
different day (held-out recordings never seen). The deployment-realistic test.

³ clean→noisy — train only on clean (`noise=N`) clips, test on each
environmental-noise class (playground/lawnmower/traffic/speech/music).

### Interpretation

- Discharge blockage level is read within one step every time (within-1 = 1.00
  across held-out days and all noise), and the exact level is near-perfect on
  cameras (CV exact = 1.00; deployed single-model in-sample 0.99) and strong on
  mics (within-1 ≥ 0.995, exact lower). The discharge throttle drives a large,
  monotonic, day-stable spectral change (the pump tone tracks load), so the 7
  levels form 7 cleanly separated clusters. This is a large, robust signal —
  cameras are the sensor to deploy for discharge.
- Suction blockage level is read within one step every time (within-1 = 1.00),
  exact ≈ 0.65 across days, MAE ≈ 0.35 ordinal steps. This is a substantial
  improvement over the previous round (honest suction MAE ≈ 1.14 with only 4
  levels). The remaining exact misses are concentrated on the two new levels
  `vin=5` and `vin=8`, each recorded on only one day, so leaving that day out
  removes the level entirely (it becomes pure extrapolation) and the model snaps to
  the nearest seen level. For suction levels present on ≥2 days it is also exact.
  Recording `vin=5/8` on a second day would close that gap.
- Confirmation that exact=1.0 is not leakage: the `leave-one-level-out` test
  (hold out a level entirely, see `fig_leave_level_out.png`) shows the model
  predicts the nearest adjacent rank with a clean monotonic ordering (MAE exactly
  1 step). Under per-recording leakage a held-out level would be predicted
  randomly, not cleanly adjacent. The features encode a smooth ordinal severity
  signal. A second check confirms it: the deployed single regularized model fits
  its own training data at exact 0.78–0.99 — below the CV exact — which is
  impossible under leakage (leakage inflates in-sample ≥ CV). within-1 stays
  ≥ 0.995 throughout.

### Aeration (corrected 2026-06-03)

Aeration on/off is answerable — see the live-detector section at the top and below
(held-out clip-level accuracy 1.000). The earlier "~34 dB quieter, just an
idle-recording artifact" claim was an averaging artifact: it compared all
aeration-ON clips (only at `vin1/vout1`) against the whole OFF set (every valve
config). At the matched config the level difference is small (mic −3 dB, cam
+6.5 dB) and the separation comes from a genuine gain-invariant spectral signature
(50–100 Hz boost on mics, tonal/harmonic shift on cameras). `predict.py` (the 60 s
blockage CLI) still carries the old low-confidence aeration flag for
back-compatibility; the `live.py` / `aeration.py` models are the validated ones —
use those. Aeration level remains not answerable (only on/off was recorded, at a
single valve config). Caveat: validity when M1 is throttled is untested (aeration
only recorded at `vin1/vout1`).

## Files

| File | Role |
|---|---|
| `build_index.py` → `measurement_index.json` | indexes all 7 folders (13,850 files, 423 signatures), handles the nested `5_19/5_20` paths and salvages 16 `noise→oise` typo'd filenames |
| `features.py` → `features_allch.csv` | enhanced gain-invariant features for every channel of every file (30 features: proven 20 + finer low-freq bands + pump-tone harmonics + band ratios) |
| `train.py` → `models/*.joblib`, `results.json`, `fig_*.png` | per-device-type severity + presence models; config-grouped, leave-campaign-out, leave-level-out, and clean→noisy evaluations; aeration report |
| `importance.py` → `fig_importance.png` | permutation feature importance (discharge ← `tone_freq` + low bands; suction ← low/high bands + `zcr`) |
| `predict.py` | CLI: `python3 run/predict.py CLIP.wav [--sensor mic\|cam] [--json]` → discharge level, suction level, aeration flag |
| `models/<target>_<kind>_<mic\|cam>.joblib` | bundles `{scaler, model, features, levels, rankmap}`; production models trained on all M1-only data |

## Reproduce

```bash
PY=~/miniconda3/envs/pool-audio/bin/python   # env with sklearn 1.8, soundfile, scipy
$PY run/build_index.py      # -> run/measurement_index.json   (~1 s)
$PY run/features.py         # -> run/features_allch.csv        (~10 min, 16 cores)
$PY run/train.py            # -> run/models/*, results.json, figures (~1 min)
$PY run/importance.py       # -> run/fig_importance.png
$PY run/predict.py SOME_CLIP.wav --sensor cam
```

## Method notes / improvements over the previous round

1. More and better-spread data: 13,850 channel-files across 7 campaigns; suction
   now spans `{1,2,3,4,5,8}` instead of `{1,2,3,4}`.
2. Gradient-boosted trees (`HistGradientBoosting`) replace RandomForest.
3. Ordinal severity modeled in evenly-spaced rank space, giving meaningful MAE /
   within-1.
4. Three honesty levels of CV (config-grouped, leave-campaign-out,
   leave-level-out) instead of a single grouped CV — separates "read a calibrated
   level" from "read an unseen day" from "read an uncalibrated level".
5. Per-device-type models throughout (mics and camera-mics never pooled).
6. Gain-invariant features: all band energies are dB relative to total power, plus
   pump-tone harmonic ratios, so models do not rely on channel gain (ablating
   `rms_db` leaves within-1 unchanged).

---

## Live detectors (aeration on/off + which machine is running)

Both questions are only exercised in the 5_25 campaign (the M2/M3/M4 × aeration
factorial, clean audio). A live monitor hears a few seconds, not 60 s, so these
models are trained on 8 s windows (4 s hop) and evaluated held out by recording
session (windows from one recording never split train/test).

### Scripts

| file | role |
|---|---|
| `window_features.py` → `features_5_25_windows.csv` | 8 s-window features for every channel of 5_25 (33,947 windows) |
| `aeration.py` → `models/aeration_live_<mic\|cam>.joblib`, `aeration_results.json`, `fig_aeration.png` | aeration on/off; matched-config control test (vin1/vout1, gain-invariant) |
| `which_pump.py` → `models/pump_<M2\|M3\|M4>_<mic\|cam>.joblib`, `which_pump_results.json`, `fig_which_pump.png` | per-machine presence (multi-label) + 8-way combo |
| `test_live.py` → `live_test_results.json` | held-out streaming test: window-level + clip-level accuracy |
| `live.py` | the live monitor (mic or file) |

### Run it

```bash
PY=~/miniconda3/envs/pool-audio/bin/python
# (re)build everything
$PY run/window_features.py && $PY run/aeration.py && $PY run/which_pump.py && $PY run/test_live.py

# LIVE from the microphone (Ctrl-C to stop)
$PY run/live.py --mic --sensor mic
# stream a recording as if live (add --realtime to play at wall-clock speed)
$PY run/live.py --file SOMECLIP.wav --task all
# just aeration, or just machines:
$PY run/live.py --mic --task aeration
$PY run/live.py --file CLIP.wav --task pump
```

Sample output (streaming an aeration-ON clip):
```
  t=  8.0s  |  aeration: ON  (p=1.00)  |  running[+M1]: M2pump:off(0.00) M3fan:off(0.00) M4fan:off(0.02)
```
`--sensor` picks which sensor family's models to use (mic vs camera-mic); for
`--mic` it defaults to `mic`. Verdicts are smoothed over the last `--smooth` (3)
windows.

### Held-out results (`test_live.py`)

Held out by recording session, so the 8 sibling mics (and 8 cams) of one recording
never split across train/test — they capture the same acoustic instant, so
splitting them would leak. (Effect size: a naive window-random split inflates
aeration window-acc 0.977 → 0.993.) `window` = one 8 s window from one sensor;
`1-mic clip` = one sensor, mean prob over its ~14 windows (the honest single-mic
live number); `8ch-fused` = all 8 same-type channels of a recording averaged (an
upgrade if you deploy several sensors — do not read it as single-mic).

| detector | sensor | window acc | 1-mic clip | 8ch-fused |
|---|---|---|---|---|
| aeration on/off (matched vin1/vout1, gain-invariant) | mic | 0.977 | 0.991 | 1.000 |
| aeration on/off | cam | 0.998 | 1.000 | 1.000 |
| M2 — 2nd large pump | mic / cam | 0.82 / 0.93 | 0.87 / 0.96 | 0.98 / 1.00 |
| M3 — exhaust fan (loud) | mic / cam | 0.94 / 0.96 | 0.96 / 0.97 | 1.00 / 1.00 |
| M4 — exhaust fan (near-silent) | mic / cam | 0.68 / 0.85 | 0.72 / 0.88 | 0.82 / 0.99 |

- Aeration on/off is the strongest result: 0.99 single-mic / 1.00 on a camera, on
  held-out recordings, and the matched-config + gain-invariant test confirms it is
  a real acoustic signal (not gain/config leakage).
- Single mic vs fusion matters for the harder machines: M2/M4 improve ~0.10 when
  the 8 sensors are averaged. With one mic, M4 (near-silent) is the weak point
  (0.72); deploy a camera-mic (0.88) or fuse channels for it.
- M1 is never a target — it is always on, so it cannot be isolated by ear; the
  "running" line always implies `+M1`.
- M4 (near-silent fan) is the hard case on a microphone (clip 0.82); a camera-mic
  reads it reliably (0.99). M2/M3 are read easily on either.

### Deployment caveats

1. Aeration was recorded at only one valve config (`vin1/vout1`). The detector is
   accurate there, but its behaviour when M1's suction/discharge is throttled is
   unverified. To make it robust, re-record aeration on/off across the valve sweep.
2. These detectors are validated on the rig's own recordings (5_25), held out by
   session — not on a different rig or room. Background-noise robustness was not
   tested for aeration/machines (5_25 is clean only); the blockage models above are
   the ones with A–E noise-robustness evidence.
3. Train/inference windows are matched (8 s) so live behaviour matches the test.

---

## Desktop app (player + independent listener + live spectrogram)

`run/app_desktop.py` — a Tkinter desktop app in two independent halves, launched
with `./run/run_gui.sh`.

Player (left). Pick an operating condition — sensor type, discharge level
(`valveOut`), suction level (`valveIn`), aeration on/off, which machines, noise —
and Find & Play: it selects a matching recording from the dataset, plays it on the
speakers (`aplay`), and shows the clip's ground-truth labels.

Listener (right, independent). Blind to the player, it captures audio and infers
the condition. Source = system-audio loopback (`parec` on the sink monitor, which
hears exactly what is playing, at training quality) or the microphone (`arecord`).
It reports which machines are running (M1 always on), aeration on/off, and the
discharge and suction blockage levels, updating ~every 2 s.

Live spectrogram (bottom). Updates from the captured audio while listening.

### How it decides (and its stated limitations)

- Sensor family is auto-detected, not selected. Mics and camera-mics are two
  acoustically distinct families (pump tone ~270 Hz vs a ~720 Hz structural tone),
  so they need different models — but the listener identifies which family it is
  hearing from the audio itself (`sensor_id.joblib`, 98% per window / 100% per
  clip, gain-invariant) and picks the matching models. It displays the detected
  family ("sensor: cam (auto-detected)"); there is no manual sensor selector, so
  the listener stays genuinely blind to the player.
- Aeration and which-machine: averaged over every 8 s window in the rolling buffer
  (their models are trained on 8 s), so verdicts smooth and stabilize quickly.
- Blockage level: read from the whole rolling buffer (up to 60 s) with the 60 s
  models — blockage needs a long window (8 s loses too much, especially on mics).
  Shown as warming up → firming → reliable as the buffer fills.
- Blockage is only asserted in the M1-only regime. The severity models were trained
  on M1-only audio; if the listener detects an auxiliary machine (M2/M3/M4) or
  aeration, the acoustic baseline differs and blockage is flagged "uncertain ·
  other equipment running" rather than reported as a (wrong) level.

### Tested against the data first (before the UI)

Leakage-controlled tests (held out by recording session, so the 8 sibling mics
never leak; broad-population end-to-end via the listener):

| output | mic | cam | how tested |
|---|---|---|---|
| discharge blockage level | within-1 0.97 / exact 0.77 | within-1 1.00 / exact 0.98 | full 60 s buffer, M1-only clips |
| suction blockage level | within-1 1.00 / exact 0.88 | within-1 1.00 / exact 0.99 | full 60 s buffer, M1-only clips |
| aeration on/off | recall 1.00, false-alarm 0.000 | recall 1.00, FA 0.000 | broad pop. after hard-negative training |
| M2 / M3 / M4 running | recall 1.00/1.00/0.91, FA 0 | recall 1.00/1.00/1.00, FA 0 | aeration-off clips, window-aggregated |

Hard-negative training (adding the M1-only blockage windows as negatives) cut the
aeration false-alarm from 19% to 0% on mics; window-aggregation recovered machine
recall. End-to-end loopback runs verified the real data path (speaker → system
audio → listener): an aeration clip reads aeration ON 0.98; a `vout=8` clip reads
discharge L8; an M3 clip reads M3 ON 0.99 with blockage correctly flagged
uncertain.

### Files
| file | role |
|---|---|
| `app_desktop.py`, `run_gui.sh` | the desktop app + launcher |
| `clip_library.py` | index-backed clip picker for the player |
| `listener.py` | listener engine (loopback/mic/file capture, auto sensor-family detection, dual-window inference); also a CLI |
| `train_sensor_id.py` → `models/sensor_id.joblib` | mic-vs-cam family auto-detector (so the listener needs no manual sensor selector) |
| `test_listener.py` → `listener_test_results.json` | broad end-to-end accuracy + false-alarm test |
| `window_features_blockage.py`, `train_blockage_windows.py`, `test_window_blockage.py` | window-length study for blockage (showed 60 s needed) |
| `train_aux_hardneg.py` → `aux_hardneg_results.json` | hard-negative retrain of aeration + pump detectors |

### Notes / caveats
- Use the `loopback — hears what's playing` source (the default). Both sources
  capture via PipeWire `parec`; the microphone source reads the OS default input,
  which on many laptops is muted/empty (reads digital silence) — in that case the
  app shows "no audio on this input" and a "no audio … switch Source to loopback"
  spectrogram overlay rather than reporting a result.
- Loopback resamples 48→44.1 kHz, a small domain shift from the (44.1 kHz) training
  clips; the conservative aeration model therefore confirms aeration-ON over
  ~15–20 s of buffer rather than instantly (it climbs to ~0.9 by 60 s) — correct,
  with no false positives. Machines/blockage are unaffected.
- For accurate listening, use loopback (digital). The microphone-from-speaker path
  re-records through speaker + room and was not validated (laptop mic ≠ AB13X
  capsule).
- Blockage validity holds for M1-only; aeration was only recorded at one valve
  config; 5_25 (aux/aeration) is clean-only (no A–E noise robustness for those).
- Dropdowns / theme: the UI uses the ttk `clam` theme with explicit light-on-dark
  combobox colours so the readonly dropdowns are legible on the dark background.

---

## Research report app (Streamlit) — `report_app.py`

A presentation-ready, multi-page report. Launch: `./run/run_report.sh` (serves on
:8501). Pages:

1. Overview — the two-axis summary (M1 flow restriction; which machine / aeration),
   key numbers.
2. Dataset & parameters — 7 campaigns / 13,850 files, the joint valve-coverage
   table, noise set.
3. Acoustic signatures — interactive mean-spectrum plots (per sensor family)
   showing how the signal changes with suction vs discharge blockage, aeration
   on/off (Δ-spectrum), and each machine M2/M3/M4 on/off. Backed by `signatures.py`
   → `signatures.npz`.
4. Mic & sensor differences — the 8 mics are not interchangeable: per-channel
   spectra plus a per-channel blockage-reading quality ranking (mic discharge ρ
   spans 0.37→0.85; camera-mics uniformly 0.81–0.93). Backed by `channels.py` →
   `channels.json` + `channels_psd.npz`. Explains why models are per-family.
5. Models & training — features, per-sensor gradient-boosted trees, the 8 s vs 60 s
   windows, leakage-honest CV (config/campaign/level grouping, sibling-mic
   handling), hard-negative training, sensor auto-detect.
6. Results — the metric tables + figures.
7. Live demo — pick a dataset clip by condition; the trained models run directly on
   the file (the reliable path) and the prediction is shown against ground truth,
   with a spectrogram.

### Aeration over loopback vs direct-file (relevant for demos)
The desktop app's loopback path resamples 48→44.1 kHz, which weakens the subtle
aeration signal — an aeration-ON clip can sit at p≈0.50 (borderline "off"). Run
directly on the file (what the report's live-demo page does) and the same clip
reads aeration ON, p=1.00. For presenting the models, use the report app's live
demo (file inference), not the loopback listener. Blockage and which-machine are
robust over loopback; only aeration's margin suffers.

8. Aeration: evidence and explanation — a direct visual A/B of real aeration OFF vs
   ON files (PSD + Δ + low-band spectrograms, refreshable) with the physical
   explanation of the indicator: a +4.5–5.5 dB 50–100 Hz boost, the dominant tone
   collapsing to the low band (~800→85 Hz on cam), and ~+8 dB of 2–40 Hz envelope
   (bubbling) modulation.

9. Blockage: evidence and explanation — like the aeration page but for the valves:
   how the spectrum evolves as each valve closes (sweeps + trend curves), and a
   suction-at-x vs discharge-at-x A/B (PSD + Δ + spectrograms). Discharge grows the
   250–500 Hz mid-band monotonically; suction collapses the tone to ~80–120 Hz
   (plus 4–8 kHz cavitation hiss on mics) — distinct fingerprints, hence two
   ordinal axes.

Build the report inputs: `python3 run/signatures.py` (then launch).

## Czech version (`run_cz/`)

`run_cz/report_app.py` is a full Czech localization of the report. It **imports the data loaders, plotting helpers and models from `run/`** (no duplication of artifacts) and translates all text. Launch: `./run_cz/run_report_cz.sh`; deploy with Main file path `run_cz/report_app.py`. Same graceful no-audio behaviour as the English app.
