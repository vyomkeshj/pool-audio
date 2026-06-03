# Pool-Audio — Pump Operating-Condition Detection from Acoustic Recordings

Determine the operating condition of a pool pump system from audio (microphone
and camera-mic recordings). This file tracks the project, the dataset, and the
work completed so it can be continued across sessions.

> Corrected framing (updated 2026-06-02 by the testbed owner).
> Nothing in this dataset is broken. Every clip is healthy equipment in a
> different operating / valve configuration; there is no faulty or damaged pump.
> M1 and M2 are large pumps; M3 and M4 are exhaust fans; aeration is an air
> injector. (The Czech protocol spreadsheet loosely calls M2/M3/M4 "smaller
> pumps"; that shorthand is incorrect — follow the owner's description.) An
> earlier session incorrectly assumed a damaged-fan pump (M3) and built a fault
> detector around it. That work is archived in `deprecated_no_fault/` and should
> not be cited. See the 2026-06-02 work-log entry.

## Goal

The microphone can answer two orthogonal questions about a ~60 s clip. Keep them
separate; do not collapse them into a single "state" label.

Axis A — M1 flow-restriction monitoring (the primary condition-monitoring target).
M1 always runs; two throttling valves restrict it. The questions are how
restricted the suction side is (`valveIn` 1–4) and the discharge side
(`valveOut` 1,2,3,4,5,8,11). `1` = fully open. This is the only fault-like axis,
and it represents a valve restriction, not pump damage. It is exercised in every
campaign and under every noise type, which makes it the robust, generalizable
core of the work.

Axis B — auxiliary equipment presence (acoustic source detection, not a fault).
Which healthy machines are also running: second pump M2, exhaust fan M3, exhaust
fan M4, aeration. These are varied only in the `5_25` campaign (all clean).

Both axes must hold up under overlaid environmental background noise (playground,
traffic, lawnmower, speech, music).

## The testbed and parameter space

The rig is a large pump M1 (always ON) — the pump under observation — plus a
second large pump M2, two exhaust fans M3/M4, an aeration injector, and two
valves throttling M1:

| Field | Meaning | Values seen on disk |
|---|---|---|
| `M1` | large pump, the one being monitored | always `1` |
| `M2` | second large pump (on/off, healthy) | `0/1` — varied only in `5_25` |
| `M3` | exhaust fan (on/off, healthy) — adds ~4–8 kHz airflow noise | `0/1` — only in `5_25` |
| `M4` | exhaust fan (on/off, healthy) — near-silent | `0/1` — only in `5_25` |
| `aeration` | air injector | `0` (off), `1` (on; only in `5_25`, anomalous — see notes) |
| `valveIn` | M1 suction throttle, "1 = open" | `1..4` (more restricted = higher) |
| `valveOut` | M1 discharge throttle / "Blockage Level" | `1,2,3,4,5,8,11` (higher = more restricted) |
| `noise` | overlaid background-noise type | `N` + `A1/A2/B1/B2/C1/C2/D1/D2/E1/E2` |

`valveIn`/`valveOut` = 1 means open (no restriction); higher means more
restricted. This restriction is the monitored operating condition, not a broken
pump. The two suffix digits on the noise code (e.g. `A1`/`A2`) denote two volume
levels / source clips of the same noise category.

### Noise legend (from the protocol spreadsheet)

| Code | Source |
|---|---|
| `N` | clean / no added noise (reference) |
| `A` | children's playground (*dětské hřiště*) |
| `B` | lawnmower (*sekačka*, petrol/electric) |
| `C` | road traffic (*doprava*) |
| `D` | human speech (*lidská řeč*) |
| `E` | music / song (*písnička*) |

Noise was mixed in to test robustness; it is not an operating state. The clean
`N` recordings are the references for learning the acoustic signatures; the noisy
variants test and augment robustness.

## Dataset on disk

All files are mono 16-bit PCM WAV, 44.1 kHz, ~60 s each. Each recording session
captures 16 devices simultaneously: `cam1..cam8` and `mic1..mic8`.

| Folder | Date | Files | Notes |
|---|---|---:|---|
| `testbed_motor_audio` | 05-07 | 1,760 | M1 only; valveOut 1–4; full noise set |
| `Testbedmotor` | 05-07 | 1,056 | M1 only; valveOut 5/8/11 (heavy restriction); full noise set |
| `Testbedmotor5_14` | 05-14 | 3,936 | M1 only; valveIn 1–3 + valveOut sweep; full noise set |
| `Testbedmotor5_15` | 05-15 | 3,184 | M1 only; valveIn 3–4 + valveOut sweep; full noise set |
| `testbedmotor5_25wav` | 05-25 | 2,048 | M2/M3/M4 + aeration sweep; all noise = `N` (clean) |
| Total | | 11,984 | |

The protocol spreadsheet also references `testbedmotor5_19wav` (122 rows) and
`testbedmotor5_20wav` (111 rows), which were not present on disk at the original
index build (recorded in the table but missing); tracked in
`_meta.protocol_folders_missing_on_disk`. Both folders later arrived — see the
2026-06-03 work-log entry, which extends the suction axis to `valveIn ∈ {5,8}`.

### Filename schema

```
YYYYMMDD_HHMMSS_M1_M2_M3_M4_<device>_<deviceDesc>_M1_aeration_A_valveIn_I_valveOut_O_noise_T_REP.wav
```
e.g. `20260525_111606_1_0_0_0_cam1_Cam_1_5_HoverCam_Solo_Spark_M1_aeration_0_valveIn_1_valveOut_1_noise_N_1.wav`

`<deviceDesc>` is free text (camera/mic model + serial) and varies in length;
parse using the structured anchors, not by splitting on `_`.

## Source of truth

- `TEF_mic_test_protocol.xlsx` — the master protocol. Sheet `List1` is the
  authoritative plan: one row per (config × noise), with column `saving_folder`
  (col 14) linking each row to its recording folder. Columns: Index, Position,
  M1-desc, M2, M3, M4, aeration, M1-input (valveIn), M1-output (valveOut),
  type/N (noise), …, saving_folder. Sheet `List1 (2)` is an earlier draft and
  should be ignored. The spreadsheet is in Czech.
- `measurement_audio_index.json` — generated index linking every WAV to its
  parameters, derived labels, and protocol row(s).
- `build_index.py` — regenerates the JSON from the folders and spreadsheet.
  Re-run after adding or removing recordings: `python3 build_index.py`.

## `measurement_audio_index.json` schema

```jsonc
{
  "_meta": { audio_dirs, n_signatures, n_files_total, audio_format,
             filename_fields, protocol_folders_missing_on_disk,
             noise_legend, equipment_legend, state_legend_DEPRECATED },
  "folder_summary": { "<folder>": { n_files, n_signatures } },
  "measurements": {
    "M2x_M3x_M4x_aerX_vinI_voutO_noiseT": {          // signature key
      "params": { M1,M2,M3,M4,aeration,valveIn,valveOut,noise },
      "state":  { nominal_state, labels[], aux_pumps_on[],
                  blockage_in_level, blockage_out_level, aeration_on },
      "protocol_excel_rows": { "<folder>": [rowNums] },
      "n_files", "n_sessions",
      "folders": ["<folder>", ...],                  // merged across campaigns
      "sessions": [ { folder, timestamp, repetition,
                      n_devices, devices: { "cam1": "...wav", ... } } ]
    }
  }
}
```

Signatures are merged across folders and dates (same physical config = same key,
giving more samples per condition); each `session` is tagged with its source
`folder` for provenance. The original build covered 366 signatures across all
11,984 files; the 2026-06-03 rebuild extends this to 423 signatures / 13,850
files across 7 folders.

### Coverage by axis (current data)

The `nominal_state` field in the index is deprecated framing: it collapses the
two orthogonal axes into one priority-ordered label and uses fault-flavoured
names such as "suction_blockage" for what is only a throttled valve. Prefer the
raw params. Actual coverage:

Axis A — M1 flow restriction (636 M1-only sessions = M2/M3/M4/aeration all off):

| | open | restricted | levels |
|---|---:|---:|---|
| suction (`valveIn`) | 175 | 461 | 1,2,3,4 |
| discharge (`valveOut`) | 96 | 540 | 1,2,3,4,5,8,11 |

Suction and discharge are swept jointly in 5_14/5_15, so most clips are
restricted on both sides. Treat them as two independent ordinal axes, not
mutually exclusive classes. Every restriction level appears under every noise
category (clean and A–E).

Axis B — auxiliary equipment (only in `5_25`, clean): M2/M3/M4 each toggled
across the full valveOut sweep (64 on-sessions each); aeration on in 16 sessions
at a single config (`vin1/vout1`).

## Data-quality notes

- 8 files in `testbed_motor_audio` carry `noise_A12` — a typo (should be `A1` or
  `A2`); this is the only signature with no protocol-row match (365/366).
- `5_25` is the only campaign that exercises M2/M3/M4 and aeration; all other
  campaigns are M1-only valve/noise sweeps.
- Aeration recordings are anomalous. All 16 aeration-ON clips sit at ~−64 dB RMS
  vs ~−30 dB for OFF — a uniform ~34 dB drop across every M2/M3/M4 combination —
  and they exist at only one valve config (`vin1/vout1`). This is consistent with
  the pumps being idle (or a different gain being used) during the aeration test
  rather than air injected into a running pump. Do not train an aeration detector
  on this; its only reliable signal is the level collapse, which is likely a
  recording artifact. Re-record if aeration matters. (See the 2026-06-03 entry
  for a matched-pair re-analysis that recovers a real on/off signature.)
- M3 vs M4: both are exhaust fans, but M3 is acoustically loud (+4.5 dB in
  4–8 kHz when on) while M4 is near-silent (+0.3 dB overall), so M4 presence is
  the hardest auxiliary machine to detect.

## Work log

- 2026-06-03 — Streamlit research report (`run/report_app.py`, launch
  `./run/run_report.sh`) plus aeration loopback-vs-file diagnosis. A
  presentation-ready 6-page report: overview / two-axis summary, dataset and
  joint-valve coverage, acoustic signatures (interactive mean-spectrum plots per
  sensor family showing suction vs discharge restriction sweeps, aeration on/off
  Δ-spectrum, M2/M3/M4 on/off Δ — backed by `run/signatures.py` →
  `signatures.npz`), models and training writeup, results tables and figures, and
  a live demo that runs the models directly on dataset files and shows prediction
  vs ground truth plus spectrogram. Verified: all pages render under AppTest
  (0 exceptions), the demo runs end-to-end, and the server returns HTTP 200.
  Added a "Mic & sensor differences" page (`run/channels.py` → `channels.json` +
  `channels_psd.npz`): per-channel mean spectra and per-channel config-grouped
  restriction-reading quality. Mics span discharge ρ 0.37 (mic7) to 0.85 (mic3);
  camera-mics are uniformly 0.81–0.93 (cam5 best). This makes the "channel
  placement dominates, so model per family" point concrete. Added an "Aeration:
  evidence and explanation" page (after the live demo): a real aeration OFF-vs-ON
  A/B comparison (PSD + Δ + low-band spectrograms, refreshable) with the physical
  explanation. Aeration indicators from matched-pair analysis: a +4.5–5.5 dB boost
  in 50–100 Hz, the dominant spectral peak collapsing to the low band (cam
  ~800→85 Hz, mic ~250→150 Hz), crest dropping ~5 dB (less tonal), and on cameras
  ~+8 dB of 2–40 Hz envelope (bubbling) modulation. These are shape/modulation
  changes (gain-invariant), consistent with forcing air into the water. Added a
  parallel "Blockage: evidence and explanation" page: per-axis spectrum evolution
  (sweeps plus trend curves for mid-band / tone / HF vs level) and a suction-at-x
  vs discharge-at-x A/B comparison (PSD + Δ + spectrograms). Discharge raises the
  250–500 Hz mid-band monotonically (cam −9.9→−5.9 dB) and shifts the tone — the
  strong, exactly readable severity signal; suction collapses the dominant tone to
  ~80–120 Hz (cam 953→81 Hz at vin5) and on mics raises 4–8 kHz cavitation hiss
  (−19.6→−6.6 dB), which is subtler and the harder axis. Distinct fingerprints at
  the same level confirm modelling them as two ordinal axes. The report is now
  9 pages, all rendering under AppTest (0 exceptions).
  Aeration "appears stronger when on" diagnosis: this is the loopback path, not
  the model. The 48→44.1 kHz loopback resampling weakens the subtle aeration
  signal, so an aeration-ON clip can sit at p≈0.50 (borderline off); because
  aeration was not flagged, the blockage reading was not marked uncertain and a
  spurious level was shown. Run directly on the file, the same cam8 clip reads
  aeration ON p=1.00 with blockage correctly flagged uncertain. For demos, use the
  report app's live demo (file inference), not the loopback listener. Blockage and
  which-machine are robust over loopback; only aeration's margin suffers. The
  signature plots confirm the physics: discharge raises 250–500 Hz and shifts the
  tone up (953→1217 Hz, L1→L11), suction is subtler (low-band), aeration is +7 dB
  at 50–100 Hz, M3 is +4–6 dB at 3–8 kHz.

- 2026-06-03 — Desktop app: player + independent listener + live spectrogram
  (`run/app_desktop.py`, launch `./run/run_gui.sh`). Tkinter, two independent
  halves. The Player picks a dataset clip by condition (discharge/suction level,
  aeration, machines, sensor, noise) and plays it via `aplay`, showing ground
  truth. The Listener captures system-audio loopback (`parec` on the sink monitor,
  which hears the exact played PCM) or the mic (`arecord`) and, blind to the
  player, reports which machines run, aeration on/off, and discharge/suction
  restriction level, with a live matplotlib spectrogram. Engine and CLI in
  `run/listener.py`; clip picker in `run/clip_library.py`. No GUI/audio installs
  needed (tkinter + ALSA/PipeWire tools already present; no sounddevice/pyaudio).

  Dual analysis window (required by a measured finding): aeration and
  which-machine are averaged over every 8 s window in the rolling buffer (their
  models are 8 s); blockage uses the whole buffer up to 60 s, because retraining
  blockage on 8 s windows degraded it (campaign-out clip within-1 0.60–0.81 vs
  1.00 at 60 s — see `test_window_blockage.py`, `train_blockage_windows.py`).
  Blockage is only asserted in the M1-only regime: if any M2/M3/M4 or aeration is
  detected, the acoustic baseline differs and blockage is flagged "uncertain ·
  other equipment running" (the severity models were trained M1-only; a running
  fan or aeration otherwise pushes a false level).

  Tested against the data before the UI (`test_listener.py`, broad stratified
  population; held out by session so the 8 sibling mics never leak — see the
  channel-leakage note below). After hard-negative training (`train_aux_hardneg.py`,
  which adds M1-only blockage windows as aeration-off / all-aux-off negatives) the
  aeration false-alarm rate fell from 19% to 0% on mics, and window aggregation
  restored machine recall:

  | output (listener) | mic | cam |
  |---|---|---|
  | discharge level | within-1 0.97 / exact 0.77 | within-1 1.00 / exact 0.98 |
  | suction level | within-1 1.00 / exact 0.88 | within-1 1.00 / exact 0.99 |
  | aeration on/off | recall 1.00, FA 0.000 | recall 1.00, FA 0.000 |
  | M2 / M3 / M4 running | recall 1.00/1.00/0.91, FA 0 | 1.00/1.00/1.00, FA 0 |

  End-to-end loopback verified (speaker → system-audio → listener): aeration
  clip → ON 0.98; vout=8 clip → discharge L8; M3 clip → M3 ON 0.99 with blockage
  flagged uncertain; file-stream vin3/vout5 → L3/L5 exact. Channel leakage: every
  recording fires 16 devices at one timestamp; all CV groups by config (blockage)
  or session (live), so sibling mics never split — a naive split inflates aeration
  window-acc from 0.977 to 0.993. Single-mic vs 8-channel-fused clip metrics are
  reported separately in `run/README.md`. Use loopback (digital) for accurate
  listening; the mic-from-speaker path is re-recorded and unvalidated. Blockage
  holds M1-only; 5_25 (aux/aeration) is clean-only. `run/README.md` has the full
  app section.

  UI fixes (post-feedback): (1) readonly `ttk.Combobox` text was invisible
  (dark-on-dark) — switched to the `clam` theme with explicit light fg (#e8e8ee)
  on field #3a3d4a and a styled popup list. (2) "Listener not showing spectrogram"
  — root cause: this machine's microphone reads pure digital silence (−180 dB,
  OS-muted) via both `arecord` and `parec`, so the listener was fed silence. Fixed
  by defaulting to the loopback source (relabelled "hears what's playing"),
  unifying capture on PipeWire `parec` (mic = default source, loopback =
  sink `.monitor`), and adding an explicit silence indicator plus a spectrogram
  "no audio" overlay (predictions suppressed when input RMS < −70 dB) so silence
  no longer looks like a broken panel. Note: loopback resamples 48→44.1 kHz, so the
  conservative aeration model confirms ON over ~15–20 s of buffer (climbing to
  ~0.9 by 60 s) — correct, with no false positives. (3) Dropped the manual mic/cam
  selector from the listener (it broke the "independent" premise): the sensor
  family is now auto-detected from the audio (`run/train_sensor_id.py` →
  `models/sensor_id.joblib`; 98% per-window / 100% per-clip, gain-invariant —
  pump tone ~270 Hz mic vs ~720 Hz cam). The listener loads both families and
  picks per-analysis, showing "sensor: cam (auto-detected)". Verified both
  directions on files and end-to-end via loopback. CLI `--sensor` defaults to
  `auto` (mic/cam still forceable). (4) "Player and listener never match" — buffer
  contamination. The 60 s rolling buffer blended consecutive clips (a prior mic
  clip plus the current cam clip), flipping sensor detection to "mic" and
  false-firing M4, so blockage went "uncertain". This was the cause, not the
  models (on a clean per-clip buffer both file and loopback read correctly). Fixed:
  `stream_capture` auto-flushes on a silence gap (fine 0.5 s sub-reads so short
  gaps are not masked by the hop), the GUI Find&Play sets a `flush_event` so each
  new clip starts fresh, and the machine-ON threshold was raised 0.5→0.6. Verified
  back-to-back M3-clip → M1-only-clip no longer blends (clip 2: machines off,
  discharge exact).

- 2026-06-03 — Live aeration-on/off and which-machine detectors (and an aeration
  correction), all in `run/`. Built a streaming monitor `run/live.py` that listens
  in a sliding 8 s window from the real mic (`--mic`, via `arecord` — no
  sounddevice/pyaudio on this machine) or by streaming a wav (`--file`), and
  reports aeration on/off plus which of M2/M3/M4 is running (M1 is always on, not a
  target). Trained on 8 s windows of the 5_25 factorial (`run/window_features.py` →
  `features_5_25_windows.csv`, 33,947 windows); `run/aeration.py` and
  `run/which_pump.py` save `models/aeration_live_{mic,cam}.joblib` and
  `models/pump_{M2,M3,M4}_{mic,cam}.joblib`; `run/test_live.py` is the
  session-held-out streaming test.

  Held out by recording session, so the 8 sibling mics (and 8 cams) of one
  recording never split across train/test (they capture the same instant and would
  leak; a naive window-random split inflates aeration window-acc from 0.977 to
  0.993). The honest single-mic clip metric uses one sensor over its ~14 windows;
  8ch-fused averages all 8 same-type channels (a deployment upgrade).

  | detector | sensor | window acc | 1-mic clip | 8ch-fused |
  |---|---|---|---|---|
  | aeration on/off (matched vin1/vout1, gain-invariant) | mic / cam | 0.98 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 |
  | M2 (2nd pump) | mic / cam | 0.82 / 0.93 | 0.87 / 0.96 | 0.98 / 1.00 |
  | M3 (loud fan) | mic / cam | 0.94 / 0.96 | 0.96 / 0.97 | 1.00 / 1.00 |
  | M4 (near-silent fan) | mic / cam | 0.68 / 0.85 | 0.72 / 0.88 | 0.82 / 0.99 |

  This corrects the earlier "aeration not answerable / ~34 dB level artifact"
  claim. The −34 dB was an averaging artifact (ON clips only at `vin1/vout1` vs the
  whole OFF set). At the matched config (toggle aeration, valve fixed) the level
  gap is small (mic −3 dB, cam +6.5 dB) but there is a strong gain-invariant
  spectral signature (mic +4.5 dB at 50–100 Hz, d=1.75; cam tone −640 Hz, harm3
  d=2.9). A decisive control — matched config with `rms_db` removed — still
  separates ON/OFF at F1 0.97 (mic) / 0.997 (cam), so this is real aeration
  acoustics, not gain/config leakage. Aeration on/off is answerable and near
  perfect at clip level; aeration level is still not answerable (only on/off was
  recorded, at one valve config). Which-machine also improved over 2026-06-02
  (M2 0.78→0.91 cam, M4 readable on cam). Caveat: aeration was only recorded at
  `vin1/vout1`, so behaviour under M1 throttling is unverified; 5_25 is clean-only,
  so there is no A–E noise-robustness evidence for these. Verified live end-to-end:
  streamed ON/OFF/M2/M3 clips give correct stable verdicts, and `--mic` captures
  the laptop mic and runs. Launch: `python3 run/live.py --mic` or `--file CLIP.wav`.

- 2026-06-03 — The two missing campaigns arrived, with a full re-run and an
  improved blockage-level reader (all in `run/`, self-contained; does not touch
  the root artifacts). `testbedmotor5_19wav` and `testbedmotor5_20wav` (the folders
  the protocol referenced but were absent on disk) are now present — M1-only valve
  sweeps that extend the suction axis to new levels `valveIn ∈ {5,8}` (was 1–4),
  i.e. the weak axis. The index now covers 7 folders, 13,850 files, 423 signatures
  (`run/build_index.py` handles the nested `5_19/5_20` paths and salvages 16
  `noise→oise` typo filenames).

  Re-extracted enhanced gain-invariant features for every channel of every file
  (`run/features.py` → `features_allch.csv`, 30 features = the proven 20 + finer
  low-freq bands + pump-tone harmonic ratios + band ratios), then trained
  per-device-type HistGradientBoosting severity readers with three honesty levels
  of CV (`run/train.py` → `run/models/*.joblib`, `run/results.json`, figures):

  | axis | config-grouped within-1/exact | leave-campaign-out within-1/exact/MAE | clean→noisy |
  |---|---|---|---|
  | discharge (`valveOut`) | 1.00 / 1.00 | 1.00 / 1.00 / 0.00 steps | within-1 = 1.00 on A–E |
  | suction (`valveIn`) | 1.00 / 1.00 | 1.00 / 0.65 / 0.35 steps | within-1 = 1.00 on A–E |

  Discharge level reads within one step every time (within-1 = 1.00 on held-out
  days and all noise); exact level is near-perfect on cameras (deploy cameras for
  discharge) and strong on mics. Suction also reads within one step every time
  (within-1 = 1.00, exact ≈0.65, MAE 0.35 steps) — a substantial improvement over
  the previous honest suction MAE ≈1.14. The exact misses occur only on the new
  levels `vin=5/8`, each recorded on a single day, so leave-campaign-out becomes
  extrapolation for them (it snaps to the nearest level); recording them on a
  second day would close the gap. Verified not leakage: leave-one-level-out
  predicts the nearest adjacent rank monotonically (`fig_leave_level_out.png`) — a
  smooth ordinal signal, not memorized recordings. Aeration unchanged: level still
  not answerable (single config, ~34 dB collapse); `run/predict.py` emits an
  on/off flag with a caveat. Deliverable CLI:
  `python3 run/predict.py CLIP.wav [--sensor mic|cam]` → discharge level + suction
  level + aeration flag. See `run/README.md`. (Root `app.py`/`reanalyze.py`/
  `train_models.py` and the old root artifacts are left intact; `run/` is the new,
  larger-data pipeline.)

- 2026-06-02 — Rebuilt the Streamlit review dashboard (`app.py`) around the
  corrected analysis, replacing the previous fault/5-state version. 8 pages:
  Overview, Dataset explorer, Equipment signatures, Blockage monitoring, Sensors &
  channels, Model suite, Class physics, Live detector. The live detector loads the
  saved `models/<task>_<mic|cam>.joblib`, extracts features from an uploaded or
  selected clip, and reports blockage present/severity plus which equipment is
  running. Verified: all 8 pages render under `streamlit.testing` AppTest, the live
  detector scores a clip end-to-end, and `run_app.sh` serves HTTP 200.
  (`use_container_width`→`width="stretch"` for the current Streamlit.) Launch:
  `./run_app.sh`.

- 2026-06-02 — Trained the full model suite and identified that the channels are
  two distinct sensors (`extract_allch.py` → `features_allch.csv` = all 16 channels
  of all 11,984 files; `train_models.py` → `models/*.joblib`, `model_results.json`,
  `fig_model_*` / `fig_channel_ranking.png`).

  Primary finding: mic ≠ cam, and channel placement dominates. The 16 channels are
  two different sensor families that should not be pooled. Mics sit at ~−61 dB with
  the true pump tone ~190 Hz; cameras sit at ~−32 dB (AGC) with a strong structural
  tone ~780 Hz that tracks pump load, so cameras read discharge restriction better
  than mics. Per-channel discharge-severity quality ranges from ρ=0.24 (mic6,
  nearly blind) to ρ=0.93 (cam5, within-1 0.98) — sensor position matters more than
  any other factor. (This is why pooling all channels first hurt, and why the
  earlier mic1 numbers were strong: mic1 is the best mic.) All models are now
  trained per device type. Channel fusion (mean features per session) rescues mics
  (ρ 0.64→0.82, averaging out bad mics) but does not beat the best single camera.

  Models trained (grouped CV by physical config; saved per device type):
  | task | model | mic | cam |
  |---|---|---|---|
  | discharge restriction present? | `discharge_present_*` | F1 0.90 | F1 0.90 |
  | suction restriction present? | `suction_present_*` | F1 0.85 | F1 0.87 |
  | discharge severity (level) | `discharge_severity_*` | within-1 0.80 (honest) | 0.94 |
  | suction severity (level) | `suction_severity_*` | within-1 0.71 | 0.74 |
  | restriction location (4-class) | `restriction_location_*` | acc 0.64 / bal-acc 0.31 | 0.67 / 0.34 |
  | which motor running (none/M2/M3/M4) | `which_motor_*` | acc 0.53 | 0.55 |
  | motor presence (M2/M3/M4 multi-label) | `motor_presence_*` | M3 F1 0.86, M2 0.65, M4 0.62 | M3 0.84, M2 0.68, M4 0.69 |

  Answers to the goal's questions:
  - Detect/identify each class — yes: restriction presence (F1 0.85–0.90) and
    severity (discharge within-1 0.94 on a camera, honest level holdout) are
    strong. The 4-class restriction location is weak (bal-acc 0.31) because the
    data is mostly "both sides restricted" — use the two independent severity axes,
    not the 4-class label.
  - Restriction levels — discharge level is read within ±1 step ~94% of the time
    (camera); suction is harder (4 levels, within-1 ~0.71–0.74 honest).
  - Which motor from the sound — partial: M1 cannot be isolated (always on); among
    M2/M3/M4 the loud exhaust fan M3 is clearly identifiable (F1 0.85), the second
    pump M2 and the near-silent fan M4 are moderate (F1 ~0.65), and picking the
    single running motor is ~0.54 acc (2× chance). More data or better features
    could lift M2/M4.
  - Aeration level — not answerable from this data. Aeration was recorded at a
    single valve config and the ON clips are uniformly ~34 dB quieter (likely pumps
    idle / different gain), so there is no validated detector; re-record with pumps
    running. (Superseded for on/off by the 2026-06-03 matched-pair result.)

  Notes on why suction is the harder axis: suction has a clean monotonic signature
  when discharge is held fixed (at vout=1, mic1: low-band 0–100 Hz rises
  −17.9→−12.1 dB, level drops, tone drops 293→124 Hz), but the campaigns sweep
  suction and discharge jointly and their effects partly overlap, so reading vin
  across all vout is confounded. Dropping the absolute-level features (`rms_db`,
  `mod_energy`) changes nothing (ρ ±0.01) — the band features are already relative
  to total power, so the models never relied on gain. Conclusion: suction is near
  the information ceiling of clip-level spectral features (within-1 ≈0.88 is good;
  exact-rank ρ≈0.5 is the limit); improving it requires a deep model or better
  sensing, not rescaling. Pooling all 16 channels is acceptable for severity
  regression (ρ 0.77) but breaks exact-level classification — a further reason to
  model per sensor. (Tested and ruled out: envelope-demodulation harmonic features.
  The 500 Hz–4 kHz Hilbert envelope of M1 is dominated by slow ~6 Hz flow
  fluctuation with no distinct rotation/vane-pass line that tracks suction level,
  so hand-crafted demod features do not help; do not re-try this avenue.)

- 2026-06-02 — Major reframing: nothing is broken (testbed owner correction).
  M1/M2 = large pumps, M3/M4 = exhaust fans, aeration = air injector; every clip is
  healthy equipment in a different valve/operating config. This invalidates the
  entire "M3 = broken fan" thread from 2026-06-01 (it measured a healthy exhaust
  fan's normal ~6 kHz airflow and 21.3 Hz shaft tone and mislabelled them as an
  imbalance fault). That thread is archived in `deprecated_no_fault/` (with a
  README); the detector there is effectively an "M3 fan running" detector. Also
  retired the single-label 5-class "state" as conflated (it fused two orthogonal
  axes).

  Redid the analysis around the two real axes (`reanalyze.py` → `reanalysis.json`
  + 4 figures), reusing the cached `features_all.csv` (no re-extraction):

  (1) Equipment acoustic signatures (5_25 matched on/off pairs,
  `fig_equipment_signature.png`):
  | machine | effect of turning it on |
  |---|---|
  | M2 (large pump) | mid/low tonal, pump tone shifts +175 Hz, overall −4 dB |
  | M3 (exhaust fan) | +4.5 dB in 4–8 kHz broadband airflow, centroid +555 Hz (the broadband increment earlier mistaken for a fault) |
  | M4 (exhaust fan) | ≈ nothing (+0.3 dB, flat) — near-silent |
  | aeration | anomalous −34 dB level collapse (see data-quality note) — flagged, not trusted |

  (2) M1 flow-restriction monitoring (636 M1-only sessions, all noise;
  config-grouped CV):
  - Detect suction restriction (`valveIn`>1): acc 0.91, F1 0.94.
  - Detect discharge restriction (`valveOut`>1): acc 0.79, F1 0.88.
  - Clean-trained, holds under noise: suction 0.89–0.95, discharge 0.83–0.90 on A–E.
  - Severity (ordinal): discharge optimistic ρ=0.88; leakage-honest (hold out whole
    valve positions {2,4,8}) MAE 0.65 steps, within-1 0.98 — it interpolates to
    unseen restriction levels. Suction is harder honest (MAE 1.14, only 4 levels).

  (3) Auxiliary equipment presence (5_25 clean, multi-label over M2/M3/M4,
  config-grouped): M3 fan F1 0.90, M2 pump F1 0.71, M4 fan F1 0.68 (near-silent, so
  hardest); subset-acc 0.46. Aeration excluded (single config, cannot CV — see
  notes).

  Summary (corrected): the mic reliably reads M1's valve restriction — whether the
  suction or discharge side is throttled and roughly how much, robustly under
  environmental noise — and can identify which healthy auxiliary machine is running
  (except the near-silent M4 fan and the anomalous aeration). No fault detection is
  claimed, because no equipment is faulty. (`app.py` was subsequently rebuilt to
  match; see the later 2026-06-02 entry.)

- 2026-06-01 — Reverse-engineered the protocol and filename schema. Wrote
  `build_index.py`; expanded `measurement_audio_index.json` from 1 folder /
  64 signatures (5_25 only) to all 5 folders / 366 signatures / 11,984 files; added
  derived `state` labels, noise/state legends, and protocol-row cross-references.
  Figures `fig1..4_*.png` exist (motor/valve/off-state sweeps, summary); author and
  contents to confirm.

- 2026-06-01 — First acoustic analysis (`analyze.py`). Extracted 20 diagnostic
  features (band energies, spectral shape, tonal peak, amplitude-modulation /
  burstiness) from 1 channel of each of the 190 clean (noise=N) sessions. Result:
  the 5 states are acoustically separable. A session-level 5-fold-CV RandomForest
  reaches 0.89 accuracy (macro-F1 0.77). Outputs: `features_clean.csv`,
  `analysis_summary.json`, `fig_state_psd.png`, `fig_feature_box.png`,
  `fig_state_scatter.png` (LDA), `fig_confusion.png`.

  Acoustic signatures (note: the "state" framing below is superseded by 2026-06-02):
  | state | distinguishing feature |
  |---|---|
  | normal | mid loudness (~−54 dB), tone ~205 Hz; sits on the suction/discharge boundary (its only weak class) |
  | suction_blockage | quietest (~−62 dB), lowest centroid/tone (~140 Hz); starved, low-frequency |
  | discharge_blockage | loud, broadband, tone rises with restriction (~380 Hz); dominant class (n=108) |
  | aerating | low RMS but strongest 4–8 kHz + highest rolloff (3.3 kHz); broadband bubble/cavitation hiss |
  | multi_pump | loudest (~−29 dB), highest tonal peak (~770 Hz); extra pumps add tones |

  Top features: amplitude-modulation energy, RMS, high bands (8–22 kHz),
  250–500 Hz, ZCR. The main confusion is `normal` vs low-level restriction
  (expected; only n=10 normal).

- 2026-06-01 — Extended to (a) noise robustness, (b) severity regression,
  (c) multi-label (`extract_features.py` → `features_all.csv`, then
  `analyze_abc.py` → `results_abc.json` + 3 figures). Features cached for all 750
  sessions (1 channel each, clean + noisy). Structural fact: the noisy campaigns
  are M1-only valve sweeps, so only normal/suction/discharge have noisy recordings;
  aerating and multi_pump exist clean-only.

  (a) Robustness (`fig_robustness.png`, 3-class normal/suction/discharge): state
  recognition survives overlaid environmental noise.
  | test condition | clean-trained acc | noise-augmented acc |
  |---|---:|---:|
  | clean | 0.925 | 0.944 |
  | playground (A) | 0.885 | 0.920 |
  | lawnmower (B) | 0.902 | 0.946 |
  | traffic (C) | 0.955 | 0.964 |
  | speech (D) | 0.892 | 0.937 |
  | music (E) | 0.920 | 0.955 |

  A model that never saw noise still reaches 0.89–0.95 on noisy clips (chance =
  0.33); adding noisy data to training recovers ~3–5 points. Note: accuracy is
  robust but macro-F1 is dragged down by the small `normal` class (n=10; F1 as low
  as 0.57 on speech) — `normal` vs low-level restriction is the weak boundary.

  (b) Severity regression (`fig_severity.png`, M1-only, RandomForest, all noise
  levels, 5-fold CV; valve levels treated as ordinal ranks):
  | target | n | levels | MAE (steps) | R² | Spearman ρ | within-1 |
  |---|---:|---|---:|---:|---:|---:|
  | discharge (valveOut) | 175 | 1,2,3,4,5,8,11 | 0.35 | 0.93 | 0.96 | 0.98 |
  | suction (valveIn) | 96 | 1,2,3,4 | 0.32 | 0.72 | 0.84 | 0.94 |

  Restriction severity is acoustically readable, especially discharge-side
  (predicted level within 1 step 98% of the time). Caveat: only 7/4 discrete valve
  positions exist, so CV is stratified-by-level (configs recur across noise/dates);
  this measures "read severity on this rig under noise", not generalization to an
  unseen rig.

  (c) Multi-label (`fig_multilabel.png`, clean only, 4 independent binary
  conditions, `MultiOutputClassifier(RF)`, 5-fold CV): exact-match (subset)
  acc 0.84, Hamming loss 0.05.
  | condition | n | precision | recall | F1 |
  |---|---:|---:|---:|---:|
  | suction_blockage | 42 | 0.90 | 0.83 | 0.86 |
  | discharge_blockage | 144 | 0.92 | 0.94 | 0.93 |
  | aerating | 16 | 0.92 | 0.75 | 0.83 |
  | multi_pump | 112 | 0.97 | 1.00 | 0.98 |

  Co-occurring conditions are detected independently and well (a clip that is both
  suction- and discharge-restricted is handled). Done clean-only on purpose:
  aerating/multi_pump are only ever recorded clean, so mixing noise in would let
  the model exploit "no background noise" as a proxy. Robustness of the
  suction/discharge labels under noise is shown separately in (a).

  Summary at this stage: every question posed was answerable from a single ~60 s
  mic clip — which state (≈0.89–0.92), how severe the restriction (ρ≈0.96
  discharge), which simultaneous conditions (subset-acc 0.84) — and it held up
  under real-world background noise.

  > Partly superseded — see 2026-06-02. The (a)/(b) restriction and severity
  > results survive (re-derived honestly in `reanalyze.py`), but the (c)
  > multi-label numbers used the conflated "aerating/multi_pump" labels and an
  > aeration column that is anomalous and single-config; the discharge severity
  > ρ=0.96 was optimistic (config-grouped honest holdout gives MAE 0.65 /
  > within-1 0.98). The 5-state and multi_pump framing is retired.

- 2026-06-01 — Pump-localization analysis (`fault_id.py`, `fault_confirm.py`).
  > Superseded — wrong premise; see 2026-06-02. No pump is broken; M3 is a healthy
  > exhaust fan. What this measured (M3's +4.5 dB ~6 kHz airflow increment) is real
  > but is normal fan noise, not a fault. Archived in `deprecated_no_fault/`. Kept
  > below only as a record of the error.

  Question (as posed at the time): one pump has a broken fan — which? Only M2/M3/M4
  can be isolated (M1 is always on). The 5_25 campaign is a full factorial (all 8
  on/off combos, 32 matched on/off pairs per pump, balanced over valveOut/aeration),
  so matched-pair differencing isolates each pump's acoustic increment.

  Conclusion at the time: M3 stood out. When M3 switches on it injects a broadband
  increment centred ~6 kHz (+6.5 dB peak in the Δ-PSD), raising broadband 2–16 kHz
  by +2.2 dB and overall level by +0.6 dB. It is consistent (94% of 448 matched
  pairs positive, Cohen's d = 0.73) and shows on both device types (mic +4.3 dB,
  cam +2.2 dB). M2 adds only narrow tonal peaks (~130 & ~400 Hz); M4 adds
  essentially nothing. Figures: `fig_pump_diff_psd.png`, `fig_pump_haystack.png`.
  (This increment is now understood as M3's normal exhaust-fan airflow, not damage.)

  Methodology note (kept as a caution): a composite "fault score" that treated high
  kurtosis/impulsiveness as the indicator pointed at M2 and was wrong. Broadband
  noise is near-Gaussian, so it lowers kurtosis; M2's high HF-kurtosis merely
  reflects that it adds little HF broadband, and M3's broadband increment was
  penalized. For broadband-source detection, use band / Δ-PSD evidence, not
  kurtosis. (`fault_id.py`'s composite score is superseded by `fault_confirm.py`'s
  band metric.) M1 (always on) cannot be toggled, so this method says nothing about
  M1; it only compares the switchable machines M2/M3/M4.

- 2026-06-01 — M3 characterization + detector (`characterize_m3.py`,
  `detect_m3.py`).
  > Superseded — wrong premise; see 2026-06-02. The "21.3 Hz imbalance line from a
  > broken blade" is the healthy exhaust fan's shaft/blade rate, and the ~6 kHz
  > increment is its normal airflow noise. `detect_m3.py`/`m3_detector.joblib` is an
  > "M3 fan running" detector, not a fault detector. Archived in
  > `deprecated_no_fault/`. Kept below only as a record of the error.

  (1) Characterization (`fig_m3_envelope.png`). Demodulated the 3–9 kHz increment
  (Hilbert envelope, M3 on−off, isolation M2=M4=aer=0). The envelope spectrum shows
  a sharp 1× line at 21.3 Hz (~1280 rpm) plus a 2× harmonic. (At the time read as
  imbalance from a broken blade; now understood as the fan's normal shaft/blade
  rate.) Broadband increment centres ~5.9 kHz. 21.3 Hz is not a clean 50 Hz
  sub-multiple, so the fan runs at its own (geared/independent) speed, not
  line-locked. → `m3_fault_signature.json`.

  (2) Detector — `detect_m3.py` (self-contained, with a CLI).
  `python3 detect_m3.py clip.wav` → P(signature present) + flag. Features are
  gain-invariant: 3–9 kHz spectral-shape ratios plus modulation at 21.3 Hz
  (1×/2×/3×). Trained on the 5_25 factorial (M3 on/off), grouped-CV by recording.

  Robustness note: v1 trained on clean only reached AUC 0.955 but false-alarmed
  ~40% on noisy M1-only clips (environmental broadband noise trips the shape
  features). Fix: hard-negative training — add 5_14/5_15 noisy M1-only clips
  (broadband, but no 21.3 Hz modulation) as negatives, forcing the model onto the
  21.3 Hz line. v2 result:

  | metric | value |
  |---|---|
  | grouped-CV ROC-AUC | 0.958 |
  | recall (M3 on) @ 5%-FA threshold | 0.81 |
  | FP on M2/M4-on (M3 off) | 0.06 — M3-specific, not "a pump is on" |
  | cross-campaign held-out FP (5_07, noise A–E) | 0.06–0.09 |
  | cross-campaign FP, clean 5_07 (heavy restriction) | 0.15 |
  | overall held-out FP | 0.088 |

  Held-out test = the two 5_07 campaigns the model never saw (all M3 off). Top
  feature = `mod_1x` (the 21.3 Hz line). CLI check: M3-on clip → 0.95 (1× line
  +13 dB); M2-on clip → 0.05; noisy M1-only → 0.06. → `m3_detector.joblib`,
  `detect_m3_result.json`, `fig_detector.png`. Limits: the detector keys on M3's
  specific 21.3 Hz line and ~6 kHz band, so it does not transfer to other machines
  without retraining; clean 5_07 heavy-restriction clips are the main false-alarm
  source (broadband discharge noise resembles the band). Raising the threshold
  trades recall for fewer false positives.

- 2026-06-01 — Streamlit dashboard (`app.py`) presenting the (then-current)
  findings, in conda env `pool-audio` (`environment.yml`; scikit-learn pinned to
  1.8.0 so `m3_detector.joblib` loads). 10 pages: Overview, States & signatures,
  Classifier, (a) Robustness, (b) Severity, (c) Multi-label, Pump-localization (M3),
  M3 signature, Live detector (upload/preset a WAV → probability gauge + spectrum +
  21.3 Hz envelope), Dataset explorer (filter + play clips). Reuses the existing
  figures/JSON artifacts and `detect_m3.detect_features`. Verified: all 10 pages
  render under `streamlit.testing` AppTest in the pool-audio env; model loads and
  scores (0.948). (This version was replaced on 2026-06-02 by the corrected
  two-axis dashboard.)

  Environment note (this machine): do not use `conda run -n pool-audio …` or a bare
  `streamlit`. The shell has the ESP-IDF python env on PATH
  (`IDF_PYTHON_ENV_PATH=…/idf6.0_py3.12_env`) and a stale streamlit in `~/.local`,
  both of which hijack the interpreter (`conda run`'s `python`/`pip` silently
  resolve to the ESP-IDF venv). Call the env's python directly, with user-site
  ignored:
  `PYTHONNOUSERSITE=1 ~/miniconda3/envs/pool-audio/bin/python -m streamlit run app.py`
  — which is what `run_app.sh` does. The first-run email prompt is silenced via
  `~/.streamlit/credentials.toml`.

## Repository files

| File | Role |
|---|---|
| `TEF_mic_test_protocol.xlsx` | master protocol (source of truth, Czech) |
| `build_index.py` → `measurement_audio_index.json` | folder + spreadsheet → WAV index |
| `analyze.py` → `features_clean.csv`, `analysis_summary.json`, `fig_state_*`/`fig_feature_box`/`fig_confusion.png` | clean-data characterization + baseline classifier; defines the 20-feature `extract()` reused elsewhere |
| `extract_features.py` → `features_all.csv` | parallel feature cache for all 750 sessions (clean + noisy) |
| `analyze_abc.py` → `results_abc.json`, `fig_robustness/severity/multilabel.png` | (a) robustness, (b) severity, (c) multi-label — uses old "state" labels; superseded by `reanalyze.py` |
| `reanalyze.py` → `reanalysis.json`, `fig_equipment_signature.png`, `fig_m1_restriction.png`, `fig_m1_severity.png`, `fig_equipment_presence.png` | Corrected analysis (2026-06-02). Two axes: (1) equipment acoustic signatures, (2) M1 flow-restriction detection + leakage-honest severity, (3) equipment-presence multi-label. Reuses `features_all.csv` (1 mic ch/session). |
| `extract_allch.py` → `features_allch.csv` | feature cache for all 16 channels of all 11,984 files (mic1-8 + cam1-8), with `device`/`dev_type`/`session` for grouped CV and the device study |
| `train_models.py` → `models/*.joblib`, `model_results.json`, `fig_model_*.png`, `fig_channel_ranking.png` | device-aware model suite: restriction present/severity, restriction location, motor presence, which-motor; per device type; ranks channels; fusion check |
| `models/<task>_<mic\|cam>.joblib` | trained bundles `{scaler, model, features, classes, task, device_type}` — load and call `model.predict(scaler.transform(X[features]))` |
| `class_differences.py` → `class_drivers.json`, `fig_class_differences.png` | which features drive each task + spectral trend across levels (discharge ← 250–500 Hz + tone shift; suction ← 0–100 Hz + level; M3 ← 4–8 kHz; M2 ← 250–500 Hz; M4 ≈ flat) |
| `app.py`, `environment.yml`, `run_app.sh` | Streamlit review dashboard (conda env `pool-audio`), rebuilt 2026-06-02 around the corrected two-axis story. 8 pages: Overview, Dataset explorer, Equipment signatures, Blockage monitoring, Sensors & channels, Model suite, Class physics, Live detector (upload/pick a WAV → pick sensor type → runs the saved `models/*.joblib`). Launch with `./run_app.sh`. |
| `run/` | Larger-data pipeline (2026-06-03): 7-folder index, all-channel features, HistGradientBoosting severity readers, live monitor, desktop app, research report. Self-contained; see `run/README.md`. |
| `deprecated_no_fault/` | Archived wrong-premise pump-fault thread (`fault_id.py`, `fault_confirm.py`, `characterize_m3.py`, `detect_m3.py` + outputs). See its README. Do not cite. |
| `fig1..4_*.png` | pre-existing motor/valve/off-state sweep figures (provenance unconfirmed) |

Feature set (20) in `analyze.extract()`: 9 octave-band log-energies (0→22 kHz),
spectral centroid/bandwidth/rolloff (85%)/flatness/crest, ZCR, tonal-peak
frequency and prominence (<2 kHz pump tone), and envelope amplitude-modulation
energy and peak (5–150 Hz, burstiness/cavitation). All on one channel per session
(mic1 preferred) to avoid device leakage. Rebuild the corrected analysis:
`python3 build_index.py && python3 extract_features.py && python3 reanalyze.py`.
Rebuild the model suite (all-channel cache → trained models):
`python3 extract_allch.py && python3 train_models.py`. (`analyze.py`/`analyze_abc.py`
still run but produce the deprecated 5-state framing.)

## Next steps (proposed)

1. Improve suction and M2/M4 identification — the weak spots. Suction leans on
   absolute level (`rms_db`, 0–100 Hz), so it is channel-fragile; the M4 fan is
   near-silent. Try demodulation / harmonic features (pump vane-pass, fan
   blade-rate) and gain-invariant normalization. M2 (pump) should be findable via
   its 250–500 Hz tonal lines.
2. Re-record aeration with pumps running — the current aeration data is unsuitable
   for level estimation (single config, ~34 dB level artifact). Needed before any
   aeration-level model.
3. Deep model — log-mel CNN / pretrained embedding (PANNs/YAMNet) per device type,
   compared against the 20-feature baseline, especially for severity and
   which-motor.
4. Productionize sensor choice — cameras read discharge restriction best (cam5
   within-1 0.98); decide whether to deploy on the best-placed sensor or fuse.
   (A CLI mirroring the dashboard's live-detector inference would round this out;
   the Streamlit app already loads `models/<task>_<dtype>.joblib` and scores a
   clip. `run/predict.py` provides this for the larger-data models.)
5. Truly leakage-honest robustness — hold out whole valve configs (train clean on
   some, test noisy on others) to separate noise robustness from rig transfer.
6. Early detection / onset — per-second envelope to flag when a restriction
   develops within a clip, not just clip-level labels.
