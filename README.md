# Pool-Pump Acoustic Diagnostics

Read the operating condition of a pool-pump testbed from a ~60 s microphone or
camera-mic clip. The system reads the main pump's valve restriction (suction and
discharge severity) and identifies which auxiliary equipment is running, robustly
under environmental background noise.

Nothing in the dataset is faulty — every recording is healthy equipment in a
different valve / operating configuration. The microphone answers two orthogonal
questions:

- **Axis A — M1 flow restriction.** How throttled the suction side (`valveIn`) and
  discharge side (`valveOut`) of the always-on main pump are. `1` = open.
- **Axis B — auxiliary equipment presence.** Which healthy machines are also
  running: second pump M2, exhaust fans M3/M4, and aeration.

## Key results (leakage-honest cross-validation)

- Discharge restriction level: read within one step every time (within-1 = 1.00 on
  held-out recording days and all noise types); exact level near-perfect on
  camera-mics.
- Suction restriction level: within-1 = 1.00, exact ≈ 0.65, MAE ≈ 0.35 ordinal
  steps.
- Aeration on/off: 0.99 single-mic / 1.00 camera at clip level.
- Which machine: M3 fan and M2 pump are read reliably; the near-silent M4 fan needs
  a camera-mic or channel fusion.
- The 16 channels are two distinct sensor families (mics vs camera-mics) and are
  modeled separately; sensor placement dominates quality.

## Streamlit apps

- `app.py` — review dashboard (dataset explorer, equipment signatures, blockage
  monitoring, sensor/channel study, model suite, live detector).
- `run/report_app.py` — presentation-ready research report with interactive
  signature plots and a live demo.

```bash
# create the environment (conda)
conda env create -f environment.yml

# launch the dashboard
./run_app.sh
# or directly:
PYTHONNOUSERSITE=1 ~/miniconda3/envs/pool-audio/bin/python -m streamlit run app.py
```

The live-detector pages require the trained `models/*.joblib` bundles, which are
not committed here (they are regenerated from the raw dataset via the analysis
scripts). See `CLAUDE.md` for the full pipeline, work log, and rebuild commands.

## Repository layout

- `build_index.py`, `extract_allch.py`, `analyze.py`, `reanalyze.py`,
  `train_models.py`, `class_differences.py` — root analysis pipeline.
- `run/` — larger-data pipeline (7 campaigns), live monitor, desktop app, and
  research report. See `run/README.md`.
- `deprecated_no_fault/` — an early analysis built on an incorrect "broken pump"
  premise, kept for provenance only. Do not cite.
- `CLAUDE.md` — detailed project documentation and work log.

## Data

The raw audio (~67 GB of WAV across 7 recording campaigns) and the proprietary
protocol spreadsheet are not included in this repository. The derived
`measurement_audio_index.json` maps every recording to its parameters.
