# Deprecated — built on a false premise (no pump is broken)

These files were produced on 2026-06-01 under the incorrect assumption that one
auxiliary pump had a broken/damaged fan and that the task was to localize the
fault. On 2026-06-02 the testbed owner corrected this:

- Nothing in the dataset is broken. Every recording is healthy equipment in a
  different operating / valve configuration.
- M1 and M2 are large pumps. M3 and M4 are exhaust fans (not "auxiliary pumps").

The central claim of this thread — "M3 = broken-fan pump, identified by a ~6 kHz
broadband increment plus a 21.3 Hz imbalance line" — is therefore wrong in
interpretation. What those analyses actually measured is real but mundane: turning
the healthy exhaust fan M3 on adds its normal broadband airflow noise (~4–8 kHz,
+4.5 dB) and a shaft-rate tone at ~21.3 Hz. That is what an exhaust fan sounds
like, not a defect. `detect_m3.py`/`m3_detector.joblib` is therefore an "M3 exhaust
fan is running" detector, not a fault detector.

The matched-pair / Δ-PSD methodology here is still sound; only the "fault" framing
is invalid. The corrected, reframed analysis lives in `../reanalyze.py` (equipment
acoustic signatures + M1 flow-restriction monitoring + equipment presence). Kept
here for provenance only — do not cite as fault detection.

Files: fault_id.py, fault_confirm.py, characterize_m3.py, detect_m3.py and their
outputs (m3_detector.joblib, *_result.json, fault_features_525.csv,
detect_features.csv, fig_pump_*.png, fig_m3_*.png, fig_detector.png).
