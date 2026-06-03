#!/usr/bin/env python3
"""Pool-pump acoustic diagnostics — review dashboard (CORRECTED, 2026-06-02).

Reviews the corrected analysis: nothing is broken; M1/M2 are large pumps, M3/M4
exhaust fans; the microphone reads M1's valve RESTRICTION (suction/discharge +
severity) and which auxiliary equipment is running. Presents the dataset, the
acoustic signatures, the trained device-aware model suite, the sensor-channel
discovery, the per-class physics, and a live detector that runs the saved models.

Launch:  ./run_app.sh     (uses the pool-audio conda env; see that script)
"""
import os, json, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from analyze import extract  # 20-feature extractor (1 channel)

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT = ["rms_db", "centroid", "bandwidth", "rolloff", "flatness", "crest", "zcr",
        "tone_freq", "tone_prom", "mod_energy", "mod_peak", "band_0_100",
        "band_100_250", "band_250_500", "band_500_1000", "band_1000_2000",
        "band_2000_4000", "band_4000_8000", "band_8000_16000", "band_16000_22050"]
NOISE = {"N": "clean (reference)", "A": "playground", "B": "lawnmower",
         "C": "traffic", "D": "speech", "E": "music"}

st.set_page_config(page_title="Pool-pump acoustic diagnostics", layout="wide",
                   page_icon="🌀")


# ----------------------------------------------------------------- loaders
@st.cache_data
def load_json(name):
    p = os.path.join(ROOT, name)
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_index():
    return load_json("measurement_audio_index.json")


@st.cache_resource
def load_models():
    md = os.path.join(ROOT, "models")
    out = {}
    if os.path.isdir(md):
        for f in os.listdir(md):
            if f.endswith(".joblib"):
                out[f[:-7]] = joblib.load(os.path.join(md, f))
    return out


def fig(name, caption=None):
    p = os.path.join(ROOT, name)
    if os.path.exists(p):
        st.image(p, caption=caption, width="stretch")
    else:
        st.info(f"(figure {name} not found — run the analysis scripts)")


idx = load_index()
reanal = load_json("reanalysis.json")
modres = load_json("model_results.json")
drivers = load_json("class_drivers.json")
MODELS = load_models()


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🌀 Pool-pump diagnostics")
st.sidebar.caption("Corrected analysis — healthy equipment, operating-condition models")
PAGE = st.sidebar.radio("Page", [
    "Overview",
    "Dataset explorer",
    "Equipment signatures",
    "Blockage monitoring",
    "Sensors & channels",
    "Model suite",
    "Class physics",
    "🔴 Live detector",
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**No fault here.** M1/M2 = large pumps · M3/M4 = exhaust fans · "
    "aeration = air injector. The earlier *broken-fan* analysis was wrong and is "
    "archived in `deprecated_no_fault/`.")


# ----------------------------------------------------------------- Overview
if PAGE == "Overview":
    st.title("Pool-pump acoustic diagnostics")
    st.markdown(
        "Reading the **operating condition** of a pool-pump testbed from a ~60 s "
        "microphone clip. Every recording is *healthy* equipment in a different "
        "valve / operating configuration — **nothing is broken**.")
    meta = idx.get("_meta", {})
    n_files = meta.get("n_files_total", 0)
    c = st.columns(4)
    c[0].metric("WAV files", f"{n_files:,}" if n_files else "—")
    c[1].metric("Configurations", meta.get("n_signatures", "—"))
    c[2].metric("Channels / session", "16 (8 mic + 8 cam)")
    c[3].metric("Trained models", len(MODELS))

    st.subheader("Two orthogonal things the mic can hear")
    a, b = st.columns(2)
    a.markdown(
        "#### 🅰 M1 flow restriction *(the monitoring target)*\n"
        "Two valves throttle the main pump M1:\n"
        "- **suction** (`valveIn` 1–4)\n- **discharge** (`valveOut` 1,2,3,4,5,8,11)\n\n"
        "`1` = open. This is a *valve restriction*, **not** pump damage.")
    b.markdown(
        "#### 🅱 Which equipment is running *(source presence)*\n"
        "- **M2** — 2nd large pump\n- **M3** — exhaust fan (loud, 4–8 kHz)\n"
        "- **M4** — exhaust fan (near-silent)\n- **aeration** — air injector "
        "*(data anomalous — see Equipment signatures)*")

    st.subheader("Headline results")
    g = st.columns(4)
    g[0].metric("Discharge blockage detect", "F1 0.90")
    g[1].metric("Discharge severity (cam)", "within-1 0.94", help="leakage-honest level holdout")
    g[2].metric("Suction blockage detect", "F1 0.85–0.87")
    g[3].metric("Exhaust fan M3 ID", "F1 0.85")
    st.info("**Biggest lesson:** the 16 channels are *two different sensors*. "
            "Mics hear the pump tone ~190 Hz; cameras (AGC) hear a ~780 Hz structural "
            "tone that tracks pump load — so **cameras read discharge blockage best**. "
            "Channel placement matters more than anything (see *Sensors & channels*).")


# ----------------------------------------------------------- Dataset explorer
elif PAGE == "Dataset explorer":
    st.title("Dataset explorer")
    st.caption("Filter the recording configurations and play individual clips.")
    ms = idx.get("measurements", {})
    rows = []
    for sig, e in ms.items():
        p = e["params"]
        for s in e["sessions"]:
            rows.append({"signature": sig, "folder": s["folder"],
                         "timestamp": s["timestamp"], **p,
                         "n_devices": s["n_devices"], "_devices": s["devices"]})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No index found.")
    else:
        c = st.columns(6)
        fM2 = c[0].selectbox("M2 pump", ["any", 0, 1])
        fM3 = c[1].selectbox("M3 fan", ["any", 0, 1])
        fM4 = c[2].selectbox("M4 fan", ["any", 0, 1])
        fAer = c[3].selectbox("aeration", ["any", 0, 1])
        fVin = c[4].selectbox("valveIn (suction)", ["any"] + sorted(df.valveIn.unique().tolist()))
        fVout = c[5].selectbox("valveOut (discharge)", ["any"] + sorted(df.valveOut.unique().tolist()))
        fNoise = st.multiselect("noise", sorted(df.noise.unique().tolist()))
        q = df.copy()
        for col, val in [("M2", fM2), ("M3", fM3), ("M4", fM4), ("aeration", fAer),
                         ("valveIn", fVin), ("valveOut", fVout)]:
            if val != "any":
                q = q[q[col] == val]
        if fNoise:
            q = q[q.noise.isin(fNoise)]
        st.write(f"**{len(q)} sessions** match.")
        st.dataframe(q[["folder", "timestamp", "M2", "M3", "M4", "aeration",
                        "valveIn", "valveOut", "noise", "n_devices"]],
                     width="stretch", height=300)
        if len(q):
            st.subheader("Play a clip")
            i = st.number_input("row #", 0, len(q) - 1, 0)
            sel = q.iloc[int(i)]
            dev = st.selectbox("device", sorted(sel["_devices"].keys()))
            path = os.path.join(ROOT, sel["folder"], sel["_devices"][dev])
            if os.path.exists(path):
                st.audio(open(path, "rb").read(), format="audio/wav")
                st.caption(f"`{sel['_devices'][dev]}`")


# ------------------------------------------------------- Equipment signatures
elif PAGE == "Equipment signatures":
    st.title("Acoustic signature of each healthy machine")
    st.caption("Matched on/off pairs in the 5_25 factorial — what turning each "
               "machine on does to the sound. No fault implied.")
    fig("fig_equipment_signature.png")
    sig = reanal.get("equipment_signatures", {})
    if sig:
        rows = []
        for m, v in sig.items():
            rows.append({"machine": m, "n_pairs": v.get("n_pairs"),
                         "Δ rms (dB)": v.get("d_rms_db"),
                         "Δ centroid (Hz)": v.get("d_centroid_hz"),
                         "loudest band Δ": (max(v["band_delta_db"], key=v["band_delta_db"].get)
                                            if "band_delta_db" in v else None)})
        st.dataframe(pd.DataFrame(rows), width="stretch")
    st.markdown(
        "- **M2** (large pump): mid/low tonal, pump tone shifts up.\n"
        "- **M3** (exhaust fan): **+4.5 dB in 4–8 kHz** airflow — *this broadband "
        "‘haystack’ was earlier mistaken for a broken-fan fault.*\n"
        "- **M4** (exhaust fan): ≈ nothing (+0.3 dB) — near-silent.\n")
    st.warning("**Aeration is anomalous.** Every aeration-ON clip is uniformly "
               "~34 dB quieter and exists at only one valve config — likely the "
               "pumps were idle (or different gain). Not a usable acoustic signature; "
               "re-record with pumps running.")


# --------------------------------------------------------- Blockage monitoring
elif PAGE == "Blockage monitoring":
    st.title("M1 flow-restriction monitoring")
    st.caption("Detect & quantify suction / discharge throttling from one clip, "
               "robust to environmental noise. 636 M1-only sessions.")
    det = reanal.get("m1_restriction", {}).get("detection", {})
    if det:
        c = st.columns(2)
        for col, name in zip(c, ["suction", "discharge"]):
            d = det.get(name, {})
            col.metric(f"{name} restriction — F1", d.get("f1", "—"),
                       help=f"acc {d.get('accuracy')} · P {d.get('precision')} · R {d.get('recall')}")
    st.subheader("Holds under environmental noise (clean-trained)")
    fig("fig_m1_restriction.png")
    st.subheader("Severity — how restricted?")
    fig("fig_m1_severity.png")
    sev = reanal.get("m1_restriction", {}).get("severity", {})
    if sev:
        rows = []
        for k, v in sev.items():
            o, h = v.get("optimistic_grouped_cv", {}), v.get("leakage_honest_holdout", {})
            rows.append({"axis": k, "levels": str(v.get("levels")),
                         "ρ (grouped CV)": o.get("spearman"),
                         "within-1 (optimistic)": o.get("within_1"),
                         "within-1 (held-out levels)": h.get("within_1"),
                         "held-out": str(h.get("held_out_levels"))})
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("Leakage-honest = whole valve positions held out of training → "
                   "tests interpolation to *unseen* blockage levels. Discharge reads "
                   "within ±1 level 94–98 % of the time on a good sensor; suction is harder.")


# ------------------------------------------------------- Sensors & channels
elif PAGE == "Sensors & channels":
    st.title("Sensors & channels — the big discovery")
    st.markdown(
        "The 16 channels are **two different sensor families** and must not be pooled:\n"
        "- **Mics** ≈ −61 dB, capture the true pump tone **~190 Hz**.\n"
        "- **Cameras** ≈ −32 dB (AGC), dominated by a structural tone **~780 Hz** that "
        "**tracks pump load** → cameras read discharge blockage *better*.")
    fig("fig_channel_ranking.png",
        "Per-channel blockage readability — placement dominates (ρ 0.24 to 0.93)")
    ci = modres.get("channel_informativeness", {})
    if ci:
        for label, rows in ci.items():
            st.subheader(label)
            st.dataframe(pd.DataFrame(rows), width="stretch", height=240)
    fus = modres.get("channel_fusion_discharge_severity", {})
    if fus:
        st.subheader("Channel fusion (mean features across a device type)")
        st.write(fus)
        st.caption("Fusion rescues the mics (averages out bad ones) but doesn't beat "
                   "the single best-placed camera.")


# --------------------------------------------------------------- Model suite
elif PAGE == "Model suite":
    st.title("Trained model suite")
    st.caption("Every task trained separately for mic and cam (different sensors). "
               "Cross-validation grouped by physical config so the same setup never "
               "sits in train and test.")
    if modres.get("blockage"):
        st.subheader("Blockage severity (ordinal)")
        rows = []
        for label, dd in modres["blockage"].items():
            for dt, v in dd.items():
                o, h = v["optimistic_grouped_cv"], v["leakage_honest_holdout"]
                rows.append({"task": label, "sensor": dt, "n": v["n"],
                             "ρ": o["spearman"], "within-1": o["within_1"],
                             "within-1 (honest)": h["within_1"]})
        st.dataframe(pd.DataFrame(rows), width="stretch")
    rd = modres.get("restriction_detect", {})
    if rd:
        st.subheader("Blockage present? (binary)")
        rows = [{"task": t, "sensor": dt, **{k: v[k] for k in ("f1", "precision", "recall", "n_pos")}}
                for t, dd in rd.items() for dt, v in dd.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch")
    mp = modres.get("motor_presence", {})
    if mp:
        st.subheader("Which motor is running? (multi-label M2/M3/M4)")
        rows = []
        for dt, v in mp.items():
            for m, mm in v["per_machine"].items():
                rows.append({"sensor": dt, "machine": m, **mm})
        st.dataframe(pd.DataFrame(rows), width="stretch")
        fig("fig_model_which_motor.png")
    if modres.get("which_motor"):
        st.subheader("Single-motor identification (none / M2 / M3 / M4)")
        st.write({dt: v.get("per_class_f1") for dt, v in modres["which_motor"].items()})
    if modres.get("aeration"):
        st.subheader("Aeration")
        st.warning(modres["aeration"].get("CAVEAT", ""))
    st.subheader("Saved models")
    st.write(sorted(MODELS.keys()))


# ------------------------------------------------------------- Class physics
elif PAGE == "Class physics":
    st.title("What acoustically distinguishes each class")
    st.caption("Which features drive each task, and how the spectrum shifts with level.")
    fig("fig_class_differences.png")
    if drivers:
        for k, v in drivers.items():
            st.subheader(k)
            if isinstance(v, dict):
                st.write(v)
            else:
                st.dataframe(pd.DataFrame(v, columns=["feature", "importance"]),
                             width="stretch")
    st.info("Discharge ← 250–500 Hz energy + tone shift · suction ← 0–100 Hz + level "
            "(so it's channel-fragile) · M3 fan ← 4–8 kHz · M2 pump ← 250–500 Hz · "
            "M4 ≈ flat (near-silent).")


# ------------------------------------------------------------- Live detector
elif PAGE == "🔴 Live detector":
    st.title("Live detector")
    st.caption("Upload a ~60 s WAV (or pick a dataset clip), choose the sensor type, "
               "and run the trained models.")
    if not MODELS:
        st.error("No models found in models/ — run `python3 train_models.py`.")
        st.stop()

    dtype = st.radio("Sensor type", ["mic", "cam"], horizontal=True,
                     help="Mics and cameras are different sensors — pick which recorded the clip.")
    src = st.radio("Audio source", ["Upload a WAV", "Pick a dataset clip"], horizontal=True)

    wav_path = None
    if src == "Upload a WAV":
        up = st.file_uploader("WAV file", type=["wav"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(up.read()); tmp.flush()
            wav_path = tmp.name
            st.audio(open(wav_path, "rb").read(), format="audio/wav")
    else:
        ms = idx.get("measurements", {})
        opts = []
        for sig, e in ms.items():
            for s in e["sessions"][:1]:
                dev = next((d for d in s["devices"] if d.startswith(dtype)), None)
                if dev:
                    p = e["params"]
                    opts.append((f"{p['noise']} | vin{p['valveIn']} vout{p['valveOut']} "
                                 f"M2{p['M2']}M3{p['M3']}M4{p['M4']} aer{p['aeration']} | "
                                 f"{dev} {s['folder']}", s["folder"], s["devices"][dev]))
        opts = opts[:400]
        if opts:
            pick = st.selectbox("clip", range(len(opts)), format_func=lambda i: opts[i][0])
            wav_path = os.path.join(ROOT, opts[pick][1], opts[pick][2])
            if os.path.exists(wav_path):
                st.audio(open(wav_path, "rb").read(), format="audio/wav")

    if wav_path and st.button("Analyse", type="primary"):
        with st.spinner("Extracting features and running models..."):
            feat, f, p = extract(wav_path)
            row = pd.DataFrame([feat])[FEAT]

        def run(task):
            b = MODELS.get(f"{task}_{dtype}")
            if b is None:
                return None
            X = b["scaler"].transform(row[b["features"]].to_numpy(float))
            return b, X

        st.subheader("🅰 M1 flow restriction")
        c = st.columns(2)
        for col, axis, sev_task, det_task in [
                (c[0], "Suction", "suction_severity", "suction_present"),
                (c[1], "Discharge", "discharge_severity", "discharge_present")]:
            present_txt, level_txt = "—", ""
            r = run(det_task)
            if r:
                b, X = r
                if hasattr(b["model"], "predict_proba"):
                    pr = b["model"].predict_proba(X)[0]
                    classes = list(b["model"].classes_)
                    pp = pr[classes.index(1)] if 1 in classes else float(b["model"].predict(X)[0])
                else:
                    pp = float(b["model"].predict(X)[0])
                present_txt = f"{pp*100:.0f}% restricted"
            rs = run(sev_task)
            if rs:
                bs, Xs = rs
                rank = float(bs["model"].predict(Xs)[0])
                levels = bs["classes"]
                lvl = levels[int(np.clip(round(rank), 0, len(levels) - 1))]
                level_txt = f"≈ level {lvl}  (of {levels[-1]})"
            col.metric(f"{axis} restriction", present_txt, level_txt)

        st.subheader("🅱 Equipment running")
        r = run("motor_presence")
        if r:
            b, X = r
            preds = b["model"].predict(X)[0]
            machines = b["classes"]
            names = {"M2": "M2 pump", "M3": "M3 fan", "M4": "M4 fan"}
            cols = st.columns(len(machines))
            for col, m, pv in zip(cols, machines, preds):
                col.metric(names.get(m, m), "ON" if int(pv) else "off")
        rw = run("which_motor")
        if rw:
            b, X = rw
            st.caption(f"Single-motor guess: **{b['model'].predict(X)[0]}**")

        st.subheader("Clip spectrum")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figp, ax = plt.subplots(figsize=(9, 3))
        ax.semilogx(f, 10 * np.log10(p + 1e-20))
        ax.set_xlim(20, 22050); ax.set_xlabel("Hz"); ax.set_ylabel("PSD (dB)")
        ax.grid(alpha=0.3, which="both")
        st.pyplot(figp)
        st.caption(f"rms {feat['rms_db']:.1f} dB · centroid {feat['centroid']:.0f} Hz · "
                   f"tone {feat['tone_freq']:.0f} Hz — note mics sit ~−61 dB, cams ~−32 dB.")
