#!/usr/bin/env python3
"""Pool-Audio — research report (Streamlit).

Tells the story of reading a pool-pump's operating condition from a ~60 s mic clip:
the two orthogonal axes (M1 flow restriction; which auxiliary machine / aeration),
how the acoustic signature changes with each, the models and how they were trained
and validated, and a live demo running the models directly on dataset clips.

Launch:  ./run/run_report.sh   (or streamlit run run/report_app.py)
"""
import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as ssignal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

st.set_page_config(page_title="Pool-Audio — research report", layout="wide")
plt.rcParams.update({"axes.facecolor": "#15161c", "figure.facecolor": "#0e0f14",
                     "axes.edgecolor": "#444", "text.color": "#e8e8ee",
                     "axes.labelcolor": "#cfcfe0", "xtick.color": "#9aa0b5",
                     "ytick.color": "#9aa0b5", "axes.titlecolor": "#e8e8ee",
                     "legend.framealpha": 0.2, "font.size": 9})


# --------------------------------------------------------------- data loaders
@st.cache_data
def load_json(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_sigs():
    d = np.load(os.path.join(HERE, "signatures.npz"))
    return {k: d[k] for k in d.files}


@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(HERE, "features_allch.csv"))


@st.cache_data
def load_index():
    return json.load(open(os.path.join(HERE, "measurement_index.json")))


@st.cache_data
def load_channels():
    p = os.path.join(HERE, "channels.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data
def load_channels_psd():
    p = os.path.join(HERE, "channels_psd.npz")
    if not os.path.exists(p):
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.files}


SR = 44100
NOISE = {"N": "clean (reference)", "A": "playground", "B": "lawnmower",
         "C": "traffic", "D": "speech", "E": "music"}


def db(p):
    return 10 * np.log10(p / p.sum() + 1e-12)


def fig_to_st(fig):
    st.pyplot(fig, clear_figure=True)


# ------------------------------------------------------------------- PAGES
def page_overview():
    st.title("🔊 Reading a pool pump's condition from sound")
    st.markdown("""
A single ~60-second microphone (or camera-mic) recording of a pumping station can
report **what the equipment is doing** — without touching it. Everything in this
dataset is **healthy equipment in different operating configurations**; there is no
broken part. The microphone answers two *orthogonal* questions:
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Axis A — M1 flow restriction")
        st.markdown("""
The main pump **M1** always runs; two throttle valves restrict it.
- **Suction** side (`valveIn`, 1–8) — how throttled is the inlet?
- **Discharge** side (`valveOut`, 1–11) — how throttled is the outlet?

`1 = fully open`, higher = more restricted. This is the core condition-monitoring
target, exercised under every background-noise type.""")
    with c2:
        st.subheader("Axis B — which equipment is running")
        st.markdown("""
Which healthy machines are on alongside M1:
- **M2** — second large pump
- **M3** — exhaust fan (loud, ~4–8 kHz airflow)
- **M4** — exhaust fan (near-silent)
- **aeration** — air injector (on/off)""")

    st.divider()
    st.subheader("Headline results (leakage-honest)")
    cols = st.columns(4)
    cols[0].metric("Discharge level", "within-1 = 1.00", "exact ≈0.98 (cam)")
    cols[1].metric("Suction level", "within-1 = 1.00", "MAE ≈0.35 steps")
    cols[2].metric("Aeration on/off", "AUC 1.00 (cam)", "F1 0.97–1.00")
    cols[3].metric("Which machine", "M3 ≈1.0, M2 ≈0.9", "M4 hard on mic")
    st.caption("13,850 channel-files · 7 recording campaigns · models per sensor "
               "family · validated by config-, campaign- and level-held-out CV. "
               "Use the sidebar to walk through the signatures, models and a live demo.")


def page_dataset():
    st.title("📁 The dataset & parameter space")
    idx = load_index()
    fs = idx["folder_summary"]
    st.markdown(f"**{idx['_meta']['n_files_total']:,} recordings** across "
                f"**{len(fs)} campaigns**; every session captures 16 devices "
                "simultaneously (8 mics + 8 camera-mics).")
    st.dataframe(pd.DataFrame([{"campaign": k, "files": v["n_files"]}
                               for k, v in fs.items()]), hide_index=True,
                 width="stretch")

    st.subheader("Joint valve coverage (M1-only sessions)")
    st.caption("Suction and discharge are swept **jointly** — most clips are "
               "restricted on both sides — so we model two independent ordinal axes, "
               "not one combined label. Cells = number of recording sessions.")
    df = load_features()
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]
    piv = (m1.drop_duplicates("session").groupby(["valveIn", "valveOut"]).size()
           .unstack(fill_value=0))
    st.dataframe(piv, width="stretch")

    st.subheader("Background-noise robustness set")
    st.markdown("Each restriction level was also recorded under five environmental "
                "noises so the models can be tested for robustness:")
    st.markdown("  ·  ".join(f"**{k}** = {v}" for k, v in NOISE.items()))
    st.info("⚠️ Aeration & the auxiliary machines (M2/M3/M4) were only recorded in "
            "one clean campaign (5_25); aeration only at one valve config "
            "(`vin1/vout1`). So aeration/which-machine are validated clean, and "
            "aeration *level* is not recoverable — only on/off.")


def _psd_sweep(sigs, dt, kind, levels, title, label):
    f = sigs["freq"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    for c, lv in zip(cmap, levels):
        key = f"{dt}_{kind}_{lv}"
        if key in sigs:
            ax.semilogx(f, db(sigs[key]), color=c, lw=1.6, label=f"{label} {lv}")
    ax.set_xlim(30, sigs["freq"].max()); ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB, rel.)"); ax.set_title(title)
    ax.legend(ncol=2, fontsize=7, title="level (1=open)"); ax.grid(alpha=0.2)
    fig.tight_layout(); return fig


def _psd_pair(sigs, dt, a, b, la, lb, title):
    f = sigs["freq"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    pa, pb = sigs[f"{dt}_{a}"], sigs[f"{dt}_{b}"]
    ax1.semilogx(f, db(pa), color="#7fd1ff", lw=1.8, label=la)
    ax1.semilogx(f, db(pb), color="#ff9d6b", lw=1.8, label=lb)
    ax1.set_title(title); ax1.set_xlabel("Hz"); ax1.set_ylabel("power (dB, rel.)")
    ax1.set_xlim(30, f.max()); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
    delta = db(pb) - db(pa)
    ax2.semilogx(f, delta, color="#46d17a", lw=1.6)
    ax2.axhline(0, color="#777", lw=0.7)
    ax2.set_title(f"Δ: ({lb}) − ({la})"); ax2.set_xlabel("Hz")
    ax2.set_ylabel("Δ power (dB)"); ax2.set_xlim(30, f.max()); ax2.grid(alpha=0.2)
    fig.tight_layout(); return fig


def page_signatures():
    st.title("〰 How the acoustic signature changes")
    sigs = load_sigs()
    dt = st.radio("Sensor family", ["cam", "mic"], horizontal=True,
                  help="Mics and camera-mics are acoustically different sensors and "
                       "are modelled separately.")
    st.caption("Mean Welch spectra over clean (noise=N) clips per condition, "
               "normalised to total power (so these are spectral *shape*, not gain).")

    st.header("Suction vs discharge blockage — they look different")
    st.markdown("""
**Discharge** throttling (outlet) drives a *large, monotonic* change: the pump tone
rises and mid-band (250–500 Hz) energy grows with severity — a strong, easily-read
signal. **Suction** throttling (inlet) is subtler: it mostly lifts the very-low band
(0–100 Hz) and drops the tone — readable, but a smaller effect (the harder axis).""")
    c1, c2 = st.columns(2)
    with c1:
        fig_to_st(_psd_sweep(sigs, dt, "discharge", [1, 2, 3, 4, 5, 8, 11],
                             "Discharge (valveOut) sweep", "vout"))
    with c2:
        fig_to_st(_psd_sweep(sigs, dt, "suction", [1, 2, 3, 4, 5],
                             "Suction (valveIn) sweep", "vin"))

    st.header("Aeration on vs off")
    st.markdown("""
Toggling **aeration** at a fixed valve config adds low-frequency / tonal structure
(a **50–100 Hz boost** on mics; a tonal/harmonic shift on cameras). The level barely
changes — the separation is in *spectral shape*, which is why a gain-invariant
detector works (and the old "−34 dB level artifact" story was an averaging mistake).""")
    fig_to_st(_psd_pair(sigs, dt, "aer_off", "aer_on", "aeration off",
                        "aeration ON", "Aeration on vs off (matched vin1/vout1)"))

    st.header("Which machine is running")
    st.markdown("""
Each auxiliary machine adds its own fingerprint. **M3** (loud exhaust fan) injects
broadband **4–8 kHz** airflow — unmistakable. **M2** (2nd pump) adds mid/low tonal
lines. **M4** (near-silent fan) barely changes the spectrum — the hardest to hear.""")
    m = st.selectbox("Machine", ["M3", "M2", "M4"],
                     format_func=lambda x: {"M3": "M3 — exhaust fan (loud)",
                                            "M2": "M2 — 2nd large pump",
                                            "M4": "M4 — exhaust fan (near-silent)"}[x])
    fig_to_st(_psd_pair(sigs, dt, f"{m}_off", f"{m}_on", f"{m} off", f"{m} ON",
                        f"{m} on vs off (matched, M1 only)"))


def page_channels():
    st.title("🎙 The 8 mics don't sound alike — placement dominates")
    st.markdown("""
Every session records the same instant on **8 microphones + 8 camera-mics**. They are
*not* interchangeable: where a sensor sits (and the camera mics' built-in AGC) changes
both what it captures and how well the blockage level can be read from it. This is the
single biggest driver of model quality — which is why we model **per sensor family**
and never pool mics with cameras.""")

    psd = load_channels_psd()
    ch = load_channels()
    if not psd or not ch:
        st.warning("Run `python3 run/channels.py` to generate the per-channel data.")
        return

    st.header("Per-channel spectra (same operating condition)")
    st.caption("Mean spectrum of each channel over M1-only clean clips, normalised to "
               "total power (spectral shape). Note how mic channels place their pump "
               "tone differently (≈130–390 Hz) while the camera mics sit much higher.")
    f = psd["freq"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    for i in range(1, 9):
        k = f"mic{i}"
        if k in psd:
            a1.semilogx(f, db(psd[k]), lw=1.3, label=k)
    a1.set_title("8 microphones"); a1.set_xlim(30, f.max()); a1.set_xlabel("Hz")
    a1.set_ylabel("power (dB, rel.)"); a1.legend(ncol=2, fontsize=7); a1.grid(alpha=0.2)
    for i in range(1, 9):
        k = f"cam{i}"
        if k in psd:
            a2.semilogx(f, db(psd[k]), lw=1.3, label=k)
    a2.set_title("8 camera-mics"); a2.set_xlim(30, f.max()); a2.set_xlabel("Hz")
    a2.legend(ncol=2, fontsize=7); a2.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)

    st.header("How well each channel reads the blockage level")
    st.caption("Per-channel, config-grouped CV (each channel alone). Higher = this one "
               "sensor reads the level better. The spread is large among mics; the "
               "camera mics are uniformly strong.")
    rows = []
    for d, v in ch.items():
        rows.append({"channel": d,
                     "family": "cam" if d.startswith("cam") else "mic",
                     "level (dB)": round(v["features"]["rms_db"], 1),
                     "pump tone (Hz)": round(v["features"]["tone_freq"]),
                     "discharge ρ": round(v["discharge"]["spearman"], 2) if v["discharge"] else None,
                     "discharge within-1": round(v["discharge"]["within1"], 2) if v["discharge"] else None,
                     "suction ρ": round(v["suction"]["spearman"], 2) if v["suction"] else None})
    tab = pd.DataFrame(rows)

    fig2, ax = plt.subplots(figsize=(11, 3.6))
    order = tab.sort_values(["family", "discharge ρ"], ascending=[True, False])
    colors = ["#7fd1ff" if fam == "cam" else "#f0b54b" for fam in order["family"]]
    ax.bar(order["channel"], order["discharge ρ"], color=colors)
    for i, (c, val) in enumerate(zip(order["channel"], order["discharge ρ"])):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("discharge-level Spearman ρ (per channel)")
    ax.set_title("Per-channel blockage-reading quality  (orange = mic, blue = camera)")
    ax.grid(axis="y", alpha=0.3); fig2.tight_layout(); fig_to_st(fig2)

    st.dataframe(tab.sort_values(["family", "discharge ρ"], ascending=[True, False]),
                 hide_index=True, width="stretch")
    best_mic = tab[tab.family == "mic"].sort_values("discharge ρ").iloc[-1]
    worst_mic = tab[tab.family == "mic"].sort_values("discharge ρ").iloc[0]
    best_cam = tab[tab.family == "cam"].sort_values("discharge ρ").iloc[-1]
    st.markdown(f"""
**Takeaways**
- Among the 8 mics, discharge-reading quality ranges from **{worst_mic['channel']}
  (ρ={worst_mic['discharge ρ']})** to **{best_mic['channel']} (ρ={best_mic['discharge ρ']})** —
  placement alone moves it by ~2×.
- The camera mics are uniformly strong (best **{best_cam['channel']} ρ={best_cam['discharge ρ']}**)
  and sit ~30 dB louder with their pump-load tone near ~800 Hz — they read discharge best.
- This is why models are trained **per family**, why pooling all 16 channels hurt exact
  level reading, and why fusing channels (or deploying the best-placed one) helps. The
  live listener auto-detects the *family*; individual placement is a deployment choice.
""")


def page_models():
    st.title("🧠 The models & how we trained them")
    st.markdown("""
### Features — 30 gain-invariant descriptors per clip
From one channel we extract: 12 log band-energies (0→22 kHz, finer below 350 Hz),
spectral centroid/bandwidth/rolloff/flatness/crest, ZCR, the dominant pump-tone
frequency + prominence + 2nd/3rd-harmonic ratios, coarse band ratios, and
envelope amplitude-modulation energy. **All but one feature are relative to total
power or are ratios**, so models don't lean on channel gain (ablating the one level
feature leaves accuracy unchanged).

### Model — gradient-boosted trees, per sensor family
`HistGradientBoosting` (regressor for severity, classifier for presence). Mics and
camera-mics are trained **separately** — placement dominates quality (cameras read
discharge best). A tiny **mic-vs-cam auto-detector** (98 % per window) lets the live
listener pick the right family by itself.

### Two analysis windows
| task | window | why |
|---|---|---|
| aeration on/off, which-machine | **8 s** | fast; trained on 8 s windows |
| blockage **level** | **up to 60 s** | needs a long window — 8 s loses too much, esp. on mics (campaign-out within-1 0.6 → **1.0** at 60 s) |

### Leakage-honest validation (three strictnesses)
- **config-grouped** — hold out a whole valve config (+ its noisy twins).
- **leave-one-campaign-out** — train on some recording *days*, test on another day.
- **leave-one-level-out** — hold out an entire restriction level (tests interpolation).

Crucially, every recording fires 16 sensors at one instant, so **all CV groups keep
those siblings together** — a naive split inflates accuracy (0.977 → 0.993).

### Hard-negative training (aeration / machines)
First versions, trained only within the clean 5_25 campaign, false-alarmed ~19 % on
mic when M1 was throttled. Adding the M1-only blockage clips as negatives (they are
definitively aeration-off / all-aux-off) cut the **false-alarm to ~0 %** while
keeping recall.
""")
    st.caption("Model bundles are saved per task and sensor under run/models/ — "
               "`{scaler, model, features, levels}`; load and predict directly.")


def _metrics_table(res):
    rows = []
    sc = res.get("severity_leave_campaign_out", {})
    for axis in ["discharge", "suction"]:
        for dv in ["mic", "cam"]:
            m = sc.get(axis, {}).get(dv, {}).get("pooled", {})
            if m:
                rows.append({"task": f"{axis} level", "sensor": dv,
                             "within-1": round(m["within1"], 3),
                             "exact": round(m["exact_acc"], 3),
                             "MAE (steps)": round(m["MAE_steps"], 2),
                             "test": "leave-campaign-out"})
    return pd.DataFrame(rows)


def page_results():
    st.title("📊 Results")
    res = load_json("results.json")
    aer = load_json("aeration_results.json")
    wp = load_json("which_pump_results.json")
    live = load_json("live_test_results.json")

    st.header("Blockage level (the headline)")
    t = _metrics_table(res)
    if not t.empty:
        st.dataframe(t, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    for col, fn, cap in [(c1, "fig_severity_discharge.png", "Discharge: predicted vs true"),
                         (c2, "fig_severity_suction.png", "Suction: predicted vs true")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")
    st.markdown("**Noise robustness** (train clean, test each environmental noise) "
                "and the leave-one-level-out interpolation check:")
    c3, c4 = st.columns(2)
    for col, fn in [(c3, "fig_robustness.png"), (c4, "fig_leave_level_out.png")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            col.image(p, width="stretch")

    st.header("Aeration on/off & which-machine")
    c5, c6 = st.columns(2)
    for col, fn, cap in [(c5, "fig_aeration.png", "Aeration: matched-config, gain-invariant"),
                         (c6, "fig_which_pump.png", "Which machine: per-machine F1")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")

    if live:
        st.subheader("End-to-end listener (held out by recording)")
        rows = []
        a = live.get("aeration_matched_gain_invariant", {})
        for dv in ["mic", "cam"]:
            if dv in a:
                rows.append({"task": "aeration on/off", "sensor": dv,
                             "1-mic clip acc": round(a[dv]["clip_acc_single_sensor"], 3),
                             "8ch-fused acc": round(a[dv]["clip_acc_fused_8ch"], 3)})
        mp = live.get("machine_presence", {})
        for mm in ["M2", "M3", "M4"]:
            for dv in ["mic", "cam"]:
                if mm in mp and dv in mp[mm]:
                    rows.append({"task": f"{mm} present", "sensor": dv,
                                 "1-mic clip acc": round(mp[mm][dv]["clip_acc_single_sensor"], 3),
                                 "8ch-fused acc": round(mp[mm][dv]["clip_acc_fused_8ch"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    p = os.path.join(HERE, "fig_importance.png")
    if os.path.exists(p):
        st.subheader("What drives each reading (feature importance)")
        st.image(p, width="stretch")


@st.cache_resource
def get_listener():
    from listener import Listener
    return Listener("auto")


@st.cache_resource
def get_library():
    from clip_library import ClipLibrary
    return ClipLibrary()


def page_demo():
    st.title("🎧 Live demo — models on real clips")
    st.caption("Runs the trained models **directly on a dataset recording** (the "
               "reliable, validated path) and compares the prediction to ground truth.")
    lib = get_library(); lis = get_listener()

    c = st.columns(5)
    dev = c[0].selectbox("Sensor", ["cam", "mic"])
    vout = c[1].selectbox("Discharge", ["any"] + lib.options("valveOut"))
    vin = c[2].selectbox("Suction", ["any"] + lib.options("valveIn"))
    aer = c[3].selectbox("Aeration", ["any", "off", "on"])
    noise = c[4].selectbox("Noise", ["any"] + list(NOISE))
    cc = st.columns(3)
    m2 = cc[0].checkbox("M2 pump"); m3 = cc[1].checkbox("M3 fan"); m4 = cc[2].checkbox("M4 fan")

    def sel(v):
        return None if v == "any" else v
    aerv = None if aer == "any" else (1 if aer == "on" else 0)

    if "demo_clip" not in st.session_state:
        st.session_state.demo_clip = None
    if st.button("🔎 Find a matching clip & run the models", type="primary"):
        clip = lib.find(dev_type=dev, valveOut=sel(vout), valveIn=sel(vin),
                        aeration=aerv, noise_cat=sel(noise),
                        M2=int(m2), M3=int(m3), M4=int(m4))
        st.session_state.demo_clip = clip

    clip = st.session_state.demo_clip
    if clip is None:
        st.info("Pick a condition and press the button. Try: Sensor=cam, "
                "Aeration=on (the example that fails over loopback but works here), "
                "or Discharge=11 for a heavy-blockage clip.")
        return

    import soundfile as sf
    x, _ = sf.read(clip["path"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    r = lis.analyze(x[:int(60 * SR)])

    st.markdown(f"**Clip:** `{clip['file']}`")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Ground truth")
        st.markdown(f"""
- suction (valveIn) = **L{clip['valveIn']}**
- discharge (valveOut) = **L{clip['valveOut']}**
- aeration = **{'ON' if clip['aeration'] else 'off'}**
- running = **M1{''.join(f' + {m}' for m in ['M2','M3','M4'] if clip[m])}**
- sensor = **{clip['dev_type']}**, noise = **{clip['noise']}**""")
    with g2:
        st.markdown("##### Model prediction (from audio)")
        b = r.get("blockage", {})
        aerp = r["aeration"]
        run = [m for m in r["running"] if m != "M1"]
        unc = b.get("discharge", {}).get("reliability") == "uncertain_aux"
        st.markdown(f"""
- sensor auto-detected = **{r['sensor']}** {'✅' if r['sensor']==clip['dev_type'] else '⚠️'}
- aeration = **{'ON' if aerp['on'] else 'off'}** (p={aerp['p']:.2f}) {'✅' if aerp['on']==bool(clip['aeration']) else '⚠️'}
- running = **M1{''.join(f' + {m}' for m in run)}**
- discharge level = **{('L'+str(b['discharge']['level'])) if b else '—'}** {'(uncertain · aux running)' if unc else ''}
- suction level = **{('L'+str(b['suction']['level'])) if b else '—'}**""")

    # spectrogram
    n = min(len(x), 12 * SR)
    f, t, Sxx = ssignal.spectrogram(x[:n], SR, nperseg=2048, noverlap=1024)
    keep = f <= 8000
    S = 10 * np.log10(Sxx[keep] + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.imshow(S, origin="lower", aspect="auto", extent=[0, n / SR, 0, 8000],
              cmap="magma", vmin=S.max() - 75, vmax=S.max())
    ax.set_ylabel("Hz"); ax.set_xlabel("time (s)"); ax.set_title("Spectrogram (first 12 s)")
    fig_to_st(fig)
    if clip["aeration"]:
        st.warning("Note: blockage models are trained on M1-only audio, so when "
                   "aeration (or an aux machine) is on, the blockage reading is "
                   "flagged uncertain rather than trusted.")


@st.cache_data
def aeration_files(dt):
    """Matched M1-only, vin1/vout1, clean aeration on/off file paths for a sensor."""
    idx = load_index()
    out = {"on": [], "off": []}
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M3"] or p["M4"] or p["valveIn"] != 1 or p["valveOut"] != 1 \
                or p["noise"] != "N":
            continue
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                d = "cam" if dev.startswith("cam") else "mic"
                if d == dt:
                    out["on" if p["aeration"] else "off"].append(
                        os.path.join(os.path.dirname(HERE), s["disk_path"], fn))
    return out


def _spec(ax, x, fmax, title):
    n = min(len(x), 12 * SR)
    f, t, S = ssignal.spectrogram(x[:n], SR, nperseg=4096, noverlap=3072)
    keep = f <= fmax
    Sd = 10 * np.log10(S[keep] + 1e-12)
    ax.imshow(Sd, origin="lower", aspect="auto", extent=[0, n / SR, 0, fmax],
              cmap="magma", vmin=Sd.max() - 70, vmax=Sd.max())
    ax.set_title(title); ax.set_xlabel("time (s)"); ax.set_ylabel("Hz")


def page_aeration_compare():
    import soundfile as sf
    st.title("💨 Aeration: see it & why")
    st.markdown("""
A direct visual A/B: the **same pump at the same valve setting**, recorded with the
**aeration air-injector OFF vs ON**. Look at the low end.""")
    dt = st.radio("Sensor", ["cam", "mic"], horizontal=True)
    if st.button("🎲 Pick a different on/off pair"):
        st.session_state.pop("aer_pair", None)
    files = aeration_files(dt)
    if not files["on"] or not files["off"]:
        st.warning("No matched aeration files found."); return
    if "aer_pair" not in st.session_state or st.session_state.get("aer_dt") != dt:
        import random
        st.session_state.aer_pair = (random.choice(files["off"]), random.choice(files["on"]))
        st.session_state.aer_dt = dt
    off_p, on_p = st.session_state.aer_pair

    def load(p):
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x
    xo, xn = load(off_p), load(on_p)

    def psd(x):
        f, p = ssignal.welch(x - x.mean(), SR, nperseg=8192, noverlap=4096)
        return f, p
    fo, po = psd(xo); fn, pn = psd(xn)

    st.subheader("Spectrum — aeration OFF vs ON")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.4))
    a1.semilogx(fo, db(po), color="#7fd1ff", lw=1.6, label="aeration OFF")
    a1.semilogx(fn, db(pn), color="#46d17a", lw=1.6, label="aeration ON")
    a1.axvspan(50, 100, color="#f0b54b", alpha=0.15)
    a1.set_xlim(30, 12000); a1.set_xlabel("Hz"); a1.set_ylabel("power (dB, rel.)")
    a1.legend(); a1.grid(alpha=0.2); a1.set_title("full spectrum (50–100 Hz band shaded)")
    a2.semilogx(fn, db(pn) - db(po), color="#f0b54b", lw=1.6)
    a2.axhline(0, color="#777", lw=0.7); a2.axvspan(50, 100, color="#f0b54b", alpha=0.15)
    a2.set_xlim(30, 12000); a2.set_xlabel("Hz"); a2.set_ylabel("Δ power (dB)")
    a2.set_title("Δ = ON − OFF"); a2.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)

    st.subheader("Spectrogram — low band (0–1500 Hz), where it shows")
    fig2, (b1, b2) = plt.subplots(1, 2, figsize=(12, 3.0))
    _spec(b1, xo, 1500, "aeration OFF")
    _spec(b2, xn, 1500, "aeration ON")
    fig2.tight_layout(); fig_to_st(fig2)

    def band_db(f, p, lo, hi):
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / p.sum() + 1e-12)
    d5010 = band_db(fn, pn, 50, 100) - band_db(fo, po, 50, 100)
    st.caption(f"OFF: `{os.path.basename(off_p)}`  ·  ON: `{os.path.basename(on_p)}`  "
               f"· measured 50–100 Hz band change here: **{d5010:+.1f} dB**")

    st.subheader("What in the signal tells you aeration is on")
    st.markdown("""
From analysing matched on/off pairs across both sensor families, three things move
together when the **air injector switches on** — and they're physically what you'd
expect from forcing air into the water:

1. **A boost in the 50–100 Hz band (≈ +4.5 dB).** The most consistent tell on *both*
   mics and cameras — a low-frequency bubbling/rumble that simply isn't there with
   aeration off. *(Shaded band in the plots above.)*
2. **The dominant spectral peak drops to the low end.** The pump's main tone collapses
   from ~800 Hz → ~85 Hz on cameras (and ~250 → ~150 Hz on mics): injected air re-loads
   the pump, so its loudest resonance moves down into that bubbling band. On the
   spectrogram you see the energy **shift downward**.
3. **The sound becomes less "tonal", more bubbly.** Spectral crest drops ~5 dB (the
   single sharp peak fills in) and — clearest on the higher-SNR camera mics — the
   **envelope picks up strong 2–40 Hz amplitude modulation (≈ +8 dB)**: the rhythmic
   burble of bubbles passing.

So the one-line answer: **aeration on = the low band (≈50–100 Hz) lights up and the
dominant tone moves down there, with an audible few-Hz bubbling modulation.** Because
all three are *shape/modulation* changes (not loudness), the detector is gain-invariant
and works even though the absolute level barely changes.

*Caveat: aeration was only recorded at one valve setting (`vin1/vout1`); when M1 is
heavily throttled the baseline spectrum already shifts low, which is why the live
listener flags blockage as uncertain whenever it hears aeration.*
""")


@st.cache_data
def blockage_file(dt, vin, vout):
    """A real clean M1-only file at exactly (vin,vout) for this sensor family."""
    import random
    idx = load_index()
    cand = []
    for sig, e in idx["measurements"].items():
        p = e["params"]
        if p["M2"] or p["M3"] or p["M4"] or p["aeration"] or p["noise"] != "N":
            continue
        if p["valveIn"] != vin or p["valveOut"] != vout:
            continue
        for s in e["sessions"]:
            for dev, fn in s["devices"].items():
                d = "cam" if dev.startswith("cam") else "mic"
                if d == dt:
                    cand.append(os.path.join(os.path.dirname(HERE), s["disk_path"], fn))
    return random.Random(vin * 100 + vout).choice(cand) if cand else None


def page_blockage_compare():
    import soundfile as sf
    st.title("🚰 Blockage: see it & why")
    st.markdown("""
M1 has two throttle valves — one on the **suction** (inlet, `valveIn`) side and one on
the **discharge** (outlet, `valveOut`) side. Both restrict flow, but they do *different*
things to the sound. Here's how each evolves as you close it (1 = open → more
restricted), and how **suction-at-x differs from discharge-at-x**.""")
    sigs = load_sigs()
    dt = st.radio("Sensor", ["cam", "mic"], horizontal=True, key="blk_dt")
    f = sigs["freq"]

    st.header("1 · How the spectrum evolves as you close each valve (x++)")
    c1, c2 = st.columns(2)
    with c1:
        fig_to_st(_psd_sweep(sigs, dt, "discharge", [1, 2, 3, 4, 5, 8, 11],
                             "DISCHARGE (valveOut) — closing the outlet", "vout"))
        st.caption("Outlet back-pressure makes the pump labour → the **250–500 Hz "
                   "mid-band grows steadily** and the tone shifts. Big, monotonic, "
                   "easy to read.")
    with c2:
        fig_to_st(_psd_sweep(sigs, dt, "suction", [1, 2, 3, 4, 5],
                             "SUCTION (valveIn) — starving the inlet", "vin"))
        st.caption("Inlet starvation → the **dominant tone collapses to a very low "
                   "frequency**" + (" and **4–8 kHz hiss (cavitation) rises**"
                   if dt == "mic" else "") + ". Subtler — the harder axis.")

    st.header("2 · The trends, quantified")

    def bdb(p, lo, hi):
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / p.sum() + 1e-12)

    def tone(p):
        lm = f < 2000
        return f[lm][np.argmax(p[lm])]
    dlv = [1, 2, 3, 4, 5, 8, 11]; slv = [1, 2, 3, 4, 5]
    dmid = [bdb(sigs[f"{dt}_discharge_{l}"], 250, 500) for l in dlv]
    smid = [bdb(sigs[f"{dt}_suction_{l}"], 250, 500) for l in slv]
    dton = [tone(sigs[f"{dt}_discharge_{l}"]) for l in dlv]
    ston = [tone(sigs[f"{dt}_suction_{l}"]) for l in slv]
    dhf = [bdb(sigs[f"{dt}_discharge_{l}"], 4000, 8000) for l in dlv]
    shf = [bdb(sigs[f"{dt}_suction_{l}"], 4000, 8000) for l in slv]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.0))
    for ax, (yd, ys, ttl, yl) in zip(axs, [
            (dmid, smid, "Mid-band 250–500 Hz", "power (dB, rel.)"),
            (dton, ston, "Dominant pump tone", "Hz"),
            (dhf, shf, "High band 4–8 kHz", "power (dB, rel.)")]):
        ax.plot(dlv, yd, "o-", color="#ff9d6b", label="discharge")
        ax.plot(slv, ys, "s-", color="#7fd1ff", label="suction")
        ax.set_title(ttl); ax.set_xlabel("valve level (1=open)"); ax.set_ylabel(yl)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)
    st.caption("Discharge ⟶ mid-band climbs monotonically (the main 'severity' signal). "
               "Suction ⟶ the tone drops sharply" +
               (" and 4–8 kHz hiss climbs" if dt == "mic" else "") +
               " — different fingerprints for the same act of 'closing a valve'.")

    st.header("3 · Suction-at-x vs Discharge-at-x — do they differ? (yes)")
    x = st.select_slider("Compare at restriction level x =", options=[2, 3, 4, 5], value=4)
    ps = sigs[f"{dt}_suction_{x}"]; pd_ = sigs[f"{dt}_discharge_{x}"]
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.3))
    a1.semilogx(f, db(ps), color="#7fd1ff", lw=1.7, label=f"SUCTION at {x} (vin={x},vout=1)")
    a1.semilogx(f, db(pd_), color="#ff9d6b", lw=1.7, label=f"DISCHARGE at {x} (vin=1,vout={x})")
    a1.axvspan(250, 500, color="#ff9d6b", alpha=0.10)
    a1.set_xlim(30, 12000); a1.set_xlabel("Hz"); a1.set_ylabel("power (dB, rel.)")
    a1.legend(fontsize=8); a1.grid(alpha=0.2); a1.set_title(f"same nominal level x={x}")
    a2.semilogx(f, db(pd_) - db(ps), color="#46d17a", lw=1.6); a2.axhline(0, color="#777", lw=0.7)
    a2.set_xlim(30, 12000); a2.set_xlabel("Hz"); a2.set_ylabel("Δ power (dB)")
    a2.set_title("Δ = discharge − suction"); a2.grid(alpha=0.2)
    fig2.tight_layout(); fig_to_st(fig2)
    st.caption(f"At the same x={x}: discharge carries **more 250–500 Hz mid-band** "
               f"(tone ≈{tone(pd_):.0f} Hz) while suction sits **lower-pitched** "
               f"(tone ≈{tone(ps):.0f} Hz). They are *not* the same sound — which is "
               "why the two axes are read by separate models.")

    st.subheader("Real clips — spectrograms (0–1500 Hz)")
    sfp = blockage_file(dt, x, 1); dfp = blockage_file(dt, 1, x)
    if sfp and dfp:
        def load(p):
            y, sr = sf.read(p, dtype="float32"); return y.mean(1) if y.ndim > 1 else y
        fig3, (b1, b2) = plt.subplots(1, 2, figsize=(12, 3.0))
        _spec(b1, load(sfp), 1500, f"SUCTION at {x} (vin={x}, vout=1)")
        _spec(b2, load(dfp), 1500, f"DISCHARGE at {x} (vin=1, vout={x})")
        fig3.tight_layout(); fig_to_st(fig3)

    st.header("What the signal tells you")
    st.markdown("""
- **Discharge (output) blockage** = back-pressure on the outlet. As you close it, the
  pump works against a wall, so its **250–500 Hz mid-band tonal energy grows steadily
  with severity** and the tone shifts up at the extremes. It's large, monotonic and
  loud → the model reads the *exact* level within ±1 essentially always (cam exact ≈0.98).
- **Suction (input) blockage** = starving the inlet. The operating point shifts so the
  **dominant tone collapses to a very low frequency** (down to ~80–120 Hz), and on
  microphones a **4–8 kHz cavitation hiss** rises as the starved inlet cavitates. The
  effect is real but subtler and noisier → readable within ±1 every time, but the exact
  step is harder (the "hard axis").
- **They are distinguishable at the same x:** discharge is *mid-band + higher tone*,
  suction is *low-tone (+ HF hiss on mics)*. So input vs output restriction are not
  interchangeable — the report models them as **two independent ordinal axes**, not one
  combined "blockage" number.
""")


PAGES = {
    "① Overview — the story": page_overview,
    "② Dataset & parameters": page_dataset,
    "③ Acoustic signatures": page_signatures,
    "④ Mic & sensor differences": page_channels,
    "⑤ Models & training": page_models,
    "⑥ Results": page_results,
    "⑦ Live demo": page_demo,
    "⑧ Aeration: see it & why": page_aeration_compare,
    "⑨ Blockage: see it & why": page_blockage_compare,
}


def main():
    st.sidebar.title("Pool-Audio")
    st.sidebar.caption("Reading pump operating condition from sound")
    choice = st.sidebar.radio("Report sections", list(PAGES))
    st.sidebar.divider()
    st.sidebar.caption("Models in run/models/ · data: 13,850 files, 7 campaigns · "
                       "all equipment healthy (operating configs, not faults).")
    PAGES[choice]()


if __name__ == "__main__":
    main()
