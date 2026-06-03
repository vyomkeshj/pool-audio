#!/usr/bin/env python3
"""Corrected acoustic analysis of the pool-pump testbed.

WHAT THIS DATA ACTUALLY IS (testbed owner, 2026-06-02):
  * Nothing is broken. Every clip is HEALTHY equipment in a different operating
    / valve configuration. There is NO faulty or damaged pump in the dataset.
  * M1, M2 = large pumps.  M3, M4 = exhaust fans.  aeration = air injector.
  * The only "fault-like" axis is flow RESTRICTION of the main pump M1, set by
    two throttling valves (valveIn = suction side, valveOut = discharge side).

So there are two orthogonal things the microphone can tell us, and we treat them
separately instead of collapsing them into one muddled "state" label:

  AXIS A  -- M1 flow restriction (condition monitoring, the real target):
            how restricted is the suction side (valveIn 1-4) and the discharge
            side (valveOut 1,2,3,4,5,8,11)?  1 = open. Exercised in every
            campaign and under every environmental-noise type -> robust core.

  AXIS B  -- auxiliary equipment running (acoustic source presence, NOT a fault):
            is the 2nd pump M2 on? exhaust fan M3? exhaust fan M4? aeration?
            Only varied in the 5_25 campaign (all clean).

This supersedes the earlier "5-state classifier" and the entire "M3 = broken
fan" thread (fault_id/fault_confirm/characterize_m3/detect_m3), which were built
on the false premise that one machine was damaged.

Reuses the cached 20-feature table (features_all.csv, 1 mic channel/session) so
no audio re-extraction is needed, except a small PSD pass for the equipment
signature figure.

Outputs: reanalysis.json,
         fig_equipment_signature.png  (what each healthy machine adds)
         fig_m1_restriction.png       (suction/discharge detection under noise)
         fig_m1_severity.png          (how-severe regression, honest vs optimistic)
         fig_equipment_presence.png   (multi-label source presence)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_absolute_error, r2_score)
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT_COLS = ["rms_db", "centroid", "bandwidth", "rolloff", "flatness", "crest",
             "zcr", "tone_freq", "tone_prom", "mod_energy", "mod_peak",
             "band_0_100", "band_100_250", "band_250_500", "band_500_1000",
             "band_1000_2000", "band_2000_4000", "band_4000_8000",
             "band_8000_16000", "band_16000_22050"]
BANDS = ["band_0_100", "band_100_250", "band_250_500", "band_500_1000",
         "band_1000_2000", "band_2000_4000", "band_4000_8000",
         "band_8000_16000", "band_16000_22050"]
BAND_HZ = [50, 175, 375, 750, 1500, 3000, 6000, 12000, 19000]  # band centres
MACHINE = {"M2": "M2 (2nd large pump)", "M3": "M3 (exhaust fan)",
           "M4": "M4 (exhaust fan)", "aeration": "aeration (air injector)"}


# ---------------------------------------------------------------- Part 1
def equipment_signatures(df, results):
    """Matched-pair acoustic increment of each healthy auxiliary machine.

    5_25 is a clean on/off factorial, so for each machine we average (on - off)
    over every configuration that differs ONLY in that machine. This isolates
    what turning each healthy machine on does to the sound -- no fault implied.
    """
    d = df[df.folder == "testbedmotor5_25wav"].copy()
    keys = ["M2", "M3", "M4", "aeration", "valveIn", "valveOut"]
    sig = {}
    print("\n=== PART 1: acoustic signature of each healthy auxiliary machine ===")
    for mach in ["M2", "M3", "M4", "aeration"]:
        others = [k for k in keys if k != mach]
        rows = []
        for _, sub in d.groupby(others):
            on, off = sub[sub[mach] == 1], sub[sub[mach] == 0]
            if len(on) and len(off):
                rows.append(on[FEAT_COLS].mean() - off[FEAT_COLS].mean())
        if not rows:
            continue
        delta = pd.DataFrame(rows).mean()
        sig[mach] = {"n_pairs": len(rows),
                     "d_rms_db": round(float(delta.rms_db), 2),
                     "d_centroid_hz": round(float(delta.centroid), 0),
                     "d_tone_freq_hz": round(float(delta.tone_freq), 0),
                     "d_mod_energy": round(float(delta.mod_energy), 2),
                     "band_delta_db": {b: round(float(delta[b]), 2) for b in BANDS}}
        peak_band = max(BANDS, key=lambda b: delta[b])
        print(f"  {MACHINE[mach]:28s} n={len(rows):2d}  "
              f"d_rms={delta.rms_db:+.1f}dB  loudest add-on band={peak_band} "
              f"({delta[peak_band]:+.1f}dB)")
    # aeration caveat: every aeration-ON clip is ~34 dB quieter and was only ever
    # recorded at one valve config (vin1/vout1) -> its "signature" is a level
    # collapse, not a stable spectral fingerprint. Likely recorded with pumps
    # idle / different gain. Flag rather than trust.
    if "aeration" in sig:
        sig["aeration"]["caveat"] = (
            "anomalous: aeration-ON clips are uniformly ~34 dB quieter and exist "
            "in only one valve config (vin1/vout1); the delta is dominated by a "
            "level collapse (pumps likely idle / different gain), not a reliable "
            "acoustic signature. Do not treat as a validated detector.")
        print("  ** aeration is anomalous (uniform -34 dB, single config) -- flagged, "
              "not a validated signature **")
    results["equipment_signatures"] = sig

    # figure: band-delta fingerprint per machine
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(BANDS))
    w = 0.2
    for i, mach in enumerate(["M2", "M3", "M4", "aeration"]):
        if mach in sig:
            vals = [sig[mach]["band_delta_db"][b] for b in BANDS]
            ax.bar(x + (i - 1.5) * w, vals, w, label=MACHINE[mach])
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("band_", "").replace("_", "–") + "Hz" for b in BANDS],
                       rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ band energy when machine ON (dB, matched pairs)")
    ax.set_title("Acoustic signature of each healthy auxiliary machine (5_25 factorial)\n"
                 "M3 fan = broadband 4–8 kHz airflow · M2 pump = mid/low tonal · "
                 "M4 fan ≈ silent · aeration = strong spectral reshape")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_equipment_signature.png"), dpi=110)
    plt.close()


# ---------------------------------------------------------------- Part 2
def m1_restriction(df, results):
    """Detect & quantify M1 flow restriction from sound, under noise.

    Restricted to M1-only clips (M2=M3=M4=aeration=0) so the auxiliary machines
    don't confound the valve acoustics. Suction (valveIn) and discharge
    (valveOut) are swept jointly, so we treat them as two INDEPENDENT axes.
    """
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)].copy()
    Xall = StandardScaler().fit_transform(m1[FEAT_COLS].to_numpy(float))
    # group by physical config so the same rig setup can't sit in train+test
    cfg = (m1.valveIn.astype(str) + "_" + m1.valveOut.astype(str)).to_numpy()
    print(f"\n=== PART 2: M1 flow-restriction monitoring "
          f"({len(m1)} M1-only sessions, all noise) ===")

    out = {"n_sessions": int(len(m1))}

    # ---- 2a/2b: binary detection of suction / discharge restriction --------
    det = {}
    for name, target in [("suction", (m1.valveIn > 1).astype(int).to_numpy()),
                         ("discharge", (m1.valveOut > 1).astype(int).to_numpy())]:
        clf = RandomForestClassifier(n_estimators=400, random_state=0,
                                     class_weight="balanced")
        # config-grouped CV = honest (configs recur across noise/date)
        gkf = GroupKFold(n_splits=5)
        yp = cross_val_predict(clf, Xall, target, cv=gkf, groups=cfg)
        det[name] = {"n_pos": int(target.sum()), "n_neg": int((1 - target).sum()),
                     "accuracy": round(accuracy_score(target, yp), 3),
                     "precision": round(precision_score(target, yp, zero_division=0), 3),
                     "recall": round(recall_score(target, yp, zero_division=0), 3),
                     "f1": round(f1_score(target, yp, zero_division=0), 3)}
        print(f"  detect {name:9s} restriction: acc={det[name]['accuracy']:.3f} "
              f"P={det[name]['precision']:.2f} R={det[name]['recall']:.2f} "
              f"F1={det[name]['f1']:.2f}  (n_pos={det[name]['n_pos']})")
    out["detection"] = det

    # ---- robustness: train ONLY on clean, test on each noise category ------
    # Consistent baseline: a held-out slice of clean clips is the "N" bar, the
    # SAME clean-trained model is then applied to every noisy category. Answers
    # "does environmental noise break a detector tuned on this rig?". Configs
    # recur across noise, so this is noise-robustness on a KNOWN rig, not
    # transfer to an unseen rig (see leakage caveat below).
    clean = (m1.noise_cat == "N").to_numpy()
    rng = np.random.RandomState(0)
    clean_idx = np.where(clean)[0]
    rng.shuffle(clean_idx)
    cut = int(0.6 * len(clean_idx))
    tr_idx, clean_te_idx = clean_idx[:cut], clean_idx[cut:]
    rob = {}
    for name, col in [("suction", "valveIn"), ("discharge", "valveOut")]:
        y = (m1[col] > 1).astype(int).to_numpy()
        clf = RandomForestClassifier(n_estimators=400, random_state=0,
                                     class_weight="balanced")
        clf.fit(Xall[tr_idx], y[tr_idx])
        per = {"N": round(accuracy_score(y[clean_te_idx], clf.predict(Xall[clean_te_idx])), 3)}
        for ncat in ["A", "B", "C", "D", "E"]:
            te = (m1.noise_cat == ncat).to_numpy()
            if te.sum():
                per[ncat] = round(accuracy_score(y[te], clf.predict(Xall[te])), 3)
        rob[name] = per
        print(f"  robustness {name:9s} (clean-trained, acc by test-noise): "
              + "  ".join(f"{k}={v}" for k, v in per.items()))
    out["robustness_clean_trained"] = rob

    # ---- 2c: severity regression, optimistic vs leakage-honest -------------
    sev = {}
    for name, col, levels in [("suction", "valveIn", [1, 2, 3, 4]),
                             ("discharge", "valveOut", [1, 2, 3, 4, 5, 8, 11])]:
        sub = m1[m1[col].isin(levels)].copy()
        Xs = StandardScaler().fit_transform(sub[FEAT_COLS].to_numpy(float))
        # ordinal rank target
        rank = {lv: i for i, lv in enumerate(levels)}
        y = sub[col].map(rank).to_numpy(float)
        gcfg = (sub.valveIn.astype(str) + "_" + sub.valveOut.astype(str)).to_numpy()

        # optimistic: config-grouped CV (same valve positions recur across noise)
        gkf = GroupKFold(5)
        yp = cross_val_predict(RandomForestRegressor(300, random_state=0),
                               Xs, y, cv=gkf, groups=gcfg)
        opt = {"mae_steps": round(mean_absolute_error(y, yp), 3),
               "r2": round(r2_score(y, yp), 3),
               "spearman": round(spearmanr(y, yp).correlation, 3),
               "within_1": round(float(np.mean(np.abs(np.round(yp) - y) <= 1)), 3)}

        # leakage-honest: hold out WHOLE valve positions (test interpolation)
        # train on alternating levels, predict the held-out ones
        hold = levels[1::2]                      # e.g. discharge holds 2,4,8
        tr = ~sub[col].isin(hold).to_numpy()
        reg = RandomForestRegressor(300, random_state=0).fit(Xs[tr], y[tr])
        yp_h = reg.predict(Xs[~tr])
        y_h = y[~tr]
        hon = {"held_out_levels": [levels[i] for i in range(1, len(levels), 2)],
               "mae_steps": round(mean_absolute_error(y_h, yp_h), 3),
               "spearman": round(spearmanr(y_h, yp_h).correlation, 3) if len(set(y_h)) > 1 else None,
               "within_1": round(float(np.mean(np.abs(np.round(yp_h) - y_h) <= 1)), 3)}
        sev[name] = {"levels": levels, "n": int(len(sub)),
                     "optimistic_grouped_cv": opt, "leakage_honest_holdout": hon}
        print(f"  severity {name:9s}: optimistic ρ={opt['spearman']:.2f} "
              f"MAE={opt['mae_steps']:.2f} | honest(holdout {hon['held_out_levels']}) "
              f"MAE={hon['mae_steps']:.2f} within1={hon['within_1']:.2f}")
    out["severity"] = sev
    results["m1_restriction"] = out

    # ---- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, ["suction", "discharge"]):
        per = rob[name]
        ax.bar(list(per.keys()), list(per.values()), color="steelblue")
        ax.axhline(0.5, color="r", ls="--", lw=1, label="chance (balanced)")
        ax.set_ylim(0, 1); ax.set_title(f"Detect {name} restriction\n(clean-trained, by noise)")
        ax.set_ylabel("accuracy"); ax.set_xlabel("test noise category"); ax.grid(alpha=0.3, axis="y")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_m1_restriction.png"), dpi=110)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, ["suction", "discharge"]):
        col = "valveIn" if name == "suction" else "valveOut"
        levels = sev[name]["levels"]
        sub = m1[m1[col].isin(levels)].copy()
        Xs = StandardScaler().fit_transform(sub[FEAT_COLS].to_numpy(float))
        rank = {lv: i for i, lv in enumerate(levels)}
        y = sub[col].map(rank).to_numpy(float)
        gcfg = (sub.valveIn.astype(str) + "_" + sub.valveOut.astype(str)).to_numpy()
        yp = cross_val_predict(RandomForestRegressor(300, random_state=0), Xs, y,
                               cv=GroupKFold(5), groups=gcfg)
        jit = (np.random.RandomState(0).rand(len(y)) - 0.5) * 0.3
        ax.scatter(np.array([levels[int(v)] for v in y]) + jit, yp, alpha=0.4, s=18)
        ax.plot(levels, range(len(levels)), "r--", label="ideal")
        ax.set_yticks(range(len(levels))); ax.set_yticklabels(levels)
        ax.set_xlabel(f"true {col} (blockage level)")
        ax.set_ylabel(f"predicted level"); ax.grid(alpha=0.3)
        ax.set_title(f"{name} severity  ρ={sev[name]['optimistic_grouped_cv']['spearman']:.2f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_m1_severity.png"), dpi=110)
    plt.close()


# ---------------------------------------------------------------- Part 3
def equipment_presence(df, results):
    """Multi-label: which auxiliary machines are running? (5_25 clean only)."""
    d = df[df.folder == "testbedmotor5_25wav"].copy()
    Xs = StandardScaler().fit_transform(d[FEAT_COLS].to_numpy(float))
    # Only M2/M3/M4 can be honestly cross-validated: each is toggled across the
    # full valveOut sweep, so config-grouped CV trains+tests on both states.
    # aeration is EXCLUDED here -- it was only ever ON at one config (vin1/vout1),
    # so any config-grouped split makes it untrainable; reported separately.
    machines = ["M2", "M3", "M4"]
    Y = d[machines].to_numpy(int)
    grp = (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).to_numpy()
    clf = MultiOutputClassifier(RandomForestClassifier(300, random_state=0,
                                                       class_weight="balanced"))
    yp = cross_val_predict(clf, Xs, Y, cv=GroupKFold(5), groups=grp)
    print("\n=== PART 3: auxiliary equipment presence (5_25 clean, multi-label) ===")
    per = {}
    for i, m in enumerate(machines):
        per[m] = {"n_on": int(Y[:, i].sum()),
                  "precision": round(precision_score(Y[:, i], yp[:, i], zero_division=0), 3),
                  "recall": round(recall_score(Y[:, i], yp[:, i], zero_division=0), 3),
                  "f1": round(f1_score(Y[:, i], yp[:, i], zero_division=0), 3)}
        print(f"  {MACHINE[m]:28s} n_on={per[m]['n_on']:3d}  "
              f"P={per[m]['precision']:.2f} R={per[m]['recall']:.2f} F1={per[m]['f1']:.2f}")
    subset_acc = float(np.mean((yp == Y).all(axis=1)))
    hamming = float(np.mean(yp != Y))
    print(f"  exact-match(subset over M2/M3/M4) acc={subset_acc:.3f}  Hamming loss={hamming:.3f}")
    # aeration: trivially separable by level alone, but only 1 config -> can't CV.
    aer = d[d.aeration == 1].rms_db
    noaer = d[d.aeration == 0].rms_db
    sep = float(noaer.mean() - aer.mean())  # mean level separation in dB
    print(f"  aeration: ON clips ~{aer.mean():.0f}dB vs OFF ~{noaer.mean():.0f}dB "
          f"(~{sep:.0f}dB quieter) BUT single config, not honestly CV-able")
    results["equipment_presence"] = {
        "cross_validated": ["M2", "M3", "M4"],
        "per_machine": per,
        "subset_accuracy_m2m3m4": round(subset_acc, 3),
        "hamming_loss_m2m3m4": round(hamming, 3),
        "aeration_note": (
            f"ON clips ({aer.mean():.0f} dB) are ~{sep:.0f} dB quieter than OFF "
            f"({noaer.mean():.0f} dB) on average, but recorded in only one valve "
            f"config -> cannot be cross-validated and the level drop is likely a "
            f"recording artifact (pumps idle / different gain), not real aeration.")}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(machines))
    for j, met in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (j - 1) * 0.25, [per[m][met] for m in machines], 0.25, label=met)
    ax.set_xticks(x); ax.set_xticklabels([MACHINE[m] for m in machines], rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Auxiliary equipment presence detection (clean)\n"
                 f"subset-acc={subset_acc:.2f} · M4 fan is near-silent → hardest")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_equipment_presence.png"), dpi=110)
    plt.close()


def main():
    df = pd.read_csv(os.path.join(ROOT, "features_all.csv"))
    results = {"_note": "Corrected analysis. Nothing is broken; M1/M2 large pumps, "
                        "M3/M4 exhaust fans. Two axes: M1 flow restriction "
                        "(monitoring) and auxiliary equipment presence."}
    equipment_signatures(df, results)
    m1_restriction(df, results)
    equipment_presence(df, results)
    json.dump(results, open(os.path.join(ROOT, "reanalysis.json"), "w"), indent=2)
    print("\nwrote reanalysis.json + 4 figures")


if __name__ == "__main__":
    main()
