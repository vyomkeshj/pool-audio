#!/usr/bin/env python3
"""Train + honestly evaluate + save operating-condition models, DEVICE-AWARE.

Nothing is broken; these read OPERATING condition of healthy equipment.
M1/M2 = large pumps, M3/M4 = exhaust fans.

KEY LESSON (2026-06-02): the 16 channels are TWO different sensor families that
must NOT be pooled. Mics sit ~-61 dB with the true pump tone ~190 Hz; cameras
sit ~-32 dB (AGC) with a strong structural tone ~780 Hz that tracks pump load.
Channel POSITION dominates: discharge-severity quality ranges from ρ=0.24
(mic6, nearly blind) to ρ=0.93 (cam5). So every model is trained PER DEVICE
TYPE, and we also rank individual channels.

Tasks (model saved as models/<task>_<mic|cam>.joblib):
  discharge_severity  -- valveOut restriction level, ordinal regressor
  suction_severity    -- valveIn restriction level, ordinal regressor
  restriction_location-- open/suction/discharge/both, 4-class
  motor_presence      -- is M2 / M3 / M4 running? multi-label
  which_motor         -- none/M2/M3/M4 (isolation), 4-class
  aeration            -- on/off (FLAGGED anomalous, not a validated detector)

CV is grouped so the same physical config never sits in train+test:
  blockage -> group by (valveIn,valveOut); equipment -> group by valveOut.

Run: python3 train_models.py
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, mean_absolute_error,
                             r2_score, balanced_accuracy_score)
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(ROOT, "models")
os.makedirs(MODELDIR, exist_ok=True)
FEAT = ["rms_db", "centroid", "bandwidth", "rolloff", "flatness", "crest", "zcr",
        "tone_freq", "tone_prom", "mod_energy", "mod_peak", "band_0_100",
        "band_100_250", "band_250_500", "band_500_1000", "band_1000_2000",
        "band_2000_4000", "band_4000_8000", "band_8000_16000", "band_16000_22050"]
DISCHARGE_LV = [1, 2, 3, 4, 5, 8, 11]
SUCTION_LV = [1, 2, 3, 4]
RESULTS = {"_note": "Device-aware operating-condition models. Healthy equipment; "
                    "M1/M2 large pumps, M3/M4 exhaust fans. Mics and cameras are "
                    "different sensors (modelled separately). CV grouped by config."}


def rf_clf():
    return RandomForestClassifier(n_estimators=400, random_state=0,
                                  class_weight="balanced", n_jobs=-1)


def cfg_groups(d):
    return (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).to_numpy()


def save_model(name, task, dtype, X, y, classes=None, multilabel=False,
               regressor=False):
    sc = StandardScaler().fit(X)
    if regressor:
        base = RandomForestRegressor(400, random_state=0, n_jobs=-1)
    elif multilabel:
        base = MultiOutputClassifier(rf_clf())
    else:
        base = rf_clf()
    base.fit(sc.transform(X), y)
    joblib.dump({"scaler": sc, "model": base, "features": FEAT, "classes": classes,
                 "task": task, "device_type": dtype},
                os.path.join(MODELDIR, f"{name}_{dtype}.joblib"))


def confusion_fig(y, yp, labels, title, path):
    cm = confusion_matrix(y, yp, labels=labels)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                     color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    plt.ylabel("true"); plt.xlabel("predicted"); plt.title(title)
    plt.colorbar(label="row-normalised"); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def severity_metrics(sub, col, levels):
    X = StandardScaler().fit_transform(sub[FEAT].to_numpy(float))
    rank = {lv: i for i, lv in enumerate(levels)}
    y = sub[col].map(rank).to_numpy(float)
    g = cfg_groups(sub)
    yp = cross_val_predict(RandomForestRegressor(300, random_state=0, n_jobs=-1),
                           X, y, cv=GroupKFold(5), groups=g)
    opt = {"spearman": round(float(spearmanr(y, yp).correlation), 3),
           "mae_steps": round(mean_absolute_error(y, yp), 3),
           "r2": round(r2_score(y, yp), 3),
           "within_1": round(float(np.mean(np.abs(np.round(yp) - y) <= 1)), 3)}
    hold = levels[1::2]
    tr = ~sub[col].isin(hold).to_numpy()
    reg = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    yph, yh = reg.predict(X[~tr]), y[~tr]
    hon = {"held_out_levels": hold,
           "mae_steps": round(mean_absolute_error(yh, yph), 3),
           "within_1": round(float(np.mean(np.abs(np.round(yph) - yh) <= 1)), 3)}
    return {"optimistic_grouped_cv": opt, "leakage_honest_holdout": hon}


# ------------------------------------------------------------ channel ranking
def channel_ranking(df):
    print("\n### CHANNEL INFORMATIVENESS (per-channel discharge & suction severity) ###")
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]
    out = {}
    for label, col, levels in [("discharge_severity", "valveOut", DISCHARGE_LV),
                              ("suction_severity", "valveIn", SUCTION_LV)]:
        rank = {lv: i for i, lv in enumerate(levels)}
        rows = []
        for dev, sub in m1.groupby("device"):
            sub = sub[sub[col].isin(levels)]
            X = StandardScaler().fit_transform(sub[FEAT].to_numpy(float))
            y = sub[col].map(rank).to_numpy(float)
            yp = cross_val_predict(RandomForestRegressor(200, random_state=0, n_jobs=-1),
                                   X, y, cv=GroupKFold(5), groups=cfg_groups(sub))
            rows.append({"device": dev, "spearman": round(float(spearmanr(y, yp).correlation), 2),
                         "within_1": round(float(np.mean(np.abs(np.round(yp) - y) <= 1)), 2)})
        rows.sort(key=lambda r: -r["spearman"])
        out[label] = rows
        print(f"  {label}: best={rows[0]['device']}(ρ{rows[0]['spearman']}) "
              f"worst={rows[-1]['device']}(ρ{rows[-1]['spearman']})")
    RESULTS["channel_informativeness"] = out

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, rows) in zip(axes, out.items()):
        devs = [r["device"] for r in rows]
        col = ["tab:orange" if d.startswith("cam") else "tab:blue" for d in devs]
        ax.bar(range(len(devs)), [r["spearman"] for r in rows], color=col)
        ax.set_xticks(range(len(devs))); ax.set_xticklabels(devs, rotation=90, fontsize=8)
        ax.set_ylabel("Spearman ρ (grouped CV)"); ax.set_title(label); ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
                        plt.Rectangle((0, 0), 1, 1, color="tab:orange")],
               labels=["mic", "cam"], loc="upper right")
    fig.suptitle("Sensor placement dominates: per-channel blockage readability")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "fig_channel_ranking.png"), dpi=110)
    plt.close()


# ------------------------------------------------------------ blockage models
def blockage(df):
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)].copy()
    print(f"\n### BLOCKAGE models (per device type) ###")
    RESULTS["blockage"] = {}
    for label, col, levels in [("discharge_severity", "valveOut", DISCHARGE_LV),
                              ("suction_severity", "valveIn", SUCTION_LV)]:
        RESULTS["blockage"][label] = {}
        for dt in ["mic", "cam"]:
            sub = m1[(m1.dev_type == dt) & (m1[col].isin(levels))]
            sev = severity_metrics(sub, col, levels)
            RESULTS["blockage"][label][dt] = {"n": int(len(sub)), **sev}
            save_model(label, f"{col} ordinal severity", dt, sub[FEAT].to_numpy(float),
                       sub[col].map({lv: i for i, lv in enumerate(levels)}).to_numpy(float),
                       classes=levels, regressor=True)
            o = sev["optimistic_grouped_cv"]; h = sev["leakage_honest_holdout"]
            print(f"  {label:18s} {dt}: ρ={o['spearman']:.2f} within1={o['within_1']:.2f} "
                  f"| honest(hold {h['held_out_levels']}) within1={h['within_1']:.2f}")

    # binary restriction detectors (the strong, balanced framing) per device type
    RESULTS["restriction_detect"] = {}
    for name, col in [("suction_present", "valveIn"), ("discharge_present", "valveOut")]:
        RESULTS["restriction_detect"][name] = {}
        for dt in ["mic", "cam"]:
            sub = m1[m1.dev_type == dt]
            X = StandardScaler().fit_transform(sub[FEAT].to_numpy(float))
            y = (sub[col] > 1).astype(int).to_numpy()
            yp = cross_val_predict(rf_clf(), X, y, cv=GroupKFold(5), groups=cfg_groups(sub))
            RESULTS["restriction_detect"][name][dt] = {
                "n_pos": int(y.sum()), "accuracy": round(accuracy_score(y, yp), 3),
                "f1": round(f1_score(y, yp, zero_division=0), 3),
                "precision": round(precision_score(y, yp, zero_division=0), 3),
                "recall": round(recall_score(y, yp, zero_division=0), 3)}
            save_model(name, f"{col}>1 binary detect", dt, sub[FEAT].to_numpy(float), y,
                       classes=[0, 1])
        a = RESULTS["restriction_detect"][name]
        print(f"  {name:18s} mic F1={a['mic']['f1']:.2f}  cam F1={a['cam']['f1']:.2f}")

    # restriction location (4-class) per device type
    def loc(r):
        s, d = r.valveIn > 1, r.valveOut > 1
        return "both" if s and d else "suction" if s else "discharge" if d else "open"
    m1 = m1.assign(rloc=m1.apply(loc, axis=1))
    labels = ["open", "suction", "discharge", "both"]
    RESULTS["restriction_location"] = {}
    for dt in ["mic", "cam"]:
        sub = m1[m1.dev_type == dt]
        X = StandardScaler().fit_transform(sub[FEAT].to_numpy(float))
        y = sub.rloc.to_numpy()
        yp = cross_val_predict(rf_clf(), X, y, cv=GroupKFold(5), groups=cfg_groups(sub))
        RESULTS["restriction_location"][dt] = {
            "n": int(len(sub)), "accuracy": round(accuracy_score(y, yp), 3),
            "balanced_accuracy": round(balanced_accuracy_score(y, yp), 3),
            "macro_f1": round(f1_score(y, yp, average="macro", labels=labels), 3),
            "per_class_f1": {c: round(f1_score(y == c, yp == c, zero_division=0), 3) for c in labels}}
        print(f"  restriction_location {dt}: acc={RESULTS['restriction_location'][dt]['accuracy']:.3f} "
              f"bal-acc={RESULTS['restriction_location'][dt]['balanced_accuracy']:.3f}")
        if dt == "cam":
            confusion_fig(y, yp, labels, f"restriction location ({dt}, acc={accuracy_score(y,yp):.2f})",
                          os.path.join(ROOT, "fig_model_restriction_location.png"))
        save_model("restriction_location", "open/suction/discharge/both", dt,
                   sub[FEAT].to_numpy(float), y, classes=labels)


# ------------------------------------------------------------ equipment models
def equipment(df):
    d = df[df.folder == "testbedmotor5_25wav"].copy()
    print(f"\n### EQUIPMENT / MOTOR-ID models (5_25 clean, per device type) ###")
    machines = ["M2", "M3", "M4"]

    # motor presence (multi-label) per device type
    RESULTS["motor_presence"] = {}
    for dt in ["mic", "cam"]:
        s = d[d.dev_type == dt]
        X = StandardScaler().fit_transform(s[FEAT].to_numpy(float))
        Y = s[machines].to_numpy(int)
        yp = cross_val_predict(MultiOutputClassifier(rf_clf()), X, Y,
                               cv=GroupKFold(5), groups=s.valveOut.to_numpy())
        per = {m: {"f1": round(f1_score(Y[:, i], yp[:, i], zero_division=0), 3),
                   "precision": round(precision_score(Y[:, i], yp[:, i], zero_division=0), 3),
                   "recall": round(recall_score(Y[:, i], yp[:, i], zero_division=0), 3)}
               for i, m in enumerate(machines)}
        RESULTS["motor_presence"][dt] = {"per_machine": per,
            "subset_accuracy": round(float(np.mean((yp == Y).all(1))), 3)}
        print(f"  motor_presence {dt}: " + " ".join(f"{m}_F1={per[m]['f1']:.2f}" for m in machines))
        save_model("motor_presence", "multi-label M2/M3/M4", dt,
                   s[FEAT].to_numpy(float), Y, classes=machines, multilabel=True)

    # which single motor (isolation) per device type
    iso = d[(d.aeration == 0)].copy()
    iso = iso[iso[machines].sum(axis=1) <= 1]
    iso["which"] = np.where(iso.M2 == 1, "M2", np.where(iso.M3 == 1, "M3",
                            np.where(iso.M4 == 1, "M4", "none")))
    labels = ["none", "M2", "M3", "M4"]
    RESULTS["which_motor"] = {}
    for dt in ["mic", "cam"]:
        s = iso[iso.dev_type == dt]
        X = StandardScaler().fit_transform(s[FEAT].to_numpy(float))
        y = s.which.to_numpy()
        yp = cross_val_predict(rf_clf(), X, y, cv=GroupKFold(5), groups=s.valveOut.to_numpy())
        RESULTS["which_motor"][dt] = {"n": int(len(s)),
            "accuracy": round(accuracy_score(y, yp), 3),
            "balanced_accuracy": round(balanced_accuracy_score(y, yp), 3),
            "per_class_f1": {c: round(f1_score(y == c, yp == c, zero_division=0), 3) for c in labels}}
        print(f"  which_motor {dt}: acc={RESULTS['which_motor'][dt]['accuracy']:.3f} "
              f"per-class F1={RESULTS['which_motor'][dt]['per_class_f1']}")
        if dt == "mic":
            confusion_fig(y, yp, labels, f"which motor running ({dt}, acc={accuracy_score(y,yp):.2f})",
                          os.path.join(ROOT, "fig_model_which_motor.png"))
        save_model("which_motor", "none/M2/M3/M4 isolation", dt,
                   s[FEAT].to_numpy(float), y, classes=labels)

    # aeration (FLAGGED)
    aer = d[d.aeration == 1].rms_db.mean(); off = d[d.aeration == 0].rms_db.mean()
    RESULTS["aeration"] = {
        "n_on_sessions": int((d.aeration == 1).sum()),
        "on_rms_db": round(float(aer), 1), "off_rms_db": round(float(off), 1),
        "CAVEAT": ("Single valve config (vin1/vout1) and a uniform level offset; "
                   "cannot be cross-validated and likely a recording artifact "
                   "(pumps idle / different gain). NOT a validated aeration detector. "
                   "Re-record with pumps running.")}
    print(f"  aeration: FLAGGED (single config, ON {aer:.0f}dB vs OFF {off:.0f}dB) -- not modelled")


def fusion_check(df):
    """Does fusing channels (mean features per session, per device type) beat the
    best single channel for discharge severity?"""
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)].copy()
    m1 = m1[m1.valveOut.isin(DISCHARGE_LV)]
    rank = {lv: i for i, lv in enumerate(DISCHARGE_LV)}
    print("\n### CHANNEL FUSION vs single (discharge severity) ###")
    fus = {}
    for dt in ["mic", "cam"]:
        s = m1[m1.dev_type == dt]
        agg = s.groupby(["session", "valveIn", "valveOut"])[FEAT].mean().reset_index()
        X = StandardScaler().fit_transform(agg[FEAT].to_numpy(float))
        y = agg.valveOut.map(rank).to_numpy(float)
        g = cfg_groups(agg)
        yp = cross_val_predict(RandomForestRegressor(300, random_state=0, n_jobs=-1),
                               X, y, cv=GroupKFold(5), groups=g)
        fus[dt] = {"n_sessions": int(len(agg)),
                   "spearman": round(float(spearmanr(y, yp).correlation), 3),
                   "within_1": round(float(np.mean(np.abs(np.round(yp) - y) <= 1)), 3)}
        print(f"  fused {dt} ({len(agg)} sessions): ρ={fus[dt]['spearman']:.2f} "
              f"within1={fus[dt]['within_1']:.2f}")
    RESULTS["channel_fusion_discharge_severity"] = fus


def main():
    df = pd.read_csv(os.path.join(ROOT, "features_allch.csv"))
    channel_ranking(df)
    blockage(df)
    equipment(df)
    fusion_check(df)
    json.dump(RESULTS, open(os.path.join(ROOT, "model_results.json"), "w"), indent=2)
    print(f"\nwrote model_results.json + models/*.joblib + fig_model_*/fig_channel_ranking.png")
    print("saved models:", sorted(os.listdir(MODELDIR)))


if __name__ == "__main__":
    main()
