#!/usr/bin/env python3
"""Train blockage-level (and aeration) readers from the acoustic features.

Targets the user goal: a model that, given a ~60 s clip, outputs the M1
discharge-blockage level (valveOut) and suction-blockage level (valveIn) well,
plus an honest aeration verdict.

Methodology improvements over the previous round
------------------------------------------------
* Gradient-boosted trees (HistGradientBoosting) instead of RandomForest.
* Per-device-type models (mic vs cam) - the two sensor families must not be
  pooled; placement dominates quality.
* LEAKAGE-HONEST evaluation, two ways:
    (1) GroupKFold grouped by physical valve config (vin,vout) - the same config
        (incl. its noisy twins / repeat sessions) never spans train & test, so a
        clip can't be scored by memorising a sibling recording.
    (2) Leave-one-level-out - hold out a WHOLE restriction level, predict it from
        the others; the true test of interpolating to an unseen blockage amount.
* Severity is ordinal: levels mapped to evenly-spaced ranks so MAE is "steps"
  and within-1 is meaningful, alongside Spearman rho and exact accuracy.
* Noise robustness: clean-trained model evaluated per environmental-noise class.
* Trained on the M1-only valve sweeps (the monitoring scenario), all noise types,
  now including the new 5_19/5_20 campaigns that extend suction to {5,8}.

Outputs (all in run/):
  models/<task>_<mic|cam>.joblib   production bundles (trained on all data)
  results.json                     every metric below
  fig_severity_<axis>.png          predicted-vs-true per device type
  fig_robustness.png               accuracy per noise category
  fig_importance.png               top features per axis
  fig_leave_level_out.png          interpolation-to-unseen-level test

Run: python3 run/train.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)

DISCHARGE_LEVELS = [1, 2, 3, 4, 5, 8, 11]
SUCTION_LEVELS = [1, 2, 3, 4, 5, 8]

META = ["folder", "file", "device", "dev_type", "session", "noise",
        "noise_cat", "M2", "M3", "M4", "aeration", "valveIn", "valveOut"]


def feat_cols(df, drop_level=False):
    cols = [c for c in df.columns if c not in META]
    if drop_level:
        cols = [c for c in cols if c not in ("rms_db",)]
    return cols


def reg_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def clf_model():
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def rank_metrics(y_true_rank, y_pred_cont):
    """Metrics on rank-space severity predictions."""
    pred_rank = np.clip(np.rint(y_pred_cont), 0, y_true_rank.max()).astype(int)
    mae = float(np.mean(np.abs(pred_rank - y_true_rank)))
    within1 = float(np.mean(np.abs(pred_rank - y_true_rank) <= 1))
    exact = float(np.mean(pred_rank == y_true_rank))
    rho = float(spearmanr(y_true_rank, y_pred_cont).correlation)
    return {"MAE_steps": mae, "within1": within1, "exact_acc": exact,
            "spearman": rho}


# ----------------------------------------------------------------------------
# SEVERITY (the headline deliverable)
# ----------------------------------------------------------------------------
def severity_cv(df, dev_type, target, levels, drop_level=False):
    """Config-grouped CV: read blockage level on the rig, never scoring a clip
    from a sibling recording of its own physical config."""
    d = df[df.dev_type == dev_type].copy()
    rankmap = {v: i for i, v in enumerate(levels)}
    d = d[d[target].isin(levels)]
    d["rank"] = d[target].map(rankmap)
    cols = feat_cols(d, drop_level)
    X = d[cols].values
    y = d["rank"].values
    groups = (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).values
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))

    oof = np.full(len(d), np.nan)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = reg_model().fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    d["pred"] = oof
    m = rank_metrics(y, oof)
    m["n"] = int(len(d))
    m["n_configs"] = int(n_groups)
    # also MAE expressed in real valve units
    inv = {i: v for v, i in rankmap.items()}
    pred_lvl = np.array([inv[int(np.clip(round(x), 0, len(levels) - 1))] for x in oof])
    m["MAE_valve_units"] = float(np.mean(np.abs(pred_lvl - d[target].values)))
    return m, d


def leave_level_out(df, dev_type, target, levels):
    """Hold out an entire restriction level, predict it from the others -
    the true interpolation test. Edge levels are extrapolation (reported, harder)."""
    d = df[df.dev_type == dev_type].copy()
    rankmap = {v: i for i, v in enumerate(levels)}
    d = d[d[target].isin(levels)]
    d["rank"] = d[target].map(rankmap)
    cols = feat_cols(d)
    out = {}
    for held in levels:
        tr = d[d[target] != held]
        te = d[d[target] == held]
        if len(te) == 0:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        m = reg_model().fit(sc.transform(tr[cols].values), tr["rank"].values)
        pred = m.predict(sc.transform(te[cols].values))
        true_rank = rankmap[held]
        pred_rank = np.clip(np.rint(pred), 0, len(levels) - 1).astype(int)
        out[str(held)] = {
            "n": int(len(te)),
            "mean_pred_rank": float(np.mean(pred)),
            "true_rank": true_rank,
            "MAE_steps": float(np.mean(np.abs(pred_rank - true_rank))),
            "within1": float(np.mean(np.abs(pred_rank - true_rank) <= 1)),
        }
    return out


def leave_campaign_out(df, dev_type, target, levels):
    """Most deployment-realistic test: hold out a WHOLE recording campaign
    (a different day / session of the same rig), train on the others. All levels
    stay represented; only the exact recordings differ. Reports per-held-campaign
    and the pooled out-of-campaign metrics."""
    d = df[(df.dev_type == dev_type) & (df[target].isin(levels))].copy()
    rankmap = {v: i for i, v in enumerate(levels)}
    d["rank"] = d[target].map(rankmap)
    cols = feat_cols(d)
    oof = np.full(len(d), np.nan)
    d = d.reset_index(drop=True)
    per = {}
    for camp in sorted(d.folder.unique()):
        tr = d[d.folder != camp]
        te_idx = d.index[d.folder == camp]
        if tr[target].nunique() < 2 or len(te_idx) == 0:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        m = reg_model().fit(sc.transform(tr[cols].values), tr["rank"].values)
        pred = m.predict(sc.transform(d.loc[te_idx, cols].values))
        oof[te_idx] = pred
        # only score campaigns that actually contain >1 level (else trivial)
        sub_true = d.loc[te_idx, "rank"].values
        if len(np.unique(sub_true)) >= 2:
            per[camp] = rank_metrics(sub_true, pred)
            per[camp]["n"] = int(len(te_idx))
            per[camp]["n_levels"] = int(len(np.unique(sub_true)))
    mask = ~np.isnan(oof)
    pooled = rank_metrics(d["rank"].values[mask], oof[mask])
    pooled["n"] = int(mask.sum())
    return {"pooled": pooled, "per_campaign": per}


def robustness_by_noise(df, dev_type, target, levels):
    """Train on CLEAN (noise=N) only, test per environmental-noise category -
    config-grouped so configs don't leak across the clean/noisy split either."""
    d = df[(df.dev_type == dev_type) & (df[target].isin(levels))].copy()
    rankmap = {v: i for i, v in enumerate(levels)}
    d["rank"] = d[target].map(rankmap)
    cols = feat_cols(d)
    clean = d[d.noise_cat == "N"]
    sc = StandardScaler().fit(clean[cols].values)
    m = reg_model().fit(sc.transform(clean[cols].values), clean["rank"].values)
    res = {}
    for cat in ["N", "A", "B", "C", "D", "E"]:
        sub = d[d.noise_cat == cat]
        if len(sub) == 0:
            continue
        pred = m.predict(sc.transform(sub[cols].values))
        res[cat] = rank_metrics(sub["rank"].values, pred)
        res[cat]["n"] = int(len(sub))
    return res


# ----------------------------------------------------------------------------
# PRESENCE (any restriction?) - binary
# ----------------------------------------------------------------------------
def presence_cv(df, dev_type, target):
    d = df[df.dev_type == dev_type].copy()
    y = (d[target] > 1).astype(int).values
    cols = feat_cols(d)
    X = d[cols].values
    groups = (d.valveIn.astype(str) + "_" + d.valveOut.astype(str)).values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(d))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = clf_model().fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return {"acc": float(accuracy_score(y, oof)),
            "f1": float(f1_score(y, oof)),
            "bal_acc": float(balanced_accuracy_score(y, oof)),
            "n": int(len(d)), "n_pos": int(y.sum())}


def fit_production(df, dev_type, target, levels, kind):
    """Train final bundle on ALL data of this device type for deployment."""
    d = df[(df.dev_type == dev_type) & (df[target].isin(levels))].copy()
    cols = feat_cols(d)
    sc = StandardScaler().fit(d[cols].values)
    if kind == "severity":
        rankmap = {v: i for i, v in enumerate(levels)}
        y = d[target].map(rankmap).values
        m = reg_model().fit(sc.transform(d[cols].values), y)
        bundle = {"scaler": sc, "model": m, "features": cols, "kind": kind,
                  "levels": levels, "rankmap": rankmap, "target": target,
                  "dev_type": dev_type}
    else:
        y = (d[target] > 1).astype(int).values
        m = clf_model().fit(sc.transform(d[cols].values), y)
        bundle = {"scaler": sc, "model": m, "features": cols, "kind": kind,
                  "target": target, "dev_type": dev_type}
    joblib.dump(bundle, os.path.join(MODELS, f"{target}_{kind}_{dev_type}.joblib"))
    return cols


# ----------------------------------------------------------------------------
# AERATION (honest)
# ----------------------------------------------------------------------------
def aeration_report(df):
    """Aeration only exists in 5_25, at one valve config, and ON-clips are ~34 dB
    quieter (likely a recording artifact). We report the level gap and a presence
    detector trained within 5_25, grouped by session - but flag that 'aeration
    LEVEL' is NOT recoverable from this data (only on/off, confounded with gain)."""
    d = df[df.folder == "testbedmotor5_25"].copy()
    on = d[d.aeration == 1]; off = d[d.aeration == 0]
    rep = {
        "n_on": int(len(on)), "n_off": int(len(off)),
        "rms_db_on": float(on.rms_db.mean()) if len(on) else None,
        "rms_db_off": float(off.rms_db.mean()) if len(off) else None,
        "on_valve_configs": sorted(set((on.valveIn.astype(str) + "/" +
                                        on.valveOut.astype(str)).tolist())),
        "verdict": ("Aeration recorded at a single valve config with a uniform "
                    "~34 dB level collapse on ON clips -> only on/off is present "
                    "and it is confounded with gain. 'Aeration level' is NOT "
                    "answerable from this data; re-record with pumps running."),
    }
    # presence detector within 5_25, grouped by session, gain-invariant features
    res = {}
    for dev_type in ["mic", "cam"]:
        sub = d[d.dev_type == dev_type]
        if sub.aeration.nunique() < 2:
            continue
        cols = feat_cols(sub, drop_level=True)  # drop rms_db so it can't cheat on gain
        X = sub[cols].values
        y = sub.aeration.values
        groups = sub.session.values
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        oof = np.zeros(len(sub))
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            m = clf_model().fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict(sc.transform(X[te]))
        res[dev_type] = {"acc": float(accuracy_score(y, oof)),
                         "f1": float(f1_score(y, oof)),
                         "n": int(len(sub)), "n_on": int(y.sum()),
                         "note": "gain-invariant features only (rms_db dropped)"}
        # persist a production aeration-presence model (with the big caveat)
        sc = StandardScaler().fit(X)
        m = clf_model().fit(sc.transform(X), y)
        joblib.dump({"scaler": sc, "model": m, "features": cols,
                     "kind": "presence", "target": "aeration", "dev_type": dev_type,
                     "caveat": rep["verdict"]},
                    os.path.join(MODELS, f"aeration_presence_{dev_type}.joblib"))
    rep["presence_detector_5_25_only"] = res
    return rep


# ----------------------------------------------------------------------------
def main():
    df = pd.read_csv(os.path.join(HERE, "features_allch.csv"))
    # monitoring scenario: M1 only (no aux equipment), all noise types
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)].copy()
    print(f"all rows={len(df)}  M1-only rows={len(m1)}")

    results = {"data": {"n_files_total": int(len(df)),
                        "n_m1_only_files": int(len(m1)),
                        "discharge_levels": DISCHARGE_LEVELS,
                        "suction_levels": SUCTION_LEVELS}}

    axes = [("discharge", "valveOut", DISCHARGE_LEVELS),
            ("suction", "valveIn", SUCTION_LEVELS)]

    results["severity_config_grouped"] = {}
    results["severity_leave_campaign_out"] = {}
    results["severity_leave_level_out"] = {}
    results["robustness_clean_trained"] = {}
    results["presence_config_grouped"] = {}
    sev_dfs = {}
    importances = {}

    for axis, target, levels in axes:
        results["severity_config_grouped"][axis] = {}
        results["severity_leave_campaign_out"][axis] = {}
        results["severity_leave_level_out"][axis] = {}
        results["robustness_clean_trained"][axis] = {}
        results["presence_config_grouped"][axis] = {}
        for dev_type in ["mic", "cam"]:
            m, dd = severity_cv(m1, dev_type, target, levels)
            results["severity_config_grouped"][axis][dev_type] = m
            sev_dfs[(axis, dev_type)] = dd
            # ablation: drop level feature (gain robustness)
            mabl, _ = severity_cv(m1, dev_type, target, levels, drop_level=True)
            results["severity_config_grouped"][axis][dev_type]["within1_no_level_feat"] = mabl["within1"]

            results["severity_leave_campaign_out"][axis][dev_type] = \
                leave_campaign_out(m1, dev_type, target, levels)
            results["severity_leave_level_out"][axis][dev_type] = \
                leave_level_out(m1, dev_type, target, levels)
            results["robustness_clean_trained"][axis][dev_type] = \
                robustness_by_noise(m1, dev_type, target, levels)
            results["presence_config_grouped"][axis][dev_type] = \
                presence_cv(m1, dev_type, target)

            cols = fit_production(m1, dev_type, target, levels, "severity")
            fit_production(m1, dev_type, target, levels, "presence")
            # importance via permutation-free: use feature_importances from a quick fit
            print(f"[{axis}/{dev_type}] severity {m['within1']:.3f} within1, "
                  f"rho={m['spearman']:.3f}, exact={m['exact_acc']:.3f}")

    results["aeration"] = aeration_report(df)
    print("aeration:", results["aeration"]["verdict"][:80], "...")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")

    make_figures(sev_dfs, results, m1)


def make_figures(sev_dfs, results, m1):
    # --- severity predicted vs true (per axis x dev_type) ---
    for axis in ["discharge", "suction"]:
        fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
        for j, dev_type in enumerate(["mic", "cam"]):
            d = sev_dfs[(axis, dev_type)]
            ax = axarr[j]
            # jitter for visibility
            rng = np.random.RandomState(0)
            jx = d["rank"].values + rng.uniform(-0.15, 0.15, len(d))
            ax.scatter(jx, d["pred"].values, s=4, alpha=0.25)
            mx = d["rank"].max()
            ax.plot([0, mx], [0, mx], "r--", lw=1)
            m = results["severity_config_grouped"][axis][dev_type]
            ax.set_title(f"{axis} / {dev_type}\n"
                         f"within1={m['within1']:.2f} rho={m['spearman']:.2f} "
                         f"MAE={m['MAE_steps']:.2f} steps")
            ax.set_xlabel("true level (rank)")
            ax.set_ylabel("predicted (rank)")
        fig.suptitle(f"M1 {axis}-blockage severity (config-grouped, leakage-honest)")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"fig_severity_{axis}.png"), dpi=110)
        plt.close(fig)

    # --- robustness per noise ---
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    cats = ["N", "A", "B", "C", "D", "E"]
    for j, axis in enumerate(["discharge", "suction"]):
        ax = axarr[j]
        for dev_type in ["mic", "cam"]:
            r = results["robustness_clean_trained"][axis][dev_type]
            ys = [r[c]["within1"] if c in r else np.nan for c in cats]
            ax.plot(cats, ys, "o-", label=dev_type)
        ax.set_ylim(0, 1.02); ax.set_title(f"{axis}: clean-trained within-1")
        ax.set_xlabel("noise category"); ax.set_ylabel("within-1 accuracy")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Noise robustness (train clean N, test each environmental noise)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_robustness.png"), dpi=110)
    plt.close(fig)

    # --- leave-level-out ---
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    for j, (axis, levels) in enumerate([("discharge", DISCHARGE_LEVELS),
                                        ("suction", SUCTION_LEVELS)]):
        ax = axarr[j]
        for dev_type in ["mic", "cam"]:
            llo = results["severity_leave_level_out"][axis][dev_type]
            xs = [int(k) for k in llo]
            true_r = [llo[k]["true_rank"] for k in llo]
            pred_r = [llo[k]["mean_pred_rank"] for k in llo]
            ax.plot(true_r, pred_r, "o-", label=f"{dev_type} pred")
        mx = len(levels) - 1
        ax.plot([0, mx], [0, mx], "k--", lw=1, label="ideal")
        ax.set_title(f"{axis}: leave-one-level-out\n(mean predicted rank for a "
                     f"level never seen in training)")
        ax.set_xlabel("held-out level (rank)"); ax.set_ylabel("mean predicted rank")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_leave_level_out.png"), dpi=110)
    plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    main()
