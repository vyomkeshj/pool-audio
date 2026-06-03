#!/usr/bin/env python3
"""a) noise robustness, b) blockage-severity regression, c) multi-label model.
Reads cached features_all.csv. Writes results_abc.json + 3 figures.

Run: python3 extract_features.py && python3 analyze_abc.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             mean_absolute_error, r2_score, confusion_matrix,
                             precision_recall_fscore_support, hamming_loss)

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "features_all.csv"))
FEAT = [c for c in df.columns if c not in (
    "state", "folder", "device", "file", "noise", "noise_cat",
    "M2", "M3", "M4", "aeration", "valveIn", "valveOut",
    "suction_blockage", "discharge_blockage", "multi_pump", "aerating")]
NOISE_CATS = ["A", "B", "C", "D", "E"]
NOISE_NAME = {"N": "clean", "A": "playground", "B": "lawnmower",
              "C": "traffic", "D": "speech", "E": "music"}
results = {}


def Xy(frame, target):
    return (frame[FEAT].to_numpy(float),
            np.asarray(frame[target], dtype=object))


# ============================================================== a) ROBUSTNESS
# 3 classes that have noisy counterparts.
print("=" * 60, "\n(a) NOISE ROBUSTNESS\n", "=" * 60)
THREE = ["normal", "suction_blockage", "discharge_blockage"]
sub = df[df.state.isin(THREE)].copy()
clean = sub[sub.noise == "N"]
scaler = StandardScaler().fit(clean[FEAT].to_numpy(float))

Xc = scaler.transform(clean[FEAT].to_numpy(float))
yc = np.asarray(clean.state, object)

# baseline: clean-trained, clean-tested (5-fold CV)
base_clf = RandomForestClassifier(n_estimators=400, random_state=0,
                                  class_weight="balanced")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
yp_clean = cross_val_predict(base_clf, Xc, yc, cv=cv)
acc_clean = accuracy_score(yc, yp_clean)
print(f"clean->clean (CV) acc={acc_clean:.3f}")

# scenario A: train on ALL clean, test on each noise category
clf_clean = RandomForestClassifier(n_estimators=400, random_state=0,
                                   class_weight="balanced").fit(Xc, yc)
robustness = {"clean_cv": round(float(acc_clean), 3)}
per_cat_clean_trained = {}
for cat in NOISE_CATS:
    g = sub[sub.noise_cat == cat]
    Xg = scaler.transform(g[FEAT].to_numpy(float))
    yg = np.asarray(g.state, object)
    pred = clf_clean.predict(Xg)
    a = accuracy_score(yg, pred)
    f = f1_score(yg, pred, average="macro", labels=THREE, zero_division=0)
    per_cat_clean_trained[cat] = {"acc": round(float(a), 3),
                                  "macroF1": round(float(f), 3), "n": int(len(g))}
    print(f"  clean-trained -> {NOISE_NAME[cat]:11s} acc={a:.3f} F1={f:.3f} (n={len(g)})")

# scenario B: noise-augmented — 5-fold CV over ALL (clean+noisy), per-cat acc
allsub = sub.copy().reset_index(drop=True)
Xa = StandardScaler().fit_transform(allsub[FEAT].to_numpy(float))
ya = np.asarray(allsub.state, object)
yp_aug = cross_val_predict(
    RandomForestClassifier(n_estimators=400, random_state=0,
                           class_weight="balanced"),
    Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0))
allsub["pred_aug"] = yp_aug
per_cat_aug = {}
for cat in ["N"] + NOISE_CATS:
    m = allsub.noise_cat == cat
    a = accuracy_score(allsub.state[m], allsub.pred_aug[m])
    per_cat_aug[cat] = round(float(a), 3)
    print(f"  augmented CV -> {NOISE_NAME[cat]:11s} acc={a:.3f} (n={int(m.sum())})")

results["a_robustness"] = {
    "classes": THREE,
    "clean_cv_acc": round(float(acc_clean), 3),
    "clean_trained_on_noise": per_cat_clean_trained,
    "augmented_cv_per_cat": per_cat_aug,
}

# figure: grouped bars
fig, ax = plt.subplots(figsize=(10, 5.5))
cats = ["N"] + NOISE_CATS
x = np.arange(len(cats))
ct = [acc_clean] + [per_cat_clean_trained[c]["acc"] for c in NOISE_CATS]
au = [per_cat_aug[c] for c in cats]
ax.bar(x - 0.2, ct, 0.4, label="clean-trained model")
ax.bar(x + 0.2, au, 0.4, label="noise-augmented model (CV)")
ax.axhline(1/3, ls="--", color="gray", lw=1, label="chance (3-class)")
ax.set_xticks(x); ax.set_xticklabels([NOISE_NAME[c] for c in cats], rotation=20)
ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("(a) State-recognition robustness vs background noise\n"
             "normal / suction_blockage / discharge_blockage")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(ct):
    ax.text(i - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate(au):
    ax.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_robustness.png"), dpi=110)
plt.close()


# ============================================================== b) SEVERITY
print("\n", "=" * 60, "\n(b) BLOCKAGE-SEVERITY REGRESSION\n", "=" * 60)
results["b_severity"] = {}


def severity(name, frame, target, levels):
    rank = {lv: i for i, lv in enumerate(levels)}
    f = frame[frame[target].isin(levels)].copy()
    f["rk"] = f[target].map(rank)
    X = StandardScaler().fit_transform(f[FEAT].to_numpy(float))
    y = f["rk"].to_numpy(int)
    reg = RandomForestRegressor(n_estimators=500, random_state=0)
    yp = cross_val_predict(reg, X, y, cv=KFold(5, shuffle=True, random_state=0))
    mae = mean_absolute_error(y, yp)
    r2 = r2_score(y, yp)
    rho = spearmanr(f[target].to_numpy(float), yp).statistic
    # exact-level accuracy by rounding
    acc = accuracy_score(y, np.clip(np.round(yp), 0, len(levels) - 1).astype(int))
    within1 = float(np.mean(np.abs(np.round(yp) - y) <= 1))
    print(f"{name}: n={len(f)} levels={levels}")
    print(f"  MAE={mae:.2f} steps  R2={r2:.2f}  Spearman={rho:.2f}  "
          f"exact={acc:.2f}  within-1={within1:.2f}")
    results["b_severity"][name] = {
        "n": int(len(f)), "levels": levels, "mae_steps": round(float(mae), 3),
        "r2": round(float(r2), 3), "spearman": round(float(rho), 3),
        "exact_acc": round(float(acc), 3), "within1_acc": round(float(within1), 3)}
    return f, y, yp, levels


# discharge: M1-only, no aeration, inlet open -> isolate outlet blockage.
disch = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) &
           (df.aeration == 0) & (df.valveIn == 1)]
d_levels = [1, 2, 3, 4, 5, 8, 11]
df_d, yd, ypd, _ = severity("discharge(valveOut)", disch, "valveOut", d_levels)

# suction: M1-only, no aeration, outlet open -> isolate inlet blockage.
suct = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) &
          (df.aeration == 0) & (df.valveOut == 1)]
s_levels = [1, 2, 3, 4]
df_s, ys, yps, _ = severity("suction(valveIn)", suct, "valveIn", s_levels)

# figure: true vs predicted scatter (jittered)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
rng = np.random.default_rng(0)
for ax, (nm, f, y, yp, levels) in zip(axes, [
        ("discharge: valveOut level", df_d, yd, ypd, d_levels),
        ("suction: valveIn level", df_s, ys, yps, s_levels)]):
    jit = rng.normal(0, 0.06, len(y))
    ax.scatter(y + jit, yp, alpha=0.4, s=18)
    ax.plot([-0.5, len(levels) - 0.5], [-0.5, len(levels) - 0.5], "r--", lw=1)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
    ax.set_xlabel("true level"); ax.set_ylabel("predicted (rank)")
    r = results["b_severity"][nm.split(":")[0].strip() +
                              ("(valveOut)" if "discharge" in nm else "(valveIn)")]
    ax.set_title(f"{nm}\nMAE={r['mae_steps']:.2f} steps  ρ={r['spearman']:.2f}  "
                 f"within-1={r['within1_acc']:.2f}")
    ax.grid(alpha=0.3)
plt.suptitle("(b) Blockage-severity regression (M1-only, all noise levels)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_severity.png"), dpi=110)
plt.close()


# ============================================================== c) MULTI-LABEL
print("\n", "=" * 60, "\n(c) MULTI-LABEL (clean data)\n", "=" * 60)
LAB = ["suction_blockage", "discharge_blockage", "aerating", "multi_pump"]
c = df[df.noise == "N"].reset_index(drop=True)
Xc2 = StandardScaler().fit_transform(c[FEAT].to_numpy(float))
Y = c[LAB].to_numpy(int)
print("label prevalence (clean):", {l: int(c[l].sum()) for l in LAB},
      "| n =", len(c))

moc = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=400, random_state=0,
                           class_weight="balanced"))
Yp = cross_val_predict(moc, Xc2, Y, cv=KFold(5, shuffle=True, random_state=0))
subset_acc = float(np.mean(np.all(Yp == Y, axis=1)))
hl = hamming_loss(Y, Yp)
print(f"subset(exact-match) acc={subset_acc:.3f}  Hamming loss={hl:.3f}")
per_label = {}
for i, l in enumerate(LAB):
    p, r, f, _ = precision_recall_fscore_support(
        Y[:, i], Yp[:, i], average="binary", zero_division=0)
    per_label[l] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                    "f1": round(float(f), 3), "prevalence": int(Y[:, i].sum())}
    print(f"  {l:20s} P={p:.2f} R={r:.2f} F1={f:.2f} (n={int(Y[:,i].sum())})")
results["c_multilabel"] = {"n": int(len(c)), "subset_acc": round(subset_acc, 3),
                           "hamming_loss": round(float(hl), 3),
                           "per_label": per_label}

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(LAB))
f1s = [per_label[l]["f1"] for l in LAB]
recs = [per_label[l]["recall"] for l in LAB]
precs = [per_label[l]["precision"] for l in LAB]
ax.bar(x - 0.25, precs, 0.25, label="precision")
ax.bar(x, recs, 0.25, label="recall")
ax.bar(x + 0.25, f1s, 0.25, label="F1")
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={per_label[l]['prevalence']})" for l in LAB],
                   fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel("score")
ax.set_title(f"(c) Multi-label condition detection (clean, 5-fold CV)\n"
             f"exact-match acc={subset_acc:.2f}, Hamming loss={hl:.2f}")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_multilabel.png"), dpi=110)
plt.close()

json.dump(results, open(os.path.join(ROOT, "results_abc.json"), "w"), indent=2)
print("\nwrote results_abc.json + fig_robustness.png, fig_severity.png, "
      "fig_multilabel.png")
