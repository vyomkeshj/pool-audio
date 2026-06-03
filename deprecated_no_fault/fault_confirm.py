#!/usr/bin/env python3
"""Confirm which aux pump is damaged, using correct broken-fan physics:
a damaged impeller/fan injects BROADBAND mid-high-frequency turbulence noise
(a 'haystack'), not impulsiveness (broadband noise actually LOWERS kurtosis).

Reloads fault_features_525.csv (no re-extraction). Tests matched-pair
consistency of the broadband increment per pump, split by device type.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "fault_features_525.csv"))
df = df[df.aeration == 0]                     # drop aeration to avoid confound
PSD_EDGES = np.logspace(np.log10(20), np.log10(22050), 81)
PSD_CTR = np.sqrt(PSD_EDGES[:-1] * PSD_EDGES[1:])
psd_cols = [f"psd{i}" for i in range(len(PSD_CTR))]
PUMPS = ["M2", "M3", "M4"]

# band index helpers
def band_idx(lo, hi):
    return [i for i, fc in enumerate(PSD_CTR) if lo <= fc < hi]
HAY = band_idx(3000, 9000)        # the haystack band
MID = band_idx(500, 9000)

def band_db(row_psd, idxs):
    # row_psd in dB; convert to linear mean then back
    lin = 10 ** (np.asarray(row_psd)[idxs] / 10)
    return 10 * np.log10(lin.mean() + 1e-20)

df["hay_db"] = df[psd_cols].apply(lambda r: band_db(r.values, HAY), axis=1)

# ---- matched-pair contrast per pump ----
# group by the OTHER variables + device; require both on & off present.
print("Matched-pair broadband-haystack (3-9 kHz) increment, pump ON - OFF\n")
summary = {}
for pmp in PUMPS:
    others = [p for p in PUMPS if p != pmp] + ["valveOut", "device"]
    diffs = []
    for key, g in df.groupby(others):
        on = g[g[pmp] == 1]["hay_db"]
        off = g[g[pmp] == 0]["hay_db"]
        if len(on) and len(off):
            diffs.append(on.mean() - off.mean())
    diffs = np.array(diffs)
    summary[pmp] = {
        "mean_dB": round(float(diffs.mean()), 3),
        "median_dB": round(float(np.median(diffs)), 3),
        "frac_positive": round(float(np.mean(diffs > 0)), 3),
        "n_pairs": int(len(diffs)),
        "cohens_d": round(float(diffs.mean() / (diffs.std() + 1e-9)), 2),
    }
    print(f"{pmp}: mean +{diffs.mean():.2f} dB  median +{np.median(diffs):.2f} dB  "
          f"frac_pos={np.mean(diffs>0):.2f}  d={diffs.mean()/(diffs.std()+1e-9):.2f}  "
          f"(n={len(diffs)} matched pairs)")

# ---- device-type split (does the haystack appear on both mic & cam?) ----
print("\nHaystack increment by device type (mean ON-OFF, dB):")
dev_split = {}
for pmp in PUMPS:
    dev_split[pmp] = {}
    for dt in ["mic", "cam"]:
        sub = df[df.devtype == dt]
        inc = sub[sub[pmp] == 1]["hay_db"].mean() - sub[sub[pmp] == 0]["hay_db"].mean()
        dev_split[pmp][dt] = round(float(inc), 3)
    print(f"  {pmp}: mic +{dev_split[pmp]['mic']:.2f}   cam +{dev_split[pmp]['cam']:.2f}")

# ---- figure: per-pump haystack increment distribution + RMS increment ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
data = []
for pmp in PUMPS:
    others = [p for p in PUMPS if p != pmp] + ["valveOut", "device"]
    d = []
    for key, g in df.groupby(others):
        on = g[g[pmp] == 1]["hay_db"]; off = g[g[pmp] == 0]["hay_db"]
        if len(on) and len(off):
            d.append(on.mean() - off.mean())
    data.append(d)
bp = ax[0].boxplot(data, labels=PUMPS, showmeans=True)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_ylabel("Δ 3–9 kHz band level, ON − OFF (dB)")
ax[0].set_title("Broadband-haystack increment per pump\n(matched pairs)")
ax[0].grid(axis="y", alpha=0.3)

rms_inc = [df[df[p] == 1]["rms_db"].mean() - df[df[p] == 0]["rms_db"].mean() for p in PUMPS]
bb_inc = [df[df[p] == 1]["bb_2_16k_db"].mean() - df[df[p] == 0]["bb_2_16k_db"].mean() for p in PUMPS]
x = np.arange(len(PUMPS))
ax[1].bar(x - 0.2, rms_inc, 0.4, label="overall RMS")
ax[1].bar(x + 0.2, bb_inc, 0.4, label="broadband 2–16 kHz (rel.)")
ax[1].set_xticks(x); ax[1].set_xticklabels(PUMPS)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("Δ level when ON (dB)")
ax[1].set_title("Loudness & broadband increment per pump")
ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "fig_pump_haystack.png"), dpi=120)
plt.close()

worst = max(summary, key=lambda p: summary[p]["mean_dB"])
out = {"metric": "broadband haystack (3-9 kHz) increment, matched pairs, aeration=0",
       "per_pump": summary, "device_split": dev_split,
       "damaged_pump": worst,
       "reasoning": ("A broken fan/impeller injects broadband mid-HF turbulence "
                     "noise. The pump whose activation adds a large, consistent "
                     "3-9 kHz haystack on both mic and cam is the damaged one.")}
json.dump(out, open(os.path.join(ROOT, "fault_id_result.json"), "w"), indent=2)
print(f"\n==> DAMAGED PUMP: {worst}")
print("wrote fault_id_result.json (updated), fig_pump_haystack.png")
