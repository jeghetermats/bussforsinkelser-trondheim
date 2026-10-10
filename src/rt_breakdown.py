"""
Hvor hjelper sanntid? Gevinsten til modell B over modell A fordelt på lead og hvor langt ut i turen stoppet ligger.

Hvis sanntidsfeaturene fanget det som er i ferd med å skje, burde gevinsten være størst på de første stoppene
ved kort lead. Leser reports/test_predictions.parquet (A) og reports/rt_test_predictions.parquet (B).
Finnes runde 1 (python src/train_rt.py --r1, samme datasett), vises også runde 2 mot runde 1 med samme inndeling
(tabellen "Runde 2 mot runde 1" i docs/experiments.md).

Resultat: reports/rt_breakdown.csv og reports/rt_breakdown.png
Kjør:     python src/rt_breakdown.py   (etter finalize.py, og etter train_rt.py --r1 for runde 1-tabellen)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REPORTS, period_label, round1_status

BINS = [-1, 5, 15, 30, 60, 1000]
LABELS = ["0-5", "5-15", "15-30", "30-60", "60+"]

a = pd.read_parquet(REPORTS / "test_predictions.parquet",
                    columns=["trip", "seq", "y", "pred_lgbm", "pred_baseline", "minute_of_day", "trip_start_min"])
b = pd.read_parquet(REPORTS / "rt_test_predictions.parquet", columns=["trip", "seq", "lead_min", "pred_rt"])
d = b.merge(a, on=["trip", "seq"])
use_r1, msg = round1_status(against="B")
if msg:
    print(msg)
if use_r1:
    r1 = pd.read_parquet(REPORTS / "rt_test_predictions_r1.parquet", columns=["trip", "seq", "lead_min", "pred_rt"])
    d = d.merge(r1.rename(columns={"pred_rt": "pred_r1"}), on=["trip", "seq", "lead_min"], how="left")
    if d.pred_r1.isna().any():
        print(f"Runde 1 mangler for {d.pred_r1.isna().mean():.1%} av radene - hopper over runde 1-tabellen")
        d = d.drop(columns="pred_r1")
d["min_fra_start"] = (d.minute_of_day - d.trip_start_min) % 1440
d["del"] = pd.cut(d.min_fra_start, BINS, labels=LABELS)
d["eA"] = (d.pred_lgbm - d.y).abs()
d["eB"] = (d.pred_rt - d.y).abs()
d["eBase"] = (d.pred_baseline - d.y).abs()
aggs = dict(n=("y", "size"), MAE_baseline=("eBase", "mean"), MAE_A=("eA", "mean"), MAE_B=("eB", "mean"))
if "pred_r1" in d:
    d["eR1"] = (d.pred_r1 - d.y).abs()
    aggs["MAE_B_runde1"] = ("eR1", "mean")

g = d.groupby(["lead_min", "del"], observed=True).agg(**aggs).reset_index()
g["B_mot_A_%"] = (1 - g.MAE_B / g.MAE_A) * 100
if "MAE_B_runde1" in g:
    g["B_mot_runde1_%"] = (1 - g.MAE_B / g.MAE_B_runde1) * 100
g.to_csv(REPORTS / "rt_breakdown.csv", index=False)
piv = g.pivot(index="lead_min", columns="del", values="B_mot_A_%")[LABELS]
print("Gevinst B mot A (%), rader = minutter før avgang, kolonner = minutter fra turstart til stoppet:")
print(piv.round(1).to_string())
if "B_mot_runde1_%" in g:
    print("\nRunde 2 mot runde 1 (%), samme inndeling:")
    print(g.pivot(index="lead_min", columns="del", values="B_mot_runde1_%")[LABELS].round(1).to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.2))
im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=max(6, float(np.nanmax(piv.to_numpy()))))
ax.set_xticks(range(len(LABELS)), LABELS)
ax.set_yticks(range(len(piv)), [f"{int(l)} min før" for l in piv.index])
ax.set_xlabel("Minutter fra turstart til stoppet")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.iat[i, j]
        ax.text(j, i, f"{v:.1f} %", ha="center", va="center", fontsize=10,
                color="white" if v > 3.5 else "#1A1A1A")
ax.set_title(f"Modell B mot modell A: lavere MAE (%) - test {period_label('test')}", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="%")
plt.tight_layout()
plt.savefig(REPORTS / "rt_breakdown.png", dpi=130)
print("Lagret reports/rt_breakdown.csv og .png")
