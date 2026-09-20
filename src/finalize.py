"""
Endelige modeller og tall til README og appen.

  Modell A (endelig): snitt av 3 seeds + blanding med baselinen, w valgt på valid (src/ensemble.py)
  Modell B (endelig): runde 2-features, snitt av 3 seeds (src/train_rt.py)

Skriver A-prognosen inn i reports/test_predictions.parquet (kolonnen pred_lgbm, som appen bruker; enkeltmodellen
beholdes som pred_a_single) og lager reports/final.json og reports/final_results.csv med KI (bootstrap over dager).

Kjør:  python src/finalize.py    (etter ensemble.py og train_rt.py)
"""
import json

import numpy as np
import pandas as pd

from common import DATA, REPORTS, bootstrap_mae_gain, metrics

ens = json.loads((REPORTS / "ensemble.json").read_text(encoding="utf-8"))["A"]
w, seeds = ens["w_valgt_på_valid"], ens["seeds"]

t = pd.read_parquet(REPORTS / "test_predictions.parquet")
base = pd.read_parquet(DATA / "test.parquet", columns=["baseline", "y"])
assert (base.y.to_numpy() == t.y.to_numpy()).all(), "test.parquet og test_predictions.parquet er ikke i samme rekkefølge"
runs = [np.load(DATA / "ens" / f"A_s{s}.npz") for s in seeds]
a_single = runs[0]["test"].astype("float64")
a_final = w * np.mean([r["test"] for r in runs], axis=0) + (1 - w) * base.baseline.to_numpy("float64")

y = t.y.to_numpy("float64")
b0 = base.baseline.to_numpy("float64")
dates = pd.to_datetime(t.date).to_numpy()
out = {"A": {"w": w, "seeds": seeds,
             "metrics": metrics(y, a_final), "metrics_én_seed": metrics(y, a_single), "metrics_baseline": metrics(y, b0),
             "mot_baseline": bootstrap_mae_gain(dates, y, a_final, b0),
             "mot_én_seed": bootstrap_mae_gain(dates, y, a_final, a_single)}}
rows = [{"lead": "dagen før", "metode": "Historisk median", **metrics(y, b0)},
        {"lead": "dagen før", "metode": "Modell A", **metrics(y, a_final)}]

# Modell B per lead, koblet på tur og stopp
t["pred_a_single"] = np.round(a_single).astype("int16")
t["pred_lgbm"] = np.round(a_final).astype("int16")
t["_a"], t["_b0"] = a_final, b0
rt = pd.read_parquet(REPORTS / "rt_test_predictions.parquet", columns=["trip", "seq", "lead_min", "pred_rt"])
d = rt.merge(t[["trip", "seq", "y", "date", "_a", "_b0"]], on=["trip", "seq"])
assert len(d) == len(rt)
r1_path = REPORTS / "rt_test_predictions_r1.parquet"
if r1_path.exists():
    r1 = pd.read_parquet(r1_path, columns=["trip", "seq", "lead_min", "pred_rt"]).rename(columns={"pred_rt": "_r1"})
    d = d.merge(r1, on=["trip", "seq", "lead_min"], how="left")
out["B"] = {}
for lead, g in d.groupby("lead_min"):
    yy, dd = g.y.to_numpy("float64"), pd.to_datetime(g.date).to_numpy()
    pb, pa, p0 = g.pred_rt.to_numpy("float64"), g._a.to_numpy(), g._b0.to_numpy()
    res = {"metrics": metrics(yy, pb), "mot_baseline": bootstrap_mae_gain(dd, yy, pb, p0),
           "mot_A": bootstrap_mae_gain(dd, yy, pb, pa)}
    if "_r1" in g and g._r1.notna().all():
        res["mot_runde1"] = bootstrap_mae_gain(dd, yy, pb, g._r1.to_numpy("float64"))
        res["metrics_runde1"] = metrics(yy, g._r1.to_numpy("float64"))
    out["B"][str(int(lead))] = res
    rows.append({"lead": f"{int(lead)} min før", "metode": "Modell B", **res["metrics"]})

t.drop(columns=["_a", "_b0"]).to_parquet(REPORTS / "test_predictions.parquet", index=False, compression="zstd")
(REPORTS / "final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(rows).to_csv(REPORTS / "final_results.csv", index=False)

a = out["A"]
print(f"Modell A (snitt {len(seeds)} seeds + blanding w={w}): MAE {a['metrics']['MAE_s']:.2f} s, "
      f"{a['mot_baseline']['forbedring_%']:.1f} % mot baseline "
      f"[{a['mot_baseline']['ci95_lav_%']:.1f}, {a['mot_baseline']['ci95_høy_%']:.1f}]")
for lead, r in out["B"].items():
    print(f"Modell B {lead} min: MAE {r['metrics']['MAE_s']:.2f} s, mot baseline {r['mot_baseline']['forbedring_%']:.1f} % "
          f"[{r['mot_baseline']['ci95_lav_%']:.1f}, {r['mot_baseline']['ci95_høy_%']:.1f}], mot A "
          f"{r['mot_A']['forbedring_%']:.1f} % [{r['mot_A']['ci95_lav_%']:.1f}, {r['mot_A']['ci95_høy_%']:.1f}]")
print(pd.DataFrame(rows).round(1).to_string(index=False))
