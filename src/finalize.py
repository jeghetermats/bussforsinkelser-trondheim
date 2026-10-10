"""
Endelige modeller og tall til README og appen.

  Modell A (endelig): snitt av 3 seeds + blanding med baselinen, w valgt på valid (src/ensemble.py)
  Modell B (endelig): runde 2-features, snitt av 3 seeds (src/train_rt.py)

Skriver A-prognosen inn i reports/test_predictions.parquet (kolonnen pred_lgbm, som appen bruker; enkeltmodellen
beholdes som pred_a_single) og lager reports/final.json og reports/final_results.csv med KI (bootstrap over dager).
I tillegg: KI med hele uker som trekkenhet (*_ukeblokker) og, hvis datasettene har den, A mot medianen fra bare
de siste 8 ukene (*_8uker), en kontroll på hvor mye av gevinsten som bare er å følge nivået.

Kjør:  python src/finalize.py    (etter ensemble.py og train_rt.py)
"""
import json
import sys

import numpy as np
import pandas as pd

from common import (DATA, REPORTS, bootstrap_mae_gain, fingerprint, load_cached, metrics, read_parquet_optional,
                    replace_when_done, round1_status)

ens = json.loads((REPORTS / "ensemble.json").read_text(encoding="utf-8"))["A"]
w, seeds = ens["w_valgt_på_valid"], ens["seeds"]

t = pd.read_parquet(REPORTS / "test_predictions.parquet")
base = read_parquet_optional(DATA / "test.parquet", ["baseline", "y"], optional=["baseline_recent"])
if len(base) != len(t) or not (base.y.to_numpy() == t.y.to_numpy()).all():
    sys.exit("data/test.parquet og reports/test_predictions.parquet har ikke de samme radene i samme rekkefølge. "
             "Kjør python src/train.py og src/ensemble.py på nytt.")
# Seed-prognosene kobles på radene etter posisjon, så de må være laget fra akkurat denne test.parquet
runs = [load_cached(DATA / "ens" / f"A_s{s}.npz", "src/ensemble.py", {"test": len(base)}, fp_test=fingerprint(base))
        for s in seeds]
a_single = runs[0]["test"].astype("float64")
a_final = w * np.mean([r["test"] for r in runs], axis=0) + (1 - w) * base.baseline.to_numpy("float64")

y = t.y.to_numpy("float64")
base_pred = base.baseline.to_numpy("float64")
dates = pd.to_datetime(t.date).to_numpy()
out = {"A": {"w": w, "seeds": seeds,
             "metrics": metrics(y, a_final), "metrics_én_seed": metrics(y, a_single), "metrics_baseline": metrics(y, base_pred),
             "mot_baseline": bootstrap_mae_gain(dates, y, a_final, base_pred),
             "mot_én_seed": bootstrap_mae_gain(dates, y, a_final, a_single),
             "mot_baseline_ukeblokker": bootstrap_mae_gain(dates, y, a_final, base_pred, block="week")}}
rows = [{"lead": "dagen før", "metode": "Historisk median", **metrics(y, base_pred)},
        {"lead": "dagen før", "metode": "Modell A", **metrics(y, a_final)}]
if "baseline_recent" in base:
    recent = base.baseline_recent.to_numpy("float64")
    out["A"]["metrics_baseline_8uker"] = metrics(y, recent)
    out["A"]["mot_baseline_8uker"] = bootstrap_mae_gain(dates, y, a_final, recent)
    rows.insert(1, {"lead": "dagen før", "metode": "Median siste 8 uker", **metrics(y, recent)})

# Modell B per lead, koblet på tur og stopp
t["pred_a_single"] = np.round(a_single).astype("int16")
t["pred_lgbm"] = np.round(a_final).astype("int16")
t["_a"], t["_base"] = a_final, base_pred
extra = []
if "baseline_recent" in base:
    t["_recent"] = base.baseline_recent.to_numpy("float64")
    extra = ["_recent"]
rt = pd.read_parquet(REPORTS / "rt_test_predictions.parquet", columns=["trip", "seq", "lead_min", "pred_rt"])
d = rt.merge(t[["trip", "seq", "y", "date", "_a", "_base"] + extra], on=["trip", "seq"])
assert len(d) == len(rt)
r1_path = REPORTS / "rt_test_predictions_r1.parquet"
use_r1, msg = round1_status(against="B")
if msg:
    print(msg)
if use_r1:
    r1 = pd.read_parquet(r1_path, columns=["trip", "seq", "lead_min", "pred_rt"]).rename(columns={"pred_rt": "_r1"})
    d = d.merge(r1, on=["trip", "seq", "lead_min"], how="left")
out["B"] = {}
for lead, g in d.groupby("lead_min"):
    yy, dd = g.y.to_numpy("float64"), pd.to_datetime(g.date).to_numpy()
    pb, pa, pbase = g.pred_rt.to_numpy("float64"), g._a.to_numpy(), g._base.to_numpy()
    res = {"metrics": metrics(yy, pb), "mot_baseline": bootstrap_mae_gain(dd, yy, pb, pbase),
           "mot_A": bootstrap_mae_gain(dd, yy, pb, pa),
           "mot_baseline_ukeblokker": bootstrap_mae_gain(dd, yy, pb, pbase, block="week")}
    if "_recent" in g:
        res["mot_baseline_8uker"] = bootstrap_mae_gain(dd, yy, pb, g._recent.to_numpy())
    if "_r1" in g and g._r1.notna().all():
        res["mot_runde1"] = bootstrap_mae_gain(dd, yy, pb, g._r1.to_numpy("float64"))
        res["metrics_runde1"] = metrics(yy, g._r1.to_numpy("float64"))
    out["B"][str(int(lead))] = res
    rows.append({"lead": f"{int(lead)} min før", "metode": "Modell B", **res["metrics"]})

replace_when_done(REPORTS / "test_predictions.parquet",     # filen leses og skrives av samme skript
                  lambda p: t.drop(columns=["_a", "_base"] + extra).to_parquet(p, index=False, compression="zstd"))
(REPORTS / "final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(rows).to_csv(REPORTS / "final_results.csv", index=False)

a = out["A"]
print(f"Modell A (snitt {len(seeds)} seeds + blanding w={w}): MAE {a['metrics']['MAE_s']:.2f} s, "
      f"{a['mot_baseline']['forbedring_%']:.1f} % mot baseline "
      f"[{a['mot_baseline']['ci95_lav_%']:.1f}, {a['mot_baseline']['ci95_høy_%']:.1f}], med ukeblokker "
      f"[{a['mot_baseline_ukeblokker']['ci95_lav_%']:.1f}, {a['mot_baseline_ukeblokker']['ci95_høy_%']:.1f}]")
if "mot_baseline_8uker" in a:
    b8 = a["mot_baseline_8uker"]
    print(f"Modell A mot median siste 8 uker: {b8['forbedring_%']:.1f} % [{b8['ci95_lav_%']:.1f}, {b8['ci95_høy_%']:.1f}]")
for lead, r in out["B"].items():
    print(f"Modell B {lead} min: MAE {r['metrics']['MAE_s']:.2f} s, mot baseline {r['mot_baseline']['forbedring_%']:.1f} % "
          f"[{r['mot_baseline']['ci95_lav_%']:.1f}, {r['mot_baseline']['ci95_høy_%']:.1f}], mot A "
          f"{r['mot_A']['forbedring_%']:.1f} % [{r['mot_A']['ci95_lav_%']:.1f}, {r['mot_A']['ci95_høy_%']:.1f}]"
          + (f", mot median siste 8 uker {r['mot_baseline_8uker']['forbedring_%']:.1f} %" if "mot_baseline_8uker" in r
             else ""))
print(pd.DataFrame(rows).round(1).to_string(index=False))
