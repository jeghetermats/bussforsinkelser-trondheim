"""
Modell B: forsinkelse ved hvert stopp, predikert 10/30/60 min før avgang med sanntidsfeatures
(se src/realtime.py). Samme protokoll som modell A (src/train.py):
early stopping på valid (sep-okt), retrening på jan-okt med fold-historikk, test nov-des 2025.

Sammenligner per lead på nøyaktig de samme testradene:
  Baseline      : historisk median (samme som før)
  Modell A      : lagret modell fra train.py (vet ingenting om dagen i dag)
  Baseline+avvik: baseline + k * (snittavvik på linjen siste 60 min - typisk avvik), k valgt på valid
                  - den enkle regelen modell B må slå for å være verdt kompleksiteten
  Modell B      : LightGBM med alle features fra A + sanntidsfeatures + lead

Kjør:  python src/train_rt.py            (full kjøring, krever data/rt_*.parquet og modell A)
       python src/train_rt.py --quick    (mindre utvalg, skriver *_quick-filer)
       python src/train_rt.py --no-rt    (kontroll uten sanntidsfeatures, skriver *_nort-filer)
"""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CAT, DATA, LGB_PARAMS, MODELS, REPORTS, bootstrap_mae_gain, ensure_dirs, feature_columns,
                    load, metrics, set_categories, use_refit_history)

QUICK = "--quick" in sys.argv
NO_RT = "--no-rt" in sys.argv          # kontroll: samme data og protokoll, men uten sanntidsfeatures
SUFFIX = ("_quick" if QUICK else "") + ("_nort" if NO_RT else "")
MAX_ROUNDS = 100 if QUICK else 3000
MODEL_PATH = MODELS / f"lgbm_realtime{SUFFIX}.txt"
A_PATH = MODELS / "lgbm_pre_departure.txt"
ensure_dirs()

t0 = time.time()
train, valid, test = load("rt_train"), load("rt_valid"), load("rt_test")
if QUICK:
    n = int(os.environ.get("QUICK_N", 1_000_000))
    train = train.sample(min(n, len(train)), random_state=0)
    valid = valid.sample(min(n * 3 // 10, len(valid)), random_state=0)
print(f"Lest inn: train {len(train):,}  valid {len(valid):,}  test {len(test):,}  ({time.time()-t0:.0f}s)")

set_categories(train, valid, test)
FEATURES = feature_columns(train)
RT_FEATURES = [c for c in FEATURES if c.startswith("rt_")] + ["lead_min", "horizon_min"]
if NO_RT:
    FEATURES = [c for c in FEATURES if c not in RT_FEATURES]
print("Features:", FEATURES)
params = dict(LGB_PARAMS)

# ---------- Enkel regel: baseline + skalert avvik på linjen siste 60 min ----------
def rule(df, k, m):
    return df.baseline.to_numpy("float64") + k * (df.rt_ld_anom60.fillna(m).to_numpy("float64") - m)

m_anom = float(train.rt_ld_anom60.median())
grid = np.round(np.arange(0, 1.01, 0.05), 2)
k_best = min(grid, key=lambda k: np.mean(np.abs(rule(valid, k, m_anom) - valid.y)))
print(f"Regel: k = {k_best} (valgt på valid), typisk avvik m = {m_anom:.1f} s")

# ---------- 1) Early stopping på valid ----------
dtrain = lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT, free_raw_data=True)
dvalid = lgb.Dataset(valid[FEATURES], valid.y, reference=dtrain)
t0 = time.time()
model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
best_iter = model.best_iteration
valid_mae = metrics(valid.y, model.predict(valid[FEATURES], num_iteration=best_iter))["MAE_s"]
print(f"Beste antall trær: {best_iter}, valid-MAE {valid_mae:.1f} s  ({time.time()-t0:.0f}s)")
del dtrain, dvalid, model

# ---------- 2) Retrening på jan-okt ----------
full = use_refit_history(pd.concat([train, valid], ignore_index=True))
del train, valid
t0 = time.time()
model = lgb.train(params, lgb.Dataset(full[FEATURES], full.y, categorical_feature=CAT), num_boost_round=best_iter)
print(f"Endelig modell trent ({time.time()-t0:.0f}s)")
model.save_model(str(MODEL_PATH))
del full

# ---------- Evaluering per lead ----------
a_meta = json.loads(A_PATH.with_suffix(".json").read_text(encoding="utf-8"))
model_a = lgb.Booster(model_file=str(A_PATH))
t0 = time.time()
pred = {
    "Baseline": test.baseline.to_numpy("float64"),
    "Modell A (før dagen)": model_a.predict(test[a_meta["features"]]),
    "Baseline + avvik siste time": rule(test, k_best, m_anom),
    "Modell B (sanntid)": model.predict(test[FEATURES]),
}
print(f"Prediksjoner ferdig ({time.time()-t0:.0f}s)")

rows, boot = [], {}
for lead in sorted(test.lead_min.unique()):
    m = (test.lead_min == lead).to_numpy()
    y = test.y.to_numpy()[m]
    for name, p in pred.items():
        rows.append({"lead_min": int(lead), "metode": name, **metrics(y, p[m])})
    dates = test.date.to_numpy()[m]
    boot[int(lead)] = {
        "B_mot_baseline": bootstrap_mae_gain(dates, y, pred["Modell B (sanntid)"][m], pred["Baseline"][m]),
        "B_mot_A": bootstrap_mae_gain(dates, y, pred["Modell B (sanntid)"][m], pred["Modell A (før dagen)"][m]),
        "B_mot_regel": bootstrap_mae_gain(dates, y, pred["Modell B (sanntid)"][m], pred["Baseline + avvik siste time"][m]),
    }
res = pd.DataFrame(rows)
res.to_csv(REPORTS / f"rt_results{SUFFIX}.csv", index=False)
info = {"quick_mode": QUICK, "test_periode": "2025-11-01 - 2025-12-31", "n_test_per_lead": int((test.lead_min == 10).sum()),
        "best_iter": int(best_iter), "valid_MAE_s": valid_mae, "regel_k": float(k_best), "regel_m": m_anom,
        "bootstrap": boot, "features": FEATURES}
(REPORTS / f"rt_metrics{SUFFIX}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nMAE (s) per lead, test nov-des 2025:")
print(res.pivot(index="metode", columns="lead_min", values="MAE_s").loc[list(pred)].round(1).to_string())
for lead, b in boot.items():
    print(f"  {lead:2d} min: B mot baseline {b['B_mot_baseline']['forbedring_%']:.1f} % "
          f"[{b['B_mot_baseline']['ci95_lav_%']:.1f}, {b['B_mot_baseline']['ci95_høy_%']:.1f}], "
          f"B mot A {b['B_mot_A']['forbedring_%']:.1f} % [{b['B_mot_A']['ci95_lav_%']:.1f}, {b['B_mot_A']['ci95_høy_%']:.1f}], "
          f"B mot regel {b['B_mot_regel']['forbedring_%']:.1f} %")

# ---------- Feature importance ----------
imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
(imp / imp.sum() * 100).rename("gain_pct").rename_axis("feature").reset_index() \
    .to_csv(REPORTS / f"rt_feature_importance{SUFFIX}.csv", index=False)
print("\nSanntidsfeatures, andel av gain: "
      f"{imp[[c for c in RT_FEATURES if c in imp.index]].sum() / imp.sum() * 100:.0f} %")

# ---------- Prognoser til appen (samme tur-id som test_predictions.parquet) ----------
trip_ids = pd.read_parquet(DATA / "test.parquet", columns=["journey_id"]).journey_id
trip_map = pd.Series(np.arange(trip_ids.nunique(), dtype="int32"), index=pd.unique(trip_ids))
out = pd.DataFrame({"trip": test.journey_id.map(trip_map).astype("Int32"),
                    "seq": test.seq.astype("int16"), "lead_min": test.lead_min.astype("int8"),
                    "pred_rt": np.round(pred["Modell B (sanntid)"]).astype("int16"),
                    "pred_rule": np.round(pred["Baseline + avvik siste time"]).astype("int16")})
out.dropna(subset=["trip"]).to_parquet(REPORTS / f"rt_test_predictions{SUFFIX}.parquet", index=False, compression="zstd")

# ---------- Figur: MAE per lead ----------
piv = res.pivot(index="lead_min", columns="metode", values="MAE_s")[list(pred)]
ax = piv.plot.bar(figsize=(8, 4), color=["#8A8F98", "#6B8FB5", "#C9A227", "#1F4E79"], rot=0)
ax.set_xlabel("Minutter før avgang"); ax.set_ylabel("MAE (sekunder)")
ax.set_ylim(piv.min().min() * 0.9, piv.max().max() * 1.02)
ax.set_title("Feil per prognosetidspunkt - test nov-des 2025"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(REPORTS / f"rt_mae_by_lead{SUFFIX}.png", dpi=130); plt.close()
print(f"\nLagret i reports/ (suffiks: '{SUFFIX}')")
