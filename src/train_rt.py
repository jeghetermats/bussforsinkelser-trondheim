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
       SEEDS=42,1,2 (miljøvariabel, standard): modell B = snitt av én modell per seed
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
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,1,2").split(",")]
ensure_dirs()

# Prognosene fra forrige versjon av B (runde 1) tas vare på for sammenligning på de samme radene
R1_PATH = REPORTS / "rt_test_predictions_r1.parquet"
if not QUICK and not NO_RT and not R1_PATH.exists() and (REPORTS / "rt_test_predictions.parquet").exists():
    import shutil
    shutil.copy(REPORTS / "rt_test_predictions.parquet", R1_PATH)

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
k_best = min(grid, key=lambda k, v=valid: np.mean(np.abs(rule(v, k, m_anom) - v.y)))
print(f"Regel: k = {k_best} (valgt på valid), typisk avvik m = {m_anom:.1f} s")

# ---------- Én modell per seed: early stopping på valid, retrening på jan-okt ----------
full = use_refit_history(pd.concat([train, valid], ignore_index=True))
pred_seeds, best_iters, valid_maes = [], [], []
for seed in SEEDS:
    t0 = time.time()
    p_seed = dict(params, seed=seed)
    dtrain = lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT)
    m = lgb.train(p_seed, dtrain, num_boost_round=MAX_ROUNDS,
                  valid_sets=[lgb.Dataset(valid[FEATURES], valid.y, reference=dtrain)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    best = m.best_iteration
    valid_maes.append(metrics(valid.y, m.predict(valid[FEATURES], num_iteration=best))["MAE_s"])
    del dtrain, m
    m = lgb.train(p_seed, lgb.Dataset(full[FEATURES], full.y, categorical_feature=CAT), num_boost_round=best)
    if seed == SEEDS[0]:
        m.save_model(str(MODEL_PATH))
        model = m
    pred_seeds.append(m.predict(test[FEATURES]))
    best_iters.append(int(best))
    print(f"Seed {seed}: {best} trær, valid-MAE {valid_maes[-1]:.2f} s  ({time.time()-t0:.0f}s)", flush=True)
best_iter, valid_mae = best_iters[0], float(np.mean(valid_maes))
del full, train, valid

# ---------- Evaluering per lead ----------
a_meta = json.loads(A_PATH.with_suffix(".json").read_text(encoding="utf-8"))
model_a = lgb.Booster(model_file=str(A_PATH))
t0 = time.time()
pred = {
    "Baseline": test.baseline.to_numpy("float64"),
    "Modell A (før dagen)": model_a.predict(test[a_meta["features"]]),
    "Baseline + avvik siste time": rule(test, k_best, m_anom),
    "Modell B (sanntid)": np.mean(pred_seeds, axis=0),
}
if len(SEEDS) > 1:
    pred[f"Modell B, én seed ({SEEDS[0]})"] = pred_seeds[0]
# Tur-id som i test_predictions.parquet (brukes også til prognosefilen til appen)
trip_ids = pd.read_parquet(DATA / "test.parquet", columns=["journey_id"]).journey_id
trip_map = pd.Series(np.arange(trip_ids.nunique(), dtype="int32"), index=pd.unique(trip_ids))
test_trip = test.journey_id.map(trip_map).astype("Int32")
# Forrige versjon av B (runde 1), koblet på tur, stopp og lead
if R1_PATH.exists() and not NO_RT:
    r1 = pd.read_parquet(R1_PATH, columns=["trip", "seq", "lead_min", "pred_rt"])
    key = pd.DataFrame({"trip": test_trip, "seq": test.seq.astype("int16"), "lead_min": test.lead_min.astype("int8")})
    merged = key.merge(r1.astype({"seq": "int16", "lead_min": "int8"}), on=["trip", "seq", "lead_min"], how="left")
    if merged.pred_rt.notna().all():
        pred["Modell B runde 1"] = merged.pred_rt.to_numpy("float64")
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
    if "Modell B runde 1" in pred:
        boot[int(lead)]["B_mot_runde1"] = bootstrap_mae_gain(dates, y, pred["Modell B (sanntid)"][m],
                                                             pred["Modell B runde 1"][m])
res = pd.DataFrame(rows)
res.to_csv(REPORTS / f"rt_results{SUFFIX}.csv", index=False)
info = {"quick_mode": QUICK, "test_periode": "2025-11-01 - 2025-12-31", "n_test_per_lead": int((test.lead_min == 10).sum()),
        "best_iter": int(best_iter), "valid_MAE_s": valid_mae, "regel_k": float(k_best), "regel_m": m_anom,
        "seeds": SEEDS, "trær_per_seed": best_iters, "valid_MAE_per_seed": valid_maes,
        "bootstrap": boot, "features": FEATURES}
(REPORTS / f"rt_metrics{SUFFIX}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nMAE (s) per lead, test nov-des 2025:")
print(res.pivot(index="metode", columns="lead_min", values="MAE_s").loc[list(pred)].round(1).to_string())
for lead, b in boot.items():
    print(f"  {lead:2d} min: B mot baseline {b['B_mot_baseline']['forbedring_%']:.1f} % "
          f"[{b['B_mot_baseline']['ci95_lav_%']:.1f}, {b['B_mot_baseline']['ci95_høy_%']:.1f}], "
          f"B mot A {b['B_mot_A']['forbedring_%']:.1f} % [{b['B_mot_A']['ci95_lav_%']:.1f}, {b['B_mot_A']['ci95_høy_%']:.1f}], "
          f"B mot regel {b['B_mot_regel']['forbedring_%']:.1f} %" +
          (f", B mot runde 1 {b['B_mot_runde1']['forbedring_%']:.1f} % [{b['B_mot_runde1']['ci95_lav_%']:.1f}, "
           f"{b['B_mot_runde1']['ci95_høy_%']:.1f}]" if "B_mot_runde1" in b else ""))

# ---------- Feature importance ----------
imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
(imp / imp.sum() * 100).rename("gain_pct").rename_axis("feature").reset_index() \
    .to_csv(REPORTS / f"rt_feature_importance{SUFFIX}.csv", index=False)
print("\nSanntidsfeatures, andel av gain: "
      f"{imp[[c for c in RT_FEATURES if c in imp.index]].sum() / imp.sum() * 100:.0f} %")

# ---------- Prognoser til appen (samme tur-id som test_predictions.parquet) ----------
out = pd.DataFrame({"trip": test_trip,
                    "seq": test.seq.astype("int16"), "lead_min": test.lead_min.astype("int8"),
                    "pred_rt": np.round(pred["Modell B (sanntid)"]).astype("int16"),
                    "pred_rule": np.round(pred["Baseline + avvik siste time"]).astype("int16")})
out.dropna(subset=["trip"]).to_parquet(REPORTS / f"rt_test_predictions{SUFFIX}.parquet", index=False, compression="zstd")

# ---------- Figur: MAE per lead ----------
PLOT = {"Baseline": "#8A8F98", "Modell A (før dagen)": "#6B8FB5", "Baseline + avvik siste time": "#C9A227",
        "Modell B runde 1": "#9DB9D5", "Modell B (sanntid)": "#1F4E79"}
cols = [c for c in PLOT if c in pred]
piv = res.pivot(index="lead_min", columns="metode", values="MAE_s")[cols]
ax = piv.plot.bar(figsize=(8, 4), color=[PLOT[c] for c in cols], rot=0)
ax.set_xlabel("Minutter før avgang"); ax.set_ylabel("MAE (sekunder)")
ax.set_ylim(piv.min().min() * 0.9, piv.max().max() * 1.02)
ax.set_title("Feil per prognosetidspunkt - test nov-des 2025"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(REPORTS / f"rt_mae_by_lead{SUFFIX}.png", dpi=130); plt.close()
print(f"\nLagret i reports/ (suffiks: '{SUFFIX}')")
