"""
Trener og evaluerer modellen for forsinkelse før avgang.

  Baseline 0: median forsinkelse i treningsdata (én konstant)
  Baseline 1: historisk median per (linje, stopp, retning, time, dagtype), med fallback
              - fra alle rader før radens måned (oppdatert månedlig), ferdig beregnet i features.py
  Modell    : LightGBM (L1-tap) med kalender, vær og historiske features (fold-historikk, se history.py).
              Early stopping på valid (sep-okt), deretter retrent på jan-okt med fold-historikk over
              jan-okt, og testet på nov-des 2025 med historikk kun fra data t.o.m. oktober.

Kjør:  python src/train.py               (full kjøring)
       python src/train.py --quick       (rask test på et mindre utvalg)
       python src/train.py --eval-only   (evaluerer lagret modell uten å trene)
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

from common import (CAT, LGB_PARAMS, MODELS, REPORTS, bootstrap_mae_gain, ensure_dirs, feature_columns, load,
                    metrics, set_categories, use_refit_history)

QUICK = "--quick" in sys.argv
EVAL_ONLY = "--eval-only" in sys.argv
SUFFIX = "_quick" if QUICK else ""        # hurtigkjøring skal ikke overskrive de ekte resultatene
MAX_ROUNDS = 100 if QUICK else 3000
MODEL_PATH = MODELS / f"lgbm_pre_departure{SUFFIX}.txt"
META_PATH = MODEL_PATH.with_suffix(".json")   # best_iter og valideringsfeil, så --eval-only har dem
ensure_dirs()

t0 = time.time()
train, valid, test = load("train"), load("valid"), load("test")
if QUICK:
    n = int(os.environ.get("QUICK_N", 1_000_000))
    train = train.sample(min(n, len(train)), random_state=0)
    valid = valid.sample(min(n * 3 // 10, len(valid)), random_state=0)
print(f"Lest inn: train {len(train):,}  valid {len(valid):,}  test {len(test):,}  ({time.time()-t0:.0f}s)")

set_categories(train, valid, test)
FEATURES = feature_columns(train)
print("Features:", FEATURES)
params = dict(LGB_PARAMS)

if EVAL_ONLY:
    model = lgb.Booster(model_file=str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    best_iter, valid_mae = model.num_trees(), meta.get("valid_MAE_s")
    full = use_refit_history(pd.concat([train, valid], ignore_index=True))
    print(f"Lastet {MODEL_PATH.name} ({best_iter} trær)")
else:
    # 1) Antall trær med early stopping på valid (sep-okt 2025)
    dtrain = lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT, free_raw_data=True)
    dvalid = lgb.Dataset(valid[FEATURES], valid.y, reference=dtrain)
    t0 = time.time()
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    best_iter = model.best_iteration
    valid_mae = metrics(valid.y, model.predict(valid[FEATURES], num_iteration=best_iter))["MAE_s"]
    print(f"Beste antall trær: {best_iter}, valid-MAE {valid_mae:.1f} s  ({time.time()-t0:.0f}s)")
    del dtrain, dvalid, model

    # 2) Retren på jan-okt 2025 med samme antall trær, med fold-historikk over hele jan-okt
    full = use_refit_history(pd.concat([train, valid], ignore_index=True))
    del train, valid
    t0 = time.time()
    model = lgb.train(params, lgb.Dataset(full[FEATURES], full.y, categorical_feature=CAT), num_boost_round=best_iter)
    print(f"Endelig modell trent ({time.time()-t0:.0f}s)")
    model.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({"best_iter": int(best_iter), "valid_MAE_s": valid_mae,
                                     "features": FEATURES}, indent=2), encoding="utf-8")

# ---------- Evaluering på test (nov-des 2025) ----------
y = test.y.to_numpy()
t0 = time.time()
print(f"Predikerer {len(test):,} testrader med {model.num_trees()} trær ...", flush=True)
pred_lgb = model.predict(test[FEATURES])
print(f"  ferdig ({time.time()-t0:.0f}s)", flush=True)
pred_b1 = test.baseline.to_numpy(dtype="float64")
pred_b0 = np.full(len(y), float(full.y.median()))

results = {
    "Baseline 0 - global median": metrics(y, pred_b0),
    "Baseline 1 - historisk median per linje/stopp/time/dagtype": metrics(y, pred_b1),
    "LightGBM": metrics(y, pred_lgb),
}
boot = bootstrap_mae_gain(test.date.to_numpy(), y, pred_lgb, pred_b1)
results["_info"] = {"quick_mode": QUICK, "test_periode": "2025-11-01 - 2025-12-31", "n_test": int(len(y)),
                    "n_train": int(len(full)), "best_iter": int(best_iter), "valid_MAE_s": valid_mae,
                    "mae_forbedring_mot_baseline": boot, "features": FEATURES}
(REPORTS / f"metrics{SUFFIX}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nResultater på test (nov-des 2025):")
print(pd.DataFrame({k: v for k, v in results.items() if not k.startswith("_")}).T.round(1).to_string())
print(f"\nMAE-forbedring mot baseline: {boot['forbedring_%']:.1f} % "
      f"(95 % KI {boot['ci95_lav_%']:.1f}-{boot['ci95_høy_%']:.1f} %, bootstrap over {boot['dager']} dager)")

# ---------- Filer til Streamlit-appen ----------
imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
(imp / imp.sum() * 100).rename("gain_pct").rename_axis("feature").reset_index() \
    .to_csv(REPORTS / f"feature_importance{SUFFIX}.csv", index=False)
app_cols = ["date", "journey_id", "line", "direction", "stop_name", "origin_name", "dest_name",
            "seq", "trip_start_min", "minute_of_day", "y"]
out = test[app_cols].copy()
out["trip"] = pd.factorize(out.pop("journey_id"))[0].astype("int32")
for c in ("seq", "trip_start_min", "minute_of_day", "y"):
    out[c] = out[c].astype("int16")
for c in ("line", "direction", "stop_name", "origin_name", "dest_name"):
    out[c] = out[c].astype("string").astype("category")
out["pred_lgbm"] = np.round(pred_lgb).astype("int16")
out["pred_baseline"] = np.round(pred_b1).astype("int16")
out.to_parquet(REPORTS / f"test_predictions{SUFFIX}.parquet", index=False, compression="zstd")

# ---------- Figurer ----------
ax = (imp / imp.sum() * 100).plot.barh(figsize=(7, 5), color="#1F4E79")
ax.set_xlabel("Andel av total gain (%)"); ax.set_title("Feature importance - LightGBM")
plt.tight_layout(); plt.savefig(REPORTS / f"feature_importance{SUFFIX}.png", dpi=130); plt.close()
by_hour = pd.DataFrame({"hour": test.hour, "b1": np.abs(pred_b1 - y), "lgb": np.abs(pred_lgb - y)}).groupby("hour").mean()
ax = by_hour.rename(columns={"b1": "Historisk median", "lgb": "LightGBM"}).plot(
    figsize=(8, 4), marker="o", color=["#999999", "#1F4E79"])
ax.set_xlabel("Time på døgnet"); ax.set_ylabel("MAE (sekunder)"); ax.set_title("Feil per time - test nov-des 2025")
plt.tight_layout(); plt.savefig(REPORTS / f"mae_per_hour{SUFFIX}.png", dpi=130); plt.close()
print(f"\nLagret resultater, prognoser og figurer i reports/ (suffiks: '{SUFFIX}')")
