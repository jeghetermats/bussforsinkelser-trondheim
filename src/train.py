"""
Trener og evaluerer modeller for forsinkelse før avgang.

  Baseline 0: median forsinkelse i treningsdata (én konstant)
  Baseline 1: historisk median per (linje, stopp, retning, time, dagtype) med fallback
  Modell    : LightGBM (L1-tap) med vær og historiske forsinkelser som features ("target encoding"),
              early stopping på valideringssettet,
              deretter retrent på train+valid og evaluert på test (nov-des).

Kjør:  python src/train.py            (full kjøring)
       python src/train.py --quick    (rask test på et mindre utvalg, ~2 min)
       python src/train.py --eval-only  (hopper over trening, evaluerer lagret modell)
       python src/train.py --gpu      (krever LightGBM bygget med GPU-støtte, ellers brukes CPU)
"""
import json
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CAT, HIST_COLS, LGB_PARAMS, MODELS, REPORTS, base_features, group_median_baseline,
                    hist_features, load, metrics, oof_hist_features, set_categories)

QUICK = "--quick" in sys.argv
MAX_ROUNDS = 100 if QUICK else 1500
GPU = "--gpu" in sys.argv
EVAL_ONLY = "--eval-only" in sys.argv
SUFFIX = "_quick" if QUICK else ""  # hurtigkjøring skal ikke overskrive de ekte figurene
MODEL_PATH = MODELS / ("lgbm_quick.txt" if QUICK else "lgbm_pre_departure.txt")


t0 = time.time()
train, valid, test = load("train"), load("valid"), load("test")
if QUICK:
    import os
    n = int(os.environ.get("QUICK_N", 1_000_000))
    train = train.sample(n, random_state=0)
    valid = valid.sample(n * 3 // 10, random_state=0)
    test = test.sample(n // 2, random_state=0)
print(f"Lest inn: train {len(train):,}  valid {len(valid):,}  test {len(test):,}  ({time.time()-t0:.0f}s)")

set_categories(train, valid, test)   # felles kategorikoder på tvers av splittene
BASE_FEATURES = base_features(train)
FEATURES = BASE_FEATURES + HIST_COLS
print("Features:", FEATURES)

if not EVAL_ONLY:
    t0 = time.time()
    train[HIST_COLS] = oof_hist_features(train)
    valid[HIST_COLS] = hist_features(train, valid)
    print(f"Historiske features beregnet ({time.time()-t0:.0f}s)")

params = dict(LGB_PARAMS)

if GPU:
    # Krever at LightGBM er kompilert med -DUSE_GPU=1 (pip-versjonen er ikke det).
    # max_bin=63 anbefales av LightGBM for GPU (raskere, nesten ingen tap i presisjon).
    params.update(device_type="gpu", max_bin=63, gpu_use_dp=False)


def train_lgb(p, dset, **kw):
    """Trener med gitte parametre; hvis GPU feiler, prøv på nytt med CPU."""
    try:
        return lgb.train(p, dset, **kw)
    except lgb.basic.LightGBMError as e:
        if p.get("device_type") != "gpu":
            raise
        print(f"GPU feilet ({e}). Faller tilbake til CPU.")
        p.update(device_type="cpu")  # beholder max_bin, datasettet er allerede bygget med den
        return lgb.train(p, dset, **kw)


print("Enhet:", params.get("device_type", "cpu"))

if EVAL_ONLY:
    model = lgb.Booster(model_file=str(MODEL_PATH))
    best_iter, valid_mae = model.num_trees(), None
    full = pd.concat([train, valid], ignore_index=True)
    del train, valid
    print(f"Lastet {MODEL_PATH.name} ({best_iter} trær)")
else:
    # 1) Finn antall trær med early stopping på valid (sep-okt 2025)
    dtrain = lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT, free_raw_data=True)
    dvalid = lgb.Dataset(valid[FEATURES], valid.y, reference=dtrain)
    t0 = time.time()
    model = train_lgb(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
    best_iter = model.best_iteration
    print(f"Beste antall trær: {best_iter}  ({time.time()-t0:.0f}s)")
    valid_mae = metrics(valid.y.to_numpy(), model.predict(valid[FEATURES], num_iteration=best_iter))["MAE_s"]
    del dtrain, dvalid, model

    # 2) Retren på train+valid (jan 2024-okt 2025) og test på nov-des 2025
    full = pd.concat([train, valid], ignore_index=True).drop(columns=HIST_COLS)
    del train, valid
    for c in CAT:
        full[c] = pd.Categorical(full[c], categories=test[c].cat.categories)
    full[HIST_COLS] = oof_hist_features(full)
    dfull = lgb.Dataset(full[FEATURES], full.y, categorical_feature=CAT)
    t0 = time.time()
    model = train_lgb(params, dfull, num_boost_round=best_iter)
    print(f"Endelig modell trent ({time.time()-t0:.0f}s)")
    model.save_model(str(MODEL_PATH))

print("Beregner historiske features for test ...", flush=True)
test[HIST_COLS] = hist_features(full, test)
y = test.y.to_numpy()
t0 = time.time()
print(f"Predikerer {len(test):,} testrader med {model.num_trees()} trær (kan ta noen minutter) ...", flush=True)
pred_lgb = model.predict(test[FEATURES])
print(f"  ferdig ({time.time()-t0:.0f}s). Beregner baselines ...", flush=True)
pred_b0 = np.full_like(y, full.y.median(), dtype="float64")
pred_b1 = group_median_baseline(full, test)

results = {
    "Baseline 0 - global median": metrics(y, pred_b0),
    "Baseline 1 - historisk median per linje/stopp/time/dagtype": metrics(y, pred_b1),
    "LightGBM": metrics(y, pred_lgb),
}
results["_info"] = {"quick_mode": QUICK, "test_periode": "2025-11-01 - 2025-12-31", "n_test": int(len(y)),
                    "n_train": int(len(full)), "best_iter": int(best_iter),
                    "valid_MAE_s": valid_mae, "features": FEATURES}
(REPORTS / ("metrics_quick.json" if QUICK else "metrics.json")).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nResultater på test (nov-des 2025):")
print(pd.DataFrame({k: v for k, v in results.items() if not k.startswith("_")}).T.round(1).to_string())

# Lagre testprediksjoner og feature importance for Streamlit-appen (app/)
imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
(imp / imp.sum() * 100).rename("gain_pct").rename_axis("feature").reset_index() \
    .to_csv(REPORTS / f"feature_importance{SUFFIX}.csv", index=False)
if "journey_id" in test.columns:
    app_cols = ["date", "journey_id", "line", "direction", "stop_name", "origin_name", "dest_name",
                "seq", "trip_start_min", "minute_of_day", "y"]
    out = test[app_cols].copy()
    out["trip"] = pd.factorize(out.pop("journey_id"))[0].astype("int32")   # kort tur-ID gir mindre fil
    out["seq"] = out["seq"].astype("int16")
    out["trip_start_min"] = out["trip_start_min"].astype("int16")
    out["minute_of_day"] = out["minute_of_day"].astype("int16")
    for c in ("line", "direction", "stop_name", "origin_name", "dest_name"):
        out[c] = out[c].astype("string").astype("category")
    out["y"] = out["y"].astype("int16")
    out["pred_lgbm"] = np.round(pred_lgb).astype("int16")
    out["pred_baseline"] = np.round(pred_b1).astype("int16")
    out.to_parquet(REPORTS / f"test_predictions{SUFFIX}.parquet", index=False, compression="zstd")
    print(f"Lagret reports/test_predictions{SUFFIX}.parquet ({len(out):,} rader) til appen")
else:
    print("Tips: kjør 'python src/features.py test' og deretter 'python src/train.py --eval-only' "
          "for å lage data til Streamlit-appen.")

# Figurer
ax = (imp / imp.sum() * 100).plot.barh(figsize=(7, 5), color="#1F4E79")
ax.set_xlabel("Andel av total gain (%)"); ax.set_title("Feature importance - LightGBM")
plt.tight_layout(); plt.savefig(REPORTS / f"feature_importance{SUFFIX}.png", dpi=130); plt.close()

by_hour = pd.DataFrame({"hour": test.hour, "b1": np.abs(pred_b1 - y), "lgb": np.abs(pred_lgb - y)}) \
            .groupby("hour").mean()
ax = by_hour.rename(columns={"b1": "Baseline 1 (historisk median)", "lgb": "LightGBM"}) \
            .plot(figsize=(8, 4), marker="o", color=["#999999", "#1F4E79"])
ax.set_xlabel("Time på døgnet"); ax.set_ylabel("MAE (sekunder)"); ax.set_title("Feil per time - test nov-des 2025")
plt.tight_layout(); plt.savefig(REPORTS / f"mae_per_hour{SUFFIX}.png", dpi=130); plt.close()
print(f"\nLagret resultater og figurer i reports/ (suffiks: '{SUFFIX}')")
