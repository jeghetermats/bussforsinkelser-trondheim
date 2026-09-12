"""
Tidsbasert evaluering med ekspanderende vindu - samme oppsett som hovedmodellen.

For hver måned M i mars-des 2025:
  * historikk: fold-historikk over alle rader før M (src/history.py); radene i M ser bare fortiden
  * trening:   rader fra jan 2025 til og med to måneder før M (20 %-utvalg, maks MAX_TRAIN rader)
  * early stopping på måneden før M, så antall trær tilpasses datamengden (ikke fast antall)
  * evaluering på M, mot baselinen (historisk median fra alle rader før M)
Januar-februar mangler fordi det trengs minst én treningsmåned og én måned til early stopping.
Desember 2024 brukes bare som historikk.

Resultatene lagres etter hver måned, så skriptet kan avbrytes og startes igjen.
Krever data/features.duckdb fra features.py.

Kjør:  python src/rolling_eval.py
       python src/rolling_eval.py --quick   (2 måneder, lite utvalg)
"""
import os
import sys
import time

import duckdb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CAT, DATA, LGB_PARAMS, REPORTS, ensure_dirs, metrics
from history import baseline_select, build_oof, build_past_monthly, feature_select, join_sql

QUICK = "--quick" in sys.argv
MONTHS = pd.period_range("2025-03", "2025-04", freq="M") if QUICK else pd.period_range("2025-03", "2025-12", freq="M")
MAX_TRAIN = 200_000 if QUICK else 4_000_000
N_ES = 100_000 if QUICK else 500_000
OUT = REPORTS / ("rolling_quick.csv" if QUICK else "rolling_eval.csv")
ensure_dirs()

con = duckdb.connect(str(DATA / "features.duckdb"))
con.execute(f"SET temp_directory='{(DATA / 'tmp').as_posix()}'")
if os.environ.get("DUCKDB_MEMORY_LIMIT"):
    con.execute(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
if os.environ.get("DUCKDB_THREADS"):
    con.execute(f"SET threads={int(os.environ['DUCKDB_THREADS'])}")

PAST = build_past_monthly(con, [f"2025-{m:02d}-01" for m in range(1, 13)])
EXCLUDE = "month, bucket, journey_id, stop_name, origin_name, dest_name"


def fetch(where, prefix, hk, n=None, seed=0):
    sample = f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed + 1})" if n else ""
    q = f"""
      SELECT c.* EXCLUDE ({EXCLUDE}), {feature_select('h')}, {baseline_select('b')}
      FROM (SELECT * FROM (SELECT * FROM clean WHERE {where}) {sample}) c
      {join_sql(prefix, hk, 'h')}
      {join_sql(PAST, "date_trunc('month', c.date)::DATE", 'b')}"""
    df = con.sql(q).df()
    for col in CAT:
        df[col] = df[col].astype("object")
    return df


done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
if not done.empty and "early_stopping" not in done.columns:
    done = pd.DataFrame()                       # gammelt format (fast antall trær) - start på nytt
rows = done.to_dict("records")

for m in MONTHS:
    if not done.empty and str(m) in set(done["month"]):
        print(f"{m}: allerede ferdig - hopper over")
        continue
    t0 = time.time()
    m0 = m.start_time.strftime("%Y-%m-%d")
    es = m - 1
    tr_end = (es.start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prefix = build_oof(con, (m.start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    fold = "dayofyear(c.date) % 5"
    train = fetch(f"bucket < 20 AND date BETWEEN DATE '2025-01-01' AND DATE '{tr_end}'", prefix, fold, MAX_TRAIN)
    val = fetch(f"bucket < 25 AND date BETWEEN DATE '{es.start_time:%Y-%m-%d}' AND DATE '{es.end_time:%Y-%m-%d}'",
                prefix, fold, N_ES)
    test = fetch(f"bucket < 25 AND date BETWEEN DATE '{m0}' AND DATE '{m.end_time:%Y-%m-%d}'", prefix, "5")
    for col in CAT:
        cats = pd.Index(pd.concat([train[col], val[col], test[col]]).unique())
        for d in (train, val, test):
            d[col] = pd.Categorical(d[col], categories=cats)
    feats = [c for c in train.columns if c not in ("y", "date", "baseline")]
    dtr = lgb.Dataset(train[feats], train.y, categorical_feature=CAT)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000,
                      valid_sets=[lgb.Dataset(val[feats], val.y, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    y = test.y.to_numpy()
    m_lgb = metrics(y, model.predict(test[feats], num_iteration=model.best_iteration))
    m_b1 = metrics(y, test.baseline.to_numpy(dtype="float64"))
    rows.append({"month": str(m), "early_stopping": True, "trees": model.best_iteration,
                 "n_train": len(train), "n_test": len(test), "mean_delay_s": float(y.mean()),
                 "MAE_baseline": m_b1["MAE_s"], "MAE_lgbm": m_lgb["MAE_s"],
                 "RMSE_baseline": m_b1["RMSE_s"], "RMSE_lgbm": m_lgb["RMSE_s"]})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{m}: {model.best_iteration} trær, {len(train):,} treningsrader | baseline {m_b1['MAE_s']:.1f}s  "
          f"LightGBM {m_lgb['MAE_s']:.1f}s ({(1 - m_lgb['MAE_s'] / m_b1['MAE_s']) * 100:+.1f} %)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows).sort_values("month")
res["forbedring_%"] = (1 - res.MAE_lgbm / res.MAE_baseline) * 100
print("\n" + res[["month", "trees", "n_train", "MAE_baseline", "MAE_lgbm", "forbedring_%"]].round(1).to_string(index=False))
tot = (1 - (res.MAE_lgbm * res.n_test).sum() / (res.MAE_baseline * res.n_test).sum()) * 100
print(f"\nVektet over alle måneder: LightGBM {tot:+.1f} % lavere MAE enn baseline; "
      f"bedre i {(res.MAE_lgbm < res.MAE_baseline).sum()} av {len(res)} måneder")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(res))
ax.plot(x, res.MAE_baseline, marker="o", color="#999999", ls="--", label="Historisk median")
ax.plot(x, res.MAE_lgbm, marker="o", color="#1F4E79", label="LightGBM")
ax.set_xticks(x, [pd.Period(p).strftime("%b") for p in res.month])
ax.set_ylabel("MAE (sekunder)"); ax.legend()
ax.set_title("Rullende evaluering 2025 - trent på tidligere måneder, testet på én måned")
plt.tight_layout(); plt.savefig(REPORTS / ("rolling_quick.png" if QUICK else "rolling_eval.png"), dpi=130); plt.close()
print(f"Lagret {OUT.name} og figur i reports/")
