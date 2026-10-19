"""
Tidsbasert evaluering med ekspanderende vindu - samme oppsett som hovedmodellen.

For hver måned M i mars-des 2025:
  * historikk: fold-historikk over alle rader før M (src/history.py); radene i M ser bare fortiden
  * trening:   rader fra jan 2025 til og med to måneder før M (20 %-utvalg, maks MAX_TRAIN rader)
  * early stopping på måneden før M, så antall trær tilpasses datamengden (ikke fast antall)
  * evaluering på M, mot baselinen (historisk median fra alle rader før M)
Januar-februar mangler fordi det trengs minst én treningsmåned og én måned til early stopping.
Desember 2024 brukes bare som historikk.

Resultatene lagres etter hver måned, så skriptet kan avbrytes og startes igjen. De er merket med når clean-tabellen
ble bygget; har features.py bygget den på nytt siden, starter evalueringen forfra.
Krever data/features.duckdb fra features.py.

Standardoppsettet stiller modellen litt svakere enn hovedmodellen: den trenes aldri på måneden som brukes til
early stopping, og early stopping på én måned stopper noen ganger svært tidlig (21-26 trær i april og august).
To valg retter på det (resultatene lagres i egne filer, så standardtallene i appen ikke endres):
  --refit           retren på trening + early stopping-måneden med beste antall trær, som hovedmodellen gjør
  --min-rounds N    bruk minst N trær (N=0: ingen nedre grense)

Kjør:  python src/rolling_eval.py
       python src/rolling_eval.py --refit --min-rounds 100   (-> reports/rolling_eval_refit_min100.csv)
       python src/rolling_eval.py --quick   (2 måneder, lite utvalg)
"""
import argparse
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CAT, DATA, HIST_MONTHS, LGB_PARAMS, REPORTS, ROLLING_MONTHS, SAMPLE_PCT, SPLITS, clean_stamp,
                    connect_duckdb, ensure_dirs, metrics, replace_when_done)
from history import baseline_select, build_oof, build_past_monthly, feature_select, join_sql

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--refit", action="store_true", help="retren på trening + early stopping-måneden")
ap.add_argument("--min-rounds", type=int, default=0, help="minst så mange trær i modellen som evalueres")
args = ap.parse_args()
QUICK = args.quick
MONTHS = pd.PeriodIndex([m[:7] for m in ROLLING_MONTHS], freq="M")          # mars-des (fra SPLITS)
if QUICK:
    MONTHS = MONTHS[:2]
MAX_TRAIN = 200_000 if QUICK else 4_000_000
N_ES = 100_000 if QUICK else 500_000
VARIANT = ("_refit" if args.refit else "") + (f"_min{args.min_rounds}" if args.min_rounds else "")
OUT = REPORTS / (f"rolling_quick{VARIANT}.csv" if QUICK else f"rolling_eval{VARIANT}.csv")
ensure_dirs()

con = connect_duckdb(DATA / "features.duckdb")
STAMP = clean_stamp(con)

PAST = build_past_monthly(con, HIST_MONTHS)
EXCLUDE = "month, bucket, journey_id, stop_name, origin_name, dest_name"


def fetch(where, prefix, hk, n=None, seed=0):
    sample = f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed + 1})" if n else ""
    q = f"""
      SELECT c.* EXCLUDE ({EXCLUDE}), {feature_select('h')}, {baseline_select('b')}
      FROM (SELECT * FROM (SELECT * FROM clean WHERE {where}) {sample}) c
      {join_sql(prefix, hk, 'h')}
      {join_sql(PAST, "date_trunc('month', c.date)::DATE", 'b')}
      ORDER BY c.date, c.journey_id, c.seq"""         # fast rekkefølge, så treningen blir lik fra gang til gang
    df = con.sql(q).df()
    for col in CAT:
        df[col] = df[col].astype("object")
    return df


done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
if not done.empty and "early_stopping" not in done.columns:
    done = pd.DataFrame()                       # gammelt format (fast antall trær) - start på nytt
if not done.empty and ("clean_bygget" not in done.columns or (done.clean_bygget.astype(str) != STAMP).any()):
    print(f"{OUT.name} er laget fra en eldre clean-tabell (features.py er kjørt på nytt siden) - starter forfra")
    done = pd.DataFrame()
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
    train = fetch(f"bucket < {SAMPLE_PCT['train']} AND date BETWEEN DATE '{SPLITS['train'][0]}' AND DATE '{tr_end}'",
                  prefix, fold, MAX_TRAIN)
    val = fetch(f"bucket < {SAMPLE_PCT['valid']} AND date BETWEEN DATE '{es.start_time:%Y-%m-%d}' "
                f"AND DATE '{es.end_time:%Y-%m-%d}'", prefix, fold, N_ES)
    test = fetch(f"bucket < {SAMPLE_PCT['test']} AND date BETWEEN DATE '{m0}' AND DATE '{m.end_time:%Y-%m-%d}'",
                 prefix, "5")
    for col in CAT:
        cats = pd.Index(pd.concat([train[col], val[col], test[col]]).unique())
        for d in (train, val, test):
            d[col] = pd.Categorical(d[col], categories=cats)
    feats = [c for c in train.columns if c not in ("y", "date", "baseline")]
    dtr = lgb.Dataset(train[feats], train.y, categorical_feature=CAT)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000,
                      valid_sets=[lgb.Dataset(val[feats], val.y, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    best = model.best_iteration
    trees = max(best, args.min_rounds)
    if args.refit:          # som hovedmodellen: retren med early stopping-måneden (den har fold-historikk)
        both = pd.concat([train, val], ignore_index=True)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(both[feats], both.y, categorical_feature=CAT), num_boost_round=trees)
    elif trees > best:      # minst min-rounds trær: tren på nytt uten early stopping
        model = lgb.train(LGB_PARAMS, lgb.Dataset(train[feats], train.y, categorical_feature=CAT),
                          num_boost_round=trees)
    y = test.y.to_numpy()
    m_lgb = metrics(y, model.predict(test[feats], num_iteration=trees))
    m_b1 = metrics(y, test.baseline.to_numpy(dtype="float64"))
    rows.append({"month": str(m), "clean_bygget": STAMP, "early_stopping": True, "trees": trees, "trees_early_stopping": best,
                 "refit": bool(args.refit),
                 "n_train": len(train) + (len(val) if args.refit else 0), "n_test": len(test),
                 "mean_delay_s": float(y.mean()),
                 "MAE_baseline": m_b1["MAE_s"], "MAE_lgbm": m_lgb["MAE_s"],
                 "RMSE_baseline": m_b1["RMSE_s"], "RMSE_lgbm": m_lgb["RMSE_s"]})
    replace_when_done(OUT, lambda p: pd.DataFrame(rows).to_csv(p, index=False))   # lagres etter hver måned
    print(f"{m}: {trees} trær, {rows[-1]['n_train']:,} treningsrader | baseline {m_b1['MAE_s']:.1f}s  "
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
ax.set_title(f"Rullende evaluering {MONTHS[0].year} - trent på tidligere måneder, testet på én måned"
             + (" (retrent med early stopping-måneden)" if args.refit else ""))
plt.tight_layout(); plt.savefig(OUT.with_suffix(".png"), dpi=130); plt.close()
print(f"Lagret {OUT.name} og figur i reports/")
