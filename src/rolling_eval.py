"""
Tidsbasert evaluering med ekspanderende vindu.

For hver måned feb-des 2025: tren på alle måneder før, test på måneden.
Historikk-features og baseline er bygget fra data før hver rads måned (features.py), så ingen
fremtidig informasjon brukes. Januar er utelatt fordi det ikke finnes treningsmåneder før den
(desember 2024 er bare historikk).

Samme LightGBM-innstillinger og antall trær som hovedmodellen (leses fra models/*.json).
Resultatene lagres etter hver måned, så skriptet kan avbrytes og startes igjen.

Kjør:  python src/rolling_eval.py
       python src/rolling_eval.py --quick   (2 måneder, lite utvalg)
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

from common import CAT, LGB_PARAMS, MODELS, REPORTS, ensure_dirs, feature_columns, load, metrics, set_categories

QUICK = "--quick" in sys.argv
MONTHS = pd.period_range("2025-03", "2025-04", freq="M") if QUICK else pd.period_range("2025-02", "2025-12", freq="M")
MAX_TRAIN = 300_000 if QUICK else 4_000_000     # tak på treningsrader per måned (tid/minne)
meta_path = MODELS / "lgbm_pre_departure.json"
ROUNDS = 50 if QUICK else (json.loads(meta_path.read_text())["best_iter"] if meta_path.exists() else 800)
OUT = REPORTS / ("rolling_quick.csv" if QUICK else "rolling_eval.csv")
ensure_dirs()

df = load("all")
set_categories(df)
FEATURES = feature_columns(df)
print(f"Lest inn {len(df):,} rader, {df.date.min():%Y-%m-%d} - {df.date.max():%Y-%m-%d}. {ROUNDS} trær per modell.")

done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
if not done.empty and done.get("rounds", pd.Series([None])).iloc[0] != ROUNDS:
    done = pd.DataFrame()              # andre innstillinger enn forrige kjøring - start på nytt
rows = done.to_dict("records")

for m in MONTHS:
    if not done.empty and str(m) in set(done["month"]):
        print(f"{m}: allerede ferdig - hopper over")
        continue
    t0 = time.time()
    past = df[df.date < m.start_time]
    test = df[(df.date >= m.start_time) & (df.date <= m.end_time)]
    train = past.sample(min(MAX_TRAIN, len(past)), random_state=0)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT),
                      num_boost_round=ROUNDS)
    y = test.y.to_numpy()
    m_lgb = metrics(y, model.predict(test[FEATURES]))
    m_b1 = metrics(y, test.baseline.to_numpy(dtype="float64"))
    rows.append({"month": str(m), "rounds": ROUNDS, "n_train": len(train), "n_test": len(test),
                 "mean_delay_s": float(y.mean()),
                 "MAE_baseline": m_b1["MAE_s"], "MAE_lgbm": m_lgb["MAE_s"],
                 "RMSE_baseline": m_b1["RMSE_s"], "RMSE_lgbm": m_lgb["RMSE_s"]})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{m}: baseline {m_b1['MAE_s']:.1f}s  LightGBM {m_lgb['MAE_s']:.1f}s  "
          f"({(1 - m_lgb['MAE_s'] / m_b1['MAE_s']) * 100:+.1f} %)  [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows).sort_values("month")
res["forbedring_%"] = (1 - res.MAE_lgbm / res.MAE_baseline) * 100
print("\n" + res[["month", "mean_delay_s", "MAE_baseline", "MAE_lgbm", "forbedring_%"]].round(1).to_string(index=False))
tot = (1 - (res.MAE_lgbm * res.n_test).sum() / (res.MAE_baseline * res.n_test).sum()) * 100
print(f"\nVektet over alle måneder: LightGBM {tot:+.1f} % lavere MAE enn baseline")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(res))
ax.plot(x, res.MAE_baseline, marker="o", color="#999999", ls="--", label="Historisk median")
ax.plot(x, res.MAE_lgbm, marker="o", color="#1F4E79", label="LightGBM")
ax.set_xticks(x, [pd.Period(p).strftime("%b") for p in res.month])
ax.set_ylabel("MAE (sekunder)"); ax.legend()
ax.set_title("Rullende evaluering 2025 - trent på tidligere måneder, testet på én måned")
plt.tight_layout(); plt.savefig(REPORTS / ("rolling_quick.png" if QUICK else "rolling_eval.png"), dpi=130); plt.close()
print(f"Lagret {OUT.name} og figur i reports/")
