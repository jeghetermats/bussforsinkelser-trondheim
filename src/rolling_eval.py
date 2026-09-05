"""
Tidsbasert kryssvalidering med rullende (ekspanderende) vindu.

For hver måned i 2025: tren på ALT som kom før måneden, test på måneden.
Slik ser vi hvor godt modellen treffer gjennom hele året - også om vinteren -
uten at fremtidig informasjon lekker inn.

Bruker data/all.parquet (bygges med: python src/features.py all).
Resultater lagres fortløpende, så skriptet kan avbrytes og startes igjen.

Kjør:  python src/rolling_eval.py
       python src/rolling_eval.py --quick   (2 måneder, lite utvalg - for å teste at alt virker)
"""
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CAT, HIST_COLS, LGB_PARAMS, REPORTS, base_features, group_median_baseline,
                    hist_features, load, metrics, oof_hist_features, set_categories)

QUICK = "--quick" in sys.argv
MONTHS = pd.period_range("2025-03", "2025-04", freq="M") if QUICK else pd.period_range("2025-01", "2025-12", freq="M")
MAX_TRAIN = 300_000 if QUICK else 4_000_000   # tak på antall treningsrader per fold (for tid/minne)
ROUNDS = 50 if QUICK else 400                   # fast antall trær (ingen early stopping per fold)
PARAMS = dict(LGB_PARAMS, learning_rate=0.15)
OUT = REPORTS / ("rolling_quick.csv" if QUICK else "rolling_eval.csv")

df = load("all")
set_categories(df)
FEATURES = base_features(df) + HIST_COLS
print(f"Lest inn {len(df):,} rader, {df.date.min():%Y-%m-%d} - {df.date.max():%Y-%m-%d}")

done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
rows = done.to_dict("records")

for m in MONTHS:
    if not done.empty and str(m) in set(done["month"]):
        print(f"{m}: allerede ferdig - hopper over")
        continue
    t0 = time.time()
    hist = df[df.date < m.start_time]                                   # alt før måneden
    test = df[(df.date >= m.start_time) & (df.date <= m.end_time)].copy()
    train = hist.sample(min(MAX_TRAIN, len(hist)), random_state=0).copy()

    train[HIST_COLS] = oof_hist_features(train)
    test[HIST_COLS] = hist_features(hist, test)                         # historikk fra ALL tidligere data

    model = lgb.train(PARAMS, lgb.Dataset(train[FEATURES], train.y, categorical_feature=CAT),
                      num_boost_round=ROUNDS)
    y = test.y.to_numpy()
    m_lgb = metrics(y, model.predict(test[FEATURES]))
    m_b1 = metrics(y, group_median_baseline(hist, test))
    rows.append({"month": str(m), "n_train_hist": len(hist), "n_test": len(test),
                 "mean_delay_s": float(y.mean()),
                 "MAE_baseline": m_b1["MAE_s"], "MAE_lgbm": m_lgb["MAE_s"],
                 "RMSE_baseline": m_b1["RMSE_s"], "RMSE_lgbm": m_lgb["RMSE_s"]})
    pd.DataFrame(rows).to_csv(OUT, index=False)                         # lagre etter hver måned
    print(f"{m}: baseline {m_b1['MAE_s']:.1f}s  LightGBM {m_lgb['MAE_s']:.1f}s  "
          f"({(1 - m_lgb['MAE_s'] / m_b1['MAE_s']) * 100:+.1f} %)  [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows).sort_values("month")
res["forbedring_%"] = (1 - res.MAE_lgbm / res.MAE_baseline) * 100
print("\n" + res[["month", "mean_delay_s", "MAE_baseline", "MAE_lgbm", "forbedring_%"]].round(1).to_string(index=False))
tot = (1 - (res.MAE_lgbm * res.n_test).sum() / (res.MAE_baseline * res.n_test).sum()) * 100
print(f"\nVektet over alle måneder: LightGBM {tot:+.1f} % lavere MAE enn baseline")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(res))
ax.plot(x, res.MAE_baseline, marker="o", color="#999999", label="Baseline (historisk median)")
ax.plot(x, res.MAE_lgbm, marker="o", color="#1F4E79", label="LightGBM")
ax.set_xticks(x, [pd.Period(p).strftime("%b") for p in res.month])
ax.set_ylabel("MAE (sekunder)")
ax.set_title("Rullende evaluering 2025 - trent på alle tidligere data, testet på én måned")
ax.legend()
plt.tight_layout(); plt.savefig(REPORTS / ("rolling_quick.png" if QUICK else "rolling_eval.png"), dpi=130); plt.close()
print(f"Lagret {OUT.name} og figur i reports/")
