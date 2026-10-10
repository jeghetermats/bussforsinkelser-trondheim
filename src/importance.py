"""
Permutasjonsviktighet for modell A: hvor mye øker feilen (MAE) på testdata når én feature stokkes om?

Gain (reports/feature_importance.csv) måler hvor mye hver feature brukes i trærne, og overdriver features med mange
verdier (stopp, linje), som kan dele opp dataene på mange måter. Her stokkes én feature om gangen mellom radene,
så modellen ikke lenger kan bruke den, og økningen i MAE måles. Det viser hva modellen faktisk trenger.

Bruker enkeltmodellen fra train.py (models/lgbm_pre_departure.txt) og et fast utvalg av testradene.
Resultat: reports/permutation_importance.csv (feature, mae_økning_s, sd_s), som appen viser i stedet for gain.
Kjør:     python src/importance.py                 (etter train.py)
          python src/importance.py --rows 300000 --repeats 3
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CAT, MODELS, REPORTS, load

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=300_000, help="antall testrader (tilfeldig, fast utvalg)")
ap.add_argument("--repeats", type=int, default=3, help="antall stokkinger per feature")
args = ap.parse_args()

model_path = MODELS / "lgbm_pre_departure.txt"
features = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))["features"]
model = lgb.Booster(model_file=str(model_path))

t0 = time.time()
test = load("test")
if len(test) > args.rows:
    test = test.sample(args.rows, random_state=0)
X = test[features].reset_index(drop=True)
for c in CAT:          # LightGBM legger kategoriene fra treningen over disse ved prediksjon
    X[c] = X[c].astype("category")
y = test.y.to_numpy("float64")
base_mae = float(np.mean(np.abs(model.predict(X) - y)))

rng = np.random.default_rng(0)
rows = []
for f in features:
    gains = []
    for _ in range(args.repeats):
        shuffled = X.copy()
        shuffled[f] = X[f].iloc[rng.permutation(len(X))].reset_index(drop=True)   # beholder datatypen
        gains.append(float(np.mean(np.abs(model.predict(shuffled) - y))) - base_mae)
    rows.append({"feature": f, "mae_økning_s": float(np.mean(gains)), "sd_s": float(np.std(gains))})
    print(f"  {f:22s} {rows[-1]['mae_økning_s']:+.2f} s", flush=True)

res = pd.DataFrame(rows).sort_values("mae_økning_s", ascending=False)
res.to_csv(REPORTS / "permutation_importance.csv", index=False)
print(f"MAE uten stokking {base_mae:.2f} s på {len(X):,} testrader ({time.time() - t0:.0f}s). "
      "Lagret reports/permutation_importance.csv")
