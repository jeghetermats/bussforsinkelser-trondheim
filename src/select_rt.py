"""
Valg av oppsett for modell B - bare på valideringsdata (sep-okt 2025), aldri på test.

Varianter (alle med early stopping på valid, snitt over seeds):
  R1            : sanntidsfeatures fra runde 1 (linje/retning, stopp, nett)
  R2            : + runde 2: forrige buss ved samme stopp, forsinkelsesvekst per strekning summert langs
                  turen, innkommende buss ved startholdeplassen (se src/realtime.py)
  R2_resid      : R2, men trent på avviket y - baseline (baselinen legges til igjen etterpå)
  R2_resid_lr05 : R2_resid med learning_rate 0,05 og min_data_in_leaf 500 (roligere læring)

Valid har én tilfeldig lead (5-90 min) per tur, så resultatet vises også for kort/middels/lang lead
og for de første stoppene på turen, der sanntid burde hjelpe mest.

Resultat: reports/rt_select.csv. Prognoser: data/ens/sel_<variant>_s<seed>.npz (kjøringen kan fortsette).
Kjør:     python src/select_rt.py --variants R1,R2,R2_resid,R2_resid_lr05 --seeds 42,1
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CAT, DATA, LGB_PARAMS, REPORTS, bootstrap_mae_gain, ensure_dirs, feature_columns, load, \
    set_categories

ROUND2 = ("rt_prev_", "rt_seg_", "rt_segcum", "rt_segcov", "rt_in_")
VARIANTS = {
    "R1": dict(round2=False, resid=False),
    "R2": dict(round2=True, resid=False),
    "R2_resid": dict(round2=True, resid=True),
    "R2_resid_lr05": dict(round2=True, resid=True, learning_rate=0.05, min_data_in_leaf=500),
}
ap = argparse.ArgumentParser()
ap.add_argument("--variants", default=",".join(VARIANTS))
ap.add_argument("--seeds", default="42,1")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
ENS = DATA / "ens"
ENS.mkdir(exist_ok=True)
ensure_dirs()

t0 = time.time()
train, valid = load("rt_train"), load("rt_valid")
set_categories(train, valid)
ALL = feature_columns(train)
print(f"Lest inn: train {len(train):,}  valid {len(valid):,}  ({time.time()-t0:.0f}s)", flush=True)
# Fingeravtrykk av valid-radene: lagrede prognoser brukes bare hvis radene er de samme og i samme rekkefølge
# (DuckDB garanterer ikke rekkefølgen når realtime.py kjøres på nytt).
FP = int((valid.y.to_numpy("int64") * (np.arange(len(valid)) % 9973 + 1)).sum())


def cached(path):
    if not path.exists():
        return False
    r = np.load(path)
    return "fp" in r.files and int(r["fp"]) == FP

for name in [v for v in args.variants.split(",") if v in VARIANTS]:
    v = VARIANTS[name]
    feats = ALL if v["round2"] else [c for c in ALL if not c.startswith(ROUND2)]
    ytr = train.y - train.baseline if v["resid"] else train.y
    yva = valid.y - valid.baseline if v["resid"] else valid.y
    params = dict(LGB_PARAMS, **{k: val for k, val in v.items() if k not in ("round2", "resid")})
    for seed in SEEDS:
        out = ENS / f"sel_{name}_s{seed}.npz"
        if cached(out):
            continue
        t0 = time.time()
        dtr = lgb.Dataset(train[feats], ytr, categorical_feature=CAT)
        m = lgb.train(dict(params, seed=seed), dtr, num_boost_round=5000,
                      valid_sets=[lgb.Dataset(valid[feats], yva, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(valid[feats], num_iteration=m.best_iteration)
        if v["resid"]:
            p = p + valid.baseline.to_numpy("float64")
        np.savez(out, pred=p.astype("float32"), best=m.best_iteration, fp=FP)
        print(f"{name} seed {seed}: {m.best_iteration} trær, valid-MAE {np.mean(np.abs(p - valid.y)):.2f} s "
              f"({len(feats)} features, {time.time()-t0:.0f}s)", flush=True)

# ---------- Oppsummering: snitt over seeds, parvis mot R1 med bootstrap over dager ----------
y = valid.y.to_numpy("float64")
lead_grp = pd.cut(valid.lead_min, [0, 20, 45, 90], labels=["5-20 min", "21-45 min", "46-90 min"])
first = (valid.sched_min_from_start < 5).to_numpy()
dates = valid.date.to_numpy()
preds, trees = {}, {}
for name in VARIANTS:
    files = [ENS / f"sel_{name}_s{s}.npz" for s in SEEDS]
    if all(cached(f) for f in files):
        rs = [np.load(f) for f in files]
        preds[name] = np.mean([r["pred"] for r in rs], axis=0).astype("float64")
        trees[name] = [int(r["best"]) for r in rs]
rows = []
for name, p in preds.items():
    e = np.abs(p - y)
    row = {"variant": name, "trær": trees[name], "MAE": e.mean(), "første 5 min av turen": e[first].mean()}
    for g in lead_grp.cat.categories:
        row[g] = e[(lead_grp == g).to_numpy()].mean()
    if name != "R1" and "R1" in preds:
        b = bootstrap_mae_gain(dates, y, p, preds["R1"])
        row["mot R1 %"], row["KI lav %"], row["KI høy %"] = b["forbedring_%"], b["ci95_lav_%"], b["ci95_høy_%"]
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(REPORTS / "rt_select.csv", index=False)
print(f"\nValid sep-okt, snitt over seeds {SEEDS} (MAE i sekunder):")
print(res.round(2).to_string(index=False))
