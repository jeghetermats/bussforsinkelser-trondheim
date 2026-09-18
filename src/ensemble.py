"""
Seed-snitt og blanding med baselinen, for modell A og B.

  1. Tren hver modell med flere seeds (samme protokoll som train.py / train_rt.py:
     early stopping på valid sep-okt, retrening på jan-okt). Seed 42 er den vanlige modellen.
  2. Seed-snitt = gjennomsnitt av prognosene.
  3. Blanding: w * modell + (1 - w) * baseline. w velges på valid (sep-okt), aldri på test.
     Som kontroll vises også w valgt over mars-okt fra ablasjonens måned-for-måned-kjøringer
     (V4, 1 mill. treningsrader, 2 seeds) - samme idé, men mindre modeller.

Resultat: reports/ensemble.csv (test nov-des) og reports/ensemble.json (valgte w, KI).
Prognoser per seed lagres i data/ens/, så en avbrutt kjøring fortsetter der den slapp.

Kjør:  python src/ensemble.py                 (A og B, seeds 42,1,2)
       python src/ensemble.py --model A --seeds 42,1
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (CAT, DATA, LGB_PARAMS, REPORTS, bootstrap_mae_gain, ensure_dirs, feature_columns, load,
                    metrics, set_categories, use_refit_history)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="A,B")
ap.add_argument("--seeds", default="42,1,2")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
W_GRID = np.round(np.arange(0.5, 1.201, 0.05), 2)
ENS = DATA / "ens"
ENS.mkdir(exist_ok=True)
ensure_dirs()
SPLITS = {"A": ("train", "valid", "test"), "B": ("rt_train", "rt_valid", "rt_test")}


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def best_w(p, b, y):
    scores = {float(w): mae(w * p + (1 - w) * b, y) for w in W_GRID}
    return min(scores, key=scores.get), scores


def train_seeds(model):
    s_tr, s_va, s_te = SPLITS[model]
    todo = [s for s in SEEDS if not (ENS / f"{model}_s{s}.npz").exists()]
    t0 = time.time()
    train, valid, test = load(s_tr), load(s_va), load(s_te)
    set_categories(train, valid, test)
    feats = feature_columns(train)
    print(f"[{model}] train {len(train):,}  valid {len(valid):,}  test {len(test):,}  ({time.time()-t0:.0f}s)",
          flush=True)
    if todo:
        full = use_refit_history(pd.concat([train, valid], ignore_index=True))
    for seed in todo:
        t0 = time.time()
        params = dict(LGB_PARAMS, seed=seed)
        dtr = lgb.Dataset(train[feats], train.y, categorical_feature=CAT)
        m = lgb.train(params, dtr, num_boost_round=3000,
                      valid_sets=[lgb.Dataset(valid[feats], valid.y, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        best = m.best_iteration
        pv = m.predict(valid[feats], num_iteration=best)
        del dtr, m
        m = lgb.train(params, lgb.Dataset(full[feats], full.y, categorical_feature=CAT), num_boost_round=best)
        pt = m.predict(test[feats])
        np.savez(ENS / f"{model}_s{seed}.npz", valid=pv.astype("float32"), test=pt.astype("float32"), best=best)
        print(f"[{model}] seed {seed}: {best} trær, valid-MAE {mae(pv, valid.y):.2f}, "
              f"test-MAE {mae(pt, test.y):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    keep = ["y", "baseline", "date"] + (["lead_min"] if model == "B" else [])
    return valid[keep].reset_index(drop=True), test[keep].reset_index(drop=True)


def evaluate(model, valid, test):
    r = [np.load(ENS / f"{model}_s{s}.npz") for s in SEEDS]
    yv, bv = valid.y.to_numpy("float64"), valid.baseline.to_numpy("float64")
    pv_single, pv_avg = r[0]["valid"].astype("float64"), np.mean([x["valid"] for x in r], axis=0)
    pt_single, pt_avg = r[0]["test"].astype("float64"), np.mean([x["test"] for x in r], axis=0)
    w, w_scores = best_w(pv_avg, bv, yv)
    bt = test.baseline.to_numpy("float64")
    preds = {"Baseline": bt, f"Én modell (seed {SEEDS[0]})": pt_single,
             f"Seed-snitt ({len(SEEDS)})": pt_avg, f"Seed-snitt + blanding (w={w:.2f})": w * pt_avg + (1 - w) * bt}
    groups = [("alle", np.ones(len(test), bool))] if model == "A" else \
        [(f"{int(L)} min", (test.lead_min == L).to_numpy()) for L in sorted(test.lead_min.unique())]
    rows, boot = [], {}
    yt = test.y.to_numpy("float64")
    for g, msk in groups:
        for name, p in preds.items():
            rows.append({"modell": model, "lead": g, "metode": name, **metrics(yt[msk], p[msk])})
        d = test.date.to_numpy()[msk]
        names = list(preds)
        boot[g] = {"snitt_mot_én": bootstrap_mae_gain(d, yt[msk], preds[names[2]][msk], preds[names[1]][msk]),
                   "blanding_mot_én": bootstrap_mae_gain(d, yt[msk], preds[names[3]][msk], preds[names[1]][msk]),
                   "blanding_mot_baseline": bootstrap_mae_gain(d, yt[msk], preds[names[3]][msk], bt[msk])}
    info = {"seeds": SEEDS, "trær": [int(x["best"]) for x in r], "w_valgt_på_valid": w,
            "valid_MAE_per_w": w_scores, "valid_MAE_én": mae(pv_single, yv), "valid_MAE_snitt": mae(pv_avg, yv),
            "bootstrap": boot}
    return rows, info


def cv_check():
    """Samme spørsmål over mars-okt fra ablasjonens måned-for-måned-kjøringer (V4, 2 seeds)."""
    files = {m: [DATA / "ablation" / "cv" / f"V4_{m}_s{s}_n1000000.npz" for s in (0, 1)]
             for m in pd.period_range("2025-03", "2025-10", freq="M").astype(str)}
    if not all(f.exists() for fs in files.values() for f in fs):
        return None
    y, b, p0, p1 = [], [], [], []
    for fs in files.values():
        a, c = np.load(fs[0], allow_pickle=True), np.load(fs[1], allow_pickle=True)
        y.append(a["y"]); b.append(a["base"]); p0.append(a["pred"]); p1.append(c["pred"])
    y, b, p0, p1 = (np.concatenate(v).astype("float64") for v in (y, b, p0, p1))
    avg = (p0 + p1) / 2
    w, scores = best_w(avg, b, y)
    return {"MAE_baseline": mae(b, y), "MAE_én_seed_snitt": (mae(p0, y) + mae(p1, y)) / 2,
            "MAE_seed_snitt": mae(avg, y), "w_best": w, "MAE_blanding": scores[w]}


all_rows, report = [], {}
for model in [m for m in args.model.split(",") if m in SPLITS]:
    valid, test = train_seeds(model)
    rows, info = evaluate(model, valid, test)
    all_rows += rows
    report[model] = info
report["cv_mars_okt_V4_1mill"] = cv_check()

res = pd.DataFrame(all_rows)
res.to_csv(REPORTS / "ensemble.csv", index=False)
(REPORTS / "ensemble.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nMAE (s), test nov-des 2025:")
print(res.pivot_table(index=["modell", "metode"], columns="lead", values="MAE_s", sort=False).round(2).to_string())
for model in report:
    if model in SPLITS:
        info = report[model]
        print(f"\n[{model}] w valgt på valid: {info['w_valgt_på_valid']}, trær per seed: {info['trær']}")
        for g, bb in info["bootstrap"].items():
            print("  " + g + ": " + ", ".join(f"{k} {v['forbedring_%']:.2f} % [{v['ci95_lav_%']:.2f}, {v['ci95_høy_%']:.2f}]"
                                          for k, v in bb.items()))
print("\nKontroll over mars-okt (V4, 1 mill. rader):", report["cv_mars_okt_V4_1mill"])
