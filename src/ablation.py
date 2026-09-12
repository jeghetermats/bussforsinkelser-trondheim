"""
Ablasjon: hvor ble de ~2 sekundene av mellom gammelt og nytt oppsett?

Kjede fra gammelt til nytt oppsett, én endring om gangen (hver endring beholdes videre):
  V0  gammelt oppsett: statistikk fra 12 %-utvalget, fold-historikk (5 dagfolder), gammel kalender,
      antall-feature, treningsrader fra des 2024-aug 2025
  V1  + statistikk fra ALLE rader
  V2  + ny kalender (julaften/nyttårsaften som helligdag)
  V3  + uten antall-featuren (hist_n_lsdhd)
  V4  + nye treningsrader (jan-aug 2025, 20 %-utvalg)
  V5  + historikk kun fra fortiden, oppdatert UKENTLIG
  V6  + historikk kun fra fortiden, oppdatert MÅNEDLIG  (= dagens pipeline, 'B')
Varianter utenfor kjeden:
  C   V6 + trening først fra mars (jan-feb bare som historikk)
  C2  V5 + trening først fra mars
  D   V6 + antall måneder med historikk som feature

Alle varianter bruker samme antall treningsrader (N_TRAIN, trukket med fast seed fra sin radpool), samme
validerings- og testrader, samme LightGBM-innstillinger og samme prosedyre:
  1) tren på treningsutvalget med early stopping på valid (sep-okt)   -> valid-MAE
  2) retren på trening+valid med samme antall trær, prediker test (nov-des) -> test-MAE

VELG VARIANT ETTER VALID-MAE. Test-tallene vises bare for åpenhet - velges det på test, gjelder ikke
konfidensintervallet lenger. Parvis bootstrap over dager mot B viser om forskjeller er mer enn støy.

Krever data/features.duckdb fra features.py (clean-tabellen).
Kjør:  python src/ablation.py                        (alle varianter, seed 0)
       python src/ablation.py --cv --seeds 0,1        (VALG: mars-okt, én måned om gangen)
       python src/ablation.py --only V5,V6 --seeds 0,1,2
       python src/ablation.py --n-train 2000000
Resultater: reports/ablation.csv (+ prognoser i data/ablation/, så ferdige varianter hoppes over).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CAT, LGB_PARAMS, REPORTS, ensure_dirs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_DIR = DATA / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ensure_dirs()

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--seeds", default="0")
ap.add_argument("--n-train", type=int, default=2_000_000)
ap.add_argument("--n-valid", type=int, default=600_000)
ap.add_argument("--cv", action="store_true",
                help="Velg variant over flere måneder (mars-okt) i stedet for én valideringsperiode")
ap.add_argument("--cv-variants", default="V3,V4,V6,D")
ap.add_argument("--n-eval", type=int, default=300_000)
args = ap.parse_args()
if args.cv and "--n-train" not in " ".join(sys.argv):
    args.n_train = 1_000_000          # standard for --cv: mindre utvalg, siden det trenes mange modeller
SEEDS = [int(s) for s in args.seeds.split(",")]

DATA_START, P1_END, P2_END = "2024-12-03", "2025-08-31", "2025-10-31"
VALID = ("2025-09-01", "2025-10-31")
TEST = ("2025-11-01", "2025-12-31")
ROWSETS = {   # (bucket-grense, fra, til)
    "old":    (12, DATA_START, P1_END),
    "new":    (20, "2025-01-01", P1_END),
    "burnin": (20, "2025-03-01", P1_END),
}
VARIANTS = {
    "V0": dict(src="sample", scheme="oof",   cal="old", hist_n=True,  rows="old",    desc="Gammelt oppsett"),
    "V1": dict(src="all",    scheme="oof",   cal="old", hist_n=True,  rows="old",    desc="+ statistikk fra alle rader"),
    "V2": dict(src="all",    scheme="oof",   cal="new", hist_n=True,  rows="old",    desc="+ ny kalender"),
    "V3": dict(src="all",    scheme="oof",   cal="new", hist_n=False, rows="old",    desc="+ uten antall-feature"),
    "V4": dict(src="all",    scheme="oof",   cal="new", hist_n=False, rows="new",    desc="+ nye treningsrader"),
    "V5": dict(src="all",    scheme="week",  cal="new", hist_n=False, rows="new",    desc="+ kun fortid, ukentlig"),
    "V6": dict(src="all",    scheme="month", cal="new", hist_n=False, rows="new",    desc="+ kun fortid, månedlig (=B)"),
    "C":  dict(src="all",    scheme="month", cal="new", hist_n=False, rows="burnin", desc="B + trening fra mars"),
    "C2": dict(src="all",    scheme="week",  cal="new", hist_n=False, rows="burnin", desc="ukentlig + trening fra mars"),
    "D":  dict(src="all",    scheme="month", cal="new", hist_n=False, rows="new",    desc="B + måneder historikk", age=True),
}
REF = "V6"
run_names = [v for v in VARIANTS if not args.only or v in args.only.split(",")]

OLD_HOLIDAYS = ["2025-01-01", "2025-04-13", "2025-04-17", "2025-04-18", "2025-04-20", "2025-04-21",
                "2025-05-01", "2025-05-17", "2025-05-29", "2025-06-08", "2025-06-09", "2024-12-25",
                "2024-12-26", "2025-12-25", "2025-12-26"]

con = duckdb.connect(str(DATA / "features.duckdb"))
con.execute(f"SET temp_directory='{(DATA / 'tmp').as_posix()}'")
if os.environ.get("DUCKDB_MEMORY_LIMIT"):
    con.execute(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
if os.environ.get("DUCKDB_THREADS"):
    con.execute(f"SET threads={int(os.environ['DUCKDB_THREADS'])}")

old_hol = ", ".join(f"DATE '{d}'" for d in OLD_HOLIDAYS)
CAL = {  # (is_holiday-uttrykk, daytype-uttrykk) for gammel og ny kalender
    "new": ("is_holiday", "daytype"),
    "old": (f"(date IN ({old_hol}))::INT",
            f"CASE WHEN weekday = 7 OR date IN ({old_hol}) THEN 2 WHEN weekday = 6 THEN 1 ELSE 0 END"),
}
for cal, (hol, day) in CAL.items():
    con.execute(f"""CREATE OR REPLACE TEMP VIEW rows_{cal} AS
        SELECT * REPLACE ({hol} AS is_holiday, {day} AS daytype), dayofyear(date) % 5 AS fold FROM clean""")

BASE = [c for c in ["line", "stop", "direction", "seq", "n_stops", "route_frac", "sched_min_from_start",
                    "trip_start_min", "minute_of_day", "hour", "weekday", "is_holiday", "daytype",
                    "is_school_break", "daylight_h", "temp", "precip", "snowfall", "snow_depth", "wind",
                    "precip_3h"]
        if c in [r[0] for r in con.sql("DESCRIBE clean").fetchall()]]
SPECS = {
    "lsdhd": (["line", "stop", "direction", "hour", "daytype"],
              ["median(y) AS hist_med_lsdhd", "count(*) AS hist_n_lsdhd"]),
    "lsd": (["line", "stop", "direction"],
            ["avg(y) AS hist_mean_lsd", "stddev_samp(y) AS hist_std_lsd", "median(y) AS med_lsd"]),
    "lhd": (["line", "hour", "daytype"], ["avg(y) AS hist_mean_lhd", "median(y) AS med_lhd"]),
    "sh":  (["stop", "hour"], ["avg(y) AS hist_mean_sh"]),
    "l":   (["line"], ["median(y) AS med_l"]),
    "g":   ([], ["median(y) AS med_g"]),
}
HIST = ["hist_med_lsdhd", "hist_mean_lsd", "hist_std_lsd", "hist_mean_lhd", "hist_mean_sh"]


def exists(name):
    return con.sql(f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{name}'").fetchone()[0] > 0


def src_where(src, period_end):
    """Hvilke rader statistikken bygges på."""
    if src == "all":
        return "TRUE"
    # Gammelt oppsett: bare radene i trenings-/valideringsutvalget (12 % t.o.m. aug, 25 % sep-okt)
    return (f"((bucket < 12 AND date <= DATE '{P1_END}') OR "
            f"(bucket < 25 AND date BETWEEN DATE '{VALID[0]}' AND DATE '{period_end}'))")


def build_hist(src, scheme, cal, period=None):
    """Lager historikktabeller. Returnerer tabellprefiks. Nøkkel 'hkey':
       oof:   0-4 = alle rader i perioden unntatt fold k, 5 = alle rader i perioden
       week/month: periodestart; statistikk fra rader med dato før periodestart."""
    prefix = f"h_{src}_{scheme}_{cal}" + (f"_{period}" if scheme == "oof" else "")
    if exists(prefix + "_g"):
        return prefix
    t0 = time.time()
    if scheme == "oof":
        if period.startswith("cv"):
            m0 = pd.Timestamp(period[2:])
            pend = (m0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            pend = P1_END if period == "p1" else P2_END
        base = f"{src_where(src, pend)} AND date BETWEEN DATE '{DATA_START}' AND DATE '{pend}'"
        keysets = [(str(f), f"{base} AND fold <> {f}") for f in range(5)] + [("5", base)]
    else:
        unit = "week" if scheme == "week" else "month"
        starts = [r[0] for r in con.sql(f"""
            SELECT DISTINCT date_trunc('{unit}', date)::DATE FROM clean
            WHERE date >= DATE '2025-01-01' ORDER BY 1""").fetchall()]
        keysets = [(f"DATE '{s}'", f"{src_where(src, P2_END)} AND date < DATE '{s}'") for s in starts]
    for name, (keys, aggs) in SPECS.items():
        ks = ", ".join(keys)
        parts = [f"SELECT {hk} AS hkey{', ' + ks if keys else ''}, {', '.join(aggs)} FROM rows_{cal} WHERE {w}"
                 + (f" GROUP BY {ks}" if keys else "") for hk, w in keysets]
        con.execute(f"CREATE OR REPLACE TABLE {prefix}_{name} AS " + " UNION ALL ".join(parts))
    print(f"  historikk {prefix}: {len(keysets)} nøkler ({time.time()-t0:.0f}s)", flush=True)
    return prefix


def sample_table(kind, key, seed, n):
    """Fast utvalg av rader (materialisert, så samme rader brukes i begge stegene)."""
    name = f"s_{kind}_{key}_{seed}_{n}"
    if not exists(name):
        if kind == "train":
            b, d0, d1 = ROWSETS[key]
        else:
            b, (d0, d1) = 25, VALID
        con.execute(f"""CREATE TABLE {name} AS SELECT * FROM (
            SELECT * FROM clean WHERE bucket < {b} AND date BETWEEN DATE '{d0}' AND DATE '{d1}'
        ) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed + 1})""")
    return name


def sample_range(tag, b, d0, d1, seed, n):
    """Fast, materialisert utvalg fra et vilkårlig datointervall."""
    name = f"s_{tag}_{d0.replace('-', '')}_{d1.replace('-', '')}_{b}_{seed}_{n}"
    if not exists(name):
        con.execute(f"""CREATE TABLE {name} AS SELECT * FROM (
            SELECT * FROM clean WHERE bucket < {b} AND date BETWEEN DATE '{d0}' AND DATE '{d1}'
        ) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed + 1})""")
    return name


def fetch(rows_sql, cal, prefix, hk_expr, hist_n, age, order=False):
    """Henter rader med kalender, historikk og baseline."""
    joins = []
    for name, (keys, _) in SPECS.items():
        on = " AND ".join([f"t_{name}.hkey = {hk_expr}"] + [f"t_{name}.{k} = r.{k}" for k in keys])
        joins.append(f"LEFT JOIN {prefix}_{name} t_{name} ON {on}")
    cols = [f"r.{c}" for c in BASE] + HIST + (["hist_n_lsdhd"] if hist_n else [])
    if age:
        cols.append(f"date_diff('month', DATE '2024-12-01', {hk_expr}) AS hist_months")
    q = f"""
      SELECT r.y, r.date, {', '.join(cols)},
             coalesce(hist_med_lsdhd, med_lsd, med_lhd, med_l, med_g) AS baseline
      FROM (SELECT * REPLACE ({CAL[cal][0]} AS is_holiday, {CAL[cal][1]} AS daytype), dayofyear(date) % 5 AS fold
            FROM ({rows_sql})) r
      {' '.join(joins)}
      {'ORDER BY r.journey_id, r.seq' if order else ''}"""
    df = con.sql(q).df()
    for c in CAT:
        df[c] = df[c].astype("category")
    return df


def align(dfs):
    for c in CAT:
        cats = pd.api.types.union_categoricals([d[c] for d in dfs]).categories
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype("object"), categories=cats)


def hk(scheme, which):
    if scheme == "oof":
        return "r.fold" if which == "fold" else "5"
    return f"date_trunc('{'week' if scheme == 'week' else 'month'}', r.date)::DATE"



# ================= Valg over flere måneder (--cv) =================
def run_cv():
    """For hver måned M i mars-okt 2025 (aldri nov-des):
         trening = rader før måneden før M, early stopping på måneden før M, evaluering på M.
       Historikk bygges bare fra data før M (fold-variantene: fold-historikk innenfor < M)."""
    months = pd.period_range("2025-03", "2025-10", freq="M")
    names = [v for v in args.cv_variants.split(",") if v in VARIANTS]
    cv_dir = OUT_DIR / "cv"
    cv_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        for m in months:
            m0 = m.start_time.strftime("%Y-%m-%d")
            inner = m - 1
            i0, i1 = inner.start_time.strftime("%Y-%m-%d"), inner.end_time.strftime("%Y-%m-%d")
            e1 = m.end_time.strftime("%Y-%m-%d")
            tr_end = (inner.start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            in_tab = sample_range("inner", 25, i0, i1, 0, args.n_eval // 2)
            ev_tab = sample_range("eval", 25, m0, e1, 0, args.n_eval)
            for name in names:
                v = VARIANTS[name]
                out = cv_dir / f"{name}_{m}_s{seed}_n{args.n_train}.npz"
                if out.exists():
                    continue
                t0 = time.time()
                b, d0, _ = ROWSETS[v["rows"]]
                tr_tab = sample_range("train", b, d0, tr_end, seed, args.n_train)
                age = v.get("age", False)
                if v["scheme"] == "oof":
                    p = build_hist(v["src"], "oof", v["cal"], "cv" + m0.replace("-", ""))
                    tr = fetch(f"SELECT * FROM {tr_tab}", v["cal"], p, hk("oof", "fold"), v["hist_n"], age)
                    va = fetch(f"SELECT * FROM {in_tab}", v["cal"], p, hk("oof", "fold"), v["hist_n"], age)
                    ev = fetch(f"SELECT * FROM {ev_tab}", v["cal"], p, hk("oof", "all"), v["hist_n"], age, order=True)
                else:
                    p = build_hist(v["src"], v["scheme"], v["cal"])
                    h = hk(v["scheme"], None)
                    tr = fetch(f"SELECT * FROM {tr_tab}", v["cal"], p, h, v["hist_n"], age)
                    va = fetch(f"SELECT * FROM {in_tab}", v["cal"], p, h, v["hist_n"], age)
                    ev = fetch(f"SELECT * FROM {ev_tab}", v["cal"], p, h, v["hist_n"], age, order=True)
                align([tr, va, ev])
                feats = [c for c in tr.columns if c not in ("y", "date", "baseline")]
                dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=CAT)
                mdl = lgb.train(dict(LGB_PARAMS, seed=seed), dtr, num_boost_round=3000,
                                valid_sets=[lgb.Dataset(va[feats], va.y, reference=dtr)],
                                callbacks=[lgb.early_stopping(50, verbose=False)])
                pe = mdl.predict(ev[feats], num_iteration=mdl.best_iteration)
                np.savez(out, pred=pe.astype("float32"), y=ev.y.to_numpy(), date=ev.date.astype(str).to_numpy(),
                         base=ev.baseline.to_numpy("float32"), best=mdl.best_iteration)
                print(f"{m} {name} seed {seed}: {mdl.best_iteration} trær, MAE {np.mean(np.abs(pe - ev.y)):.2f} s "
                      f"(baseline {np.mean(np.abs(ev.baseline - ev.y)):.2f}) [{time.time()-t0:.0f}s]", flush=True)

    # ---- oppsummering: snitt over måneder og seeds, parvis mot V6 med bootstrap over dager ----
    recs, err = [], {}
    for f in sorted(cv_dir.glob(f"*_n{args.n_train}.npz")):
        name, mon, sd = f.stem.split("_")[0], f.stem.split("_")[1], int(f.stem.split("_")[2][1:])
        r = np.load(f, allow_pickle=True)
        e = np.abs(r["pred"] - r["y"])
        recs.append({"variant": name, "måned": mon, "seed": sd, "MAE": e.mean(),
                     "baseline_MAE": np.abs(r["base"] - r["y"]).mean(), "trær": int(r["best"])})
        err.setdefault((name, mon), []).append((e, r["date"]))
    res = pd.DataFrame(recs)
    if res.empty:
        raise SystemExit("Ingen cv-resultater ennå.")
    res.to_csv(REPORTS / "ablation_cv_months.csv", index=False)
    per_month = res.groupby(["måned", "variant"]).MAE.mean().unstack()
    print("\nMAE per måned (snitt over seeds):\n" + per_month.round(2).to_string())

    summary = []
    for name in per_month.columns:
        common = [mon for mon in per_month.index if (name, mon) in err and (REF, mon) in err]
        if not common:
            continue
        a = np.concatenate([np.mean([x[0] for x in err[(name, mon)]], axis=0) for mon in common])
        b = np.concatenate([np.mean([x[0] for x in err[(REF, mon)]], axis=0) for mon in common])
        d = np.concatenate([err[(name, mon)][0][1] for mon in common])
        lo, hi = (0.0, 0.0) if name == REF else paired_ci(a, b, d)
        summary.append({"variant": name, "beskrivelse": VARIANTS[name]["desc"], "måneder": len(common),
                        "MAE_snitt": a.mean(), "diff_mot_B": a.mean() - b.mean(), "KI_lav": lo, "KI_høy": hi,
                        "seeds": int(res[res.variant == name].seed.nunique())})
    sm = pd.DataFrame(summary).sort_values("MAE_snitt")
    sm.to_csv(REPORTS / "ablation_cv.csv", index=False)
    print("\nVALG (mars-okt, aldri testperioden): lavest MAE_snitt vinner. KI er parvis bootstrap over dager mot B.")
    print(sm.round(2).to_string(index=False))


def paired_ci(a, b, d, n_boot=2000, seed=0):
    """95 %-KI for MAE(a) - MAE(b), bootstrap over dager."""
    df = pd.DataFrame({"d": d, "x": a - b}).groupby("d")["x"].agg(["sum", "count"])
    sm_, n = df["sum"].to_numpy(), df["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sm_), (n_boot, len(sm_)))
    diffs = sm_[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


if args.cv:
    run_cv()
    raise SystemExit

test_rows = f"SELECT * FROM clean WHERE bucket < 25 AND date BETWEEN DATE '{TEST[0]}' AND DATE '{TEST[1]}'"
meta_path = OUT_DIR / "test_meta.npz"
results = []

for seed in SEEDS:
    for name in run_names:
        v = VARIANTS[name]
        out = OUT_DIR / f"{name}_s{seed}_n{args.n_train}.npz"
        if out.exists():
            print(f"{name} seed {seed}: ferdig fra før - hopper over")
            continue
        t0 = time.time()
        print(f"\n=== {name} (seed {seed}): {v['desc']} ===", flush=True)
        tr_tab = sample_table("train", v["rows"], seed, args.n_train)
        va_tab = sample_table("valid", "sepokt", 0, args.n_valid)   # fast for alle
        age = v.get("age", False)
        if v["scheme"] == "oof":
            p1 = build_hist(v["src"], "oof", v["cal"], "p1")
            p2 = build_hist(v["src"], "oof", v["cal"], "p2")
            tr1 = fetch(f"SELECT * FROM {tr_tab}", v["cal"], p1, hk("oof", "fold"), v["hist_n"], age)
            va1 = fetch(f"SELECT * FROM {va_tab}", v["cal"], p1, hk("oof", "all"), v["hist_n"], age, order=True)
            tr2 = fetch(f"SELECT * FROM {tr_tab}", v["cal"], p2, hk("oof", "fold"), v["hist_n"], age)
            va2 = fetch(f"SELECT * FROM {va_tab}", v["cal"], p2, hk("oof", "fold"), v["hist_n"], age)
            te = fetch(test_rows, v["cal"], p2, hk("oof", "all"), v["hist_n"], age, order=True)
        else:
            p = build_hist(v["src"], v["scheme"], v["cal"])
            h = hk(v["scheme"], None)
            tr1 = fetch(f"SELECT * FROM {tr_tab}", v["cal"], p, h, v["hist_n"], age)
            va1 = fetch(f"SELECT * FROM {va_tab}", v["cal"], p, h, v["hist_n"], age, order=True)
            tr2, va2 = tr1, va1
            te = fetch(test_rows, v["cal"], p, h, v["hist_n"], age, order=True)
        align([tr1, va1, tr2, va2, te])
        feats = [c for c in tr1.columns if c not in ("y", "date", "baseline")]
        params = dict(LGB_PARAMS, seed=seed)

        dtr = lgb.Dataset(tr1[feats], tr1.y, categorical_feature=CAT)
        m1 = lgb.train(params, dtr, num_boost_round=3000,
                       valid_sets=[lgb.Dataset(va1[feats], va1.y, reference=dtr)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        best = m1.best_iteration
        pv = m1.predict(va1[feats], num_iteration=best)
        full = pd.concat([tr2, va2], ignore_index=True)
        m2 = lgb.train(params, lgb.Dataset(full[feats], full.y, categorical_feature=CAT), num_boost_round=best)
        pt = m2.predict(te[feats])

        if not meta_path.exists():
            np.savez(meta_path, y=te.y.to_numpy(), date=te.date.astype(str).to_numpy(),
                     yv=va1.y.to_numpy(), datev=va1.date.astype(str).to_numpy())
        np.savez(out, test=pt.astype("float32"), valid=pv.astype("float32"),
                 base_test=te.baseline.to_numpy("float32"), base_valid=va1.baseline.to_numpy("float32"),
                 best=best, n_test=len(te), n_valid=len(va1))
        print(f"  {best} trær | valid-MAE {np.mean(np.abs(pv - va1.y)):.2f} s | "
              f"test-MAE {np.mean(np.abs(pt - te.y)):.2f} s (baseline {np.mean(np.abs(te.baseline - te.y)):.2f} s) "
              f"[{time.time()-t0:.0f}s]", flush=True)

# ================= Oppsummering =================
meta = np.load(meta_path, allow_pickle=True)
y, dates, yv, datesv = meta["y"], meta["date"], meta["yv"], meta["datev"]
special = np.isin(dates, ["2025-12-24", "2025-12-31"])


rows = []
for f in sorted(OUT_DIR.glob(f"*_n{args.n_train}.npz")):
    name, seed = f.stem.split("_")[0], int(f.stem.split("_")[1][1:])
    r = np.load(f)
    if len(r["test"]) != len(y) or len(r["valid"]) != len(yv):
        continue
    et, ev = np.abs(r["test"] - y), np.abs(r["valid"] - yv)
    rows.append({"variant": name, "seed": seed, "beskrivelse": VARIANTS.get(name, {}).get("desc", ""),
                 "trær": int(r["best"]), "valid_MAE": ev.mean(), "test_MAE": et.mean(),
                 "test_baseline_MAE": np.abs(r["base_test"] - y).mean(),
                 "test_MAE_24_31_des": et[special].mean(), "_et": et, "_ev": ev})
res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit("Ingen resultater ennå.")
res["forbedring_%"] = (1 - res.test_MAE / res.test_baseline_MAE) * 100

for c in ("valid_diff_mot_B", "test_diff_mot_B"):
    res[c] = np.nan
for c in ("valid_KI", "test_KI"):
    res[c] = ""
ref = res[res.variant == REF]
for i, r in res.iterrows():
    rr = ref[ref.seed == r.seed]
    if len(rr) and r.variant != REF:
        lo, hi = paired_ci(r["_ev"], rr.iloc[0]["_ev"], datesv)
        res.loc[i, "valid_diff_mot_B"] = r.valid_MAE - rr.iloc[0].valid_MAE
        res.loc[i, "valid_KI"] = f"[{lo:+.2f}, {hi:+.2f}]"
        lo, hi = paired_ci(r["_et"], rr.iloc[0]["_et"], dates)
        res.loc[i, "test_diff_mot_B"] = r.test_MAE - rr.iloc[0].test_MAE
        res.loc[i, "test_KI"] = f"[{lo:+.2f}, {hi:+.2f}]"
order = list(VARIANTS)
res["_o"] = res.variant.map(order.index)
res = res.sort_values(["seed", "_o"]).drop(columns=["_et", "_ev", "_o"])
res.to_csv(REPORTS / "ablation.csv", index=False)

pd.set_option("display.width", 200)
print("\n" + res.round(2).to_string(index=False))
print("\nVelg variant etter valid_MAE. valid_/test_diff_mot_B < 0 betyr bedre enn B (V6); "
      "KI er parvis bootstrap over dager. Forskjeller der KI dekker 0 er støy.")
chain = res[res.variant.isin([f"V{i}" for i in range(7)])].groupby("variant")[["valid_MAE", "test_MAE"]].mean()
chain = chain.reindex([v for v in order if v in chain.index])
chain["diff_test_fra_forrige"] = chain.test_MAE.diff()
chain["diff_valid_fra_forrige"] = chain.valid_MAE.diff()
print("\nKjeden (snitt over seeds):\n" + chain.round(2).to_string())
