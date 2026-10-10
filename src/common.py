"""Felles kode for trening og evaluering (datainnlasting, features, metrikker)."""
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
# DATA_DIR=<mappe> lar skriptene jobbe i en annen datamappe (brukes av testene med syntetiske data)
CUSTOM_DATA = bool(os.environ.get("DATA_DIR"))
if CUSTOM_DATA:
    DATA = Path(os.environ["DATA_DIR"]).resolve()

CAT = ["line", "stop", "direction"]

# Utvalg og perioder, felles for alle skriptene. Hvilke turer som havner i hvert utvalg avgjøres av
# bucket(journey_id) < SAMPLE_PCT (hele turer holdes samlet), se split_hash_sql.
SAMPLE_PCT = {"train": 20, "valid": 25, "test": 25}
SPLITS = {"train": ("2025-01-01", "2025-08-31"),
          "valid": ("2025-09-01", "2025-10-31"),
          "test":  ("2025-11-01", "2025-12-31")}
# Alle andre datoer avledes fra SPLITS, så en ny periode bare trenger endringer her:
DATA_START = "2024-12-03"                  # første dag i Enturs datasett; bare historikk før treningsperioden
P1_END = SPLITS["train"][1]                # slutt på treningsperioden (fold-historikk for early stopping)
P2_END = SPLITS["valid"][1]                # slutt på trening + valid (fold-historikk for retreningen)
TEST_PERIOD = f"{SPLITS['test'][0]} - {SPLITS['test'][1]}"
MND = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]


def months(start, end):
    """Månedsstarter ('YYYY-MM-01') fra måneden til start til og med måneden til end."""
    return [str(p.start_time.date()) for p in pd.period_range(start[:7], end[:7], freq="M")]


HIST_MONTHS = months(SPLITS["train"][0], SPLITS["test"][1])    # måneder som trenger baseline-historikk
# Måned-for-måned-evaluering: hver måned M trenes på månedene før M-1 og stoppes tidlig på M-1, så den første
# måneden som kan evalueres er to måneder etter treningsstart (mars). CV-valget bruker bare månedene før testen.
ROLLING_MONTHS = HIST_MONTHS[2:]                                # mars-des
CV_MONTHS = months(ROLLING_MONTHS[0], SPLITS["valid"][1])       # mars-okt


def period_label(split):
    """'nov-des 2025' for test, osv. (til utskrifter og figurer)."""
    a, b = (pd.Timestamp(d) for d in SPLITS[split])
    return f"{MND[a.month - 1]}-{MND[b.month - 1]} {b.year}" if a.year == b.year else \
        f"{MND[a.month - 1]} {a.year}-{MND[b.month - 1]} {b.year}"


# Hvordan turene fordeles på utvalg (og lead i realtime.py). duckdb (standard): DuckDBs hash(), som gir de
# publiserte resultatene, men ikke er garantert lik mellom DuckDB-versjoner (derfor er duckdb låst i
# requirements.txt). md5: de første 15 hex-sifrene i md5(journey_id), likt i alle versjoner og utenfor DuckDB
# (int(hashlib.md5(id.encode()).hexdigest()[:15], 16) % 100). Bytte gir andre turer i hvert utvalg, så hele
# pipelinen må kjøres på nytt.
def split_hash_sql(expr, modulo, method):
    if method == "md5":
        return f"(('0x' || left(md5({expr}), 15))::BIGINT % {modulo})"
    if method == "duckdb":
        return f"(hash({expr}) % {modulo})"
    raise SystemExit(f"SPLIT_HASH må være duckdb eller md5, ikke {method!r}")

# Sanntidsfeatures fra runde 2 av modell B (src/realtime.py); runde 1 = alle andre rt_-features
RT_ROUND2 = ("rt_prev_", "rt_seg_", "rt_segcum", "rt_segcov", "rt_in_")

# Kolonner som ikke er features: mål, dato, baseline-prognosene og id-/visningskolonner til appen
NON_FEATURES = ("y", "date", "baseline", "baseline_recent", "journey_id", "stop_name", "origin_name", "dest_name")

LGB_PARAMS = dict(objective="l1", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  cat_smooth=20, max_cat_threshold=64, max_bin=255,
                  num_threads=0, verbose=-1, seed=42)

# GPU (OpenCL): sett miljøvariabelen LGB_DEVICE=gpu. Krever en LightGBM-build med GPU-støtte.
# NB: GPU-versjonen tåler maks 256 bins per feature. 'stop' har tusenvis av kategorier, så
# modellene i dette prosjektet må kjøres på CPU ('bin size ... cannot run on GPU').
# max_bin=63 er LightGBMs anbefaling for GPU - det endrer modellen litt, så sammenlign bare
# kjøringer med samme enhet.
if os.environ.get("LGB_DEVICE", "").lower() == "gpu":
    LGB_PARAMS.update(device_type="gpu", max_bin=63, gpu_use_dp=False)
    os.environ.setdefault("BOOST_COMPUTE_USE_OFFLINE_CACHE", "0")   # unngår feil i OpenCL-kernel-cachen når Windows ikke bruker UTF-8 (f.eks. japansk systemspråk)


def raw_files():
    """Rådata: månedsfiler i data/raw/ (fra retrieval.py) + ev. den eldre atb_2025.parquet i prosjektroten."""
    files = [f for f in sorted((DATA / "raw").glob("atb_*.parquet")) if f.stat().st_size > 10_000]  # hopp over tomme
    if (ROOT / "atb_2025.parquet").exists() and not CUSTOM_DATA:
        files = [f for f in files if not f.name.startswith("atb_2025")] + [ROOT / "atb_2025.parquet"]
    return files


def raw_sql(files):
    """DuckDB-uttrykk som leser alle rådatafilene som én tabell."""
    return "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "], union_by_name=true)"


def connect_duckdb(path):
    """DuckDB-tilkobling med felles oppsett. Valgfritt: DUCKDB_MEMORY_LIMIT=8GB og DUCKDB_THREADS=8."""
    import duckdb
    con = duckdb.connect(str(path))
    con.execute(f"SET temp_directory='{(DATA / 'tmp').as_posix()}'")
    if os.environ.get("DUCKDB_MEMORY_LIMIT"):
        con.execute(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
    if os.environ.get("DUCKDB_THREADS"):
        con.execute(f"SET threads={int(os.environ['DUCKDB_THREADS'])}")
    return con


def ensure_dirs():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)


def load(split):
    """Leser data/<split>.parquet med kompakte datatyper."""
    df = pd.read_parquet(DATA / f"{split}.parquet")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(["int64", "int32"]).columns:
        if c != "y":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    df["date"] = pd.to_datetime(df["date"])
    return df


def write_dataset_meta(name, **options):
    """Skriver data/<name>_meta.json (oppsett og byggetidspunkt) når features.py/realtime.py har bygget datasettene.
    Lagrede prognoser (data/ens/) knyttes til dette, så de ikke gjenbrukes etter at datasettene er bygget på nytt."""
    meta = dict(options, bygget=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    (DATA / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def dataset_meta(name):
    """Innholdet i data/<name>_meta.json, eller {} for datasett bygget før filen fantes."""
    p = DATA / f"{name}_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fingerprint(df):
    """Fingeravtrykk av radene: målet y vektet med posisjonen. Endres når radene byttes ut eller kommer i en
    annen rekkefølge (DuckDB garanterer ikke rekkefølgen når datasettene bygges på nytt)."""
    y = np.asarray(df["y"], dtype="int64")
    return int((y * (np.arange(len(y)) % 9973 + 1)).sum())


def signature(*parts):
    """Kort hash av oppsettet (features, parametre, datasett), lagres sammen med prognosene."""
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_npz(path, **arrays):
    """np.savez via en midlertidig fil, så en avbrutt kjøring aldri etterlater en halvskrevet fil."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def replace_when_done(path, write):
    """Skriver filen via en midlertidig fil (write(tmp_path)) og bytter den inn først når den er ferdig."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def cache_ok(path, allow_missing=(), **expected):
    """True hvis lagrede prognoser i path finnes og ble laget for de samme radene og det samme oppsettet
    (f.eks. fp_valid=..., fp_test=..., sig=...). Ellers skrives årsaken, og prognosene må lages på nytt."""
    if not path.exists():
        return False
    try:
        r = np.load(path)
        r.files
    except Exception:  # noqa: BLE001 - f.eks. en fil fra en avbrutt kjøring før skrivingen ble atomisk
        print(f"  {path.name}: kan ikke leses - kjøres på nytt", flush=True)
        return False
    for k, v in expected.items():
        if k not in r.files:
            if k in allow_missing:
                continue
            print(f"  {path.name}: '{k}' mangler (laget før denne sjekken fantes) - kjøres på nytt", flush=True)
            return False
        if r[k].item() != v:
            print(f"  {path.name}: {k} stemmer ikke (datasett eller oppsett er endret) - kjøres på nytt", flush=True)
            return False
    return True


def load_cached(path, script, rows, **expected):
    """Leser lagrede prognoser og stopper hvis de ikke passer til radene de skal kobles mot.
    rows = {array: antall rader}, expected = fingeravtrykk (f.eks. fp_test=...).
    Filer laget før fingeravtrykkene fantes gir bare en advarsel."""
    if not path.exists():
        sys.exit(f"Mangler {path} - kjør python {script} først.")
    try:
        r = np.load(path)
        r.files
    except Exception:  # noqa: BLE001
        sys.exit(f"{path.name} kan ikke leses - kjør python {script} på nytt.")
    for key, n in rows.items():
        if len(r[key]) != n:
            sys.exit(f"{path.name}: {len(r[key]):,} prognoser, men {n:,} rader. Kjør python {script} på nytt.")
    for k, v in expected.items():
        if k not in r.files:
            print(f"Advarsel: {path.name} har ikke fingeravtrykk, så det kan ikke sjekkes at prognosene passer til "
                  f"radene. Kjør python {script} på nytt for å være sikker.", flush=True)
            break
        if r[k].item() != v:
            sys.exit(f"{path.name} passer ikke til radene i data/ (er datasettene bygget på nytt?). "
                     f"Kjør python {script} på nytt.")
    return r


def round1_status(against="realtime"):
    """(kan brukes, melding) for runde 1-prognosene til modell B (python src/train_rt.py --r1): de må være laget av
    de samme datasettene som det de sammenlignes med. against = "realtime" (datasettene nå) eller "B" (prognosene
    i reports/rt_metrics.json)."""
    if not (REPORTS / "rt_test_predictions_r1.parquet").exists():
        return False, "Ingen sammenligning med runde 1: kjør python src/train_rt.py --r1 først"
    meta_path = REPORTS / "rt_metrics_r1.json"
    if not meta_path.exists():
        return True, ("Advarsel: runde 1-prognosene har ingen metadata, så det kan ikke sjekkes at de er fra de samme "
                      "datasettene (kjør python src/train_rt.py --r1 for å være sikker)")
    built = json.loads(meta_path.read_text(encoding="utf-8")).get("datasett")
    if against == "B":
        b_path = REPORTS / "rt_metrics.json"
        now = json.loads(b_path.read_text(encoding="utf-8")).get("datasett") if b_path.exists() else None
    else:
        now = dataset_meta("realtime")
    if built != now:
        return False, ("Ingen sammenligning med runde 1: prognosene er fra en annen bygging av datasettene "
                       "(kjør python src/train_rt.py --r1 på nytt)")
    return True, ""


def read_parquet_optional(path, columns, optional=()):
    """Leser kolonnene, pluss de valgfrie hvis filen har dem (f.eks. baseline_recent i datasett bygget før den
    fantes)."""
    try:
        return pd.read_parquet(path, columns=list(columns) + list(optional))
    except Exception:  # noqa: BLE001 - manglende kolonne gir ulike unntak i ulike pyarrow-versjoner
        return pd.read_parquet(path, columns=list(columns))


def set_categories(*dfs):
    """Gir kategorikolonnene like koder på tvers av datasett (nødvendig for LightGBM)."""
    for c in CAT:
        cats = pd.Index(pd.concat([d[c].astype("object") for d in dfs]).unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype("object"), categories=cats)


def feature_columns(df):
    """Feature-kolonner. refit_* er historikk til retreningssteget (erstatter hist_* der)."""
    return [c for c in df.columns if c not in NON_FEATURES and not c.startswith("refit_")]


def use_refit_history(df):
    """Bytter hist_* med refit_hist_* (fold-historikk over trening+valid) før retrening."""
    for c in [c for c in df.columns if c.startswith("refit_")]:
        df[c[len("refit_"):]] = df.pop(c)
    return df


def metrics(y, pred):
    err = np.asarray(pred, dtype="float64") - np.asarray(y, dtype="float64")
    return {"MAE_s": float(np.mean(np.abs(err))),
            "RMSE_s": float(np.sqrt(np.mean(err ** 2))),
            "innen_1min_%": float(np.mean(np.abs(err) <= 60) * 100),
            "innen_2min_%": float(np.mean(np.abs(err) <= 120) * 100)}


def bootstrap_mae_gain(dates, y, pred_model, pred_base, n_boot=2000, seed=0, block="day"):
    """95 %-intervall for relativ MAE-forbedring, med hele dager som trekkenhet
    (rader samme dag er ikke uavhengige: samme vær, trafikk og hendelser).
    block="week": hele uker (mandag-søndag) som trekkenhet. Dagene er heller ikke uavhengige - de verste dagene
    kommer samlet (andre halvdel av november) - så ukeintervallet er ærligere, men bredere og grovere."""
    d = pd.DataFrame({"date": dates, "em": np.abs(pred_model - y), "eb": np.abs(pred_base - y)})
    if block == "week":
        key = pd.to_datetime(d["date"]).dt.to_period("W-SUN")
    elif block == "day":
        key = d["date"]
    else:
        raise ValueError(f"block må være day eller week, ikke {block!r}")
    per_block = d.groupby(key)[["em", "eb"]].sum()
    sm, sb = per_block["em"].to_numpy(), per_block["eb"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sm), size=(n_boot, len(sm)))
    gain = 1 - sm[idx].sum(1) / sb[idx].sum(1)
    out = {"forbedring_%": float((1 - sm.sum() / sb.sum()) * 100),
           "ci95_lav_%": float(np.percentile(gain, 2.5) * 100),
           "ci95_høy_%": float(np.percentile(gain, 97.5) * 100),
           "dager": int(d["date"].nunique())}
    if block == "week":
        out["uker"] = int(len(sm))
    return out
