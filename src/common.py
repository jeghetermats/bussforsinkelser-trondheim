"""Felles kode for trening og evaluering (datainnlasting, features, metrikker)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"

CAT = ["line", "stop", "direction"]

# Historiske features (se src/history.py)
HIST_COLS = ["hist_med_lsdhd", "hist_mean_lsd", "hist_std_lsd", "hist_mean_lhd", "hist_mean_sh"]

# Kolonner som ikke er features: mål, dato, baseline-prognosen og id-/visningskolonner til appen
NON_FEATURES = ("y", "date", "baseline", "journey_id", "stop_name", "origin_name", "dest_name")

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


def bootstrap_mae_gain(dates, y, pred_model, pred_base, n_boot=2000, seed=0):
    """95 %-intervall for relativ MAE-forbedring, med hele dager som trekkenhet
    (rader samme dag er ikke uavhengige: samme vær, trafikk og hendelser)."""
    d = pd.DataFrame({"date": dates, "em": np.abs(pred_model - y), "eb": np.abs(pred_base - y)})
    per_day = d.groupby("date")[["em", "eb"]].agg(["sum", "count"])
    sm, sb, n = per_day[("em", "sum")].to_numpy(), per_day[("eb", "sum")].to_numpy(), per_day[("em", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sm), size=(n_boot, len(sm)))
    gain = 1 - sm[idx].sum(1) / sb[idx].sum(1)
    return {"forbedring_%": float((1 - sm.sum() / sb.sum()) * 100),
            "ci95_lav_%": float(np.percentile(gain, 2.5) * 100),
            "ci95_høy_%": float(np.percentile(gain, 97.5) * 100),
            "dager": int(len(sm))}
