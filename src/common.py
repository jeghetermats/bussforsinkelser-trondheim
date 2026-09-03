"""Felles kode for trening og evaluering (features, baselines, metrikker)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

CAT = ["line", "stop", "direction"]

# Historiske forsinkelser per gruppe, brukt som features ("target encoding").
# (navn, grupperingsnøkler, aggregering)
HIST_SPECS = [
    ("hist_med_lsdhd", ["line", "stop", "direction", "hour", "daytype"], "median"),
    ("hist_n_lsdhd",   ["line", "stop", "direction", "hour", "daytype"], "size"),
    ("hist_mean_lsd",  ["line", "stop", "direction"], "mean"),
    ("hist_std_lsd",   ["line", "stop", "direction"], "std"),
    ("hist_mean_lhd",  ["line", "hour", "daytype"], "mean"),
    ("hist_mean_sh",   ["stop", "hour"], "mean"),
]
HIST_COLS = [name for name, _, _ in HIST_SPECS]

LGB_PARAMS = dict(objective="l1", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  cat_smooth=20, max_cat_threshold=64, max_bin=255,
                  num_threads=0, verbose=-1, seed=42)


def load(split):
    """Leser data/<split>.parquet og gjør klar kolonnetyper og dagtype."""
    df = pd.read_parquet(DATA / f"{split}.parquet")
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        df[c] = df[c].astype("int32")
    df["date"] = pd.to_datetime(df["date"])
    df["daytype"] = np.select([df.weekday == 7, df.weekday == 6], [2, 1], 0).astype("int8")  # hverdag/lør/søn
    df.loc[df.is_holiday == 1, "daytype"] = 2                                                   # helligdag ~ søndag
    return df


def set_categories(*dfs):
    """Gir kategorikolonnene like koder på tvers av datasett (nødvendig for LightGBM)."""
    for c in CAT:
        cats = pd.Index(pd.concat([d[c].astype("object") for d in dfs]).unique())
        for d in dfs:
            d[c] = pd.Categorical(d[c].astype("object"), categories=cats)


ID_COLS = ("y", "date", "journey_id", "stop_name", "origin_name", "dest_name")   # ikke features


def base_features(df):
    return [c for c in df.columns if c not in ID_COLS and c not in HIST_COLS]


def metrics(y, pred):
    err = pred - y
    return {"MAE_s": float(np.mean(np.abs(err))),
            "RMSE_s": float(np.sqrt(np.mean(err ** 2))),
            "innen_1min_%": float(np.mean(np.abs(err) <= 60) * 100),
            "innen_2min_%": float(np.mean(np.abs(err) <= 120) * 100)}


def group_median_baseline(train, test):
    """Historisk median med fallback til grovere grupper når en kombinasjon mangler."""
    levels = [["line", "stop", "direction", "hour", "daytype"],
              ["line", "stop", "direction"],
              ["line", "hour", "daytype"],
              ["line"]]
    pred = pd.Series(np.nan, index=test.index, dtype="float64")
    for keys in levels:
        med = train.groupby(keys, observed=True)["y"].median().rename("m")
        pred = pred.fillna(test[keys].join(med, on=keys)["m"])
    return pred.fillna(train.y.median()).to_numpy()


def hist_features(fit, apply):
    """Beregner historiske statistikker på `fit` og slår dem opp for radene i `apply`."""
    out = pd.DataFrame(index=apply.index)
    for name, keys, agg in HIST_SPECS:
        stat = fit.groupby(keys, observed=True)["y"].agg(agg).rename(name).astype("float32")
        out[name] = apply[keys].join(stat, on=keys)[name]
    return out


def oof_hist_features(df, k=5):
    """Out-of-fold: hver rad får statistikk beregnet UTEN sin egen fold.
    Uten dette ville radens egen forsinkelse lekke inn i featuren, og modellen ville
    stole altfor mye på den (overfitting). Foldene er hele dager, så en tur aldri deler seg."""
    fold = df["date"].dt.dayofyear.to_numpy() % k
    out = pd.DataFrame(index=df.index, columns=HIST_COLS, dtype="float32")
    for f in range(k):
        mask = fold == f
        out.loc[mask, HIST_COLS] = hist_features(df.loc[~mask], df.loc[mask]).to_numpy()
    return out
