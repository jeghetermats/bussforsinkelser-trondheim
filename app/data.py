"""Datainnlasting og hjelpefunksjoner for appen. Alt er forhåndsberegnet av src/train.py."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# Farger brukt konsekvent i alle figurer
COLOR_ACTUAL = "#1A1A1A"
COLOR_MODEL = "#1F4E79"
COLOR_BASELINE = "#8A8F98"


def _find(name: str) -> Path:
    """Bruker den fulle filen hvis den finnes, ellers hurtigversjonen (_quick)."""
    full = REPORTS / name
    if full.exists():
        return full
    return full.with_name(full.stem + "_quick" + full.suffix)


def line_label(line_ref: str) -> str:
    """'ATB:Line:2_10' -> '10'"""
    return str(line_ref).rsplit("_", 1)[-1]


def fmt_delay(seconds: float) -> str:
    """Forsinkelse som tekst: '46 s', '-46 s', '1 min 07 s'."""
    sign = "-" if seconds < 0 else ""
    s = int(round(abs(seconds)))
    if s < 60:
        return f"{sign}{s} s"
    return f"{sign}{s // 60} min {s % 60:02d} s"


UKEDAGER = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
MÅNEDER = ["januar", "februar", "mars", "april", "mai", "juni", "juli", "august", "september",
           "oktober", "november", "desember"]


def fmt_date(d) -> str:
    """date -> 'mandag 1. desember 2025'"""
    return f"{UKEDAGER[d.weekday()]} {d.day}. {MÅNEDER[d.month - 1]} {d.year}"


def fmt_clock(minute_of_day: int) -> str:
    m = int(minute_of_day) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


@st.cache_data(show_spinner="Laster testdata ...")
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(_find("test_predictions.parquet"))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["line_no"] = df["line"].astype(str).map(line_label)
    return df


@st.cache_data
def load_trips() -> pd.DataFrame:
    """Én rad per tur: linje, retning, dato, avgangstid, fra/til."""
    df = load_predictions()
    trips = (df.groupby("trip", observed=True)
               .agg(line_no=("line_no", "first"), direction=("direction", "first"),
                    date=("date", "first"), start=("trip_start_min", "first"),
                    origin=("origin_name", "first"), dest=("dest_name", "first"))
               .reset_index())
    trips["origin"] = trips["origin"].astype(str)
    trips["dest"] = trips["dest"].astype(str)
    return trips


@st.cache_data
def load_metrics() -> dict:
    return json.loads(_find("metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def load_rolling() -> pd.DataFrame | None:
    p = _find("rolling_eval.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_importance() -> pd.DataFrame | None:
    p = _find("feature_importance.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_oracle() -> pd.DataFrame | None:
    p = _find("oracle.csv")
    return pd.read_csv(p) if p.exists() else None
