"""Datainnlasting og hjelpefunksjoner for appen. Alt er forhåndsberegnet av src/train.py."""
import json
from pathlib import Path

import numpy as np
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


def fmt_num(x: float, dec: int = 1, sign: bool = False) -> str:
    """Tall med desimalkomma: 3.64 -> '3,6', med sign=True '+3,6'."""
    return (f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}").replace(".", ",")


UKEDAGER = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
MÅNEDER = ["januar", "februar", "mars", "april", "mai", "juni", "juli", "august", "september",
           "oktober", "november", "desember"]
MND = [m[:3] for m in MÅNEDER]


def fmt_date(d) -> str:
    """date -> 'mandag 1. desember 2025'"""
    return f"{UKEDAGER[d.weekday()]} {d.day}. {MÅNEDER[d.month - 1]} {d.year}"


def fmt_clock(minute_of_day: int) -> str:
    m = int(minute_of_day) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


# De store tabellene caches med cache_resource: én delt kopi for alle besøkende og omkjøringer.
# cache_data lager en ny kopi ved hvert kall, og da gikk appen over minnegrensen på Streamlit Cloud.
# Tabellene skal derfor bare leses, aldri endres på stedet (bruk .copy() ved behov).
@st.cache_resource(show_spinner="Laster testdata ...")
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(_find("test_predictions.parquet"))
    # Linjenummer som kategori (noen hundre verdier), ikke 1,7 mill. Python-strenger
    df["line_no"] = df["line"].cat.rename_categories(lambda c: line_label(c))
    return df


@st.cache_resource
def load_trips() -> pd.DataFrame:
    """Én rad per tur: linje, retning, dato, avgangstid, fra/til."""
    df = load_predictions()
    trips = (df.groupby("trip", observed=True)
               .agg(line_no=("line_no", "first"), direction=("direction", "first"),
                    date=("date", "first"), start=("trip_start_min", "first"),
                    origin=("origin_name", "first"), dest=("dest_name", "first"), mean_y=("y", "mean"))
               .reset_index())
    trips["line_no"] = trips["line_no"].astype(str)
    trips["date"] = pd.to_datetime(trips["date"]).dt.date
    trips["origin"] = trips["origin"].astype(str)
    trips["dest"] = trips["dest"].astype(str)
    return trips


def _index_by_trip(trip_ids: np.ndarray):
    """Radnumrene sortert etter tur, så radene til én tur finnes med binærsøk i stedet for å filtrere alle radene."""
    order = np.argsort(trip_ids, kind="stable")
    return order, trip_ids[order]


@st.cache_resource
def _trip_index():
    return _index_by_trip(load_predictions()["trip"].to_numpy())


@st.cache_resource
def _rt_index():
    return _index_by_trip(load_rt_predictions()["trip"].to_numpy())


def _rows(order, keys, trip):
    lo, hi = np.searchsorted(keys, [trip, trip + 1])
    return order[lo:hi]


def trip_rows(df: pd.DataFrame, trip: int) -> pd.DataFrame:
    """Radene til én tur (df = load_predictions()), sortert etter stoppnummer."""
    order, keys = _trip_index()
    return df.iloc[_rows(order, keys, int(trip))].sort_values("seq")


def rt_trip_rows(rt: pd.DataFrame, trip: int, lead: int) -> pd.DataFrame:
    """Modell B-prognosene for én tur og ett prognosetidspunkt (rt = load_rt_predictions())."""
    order, keys = _rt_index()
    r = rt.iloc[_rows(order, keys, int(trip))]
    return r[r.lead_min == lead]


@st.cache_data(max_entries=500)
def hourly_profile(line: str, direction: str, lead: int | None) -> pd.DataFrame:
    """Median faktisk forsinkelse, prognose og baseline per time for én linje og retning (minutter).
    lead=None: modell A; ellers modell B med prognosen laget lead minutter før avgang (A der B mangler)."""
    df = load_predictions()
    d = df.loc[(df.line_no == line) & (df.direction == direction),
               ["trip", "seq", "minute_of_day", "y", "pred_lgbm", "pred_baseline"]]
    d = d.assign(pred=d["pred_lgbm"])
    rt = load_rt_predictions() if lead is not None else None
    if rt is not None:
        order, keys = _rt_index()
        tu = np.unique(d.trip.to_numpy())
        lo, hi = np.searchsorted(keys, tu), np.searchsorted(keys, tu + 1)
        rb = rt.iloc[np.concatenate([order[a:b] for a, b in zip(lo, hi)] + [np.array([], dtype=np.int64)])]
        rb = rb.loc[rb.lead_min == lead, ["trip", "seq", "pred_rt"]]
        d = d.merge(rb, on=["trip", "seq"], how="left")
        d["pred"] = d["pred_rt"].fillna(d["pred_lgbm"])
    return (d.assign(time=d.minute_of_day // 60)
             .groupby("time")[["y", "pred", "pred_baseline"]].median() / 60).reset_index()


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
def load_permutation_importance() -> pd.DataFrame | None:
    """Permutasjonsviktighet fra src/importance.py (økning i MAE når en feature stokkes), hvis den finnes."""
    p = REPORTS / "permutation_importance.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_oracle() -> pd.DataFrame | None:
    p = _find("oracle.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Laster sanntidsprognoser ...")
def load_rt_predictions() -> pd.DataFrame | None:
    """Modell B: én rad per (tur, stopp, lead). Kolonner trip, seq, lead_min, pred_rt."""
    p = _find("rt_test_predictions.parquet")
    if not p.exists():
        return None
    rt = pd.read_parquet(p, columns=["trip", "seq", "lead_min", "pred_rt"])
    rt["trip"] = rt["trip"].astype("int32")   # alle turer har id; vanlig int32 i stedet for nullbar Int32
    return rt


@st.cache_data
def load_rt_results() -> pd.DataFrame | None:
    p = _find("rt_results.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_final() -> dict | None:
    """Endelige modeller (src/finalize.py): A = seed-snitt + blanding, B = runde 2 + seed-snitt."""
    p = _find("final.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
