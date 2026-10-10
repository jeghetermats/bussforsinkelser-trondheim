"""
Henter timesvær for Trondheim fra Open-Meteo (gratis, ingen API-nøkkel) for perioden rådataene dekker.

Standard er målt vær (Historical Weather API). Det er ikke kjent dagen før, så modell A får litt informasjon den ikke
ville hatt i drift. Med --forecast hentes i stedet værvarselet som ble laget 24 timer før (Previous Runs API,
variablene *_previous_day1, arkivert fra januar 2024). Kolonnene får samme navn, pluss 'source', så features.py
kan brukes uendret, og data/features_meta.json viser hvilken kilde datasettene ble bygget med.

Kjør:  python src/weather.py              (målt vær)
       python src/weather.py --forecast   (varsel fra dagen før)
og deretter  python src/features.py  på nytt (og resten av pipelinen).
"""
import calendar
import datetime as dt
import sys

import pandas as pd
import requests

from common import DATA, raw_files


def data_period(files):
    """Første og siste dag i rådatafilene (atb_ÅÅÅÅ_MM.parquet = én måned, atb_ÅÅÅÅ.parquet = hele året)."""
    months = []
    for f in files:
        parts = f.stem.split("_")
        year = int(parts[1])
        months += [(year, int(parts[2]))] if len(parts) > 2 else [(year, 1), (year, 12)]
    (y0, m0), (y1, m1) = min(months), max(months)
    end = dt.date(y1, m1, calendar.monthrange(y1, m1)[1])
    return dt.date(y0, m0, 1), min(end, dt.date.today() - dt.timedelta(days=2))   # arkivet ligger et par dager bak


FORECAST = "--forecast" in sys.argv
VARS = ["temperature_2m", "precipitation", "snowfall", "snow_depth", "wind_speed_10m"]

files = raw_files()
if not files:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
start, end = data_period(files)
if FORECAST:
    URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
    hourly = [f"{v}_previous_day1" for v in VARS]           # varselet laget 24 timer før
    start = max(start, dt.date(2024, 1, 1))                 # arkivet starter januar 2024
    source = "varsel dagen før"
else:
    URL = "https://archive-api.open-meteo.com/v1/archive"
    hourly = VARS
    source = "målt"
params = {
    "latitude": 63.43, "longitude": 10.40,          # Trondheim sentrum
    "start_date": start.isoformat(), "end_date": end.isoformat(),
    "hourly": ",".join(hourly),
    "timezone": "Europe/Oslo",
}
r = requests.get(URL, params=params, timeout=120)
if not r.ok:
    sys.exit(f"Open-Meteo svarte {r.status_code}: {r.text[:300]}")
df = pd.DataFrame(r.json()["hourly"]).rename(columns=dict(zip(hourly, VARS)))
for v in VARS:
    if v not in df:
        df[v] = float("nan")
    if df[v].isna().mean() > 0.5:
        print(f"Advarsel: {df[v].isna().mean():.0%} av timene mangler {v}" + (" (finnes ikke i varselarkivet)"
                                                                              if FORECAST else ""))
df["time"] = pd.to_datetime(df["time"])
df = df.drop_duplicates("time")  # sommertid-overgang gir duplikate timer
df["precip_3h"] = df["precipitation"].rolling(3, min_periods=1).sum()   # nedbør siste 3 timer
df["source"] = source
out = DATA / "weather_trondheim.parquet"
out.parent.mkdir(exist_ok=True)
df.to_parquet(out, index=False)
print(f"Lagret {len(df):,} timer med vær ({source}, {start} - {end}) -> {out}")
