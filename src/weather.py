"""
Henter timesvær for Trondheim fra Open-Meteo (gratis, ingen API-nøkkel) for perioden rådataene dekker.
Kjør:  python src/weather.py   og deretter  python src/features.py  på nytt.
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


files = raw_files()
if not files:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
start, end = data_period(files)
URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 63.43, "longitude": 10.40,          # Trondheim sentrum
    "start_date": start.isoformat(), "end_date": end.isoformat(),
    "hourly": "temperature_2m,precipitation,snowfall,snow_depth,wind_speed_10m",
    "timezone": "Europe/Oslo",
}
r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()
df = pd.DataFrame(r.json()["hourly"])
df["time"] = pd.to_datetime(df["time"])
df = df.drop_duplicates("time")  # sommertid-overgang gir duplikate timer
df["precip_3h"] = df["precipitation"].rolling(3, min_periods=1).sum()   # nedbør siste 3 timer
out = DATA / "weather_trondheim.parquet"
out.parent.mkdir(exist_ok=True)
df.to_parquet(out, index=False)
print(f"Lagret {len(df):,} timer med vær ({start} - {end}) -> {out}")
