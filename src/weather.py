"""
Henter timesvær for Trondheim 2024-2025 fra Open-Meteo (gratis, ingen API-nøkkel).
Kjør:  python src/weather.py   og deretter  python src/features.py  på nytt.
"""
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 63.43, "longitude": 10.40,          # Trondheim sentrum
    "start_date": "2024-01-01", "end_date": "2025-12-31",
    "hourly": "temperature_2m,precipitation,snowfall,snow_depth,wind_speed_10m",
    "timezone": "Europe/Oslo",
}
r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()
df = pd.DataFrame(r.json()["hourly"])
df["time"] = pd.to_datetime(df["time"])
df = df.drop_duplicates("time")  # sommertid-overgang gir duplikate timer
df["precip_3h"] = df["precipitation"].rolling(3, min_periods=1).sum()   # nedbør siste 3 timer
out = ROOT / "data" / "weather_trondheim.parquet"
out.parent.mkdir(exist_ok=True)
df.to_parquet(out, index=False)
print(f"Lagret {len(df):,} timer med vær -> {out}")
