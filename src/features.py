"""
Bygger trenings-, validerings- og testsett for oppgaven:
  "Forutsi ankomstforsinkelsen ved hvert stopp FØR bussen har startet turen."

Kun informasjon som er kjent før avgang brukes (rute, stopp, rutetid, kalender, vær).
Forsinkelse ved forrige stopp brukes IKKE.

Tidsbasert splitt:
  train = jan 2024-aug 2025, valid = sep-okt 2025, test = nov-des 2025
  all   = hele perioden (mindre utvalg), brukes av src/rolling_eval.py

Kjør:  python src/features.py                 (bygger alle)
       python src/features.py train test      (bare utvalgte)
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
# Rådata: månedsfiler i data/raw/ (fra retrieval.py) + ev. den gamle atb_2025.parquet.
# Hvis atb_2025.parquet finnes, brukes den for 2025 og månedsfiler for 2025 ignoreres (unngår duplikater).
raw_files = sorted((ROOT / "data" / "raw").glob("atb_*.parquet"))
if (ROOT / "atb_2025.parquet").exists():
    raw_files = [f for f in raw_files if not f.name.startswith("atb_2025")] + [ROOT / "atb_2025.parquet"]
if not raw_files:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
RAW_SQL = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in raw_files) + "], union_by_name=true)"
print("Rådata:", ", ".join(f.name for f in raw_files))
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

# Andel av turene som tas med (sampling på turnivå, så hele turer holdes samlet)
SAMPLE_PCT = {"train": 12, "valid": 25, "test": 25, "all": 8}

# Norske helligdager 2024-2025
HOLIDAYS = ["2024-01-01", "2024-03-24", "2024-03-28", "2024-03-29", "2024-03-31", "2024-04-01",
            "2024-05-01", "2024-05-09", "2024-05-17", "2024-05-19", "2024-05-20", "2024-12-25",
            "2024-12-26",
            "2025-01-01", "2025-04-13", "2025-04-17", "2025-04-18", "2025-04-20",
            "2025-04-21", "2025-05-01", "2025-05-17", "2025-05-29", "2025-06-08",
            "2025-06-09", "2025-12-25", "2025-12-26"]
# Skoleferier i Trondheim 2024-2025 (omtrentlig - sjekk mot trondheim.kommune.no)
SCHOOL_BREAKS = [("2024-01-01", "2024-01-02"), ("2024-02-26", "2024-03-01"),
                 ("2024-03-25", "2024-04-01"), ("2024-06-21", "2024-08-18"),
                 ("2024-09-30", "2024-10-04"), ("2024-12-21", "2024-12-31"),
                 ("2025-01-01", "2025-01-02"), ("2025-02-24", "2025-02-28"),
                 ("2025-04-14", "2025-04-22"), ("2025-06-20", "2025-08-18"),
                 ("2025-09-29", "2025-10-03"), ("2025-12-20", "2025-12-31")]

con = duckdb.connect()
con.execute(f"SET memory_limit='1500MB'; SET threads=2; SET temp_directory='{(ROOT / 'data' / 'tmp').as_posix()}'")

holidays_sql = ", ".join(f"DATE '{d}'" for d in HOLIDAYS)
school_sql = " OR ".join(f"(operatingDate BETWEEN DATE '{a}' AND DATE '{b}')" for a, b in SCHOOL_BREAKS)

weather_path = ROOT / "data" / "weather_trondheim.parquet"
if not weather_path.exists():                                   # eldre filnavn
    weather_path = ROOT / "data" / "weather_trondheim_2025.parquet"
has_weather = weather_path.exists()
weather_join = (f"LEFT JOIN '{weather_path.as_posix()}' w ON w.time = date_trunc('hour', f.aimed_arrival_local)"
                if has_weather else "")
weather_cols = (", w.temperature_2m AS temp, w.precipitation AS precip, w.snowfall AS snowfall, "
                "w.snow_depth AS snow_depth, w.wind_speed_10m AS wind, w.precip_3h AS precip_3h"
                if has_weather else "")
print("Værdata:", "ja" if has_weather else "nei (kjør src/weather.py for å legge til)")

def build(split, date_from, date_to):
    pct = SAMPLE_PCT[split]
    out = OUT / f"{split}.parquet"
    # Test-splitten får også tur-ID og stoppnavn (ikke features - brukes av Streamlit-appen)
    id_cols = (", f.journey_id, f.stopPointName AS stop_name, f.origin_name, f.dest_name"
               if split == "test" else "")
    con.execute(f"""
    COPY (
      WITH base AS (
        SELECT * FROM {RAW_SQL}
        WHERE operatingDate BETWEEN DATE '{date_from}' AND DATE '{date_to}'
          AND hash(journey_id) % 100 < {pct}
      ),
      trip AS (
        SELECT *,
          max(seq) OVER j AS n_stops,
          min(aimed_arrival_local) OVER j AS trip_start,
          first_value(stopPointName) OVER js AS origin_name,
          last_value(stopPointName) OVER js  AS dest_name
        FROM base
        WINDOW j AS (PARTITION BY journey_id),
               js AS (PARTITION BY journey_id ORDER BY seq
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
      ),
      f AS (
        SELECT * FROM trip
        WHERE arrival_delay_s BETWEEN -1800 AND 3600   -- fjern åpenbare datafeil
      )
      SELECT
        f.arrival_delay_s                                   AS y,
        f.operatingDate                                     AS date,
        f.lineRef                                           AS line,
        f.stopPointRef                                      AS stop,
        f.directionRef                                      AS direction,
        f.seq,
        f.n_stops,
        f.seq / f.n_stops::DOUBLE                           AS route_frac,
        date_diff('second', f.trip_start, f.aimed_arrival_local) / 60.0 AS sched_min_from_start,
        hour(f.trip_start) * 60 + minute(f.trip_start)      AS trip_start_min,
        hour(f.aimed_arrival_local) * 60 + minute(f.aimed_arrival_local) AS minute_of_day,
        hour(f.aimed_arrival_local)                         AS hour,
        isodow(f.operatingDate)                             AS weekday,
        -- Timer dagslys i Trondheim (63.43 grader nord). Brukes i stedet for måned/dag i året, fordi
        -- test (nov-des) ellers ville hatt verdier modellen aldri har sett i trening (jan-okt).
        2 * degrees(acos(greatest(-1, least(1,
            -tan(radians(63.43)) * tan(radians(23.44 * sin(2 * pi() * (284 + dayofyear(f.operatingDate)) / 365)))
        )))) / 15                                           AS daylight_h,
        (f.operatingDate IN ({holidays_sql}))::INT          AS is_holiday,
        ({school_sql.replace('operatingDate', 'f.operatingDate')})::INT AS is_school_break
        {weather_cols}
        {id_cols}
      FROM f
      {weather_join}
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    print(f"{split:5s}: {n:,} rader -> {out.name}")

SPLITS = {"train": ("2024-01-01", "2025-08-31"),
          "valid": ("2025-09-01", "2025-10-31"),
          "test":  ("2025-11-01", "2025-12-31"),
          "all":   ("2024-01-01", "2025-12-31")}
for name in (sys.argv[1:] or SPLITS):
    build(name, *SPLITS[name])
