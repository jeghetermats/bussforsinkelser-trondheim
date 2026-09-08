"""
Bygger datasettene for oppgaven:
  "Forutsi ankomstforsinkelsen ved hvert stopp FØR bussen har startet turen."

Kun informasjon som er kjent før avgang brukes (rute, stopp, rutetid, kalender, vær og historikk).
Forsinkelse ved forrige stopp brukes IKKE.

Tre steg, alt i DuckDB (data/features.duckdb):
  1. clean  - ALLE rensede stoppanløp (~43 mill.) med features.
  2. hist_* - historisk statistikk per måned, beregnet KUN fra data før måneden starter
              (månedlig oppdatert historikk, slik en ekte tjeneste ville hatt).
              Statistikken bruker alle rader, ikke et utvalg.
  3. Utvalg av turer skrives til parquet, med historikk og baseline slått opp for radens måned.

Baseline = historisk median for (linje, stopp, retning, time, dagtype), med fallback til grovere
grupper - beregnet med nøyaktig samme månedlige historikk som modellens features.

Splitter (data fra 3. des 2024; desember 2024 brukes bare som historikk):
  train = jan-aug 2025, valid = sep-okt 2025, test = nov-des 2025
  all   = hele 2025 (mindre utvalg), brukes av src/rolling_eval.py

Kjør:  python src/features.py              (bygger alt)
       python src/features.py --reuse      (gjenbruker clean-tabellen fra forrige kjøring)
Valgfritt: DUCKDB_MEMORY_LIMIT=8GB og DUCKDB_THREADS=8 som miljøvariabler.
"""
import os
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ---------- Rådata ----------
# Månedsfiler i data/raw/ (fra retrieval.py) + ev. den gamle atb_2025.parquet i prosjektroten.
raw_files = [f for f in sorted((DATA / "raw").glob("atb_*.parquet")) if f.stat().st_size > 10_000]  # hopp over tomme
if (ROOT / "atb_2025.parquet").exists():
    raw_files = [f for f in raw_files if not f.name.startswith("atb_2025")] + [ROOT / "atb_2025.parquet"]
if not raw_files:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
RAW_SQL = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in raw_files) + "], union_by_name=true)"

# Andel av turene som tas med i hver split (sampling på turnivå, så hele turer holdes samlet).
# Samme hash som før, så testturene er de samme som i tidligere kjøringer.
SAMPLE_PCT = {"train": 20, "valid": 25, "test": 25, "all": 10}
SPLITS = {"train": ("2025-01-01", "2025-08-31"),
          "valid": ("2025-09-01", "2025-10-31"),
          "test":  ("2025-11-01", "2025-12-31"),
          "all":   ("2025-01-01", "2025-12-31")}
HIST_MONTHS = [f"2025-{m:02d}-01" for m in range(1, 13)]   # måneder vi trenger historikk for

# Norske helligdager (inkl. julaften/nyttårsaften, som har egne ruter) og skoleferier i Trondheim (omtrentlig)
HOLIDAYS = ["2024-12-24", "2024-12-25", "2024-12-26", "2024-12-31",
            "2025-01-01", "2025-04-13", "2025-04-17", "2025-04-18", "2025-04-20", "2025-04-21",
            "2025-05-01", "2025-05-17", "2025-05-29", "2025-06-08", "2025-06-09",
            "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31"]
SCHOOL_BREAKS = [("2024-12-21", "2024-12-31"),
                 ("2025-01-01", "2025-01-02"), ("2025-02-24", "2025-02-28"),
                 ("2025-04-14", "2025-04-22"), ("2025-06-20", "2025-08-18"),
                 ("2025-09-29", "2025-10-03"), ("2025-12-20", "2025-12-31")]

# ---------- Historiske statistikker ----------
# tabell: (grupperingsnøkler, [(kolonnenavn, SQL-aggregat)])
HIST_TABLES = {
    "hist_lsdhd": (["line", "stop", "direction", "hour", "daytype"], [("hist_med_lsdhd", "median(y)")]),
    "hist_lsd":   (["line", "stop", "direction"], [("hist_mean_lsd", "avg(y)"), ("hist_std_lsd", "stddev_samp(y)"),
                                                    ("med_lsd", "median(y)")]),
    "hist_lhd":   (["line", "hour", "daytype"], [("hist_mean_lhd", "avg(y)"), ("med_lhd", "median(y)")]),
    "hist_sh":    (["stop", "hour"], [("hist_mean_sh", "avg(y)")]),
    "hist_l":     (["line"], [("med_l", "median(y)")]),
    "hist_g":     ([], [("med_g", "median(y)")]),
}
FEATURE_HIST_COLS = ["hist_med_lsdhd", "hist_mean_lsd", "hist_std_lsd", "hist_mean_lhd", "hist_mean_sh"]


def log(msg, t0=None):
    print(msg + (f"  ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


con = duckdb.connect(str(DATA / "features.duckdb"))
con.execute(f"SET temp_directory='{(DATA / 'tmp').as_posix()}'")
if os.environ.get("DUCKDB_MEMORY_LIMIT"):
    con.execute(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
if os.environ.get("DUCKDB_THREADS"):
    con.execute(f"SET threads={int(os.environ['DUCKDB_THREADS'])}")

weather_path = DATA / "weather_trondheim.parquet"
has_weather = weather_path.exists()
weather_join = (f"LEFT JOIN '{weather_path.as_posix()}' w ON w.time = date_trunc('hour', f.aimed_arrival_local)"
                if has_weather else "")
weather_cols = ("w.temperature_2m AS temp, w.precipitation AS precip, w.snowfall AS snowfall, "
                "w.snow_depth AS snow_depth, w.wind_speed_10m AS wind, w.precip_3h AS precip_3h,"
                if has_weather else "")
log(f"Rådata: {', '.join(f.name for f in raw_files)}")
log(f"Værdata: {'ja' if has_weather else 'nei (kjør src/weather.py for å legge til)'}")

holidays_sql = ", ".join(f"DATE '{d}'" for d in HOLIDAYS)
school_sql = " OR ".join(f"(f.operatingDate BETWEEN DATE '{a}' AND DATE '{b}')" for a, b in SCHOOL_BREAKS)

# ================= 1. clean: alle rensede rader =================
has_clean = con.sql("SELECT count(*) FROM information_schema.tables WHERE table_name = 'clean'").fetchone()[0] > 0
if "--reuse" in sys.argv and has_clean:
    log("Gjenbruker clean-tabellen (--reuse)")
else:
    t0 = time.time()
    log("Bygger clean-tabellen av alle rader (kan ta noen minutter) ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE clean AS
    WITH trip AS (
      SELECT *,
        max(seq) OVER j AS n_stops,
        min(aimed_arrival_local) OVER j AS trip_start,
        first_value(stopPointName) OVER js AS origin_name,
        last_value(stopPointName) OVER js  AS dest_name
      FROM {RAW_SQL}
      WINDOW j AS (PARTITION BY journey_id),
             js AS (PARTITION BY journey_id ORDER BY seq ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ),
    f AS (SELECT * FROM trip WHERE arrival_delay_s BETWEEN -1800 AND 3600)   -- fjern åpenbare datafeil
    SELECT
      f.arrival_delay_s::INTEGER                           AS y,
      f.operatingDate                                     AS date,
      date_trunc('month', f.operatingDate)::DATE          AS month,
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
      (f.operatingDate IN ({holidays_sql}))::INT          AS is_holiday,
      CASE WHEN isodow(f.operatingDate) = 7 OR f.operatingDate IN ({holidays_sql}) THEN 2
           WHEN isodow(f.operatingDate) = 6 THEN 1 ELSE 0 END AS daytype,   -- hverdag / lørdag / søn- og helligdag
      ({school_sql})::INT                                 AS is_school_break,
      -- Timer dagslys i Trondheim (63.43 grader nord), i stedet for måned (test-månedene finnes ikke i treningen)
      2 * degrees(acos(greatest(-1, least(1,
          -tan(radians(63.43)) * tan(radians(23.44 * sin(2 * pi() * (284 + dayofyear(f.operatingDate)) / 365)))
      )))) / 15                                           AS daylight_h,
      {weather_cols}
      hash(f.journey_id) % 100                            AS bucket,
      f.journey_id,
      f.stopPointName AS stop_name, f.origin_name, f.dest_name
    FROM f
    {weather_join}
    """)
    n, d0, d1 = con.sql("SELECT count(*), min(date), max(date) FROM clean").fetchone()
    log(f"clean: {n:,} rader, {d0} - {d1}", t0)

# ================= 2. Månedlig historikk, kun fra tidligere datoer =================
t0 = time.time()
log("Beregner historikk per måned (kun data fra før måneden) ...")
for table, (keys, aggs) in HIST_TABLES.items():
    key_sql = ", ".join(keys)
    agg_sql = ", ".join(f"{expr}::FLOAT AS {name}" for name, expr in aggs)
    parts = []
    for m in HIST_MONTHS:
        sel = f"SELECT DATE '{m}' AS month{', ' + key_sql if keys else ''}, {agg_sql} FROM clean WHERE date < DATE '{m}'"
        parts.append(sel + (f" GROUP BY {key_sql}" if keys else ""))
    con.execute(f"CREATE OR REPLACE TABLE {table} AS " + " UNION ALL ".join(parts))
    log(f"  {table}: {con.sql(f'SELECT count(*) FROM {table}').fetchone()[0]:,} rader", t0)

# Sjekk mot lekkasje: historikken for en måned skal bare bygge på data fra før måneden
first = con.sql("SELECT min(date) FROM clean").fetchone()[0]
n_first = con.sql(f"SELECT count(*) FROM hist_g WHERE month <= DATE '{first}' AND med_g IS NOT NULL").fetchone()[0]
assert n_first == 0, "Lekkasje: historikk finnes for en måned uten tidligere data"

# ================= 3. Skriv utvalgte splitter =================
joins = []
for table, (keys, _) in HIST_TABLES.items():
    on = " AND ".join([f"{table}.month = c.month"] + [f"{table}.{k} = c.{k}" for k in keys])
    joins.append(f"LEFT JOIN {table} ON {on}")
join_sql = "\n".join(joins)
exclude = "month, bucket, journey_id, stop_name, origin_name, dest_name"

for name, (d_from, d_to) in SPLITS.items():
    t0 = time.time()
    out = DATA / f"{name}.parquet"
    id_cols = ", c.journey_id, c.stop_name, c.origin_name, c.dest_name" if name == "test" else ""
    con.execute(f"""
    COPY (
      SELECT c.* EXCLUDE ({exclude}),
             {', '.join(FEATURE_HIST_COLS)},
             coalesce(hist_med_lsdhd, med_lsd, med_lhd, med_l, med_g) AS baseline
             {id_cols}
      FROM clean c
      {join_sql}
      WHERE c.bucket < {SAMPLE_PCT[name]} AND c.date BETWEEN DATE '{d_from}' AND DATE '{d_to}'
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    log(f"{name:5s}: {n:,} rader -> {out.name}", t0)

log("Ferdig. data/features.duckdb kan slettes for å spare plass (bygges på nytt ved neste kjøring).")
