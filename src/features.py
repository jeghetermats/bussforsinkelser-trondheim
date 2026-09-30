"""
Bygger datasettene for oppgaven:
  "Forutsi ankomstforsinkelsen ved hvert stopp FØR bussen har startet turen."

Kun informasjon som er kjent før avgang brukes (rute, stopp, rutetid, kalender, vær og historikk).
Forsinkelse ved forrige stopp brukes IKKE.

Steg, alt i DuckDB (data/features.duckdb):
  1. clean - ALLE rensede stoppanløp (~43 mill.) med features.
  2. Historikk over alle rader (se src/history.py):
       fold-historikk for periodene t.o.m. aug og t.o.m. okt  -> modellens historikk-features
       månedlig fortid                                        -> baseline (historisk median)
  3. Utvalg av turer skrives til parquet:
       train (jan-aug 2025): hist_* = fold-historikk t.o.m. aug,   refit_hist_* = fold-historikk t.o.m. okt
       valid (sep-okt 2025): hist_* = alt t.o.m. aug (kun fortid), refit_hist_* = fold-historikk t.o.m. okt
       test  (nov-des 2025): hist_* = alt t.o.m. okt (kun fortid)
     'baseline' = historisk median per (linje, stopp, retning, time, dagtype) fra alle rader før
     radens måned, med fallback - den sterkeste ærlige baselinen vi har.

Valget av fold-historikk er gjort med src/ablation.py --cv (mars-okt), ikke på testperioden.

Kjør:  python src/features.py              (bygger alt)
       python src/features.py --reuse      (gjenbruker clean og historikktabeller fra forrige kjøring)
Valgfritt: DUCKDB_MEMORY_LIMIT=8GB og DUCKDB_THREADS=8 som miljøvariabler.
"""
import sys
import time

from common import DATA, SAMPLE_PCT, SPLITS, connect_duckdb, raw_files, raw_sql
from history import baseline_select, build_oof, build_past_monthly, feature_select, join_sql

DATA.mkdir(exist_ok=True)

# ---------- Rådata ----------
RAW_FILES = raw_files()
if not RAW_FILES:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
RAW_SQL = raw_sql(RAW_FILES)

# Utvalg (SAMPLE_PCT) og perioder (SPLITS) ligger i common.py, felles med realtime.py.
P1_END, P2_END = "2025-08-31", "2025-10-31"                # slutt på treningsperiode / trening+valid
HIST_MONTHS = [f"2025-{m:02d}-01" for m in range(1, 13)]   # måneder vi trenger baseline-historikk for

# Norske helligdager (inkl. julaften/nyttårsaften, som har egne ruter) og skoleferier i Trondheim (omtrentlig)
HOLIDAYS = ["2024-12-24", "2024-12-25", "2024-12-26", "2024-12-31",
            "2025-01-01", "2025-04-13", "2025-04-17", "2025-04-18", "2025-04-20", "2025-04-21",
            "2025-05-01", "2025-05-17", "2025-05-29", "2025-06-08", "2025-06-09",
            "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31"]
SCHOOL_BREAKS = [("2024-12-21", "2024-12-31"),
                 ("2025-01-01", "2025-01-02"), ("2025-02-24", "2025-02-28"),
                 ("2025-04-14", "2025-04-22"), ("2025-06-20", "2025-08-18"),
                 ("2025-09-29", "2025-10-03"), ("2025-12-20", "2025-12-31")]

def log(msg, t0=None):
    print(msg + (f"  ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


con = connect_duckdb(DATA / "features.duckdb")

weather_path = DATA / "weather_trondheim.parquet"
has_weather = weather_path.exists()
weather_join = (f"LEFT JOIN '{weather_path.as_posix()}' w ON w.time = date_trunc('hour', f.aimed_arrival_local)"
                if has_weather else "")
weather_cols = ("w.temperature_2m AS temp, w.precipitation AS precip, w.snowfall AS snowfall, "
                "w.snow_depth AS snow_depth, w.wind_speed_10m AS wind, w.precip_3h AS precip_3h,"
                if has_weather else "")
log(f"Rådata: {', '.join(f.name for f in RAW_FILES)}")
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

# ================= 2. Historikk over alle rader =================
t0 = time.time()
rebuild = "--reuse" not in sys.argv
log("Beregner historikk over alle rader ...")
P1 = build_oof(con, P1_END, rebuild)                 # fold-historikk jan-aug (+ des 2024)
P2 = build_oof(con, P2_END, rebuild)                 # fold-historikk t.o.m. okt (til retrening)
PAST = build_past_monthly(con, HIST_MONTHS, rebuild)  # baseline: alt før radens måned
log("Historikk ferdig", t0)

# ================= 3. Skriv utvalgte splitter =================
FOLD = "dayofyear(c.date) % 5"
MONTH = "date_trunc('month', c.date)::DATE"
EXCLUDE = "month, bucket, journey_id, stop_name, origin_name, dest_name"
plans = {   # split: [(prefiks, hkey-uttrykk, kolonneprefiks)]
    "train": [(P1, FOLD, ""), (P2, FOLD, "refit_")],
    "valid": [(P1, "5", ""), (P2, FOLD, "refit_")],
    "test":  [(P2, "5", "")],
}
for name, (d_from, d_to) in SPLITS.items():
    t0 = time.time()
    out = DATA / f"{name}.parquet"
    joins, cols = [join_sql(PAST, MONTH, "b")], [baseline_select("b")]
    for i, (prefix, hk, colpref) in enumerate(plans[name]):
        joins.append(join_sql(prefix, hk, f"h{i}"))
        cols.append(feature_select(f"h{i}", colpref))
    id_cols = ", c.journey_id, c.stop_name, c.origin_name, c.dest_name" if name == "test" else ""
    con.execute(f"""
    COPY (
      SELECT c.* EXCLUDE ({EXCLUDE}), {', '.join(cols)} {id_cols}
      FROM clean c
      {' '.join(joins)}
      WHERE c.bucket < {SAMPLE_PCT[name]} AND c.date BETWEEN DATE '{d_from}' AND DATE '{d_to}'
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    log(f"{name:5s}: {n:,} rader -> {out.name}", t0)

log("Ferdig. data/features.duckdb kan slettes for å spare plass (bygges på nytt ved neste kjøring).")
