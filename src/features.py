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

Kolonnen 'baseline_recent' er en ekstra baseline (ikke feature): median fra bare de siste 8 ukene før radens
måned, samme grupper og fallback. Den fanger opp at forsinkelsene stiger gjennom høsten, så den viser hvor mye av
gevinsten til modellen som bare er å følge nivået (se train.py og finalize.py).

Kjør:  python src/features.py              (bygger alt)
       python src/features.py --reuse      (gjenbruker clean og historikktabeller fra forrige kjøring)
Valgfritt: DUCKDB_MEMORY_LIMIT=8GB og DUCKDB_THREADS=8 som miljøvariabler.
Varianter for å teste evalueringen (miljøvariabler, standard i parentes; realtime.py følger det features.py brukte):
  HIST_FOLDS=week     (day)    folder av hele uker i stedet for dag i året % 5
  HIST_AFTER=monthly  (frozen) valid/test-rader får historikk oppdatert hver måned (alt før radens måned), som
                               baselinen, i stedet for historikk frosset ved slutten av treningsperioden
  SPLIT_HASH=md5      (duckdb) fordeling av turer på utvalg uavhengig av DuckDB-versjon (se common.py; gir andre
                               turer i hvert utvalg, så alle resultater må lages på nytt)
"""
import datetime as dt
import os
import sys
import time

from common import (DATA, HIST_MONTHS, P1_END, P2_END, SAMPLE_PCT, SPLITS, connect_duckdb, raw_files, raw_sql,
                    split_hash_sql, write_dataset_meta)
from history import (baseline_select, build_oof, build_past_monthly, build_recent_monthly, feature_select, fold_expr,
                     join_sql)

HIST_FOLDS = os.environ.get("HIST_FOLDS", "day")
HIST_AFTER = os.environ.get("HIST_AFTER", "frozen")
if HIST_FOLDS not in ("day", "week") or HIST_AFTER not in ("frozen", "monthly"):
    sys.exit("HIST_FOLDS må være day eller week, HIST_AFTER må være frozen eller monthly")
SPLIT_HASH = os.environ.get("SPLIT_HASH", "duckdb")
RECENT_DAYS = 56                                           # 'baseline_recent': median siste 8 uker

DATA.mkdir(exist_ok=True)

# ---------- Rådata ----------
RAW_FILES = raw_files()
if not RAW_FILES:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
RAW_SQL = raw_sql(RAW_FILES)

# Utvalg (SAMPLE_PCT), perioder (SPLITS, P1_END, P2_END) og måneder (HIST_MONTHS) ligger i common.py.


def easter(year):
    """Påskedag (gregoriansk kalender, anonym algoritme)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def holidays(year):
    """Norske helligdager, pluss palmesøndag, julaften og nyttårsaften (egne ruter)."""
    e = easter(year)
    moving = [e + dt.timedelta(days=n) for n in (-7, -3, -2, 0, 1, 39, 49, 50)]  # palmes., skjærtors. ... 2. pinsed.
    fixed = [dt.date(year, m, d) for m, d in [(1, 1), (5, 1), (5, 17), (12, 24), (12, 25), (12, 26), (12, 31)]]
    return sorted(fixed + moving)


# Skoleferier i Trondheim (omtrentlig), per år. Må legges inn for hvert nytt år i dataene (sjekkes under).
SCHOOL_BREAKS = {
    2024: [("2024-12-21", "2024-12-31")],
    2025: [("2025-01-01", "2025-01-02"), ("2025-02-24", "2025-02-28"),
           ("2025-04-14", "2025-04-22"), ("2025-06-20", "2025-08-18"),
           ("2025-09-29", "2025-10-03"), ("2025-12-20", "2025-12-31")],
}

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
weather_source = "ingen"
if has_weather:     # målt vær (standard) eller værvarselet fra dagen før (src/weather.py --forecast)
    cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM '{weather_path.as_posix()}'").fetchall()]
    weather_source = (", ".join(r[0] for r in con.sql(f"SELECT DISTINCT source FROM '{weather_path.as_posix()}'")
                                .fetchall()) if "source" in cols else "målt")
log(f"Værdata: {'ja (' + weather_source + ')' if has_weather else 'nei (kjør src/weather.py for å legge til)'}")
log(f"Historikk: folder={HIST_FOLDS}, valid/test={HIST_AFTER}"
    + ("" if (HIST_FOLDS, HIST_AFTER) == ("day", "frozen") else "  (avviker fra oppsettet i README)"))

# Kalenderen må dekke alle årene i rådataene (helligdagene beregnes, skoleferiene må legges inn)
YEARS = [r[0] for r in con.sql(f"SELECT DISTINCT year(operatingDate) FROM {RAW_SQL} ORDER BY 1").fetchall()]
missing = [y for y in YEARS if y not in SCHOOL_BREAKS]
if missing:
    sys.exit(f"Skoleferiene i SCHOOL_BREAKS (features.py) mangler for {', '.join(map(str, missing))}. "
             "Legg dem inn (Trondheim kommunes skolerute) og kjør på nytt.")
holidays_sql = ", ".join(f"DATE '{d}'" for y in YEARS for d in holidays(y))
school_sql = " OR ".join(f"(f.operatingDate BETWEEN DATE '{a}' AND DATE '{b}')"
                         for y in YEARS for a, b in SCHOOL_BREAKS[y])

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
      {split_hash_sql('f.journey_id', 100, SPLIT_HASH)}  AS bucket,
      f.journey_id,
      f.stopPointName AS stop_name, f.origin_name, f.dest_name
    FROM f
    {weather_join}
    ORDER BY f.operatingDate, f.journey_id, f.seq      -- fast rekkefølge (og raskere datofiltre senere)
    """)
    n, d0, d1 = con.sql("SELECT count(*), min(date), max(date) FROM clean").fetchone()
    log(f"clean: {n:,} rader, {d0} - {d1}", t0)

# ================= 2. Historikk over alle rader =================
t0 = time.time()
rebuild = "--reuse" not in sys.argv
log("Beregner historikk over alle rader ...")
P1 = build_oof(con, P1_END, rebuild, HIST_FOLDS)     # fold-historikk jan-aug (+ des 2024)
P2 = build_oof(con, P2_END, rebuild, HIST_FOLDS)     # fold-historikk t.o.m. okt (til retrening)
PAST = build_past_monthly(con, HIST_MONTHS, rebuild)  # baseline: alt før radens måned
RECENT = build_recent_monthly(con, HIST_MONTHS, RECENT_DAYS, rebuild)   # ekstra baseline: siste 8 uker
log("Historikk ferdig", t0)

# ================= 3. Skriv utvalgte splitter =================
FOLD = fold_expr("c.date", HIST_FOLDS)
MONTH = "date_trunc('month', c.date)::DATE"
EXCLUDE = "month, bucket, journey_id, stop_name, origin_name, dest_name"
AFTER_P1 = (P1, "5", "") if HIST_AFTER == "frozen" else (PAST, MONTH, "")    # valid-rader i early stopping
AFTER_P2 = (P2, "5", "") if HIST_AFTER == "frozen" else (PAST, MONTH, "")    # testrader
plans = {   # split: [(prefiks, hkey-uttrykk, kolonneprefiks)]
    "train": [(P1, FOLD, ""), (P2, FOLD, "refit_")],
    "valid": [AFTER_P1, (P2, FOLD, "refit_")],
    "test":  [AFTER_P2],
}
for name, (d_from, d_to) in SPLITS.items():
    t0 = time.time()
    out = DATA / f"{name}.parquet"
    joins = [join_sql(PAST, MONTH, "b"), join_sql(RECENT, MONTH, "r")]
    cols = [baseline_select("b"), baseline_select("r", "baseline_recent")]
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
      ORDER BY c.date, c.journey_id, c.seq          -- fast rekkefølge mellom kjøringer (DuckDB garanterer den ikke)
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    log(f"{name:5s}: {n:,} rader -> {out.name}", t0)

# Oppsettet lagres sammen med datasettene; lagrede prognoser i data/ens/ knyttes til dette byggetidspunktet
write_dataset_meta("features", HIST_FOLDS=HIST_FOLDS, HIST_AFTER=HIST_AFTER, SPLIT_HASH=SPLIT_HASH, vær=weather_source)
log("Ferdig. data/features.duckdb kan slettes for å spare plass (bygges på nytt ved neste kjøring).")
