"""
Sanntidsfeatures for modell B: 'Hva vet vi om trafikken akkurat nå, X minutter før avgang?'

Modell A (features.py/train.py) bruker bare det som er kjent dagen før: rute, kalender, vær, historikk.
Oraklet (src/oracle.py) viste at informasjon innen dagen - hvordan linjen og retningen går i dag -
er det som gir mest. Modell B bruker derfor også registrerte ankomster fra de siste timene.

Prognosetidspunkt p = turens planlagte avgang - lead (lead = 10/30/60 min i testen).
Vi bruker bare observasjoner med faktisk ankomsttid < p - BUFFER_MIN, for å ta høyde for at
sanntidsdata kommer litt forsinket inn. Turen selv har ikke startet ennå, så den er aldri med.

Observasjoner: alle rader i clean med faktisk tid = planlagt ankomst + forsinkelse.
'Avvik' = forsinkelse - baseline (historisk median fra månedene før), klippet til [-600, 900] s,
slik at en rushtur som alltid er 3 min sen ikke ser ut som et avvik.

Features (alle beregnet på p - BUFFER_MIN):
  per linje+retning : antall obs og snittavvik siste 20/60/180 min, siste obs (avvik, alder i min)
  per stopp         : antall obs og snittavvik siste 60 min (alle linjer, samme stopp)
  hele nettet       : antall obs siste 15 min, snittavvik siste 15/60 min
  lead_min, horizon_min (= lead + planlagte minutter fra turstart til stoppet)
Runde 2 - knyttet til hvor på ruten og hvilken buss:
  forrige buss      : avvik for siste buss på samme linje+retning ved SAMME stopp, og hvor lenge siden
  strekninger       : vekst i avvik per strekning (stopp k-1 -> k, alle linjer) siste 30/90 min,
                      summert langs turen fra første stopp til stopp k (rt_segcum*), og andel dekket
  innkommende buss  : siste planlagte tur på samme linje som ender ved turens startholdeplass (navn) før avgang
                      (sannsynligvis bussen som kjører turen): planlagt pause, siste kjente forsinkelse,
                      hvor langt den har kommet og pausen som er igjen etter forsinkelsen

Utvalg og historikk er de samme som i features.py (samme turer, samme fold-historikk).
Trening/valid: én tilfeldig lead per tur (5-90 min, fast hash). Test: hver rad med lead 10, 30 og 60.

Resultat: data/rt_train.parquet, rt_valid.parquet, rt_test.parquet
Tabeller: data/realtime.duckdb (features.duckdb åpnes bare for lesing).

Kjør:  python src/realtime.py
Valgfritt (testing): RT_FROM/RT_TO=YYYY-MM-DD begrenser datoene, RT_SPLITS=test, RT_DB=sti
"""
import os
import time
from pathlib import Path

import duckdb

from history import baseline_select, feature_select, join_sql

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IN_MAX_GAP = 90                     # innkommende buss: maks planlagt pause (min) før vi regner den som ukjent
BUFFER_MIN = 2                      # rapporteringsforsinkelse vi antar i sanntidsstrømmen
TEST_LEADS = [10, 30, 60]
LEAD_MIN, LEAD_MAX = 5, 90          # lead-intervall for trening/valid
SAMPLE_PCT = {"train": 20, "valid": 25, "test": 25}      # samme som features.py
SPLITS = {"train": ("2025-01-01", "2025-08-31"),
          "valid": ("2025-09-01", "2025-10-31"),
          "test":  ("2025-11-01", "2025-12-31")}
P1, P2, PAST = "f.hoof_20250831", "f.hoof_20251031", "f.hpast"   # bygget av features.py
FOLD = "dayofyear(c.date) % 5"
MONTH = "date_trunc('month', c.date)::DATE"
PLANS = {"train": [(P1, FOLD, ""), (P2, FOLD, "refit_")],
         "valid": [(P1, "5", ""), (P2, FOLD, "refit_")],
         "test":  [(P2, "5", "")]}

D_FROM, D_TO = os.environ.get("RT_FROM"), os.environ.get("RT_TO")
RUN_SPLITS = os.environ.get("RT_SPLITS", "train,valid,test").split(",")
DB = Path(os.environ.get("RT_DB", DATA / "realtime.duckdb"))


def log(msg, t0=None):
    print(msg + (f"  ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


def date_filter(col):
    parts = []
    if D_FROM:
        parts.append(f"{col} >= DATE '{D_FROM}' - 1")    # dagen før, for vinduer over midnatt
    if D_TO:
        parts.append(f"{col} <= DATE '{D_TO}'")
    return " AND ".join(parts) if parts else "TRUE"


con = duckdb.connect(str(DB))
con.execute(f"SET temp_directory='{(DATA / 'tmp').as_posix()}'")
if os.environ.get("DUCKDB_MEMORY_LIMIT"):
    con.execute(f"SET memory_limit='{os.environ['DUCKDB_MEMORY_LIMIT']}'")
if os.environ.get("DUCKDB_THREADS"):
    con.execute(f"SET threads={int(os.environ['DUCKDB_THREADS'])}")
con.execute(f"ATTACH '{(DATA / 'features.duckdb').as_posix()}' AS f (READ_ONLY)")

# ================= 1. Observasjoner med faktisk tidspunkt =================
t0 = time.time()
raw_files = [p for p in sorted((DATA / "raw").glob("atb_*.parquet")) if p.stat().st_size > 10_000]
if (ROOT / "atb_2025.parquet").exists():
    raw_files = [p for p in raw_files if not p.name.startswith("atb_2025")] + [ROOT / "atb_2025.parquet"]
raw_sql = "read_parquet([" + ", ".join(f"'{p.as_posix()}'" for p in raw_files) + "], union_by_name=true)"
con.execute(f"""
CREATE OR REPLACE TABLE trip_ts AS
SELECT journey_id, any_value(lineRef) AS line,
       min(aimed_arrival_local)::TIMESTAMP AS trip_start_ts, max(aimed_arrival_local)::TIMESTAMP AS trip_end_ts,
       arg_min(stopPointName, seq) AS origin_name, arg_max(stopPointName, seq) AS dest_name, max(seq) AS n_stops
FROM {raw_sql} WHERE {date_filter('operatingDate')} GROUP BY journey_id
""")
con.execute(f"""
CREATE OR REPLACE TABLE obs AS
SELECT *, greatest(-300, least(600, anom - lag(anom) OVER j))::FLOAT AS growth,
       lag(stop) OVER j AS prev_stop
FROM (
SELECT journey_id, seq, line, direction, stop, y,
       trip_start_ts + to_seconds(CAST(round(sched_min_from_start * 60) AS BIGINT) + y) AS obs_ts,
       greatest(-600, least(900, y - baseline))::FLOAT AS anom
FROM (
  SELECT c.journey_id, c.seq, c.line, c.direction, c.stop, c.sched_min_from_start, c.y, t.trip_start_ts,
         {baseline_select('b')}
  FROM f.clean c JOIN trip_ts t USING (journey_id)
  {join_sql(PAST, MONTH, 'b')}
  WHERE {date_filter('c.date')}
)
) WINDOW j AS (PARTITION BY journey_id ORDER BY seq)
""")
log(f"Observasjoner: {con.sql('SELECT count(*) FROM obs').fetchone()[0]:,}", t0)

# ================= 2. Kumulative summer per tidsbøtte =================
# cum-tabellene har én rad per (nøkkel, bøtteslutt t) med løpende summer. Summen i vinduet (T-W, T]
# er cum(T) - cum(T-W), som slås opp med ASOF JOIN (siste bøtte med t <= tidspunktet).
t0 = time.time()
for name, keys, minutes, val in [("cum_ld", ["line", "direction"], 5, "anom"),
                                 ("cum_stop", ["stop"], 15, "anom"),
                                 ("cum_net", [], 5, "anom"),
                                 ("cum_seg", ["prev_stop", "stop"], 5, "growth")]:
    ks = ", ".join(keys)
    part = f"PARTITION BY {ks} " if keys else ""
    con.execute(f"""
    CREATE OR REPLACE TABLE {name} AS
    WITH b AS (
      SELECT {ks + ', ' if keys else ''}
             time_bucket(INTERVAL {minutes} MINUTE, obs_ts) + INTERVAL {minutes} MINUTE AS t,
             count(*) AS n, count({val}) AS na, sum({val}) AS sa
      FROM obs {"WHERE prev_stop IS NOT NULL" if name == "cum_seg" else ""} GROUP BY ALL
    )
    SELECT {ks + ', ' if keys else ''} t,
           sum(n) OVER w AS cn, sum(na) OVER w AS cna, sum(sa) OVER w AS csa
    FROM b WINDOW w AS ({part}ORDER BY t ROWS UNBOUNDED PRECEDING)
    """)
con.execute("CREATE OR REPLACE TABLE obs_ld AS SELECT line, direction, obs_ts, anom FROM obs WHERE anom IS NOT NULL")
con.execute("CREATE OR REPLACE TABLE obs_stop AS "
            "SELECT line, direction, stop, obs_ts, anom FROM obs WHERE anom IS NOT NULL")
con.execute("CREATE OR REPLACE TABLE obs_j AS SELECT journey_id, obs_ts, y, seq FROM obs")
log("Kumulative tabeller ferdig", t0)


def window_feats(table, alias_q, keys, windows, prefix, with_n=True):
    """ASOF-joins og SELECT-uttrykk for vinduer (minutter) over en cum-tabell."""
    joins, cols = [], []
    on_keys = [f"{{a}}.{k} = {alias_q}.{k}" for k in keys]
    now = f"a_{prefix}"
    joins.append(f"ASOF LEFT JOIN {table} {now} ON " +
                 " AND ".join([k.format(a=now) for k in on_keys] + [f"{now}.t <= {alias_q}.T"]))
    for w in windows:
        a = f"a_{prefix}_{w}"
        joins.append(f"ASOF LEFT JOIN {table} {a} ON " +
                     " AND ".join([k.format(a=a) for k in on_keys] + [f"{a}.t <= {alias_q}.T - INTERVAL {w} MINUTE"]))
        n = f"(coalesce({now}.cn, 0) - coalesce({a}.cn, 0))"
        na = f"(coalesce({now}.cna, 0) - coalesce({a}.cna, 0))"
        sa = f"(coalesce({now}.csa, 0) - coalesce({a}.csa, 0))"
        if with_n:
            cols.append(f"{n}::INTEGER AS rt_{prefix}_n{w}")
        cols.append(f"({sa} / nullif({na}, 0))::FLOAT AS rt_{prefix}_anom{w}")
    return "\n".join(joins), ", ".join(cols)


# ================= 3. Datasett =================
EXCLUDE = "month, bucket, journey_id, stop_name, origin_name, dest_name"
for split in RUN_SPLITS:
    t0 = time.time()
    d_from, d_to = SPLITS[split]
    if D_FROM:
        d_from = max(d_from, D_FROM)
    if D_TO:
        d_to = min(d_to, D_TO)
    joins, cols = [join_sql(PAST, MONTH, "b")], [baseline_select("b")]
    for i, (prefix, hk, colpref) in enumerate(PLANS[split]):
        joins.append(join_sql(prefix, hk, f"h{i}"))
        cols.append(feature_select(f"h{i}", colpref))
    if split == "test":
        lead_sql = f"CROSS JOIN (SELECT unnest({TEST_LEADS}) AS lead_min) l"
        lead_col = "l.lead_min"
    else:
        lead_sql = ""
        lead_col = f"({LEAD_MIN} + hash(c.journey_id || '#lead') % {LEAD_MAX - LEAD_MIN + 1})::INTEGER"
    # 3a. Rader med historikk, lead og tidspunktet T vi har data fram til
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE q AS
    SELECT c.* EXCLUDE ({EXCLUDE}), {', '.join(cols)}, c.journey_id,
           {lead_col} AS lead_min,
           t.trip_start_ts - to_minutes({lead_col} + {BUFFER_MIN}) AS T,
           t.trip_start_ts, t.origin_name AS origin_stopname,
           lag(c.stop) OVER (PARTITION BY c.journey_id, {lead_col} ORDER BY c.seq) AS prev_stop
    FROM f.clean c
    JOIN trip_ts t USING (journey_id)
    {lead_sql}
    {' '.join(joins)}
    WHERE c.bucket < {SAMPLE_PCT[split]} AND c.date BETWEEN DATE '{d_from}' AND DATE '{d_to}'
    """)
    # 3b. Features per tur (linje+retning og nettet) - samme for alle stopp på turen
    j_ld, c_ld = window_feats("cum_ld", "tr", ["line", "direction"], [20, 60, 180], "ld")
    j_net, c_net = window_feats("cum_net", "tr", [], [15, 60], "net")
    c_net = c_net.replace("rt_net_n60", "rt_net_n60_unused")   # antall siste 15 min holder
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE trip_feats0 AS
    SELECT tr.journey_id, tr.lead_min, tr.T, tr.trip_start_ts, {c_ld}, {c_net},
           last.anom AS rt_ld_last_anom,
           (date_diff('second', last.obs_ts, tr.T) / 60.0)::FLOAT AS rt_ld_last_age_min,
           inc.journey_id AS in_journey, inc.trip_end_ts AS in_end, inc.n_stops AS in_n_stops
    FROM (SELECT DISTINCT journey_id, lead_min, line, direction, T, trip_start_ts, origin_stopname FROM q) tr
    {j_ld}
    {j_net}
    ASOF LEFT JOIN obs_ld last ON last.line = tr.line AND last.direction = tr.direction AND last.obs_ts <= tr.T
    ASOF LEFT JOIN trip_ts inc ON inc.line = tr.line AND inc.dest_name = tr.origin_stopname
                               AND inc.trip_end_ts < tr.trip_start_ts
    """)
    # Innkommende buss: siste kjente forsinkelse før T
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE trip_feats AS
    SELECT tf.* EXCLUDE (T, trip_start_ts, in_journey, in_end, in_n_stops),
           (date_diff('second', tf.in_end, tf.trip_start_ts) / 60.0)::FLOAT AS rt_in_gap_min,
           o.y::FLOAT AS rt_in_delay,
           (date_diff('second', o.obs_ts, tf.T) / 60.0)::FLOAT AS rt_in_age_min,
           (o.seq / tf.in_n_stops::DOUBLE)::FLOAT AS rt_in_progress,
           (date_diff('second', tf.in_end, tf.trip_start_ts) / 60.0 - o.y / 60.0)::FLOAT AS rt_in_slack_min
    FROM (SELECT * REPLACE (CASE WHEN date_diff('minute', in_end, trip_start_ts) <= {IN_MAX_GAP}
                                 THEN in_journey END AS in_journey,
                            CASE WHEN date_diff('minute', in_end, trip_start_ts) <= {IN_MAX_GAP}
                                 THEN in_end END AS in_end)
          FROM trip_feats0) tf
    ASOF LEFT JOIN obs_j o ON o.journey_id = tf.in_journey AND o.obs_ts <= tf.T
    """)
    # 3c. Features per stopp (alle linjer) og ferdig datasett
    j_st, c_st = window_feats("cum_stop", "q", ["stop"], [60], "stop")
    j_sg, c_sg = window_feats("cum_seg", "q", ["prev_stop", "stop"], [30, 90], "seg")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE r AS
    SELECT q.*, {c_st}, {c_sg},
           prev.anom AS rt_prev_anom,
           (date_diff('second', prev.obs_ts, q.T) / 60.0)::FLOAT AS rt_prev_age_min
    FROM q
    {j_st}
    {j_sg}
    ASOF LEFT JOIN obs_stop prev ON prev.line = q.line AND prev.direction = q.direction
                                 AND prev.stop = q.stop AND prev.obs_ts <= q.T
    """)
    keep_ids = "r.journey_id, " if split == "test" else ""
    out = DATA / f"rt_{split}.parquet"
    seg_cum = ", ".join(
        f"sum(coalesce(r.rt_seg_anom{w}, 0)) OVER tw ::FLOAT AS rt_segcum{w}, "
        f"avg((r.rt_seg_n{w} > 0)::INT) OVER tw ::FLOAT AS rt_segcov{w}" for w in (30, 90))
    con.execute(f"""
    COPY (
      SELECT r.* EXCLUDE (journey_id, T, trip_start_ts, origin_stopname, prev_stop), {keep_ids}
             (r.lead_min + r.sched_min_from_start)::FLOAT AS horizon_min,
             {seg_cum},
             tf.* EXCLUDE (journey_id, lead_min, rt_net_n60_unused)
      FROM r
      JOIN trip_feats tf ON tf.journey_id = r.journey_id AND tf.lead_min = r.lead_min
      WINDOW tw AS (PARTITION BY r.journey_id, r.lead_min ORDER BY r.seq ROWS UNBOUNDED PRECEDING)
      ORDER BY r.date, r.journey_id, r.lead_min, r.seq          -- fast rekkefølge mellom kjøringer
    ) TO '{out.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.sql(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    log(f"{split:5s}: {n:,} rader -> {out.name}", t0)

log("Ferdig.")
