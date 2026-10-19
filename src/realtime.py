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
                      hvor langt den har kommet og pausen som er igjen etter forsinkelsen. Endeholdeplass, sluttid
                      og antall stopp er de planlagte (samme tur tidligere dager, se trip_plan i features.py)

Utvalg og historikk er de samme som i features.py (samme turer, samme fold-historikk; HIST_FOLDS og HIST_AFTER
leses fra data/features_meta.json, så de følger det features.py ble kjørt med).
Trening/valid: én tilfeldig lead per tur (5-90 min, fast hash). Test: hver rad med lead 10, 30 og 60.

Resultat: data/rt_train.parquet, rt_valid.parquet, rt_test.parquet
Tabeller: data/realtime.duckdb (features.duckdb åpnes bare for lesing).

Kjør:  python src/realtime.py
Valgfritt (testing): RT_FROM/RT_TO=YYYY-MM-DD begrenser datoene, RT_SPLITS=test, RT_DB=sti
Variant: RT_INCOMING=alle   innkommende buss kan også være en annen linje som ender ved startholdeplassen (brukes
                            når samme linje ikke har en tur som ender der innen IN_MAX_GAP), med flagget
                            rt_in_same_line (1 = samme linje, 0 = annen linje). Standard: linje (bare samme linje).
"""
import os
import time
from pathlib import Path

from common import (DATA, P1_END, P2_END, SAMPLE_PCT, SPLITS, connect_duckdb, dataset_meta, raw_files, raw_sql,
                    split_hash_sql, write_dataset_meta)
from history import baseline_select, feature_select, fold_expr, join_sql, oof_name

IN_MAX_GAP = 90                     # innkommende buss: maks planlagt pause (min) før vi regner den som ukjent
BUFFER_MIN = 2                      # rapporteringsforsinkelse vi antar i sanntidsstrømmen
TEST_LEADS = [10, 30, 60]
LEAD_MIN, LEAD_MAX = 5, 90          # lead-intervall for trening/valid
FEATURES_META = dataset_meta("features")          # oppsettet features.py ble kjørt med
HIST_FOLDS = FEATURES_META.get("HIST_FOLDS", "day")
HIST_AFTER = FEATURES_META.get("HIST_AFTER", "frozen")
SPLIT_HASH = FEATURES_META.get("SPLIT_HASH", "duckdb")
for k, v in (("HIST_FOLDS", HIST_FOLDS), ("HIST_AFTER", HIST_AFTER), ("SPLIT_HASH", SPLIT_HASH)):
    if os.environ.get(k, v) != v:
        raise SystemExit(f"{k}={os.environ[k]}, men features.py ble kjørt med {k}={v}. Kjør features.py på nytt.")
RT_INCOMING = os.environ.get("RT_INCOMING", "linje")
if RT_INCOMING not in ("linje", "alle"):
    raise SystemExit("RT_INCOMING må være linje eller alle")
P1 = "f." + oof_name(P1_END, HIST_FOLDS)     # bygget av features.py
P2 = "f." + oof_name(P2_END, HIST_FOLDS)
PAST = "f.hpast"
FOLD = fold_expr("c.date", HIST_FOLDS)
MONTH = "date_trunc('month', c.date)::DATE"
AFTER_P1 = (P1, "5", "") if HIST_AFTER == "frozen" else (PAST, MONTH, "")
AFTER_P2 = (P2, "5", "") if HIST_AFTER == "frozen" else (PAST, MONTH, "")
PLANS = {"train": [(P1, FOLD, ""), (P2, FOLD, "refit_")],
         "valid": [AFTER_P1, (P2, FOLD, "refit_")],
         "test":  [AFTER_P2]}

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


con = connect_duckdb(DB)
con.execute(f"ATTACH '{(DATA / 'features.duckdb').as_posix()}' AS f (READ_ONLY)")
if not con.sql("SELECT count(*) FROM information_schema.tables WHERE table_catalog = 'f' AND table_name = 'trip_plan'"
               ).fetchone()[0]:
    raise SystemExit("data/features.duckdb er fra en eldre versjon uten planlagt turstruktur. Kjør features.py på nytt.")

# ================= 1. Observasjoner med faktisk tidspunkt =================
# Endeholdeplass, sluttid og antall stopp er de planlagte fra features.py (trip_plan, tidligere dager), ikke de
# registrerte: blir en tur avkortet underveis, er det ikke kjent ved prognosetidspunktet.
t0 = time.time()
RAW_SQL = raw_sql(raw_files())
con.execute(f"""
CREATE OR REPLACE TABLE trip_ts AS
SELECT r.*, p.dest_name, p.n_stops,
       r.trip_start_ts + to_seconds(CAST(round(p.dur_min * 60) AS BIGINT)) AS trip_end_ts
FROM (SELECT journey_id, any_value(lineRef) AS line, min(aimed_arrival_local)::TIMESTAMP AS trip_start_ts,
             arg_min(stopPointName, seq) AS origin_name
      FROM {RAW_SQL} WHERE {date_filter('operatingDate')} GROUP BY journey_id) r
LEFT JOIN f.trip_plan p USING (journey_id)
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
# Tabellene til ASOF-oppslagene har én rad per nøkkel og tidspunkt. Flere observasjoner med nøyaktig samme
# tidspunkt ville ellers gitt et tilfeldig valg mellom dem, og ulike features fra kjøring til kjøring.
# Ved like tidspunkt brukes observasjonen fra turen med høyest id (og høyest seq).
con.execute("""CREATE OR REPLACE TABLE obs_ld AS SELECT line, direction, obs_ts, anom FROM obs
               WHERE anom IS NOT NULL
               QUALIFY row_number() OVER (PARTITION BY line, direction, obs_ts ORDER BY journey_id DESC, seq DESC) = 1""")
con.execute("""CREATE OR REPLACE TABLE obs_stop AS SELECT line, direction, stop, obs_ts, anom FROM obs
               WHERE anom IS NOT NULL
               QUALIFY row_number() OVER (PARTITION BY line, direction, stop, obs_ts
                                          ORDER BY journey_id DESC, seq DESC) = 1""")
con.execute("""CREATE OR REPLACE TABLE obs_j AS SELECT journey_id, obs_ts, y, seq FROM obs
               QUALIFY row_number() OVER (PARTITION BY journey_id, obs_ts ORDER BY seq DESC) = 1""")
con.execute("""CREATE OR REPLACE TABLE trip_end AS SELECT * FROM trip_ts WHERE trip_end_ts IS NOT NULL
               QUALIFY row_number() OVER (PARTITION BY line, dest_name, trip_end_ts ORDER BY journey_id DESC) = 1""")
if RT_INCOMING == "alle":
    con.execute("""CREATE OR REPLACE TABLE trip_end_any AS SELECT * FROM trip_ts WHERE trip_end_ts IS NOT NULL
                   QUALIFY row_number() OVER (PARTITION BY dest_name, trip_end_ts ORDER BY journey_id DESC) = 1""")
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
        lead_hash = split_hash_sql("c.journey_id || '#lead'", LEAD_MAX - LEAD_MIN + 1, SPLIT_HASH)
        lead_col = f"({LEAD_MIN} + {lead_hash})::INTEGER"
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
    any_cols, any_join = "", ""
    if RT_INCOMING == "alle":   # også siste tur fra en annen linje som ender ved startholdeplassen
        any_cols = (", inc2.journey_id AS in2_journey, inc2.trip_end_ts AS in2_end, inc2.n_stops AS in2_n_stops, "
                    "(inc2.line = tr.line)::INT AS in2_same")
        any_join = ("ASOF LEFT JOIN trip_end_any inc2 ON inc2.dest_name = tr.origin_stopname "
                    "AND inc2.trip_end_ts < tr.trip_start_ts")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE trip_feats0 AS
    SELECT tr.journey_id, tr.lead_min, tr.T, tr.trip_start_ts, {c_ld}, {c_net},
           last.anom AS rt_ld_last_anom,
           (date_diff('second', last.obs_ts, tr.T) / 60.0)::FLOAT AS rt_ld_last_age_min,
           inc.journey_id AS in_journey, inc.trip_end_ts AS in_end, inc.n_stops AS in_n_stops {any_cols}
    FROM (SELECT DISTINCT journey_id, lead_min, line, direction, T, trip_start_ts, origin_stopname FROM q) tr
    {j_ld}
    {j_net}
    ASOF LEFT JOIN obs_ld last ON last.line = tr.line AND last.direction = tr.direction AND last.obs_ts <= tr.T
    ASOF LEFT JOIN trip_end inc ON inc.line = tr.line AND inc.dest_name = tr.origin_stopname
                               AND inc.trip_end_ts < tr.trip_start_ts
    {any_join}
    """)
    ok1 = f"date_diff('minute', in_end, trip_start_ts) <= {IN_MAX_GAP}"
    if RT_INCOMING == "alle":   # samme linje først, ellers en annen linje
        ok2 = f"date_diff('minute', in2_end, trip_start_ts) <= {IN_MAX_GAP}"
        chosen = f"""(SELECT * EXCLUDE (in2_journey, in2_end, in2_n_stops, in2_same)
                             REPLACE (CASE WHEN {ok1} THEN in_journey WHEN {ok2} THEN in2_journey END AS in_journey,
                                      CASE WHEN {ok1} THEN in_end WHEN {ok2} THEN in2_end END AS in_end,
                                      CASE WHEN {ok1} THEN in_n_stops WHEN {ok2} THEN in2_n_stops END AS in_n_stops),
                             (CASE WHEN {ok1} THEN 1 WHEN {ok2} THEN in2_same END)::FLOAT AS rt_in_same_line
                      FROM trip_feats0)"""
    else:
        chosen = f"""(SELECT * REPLACE (CASE WHEN {ok1} THEN in_journey END AS in_journey,
                                       CASE WHEN {ok1} THEN in_end END AS in_end)
                      FROM trip_feats0)"""
    # Innkommende buss: siste kjente forsinkelse før T
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE trip_feats AS
    SELECT tf.* EXCLUDE (T, trip_start_ts, in_journey, in_end, in_n_stops),
           (date_diff('second', tf.in_end, tf.trip_start_ts) / 60.0)::FLOAT AS rt_in_gap_min,
           o.y::FLOAT AS rt_in_delay,
           (date_diff('second', o.obs_ts, tf.T) / 60.0)::FLOAT AS rt_in_age_min,
           (o.seq / tf.in_n_stops::DOUBLE)::FLOAT AS rt_in_progress,
           (date_diff('second', tf.in_end, tf.trip_start_ts) / 60.0 - o.y / 60.0)::FLOAT AS rt_in_slack_min
    FROM {chosen} tf
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

# Lagrede prognoser i data/ens/ knyttes til dette byggetidspunktet (også delvise ombygginger med RT_SPLITS)
write_dataset_meta("realtime", splits=RUN_SPLITS, RT_FROM=D_FROM, RT_TO=D_TO, RT_INCOMING=RT_INCOMING,
                   HIST_FOLDS=HIST_FOLDS, HIST_AFTER=HIST_AFTER, SPLIT_HASH=SPLIT_HASH)
log(f"Ferdig. {DB.name} brukes ikke av de andre skriptene og kan slettes for å spare plass.")
