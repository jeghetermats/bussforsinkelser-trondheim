"""
Avslører turstrukturen hva som skjedde etter avgang?

features.py beregner n_stops (høyeste seq), route_frac, trip_start_min og origin/dest fra stoppene som faktisk er
registrert. Mangler de siste stoppene på en tur (f.eks. fordi en forsinket buss ble snudd før endeholdeplassen),
blir n_stops lavere og dest_name en annen, og det er ikke kjent før avgang. Mangler de første stoppene, flyttes
trip_start_min og origin_name.

Hver tur sammenlignes med samme planlagte tur (linje, retning, dagtype og planlagt avgang) på minst 3 andre dager i
samme måned:
  * mangler slutten: høyeste seq er lavere enn det vanlige for den planlagte turen
  * mangler starten: laveste seq er høyere enn det vanlige (turer der første stopp mangler får en senere
    'planlagt avgang' og finner ofte ikke sin planlagte tur; derfor vises også andelen med laveste seq > 1)
Hvis avkortede turer er mer forsinket enn hele turer på samme linje (målt på de tre første registrerte stoppene,
som begge har), lekker strukturen litt informasjon om forsinkelsen. Da bør n_stops/route_frac/origin/dest hentes
fra rutetabellen i stedet.

Resultat: reports/trip_structure_check.csv (per linje) og en oppsummering i terminalen.
Kjør:     python src/check_trip_structure.py      (leser rådataene i data/raw/, trenger ikke features.py)
"""
import sys

import pandas as pd

from common import REPORTS, connect_duckdb, ensure_dirs, raw_files, raw_sql

files = raw_files()
if not files:
    sys.exit("Fant ingen rådata. Kjør retrieval.py først.")
ensure_dirs()
con = connect_duckdb(":memory:")
con.execute(f"""
CREATE TABLE trips AS
SELECT journey_id, any_value(lineRef) AS line, any_value(directionRef) AS direction, any_value(operatingDate) AS date,
       min(seq) AS min_seq, max(seq) AS max_seq, count(*) AS n_rec, min(aimed_arrival_local) AS start_ts,
       avg(arrival_delay_s) FILTER (WHERE arrival_delay_s BETWEEN -1800 AND 3600) AS mean_delay,
       avg(arrival_delay_s) FILTER (WHERE arrival_delay_s BETWEEN -1800 AND 3600 AND seq <= 3) AS delay_first3
FROM {raw_sql(files)} GROUP BY journey_id
""")
con.execute("""
CREATE TABLE keyed AS
SELECT *, date_trunc('month', date) AS month,
       CASE WHEN isodow(date) = 7 THEN 2 WHEN isodow(date) = 6 THEN 1 ELSE 0 END AS daytype,
       strftime(start_ts, '%H:%M') AS start_hm
FROM trips
""")
con.execute("""
CREATE TABLE planned AS
SELECT line, direction, month, daytype, start_hm, count(*) AS n_days,
       mode(max_seq) AS usual_max, mode(min_seq) AS usual_min
FROM keyed GROUP BY ALL HAVING count(*) >= 3
""")
d = con.sql("""
SELECT k.line, k.direction, k.date, k.n_rec, k.min_seq, k.mean_delay, k.delay_first3,
       p.n_days IS NOT NULL AS compared,
       coalesce(k.max_seq < p.usual_max, FALSE) AS missing_end,
       coalesce(k.min_seq > p.usual_min, FALSE) AS missing_start
FROM keyed k LEFT JOIN planned p USING (line, direction, month, daytype, start_hm)
""").df()
con.close()

d["truncated"] = d.missing_end | d.missing_start
d["first_seq_gt_1"] = d.min_seq > 1
c = d[d.compared]
if c.empty:
    sys.exit("Ingen planlagte turer finnes på minst 3 dager i samme måned - for lite data til sammenligningen.")


def share(mask, weights=None):
    return float((mask * (1 if weights is None else weights)).sum() / (len(mask) if weights is None else weights.sum())
                 * 100)


print(f"{len(c):,} av {len(d):,} turer sammenlignet med samme planlagte tur samme måned (minst 3 dager).")
print(f"Avkortet: {share(c.truncated):.2f} % av turene ({share(c.truncated, c.n_rec):.2f} % av radene); "
      f"mangler siste stopp {share(c.missing_end):.2f} %, første stopp {share(c.missing_start):.2f} %.")
print(f"Laveste registrerte seq > 1: {share(d.first_seq_gt_1):.2f} % av alle turer.")
full, trunc = c[~c.truncated], c[c.truncated]
within = (trunc.groupby("line").delay_first3.mean() - full.groupby("line").delay_first3.mean()).dropna()
weights = trunc.groupby("line").size().reindex(within.index)
diff = float((within * weights).sum() / weights.sum()) if len(within) else float("nan")
print(f"Forsinkelse på de tre første stoppene: hele turer {full.delay_first3.mean():.0f} s, avkortede "
      f"{trunc.delay_first3.mean():.0f} s (forskjell innen samme linje: {diff:+.0f} s).")

per_line = c.groupby("line").apply(lambda g: pd.Series({
    "turer": len(g),
    "andel_avkortet_%": share(g.truncated),
    "andel_mangler_slutt_%": share(g.missing_end),
    "andel_mangler_start_%": share(g.missing_start),
    "andel_rader_%": share(g.truncated, g.n_rec),
    "forsinkelse_3_første_hele_s": g.delay_first3[~g.truncated].mean(),
    "forsinkelse_3_første_avkortet_s": g.delay_first3[g.truncated].mean(),
}), include_groups=False).sort_values("turer", ascending=False)
per_line.to_csv(REPORTS / "trip_structure_check.csv")
pd.set_option("display.width", 200)
print("\nLinjene med flest turer:")
print(per_line.head(10).round(1).to_string())
print(f"\nLagret {REPORTS / 'trip_structure_check.csv'}")
