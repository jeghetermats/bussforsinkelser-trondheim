"""
Historiske statistikker i DuckDB, brukt av features.py og rolling_eval.py.

To typer historikk:
  * fold-historikk ('oof'): innenfor en periode deles dagene i 5 folder (dag i året % 5, eller hele uker med
    folds="week", se fold_expr). En rad får statistikk fra de andre 4 foldene, så den aldri ser sin egen fasit.
    hkey 0-4 = 'alle rader i perioden unntatt fold k', hkey 5 = 'alle rader i perioden'.
    Brukes for treningsrader (hkey = radens fold) og for rader ETTER perioden (hkey 5),
    som da bare ser fortiden. Valgt fordi den slo historikk kun fra fortiden i valget over
    mars-okt (se src/ablation.py --cv og reports/ablation_cv.csv).
  * månedlig fortid ('past'): hkey = månedsstart, statistikk fra alle rader før måneden.
    Brukes for baselinen (historisk median, oppdatert hver måned).
  * nylig fortid ('recent'): som 'past', men bare fra de siste N dagene før måneden. Gir en ekstra baseline
    (median siste 8 uker) som fanger opp at forsinkelsene stiger gjennom høsten.
"""
import datetime as dt
import time

from common import DATA_START

# Ukefolder teller hele uker fra mandagen i uken datasettet starter
WEEK0 = dt.date.fromisoformat(DATA_START) - dt.timedelta(days=dt.date.fromisoformat(DATA_START).weekday())

# (grupperingsnøkler, aggregater)
SPECS = {
    "lsdhd": (["line", "stop", "direction", "hour", "daytype"], ["median(y) AS hist_med_lsdhd"]),
    "lsd":   (["line", "stop", "direction"],
              ["avg(y) AS hist_mean_lsd", "stddev_samp(y) AS hist_std_lsd", "median(y) AS med_lsd"]),
    "lhd":   (["line", "hour", "daytype"], ["avg(y) AS hist_mean_lhd", "median(y) AS med_lhd"]),
    "sh":    (["stop", "hour"], ["avg(y) AS hist_mean_sh"]),
    "l":     (["line"], ["median(y) AS med_l"]),
    "g":     ([], ["median(y) AS med_g"]),
}
FEATURE_COLS = ["hist_med_lsdhd", "hist_mean_lsd", "hist_std_lsd", "hist_mean_lhd", "hist_mean_sh"]


def _exists(con, name):
    return con.sql(f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{name}'").fetchone()[0] > 0


def _build(con, prefix, keysets, rebuild):
    if not rebuild and _exists(con, prefix + "_g"):
        return prefix
    t0 = time.time()
    for name, (keys, aggs) in SPECS.items():
        ks = ", ".join(keys)
        parts = [f"SELECT {hk} AS hkey{', ' + ks if keys else ''}, {', '.join(f'{a}' for a in aggs)} "
                 f"FROM clean WHERE {w}" + (f" GROUP BY {ks}" if keys else "") for hk, w in keysets]
        sel = " UNION ALL ".join(parts)
        con.execute(f"CREATE OR REPLACE TABLE {prefix}_{name} AS SELECT * REPLACE ("
                    + ", ".join(f"{a.split(' AS ')[1]}::FLOAT AS {a.split(' AS ')[1]}" for a in aggs)
                    + f") FROM ({sel})")
    print(f"  historikk {prefix}: {len(keysets)} nøkler ({time.time()-t0:.0f}s)", flush=True)
    return prefix


def fold_expr(col="date", folds="day"):
    """Hvilken av de 5 foldene en dato hører til. day: dag i året % 5 (nabodager i ulike folder).
    week: hele uker (mandag-søndag) i samme fold, så en ukelang hendelse ikke havner i både en rads egen fold og
    historikken den får fra de andre foldene."""
    if folds == "week":
        return f"((date_diff('day', DATE '{WEEK0}', {col}) // 7) % 5)"
    if folds != "day":
        raise ValueError(f"folds må være day eller week, ikke {folds!r}")
    return f"dayofyear({col}) % 5"


def build_oof(con, period_end, rebuild=False, folds="day"):
    """Fold-historikk over [DATA_START, period_end]. Returnerer tabellprefiks."""
    base = f"date BETWEEN DATE '{DATA_START}' AND DATE '{period_end}'"
    keysets = [(str(f), f"{base} AND {fold_expr('date', folds)} <> {f}") for f in range(5)] + [("5", base)]
    return _build(con, oof_name(period_end, folds), keysets, rebuild)


def oof_name(period_end, folds="day"):
    """Tabellprefikset build_oof bruker (realtime.py slår opp i tabellene features.py har bygget)."""
    return ("hoofw_" if folds == "week" else "hoof_") + period_end.replace("-", "")


def build_past_monthly(con, months, rebuild=False):
    """Historikk fra alle rader før hver måned i `months` (liste av 'YYYY-MM-01')."""
    keysets = [(f"DATE '{m}'", f"date < DATE '{m}'") for m in months]
    return _build(con, "hpast", keysets, rebuild)


def build_recent_monthly(con, months, days=56, rebuild=False):
    """Historikk fra de siste `days` dagene før hver måned i `months` (hkey = månedsstart)."""
    keysets = [(f"DATE '{m}'", f"date >= DATE '{m}' - INTERVAL {days} DAY AND date < DATE '{m}'") for m in months]
    return _build(con, f"hrecent{days}", keysets, rebuild)


def join_sql(prefix, hk_expr, alias):
    """LEFT JOIN-er for alle tabellene i et prefiks. Kolonner hentes som <alias>_<tabell>.<kol>."""
    out = []
    for name, (keys, _) in SPECS.items():
        t = f"{alias}_{name}"
        on = " AND ".join([f"{t}.hkey = {hk_expr}"] + [f"{t}.{k} = c.{k}" for k in keys])
        out.append(f"LEFT JOIN {prefix}_{name} {t} ON {on}")
    return "\n".join(out)


def feature_select(alias, suffix=""):
    """SELECT-uttrykk for feature-kolonnene fra et alias (valgfritt navnesuffiks/prefiks)."""
    src = {"hist_med_lsdhd": "lsdhd", "hist_mean_lsd": "lsd", "hist_std_lsd": "lsd",
           "hist_mean_lhd": "lhd", "hist_mean_sh": "sh"}
    return ", ".join(f"{alias}_{src[c]}.{c} AS {suffix}{c}" for c in FEATURE_COLS)


def baseline_select(alias, name="baseline"):
    return (f"coalesce({alias}_lsdhd.hist_med_lsdhd, {alias}_lsd.med_lsd, {alias}_lhd.med_lhd, "
            f"{alias}_l.med_l, {alias}_g.med_g) AS {name}")
