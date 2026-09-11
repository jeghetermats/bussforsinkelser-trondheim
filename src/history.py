"""
Historiske statistikker i DuckDB, brukt av features.py og rolling_eval.py.

To typer historikk:
  * fold-historikk ('oof'): innenfor en periode deles dagene i 5 folder (dag i året % 5).
    En rad får statistikk fra de andre 4 foldene, så den aldri ser sin egen fasit.
    hkey 0-4 = 'alle rader i perioden unntatt fold k', hkey 5 = 'alle rader i perioden'.
    Brukes for treningsrader (hkey = radens fold) og for rader ETTER perioden (hkey 5),
    som da bare ser fortiden. Valgt fordi den slo historikk kun fra fortiden i valget over
    mars-okt (se src/ablation.py --cv og reports/ablation_cv.csv).
  * månedlig fortid ('past'): hkey = månedsstart, statistikk fra alle rader før måneden.
    Brukes for baselinen (historisk median, oppdatert hver måned).
"""
import time

DATA_START = "2024-12-03"

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
BASELINE_SQL = "coalesce(hist_med_lsdhd, med_lsd, med_lhd, med_l, med_g)"


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


def build_oof(con, period_end, rebuild=False):
    """Fold-historikk over [DATA_START, period_end]. Returnerer tabellprefiks."""
    base = f"date BETWEEN DATE '{DATA_START}' AND DATE '{period_end}'"
    keysets = [(str(f), f"{base} AND dayofyear(date) % 5 <> {f}") for f in range(5)] + [("5", base)]
    return _build(con, f"hoof_{period_end.replace('-', '')}", keysets, rebuild)


def build_past_monthly(con, months, rebuild=False):
    """Historikk fra alle rader før hver måned i `months` (liste av 'YYYY-MM-01')."""
    keysets = [(f"DATE '{m}'", f"date < DATE '{m}'") for m in months]
    return _build(con, "hpast", keysets, rebuild)


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


def baseline_select(alias):
    return (f"coalesce({alias}_lsdhd.hist_med_lsdhd, {alias}_lsd.med_lsd, {alias}_lhd.med_lhd, "
            f"{alias}_l.med_l, {alias}_g.med_g) AS baseline")
