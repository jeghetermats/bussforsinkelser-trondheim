"""Enhetstester for historikken i src/history.py på små, oppdiktede data i DuckDB (i minnet).

Den sentrale påstanden: en rad ser aldri sin egen fasit. Fold-historikken for en treningsrad skal komme fra de
andre foldene (andre dager), og baselinen bare fra månedene før radens måned.
"""
import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from history import (DATA_START, baseline_select, build_oof, build_past_monthly,  # noqa: E402
                     feature_select, fold_expr, join_sql)

PERIOD_END = "2025-02-28"


@pytest.fixture(scope="module")
def con():
    """clean-tabell med samme kolonner som features.py bruker i historikken: 2 linjer x 3 stopp, hver dag
    des 2024-mar 2025, tilfeldig y (så ulike utvalg av dager gir ulike snitt og medianer)."""
    rng = np.random.default_rng(0)
    rows = []
    for date in pd.date_range(DATA_START, "2025-03-31"):
        for line in ("L1", "L2"):
            for stop in ("S1", "S2", "S3"):
                for hour in (8, 16):
                    rows.append((int(rng.integers(-120, 600)), date.date(), line, stop, "Outbound", hour,
                                 int(date.dayofweek >= 5)))
    df = pd.DataFrame(rows, columns=["y", "date", "line", "stop", "direction", "hour", "daytype"])
    c = duckdb.connect()
    values = ", ".join(f"({y}, DATE '{d}', '{l}', '{s}', '{r}', {h}, {t})" for y, d, l, s, r, h, t in rows)
    c.execute(f"CREATE TABLE clean AS SELECT * FROM (VALUES {values}) "
              "v(y, date, line, stop, direction, hour, daytype)")
    yield c, df
    c.close()


def _fold(dates):
    return pd.to_datetime(pd.Series(dates)).dt.dayofyear.to_numpy() % 5


def test_fold_historikk_ser_aldri_radens_egen_dag(con):
    con, df = con
    prefix = build_oof(con, PERIOD_END, rebuild=True)
    period = df[(pd.to_datetime(df.date) >= DATA_START) & (pd.to_datetime(df.date) <= PERIOD_END)]
    fold = _fold(period.date)
    for k in range(6):
        src = period if k == 5 else period[fold != k]       # hkey 5 = hele perioden (for rader etter perioden)
        exp = src.groupby(["line", "stop", "direction"]).y.mean()
        got = con.sql(f"SELECT line, stop, direction, hist_mean_lsd FROM {prefix}_lsd WHERE hkey = {k}").df()
        got = got.set_index(["line", "stop", "direction"]).hist_mean_lsd
        pd.testing.assert_series_equal(got.sort_index(), exp.sort_index().astype("float32"),
                                       check_names=False, check_index_type=False, rtol=1e-6)


def test_treningsrad_faar_historikk_uten_egen_fold(con):
    """Koblingen slik features.py gjør den for treningsrader (hkey = radens fold): ingen rad ser sin egen dag."""
    con, df = con
    prefix = build_oof(con, PERIOD_END)
    rows = con.sql(f"""
        SELECT c.y, c.date, c.line, c.stop, c.direction, {feature_select('h')}
        FROM clean c {join_sql(prefix, "dayofyear(c.date) % 5", 'h')}
        WHERE c.date <= DATE '{PERIOD_END}'""").df()
    period = df[pd.to_datetime(df.date) <= PERIOD_END]
    pfold = _fold(period.date)
    for _, r in rows.sample(40, random_state=1).iterrows():
        own_fold = _fold([r.date])[0]
        same = period[(period.line == r.line) & (period.stop == r.stop) & (period.direction == r.direction)]
        allowed = same[pfold[same.index] != own_fold]
        assert pd.Timestamp(r.date) not in set(pd.to_datetime(allowed.date))
        assert r.hist_mean_lsd == pytest.approx(allowed.y.mean(), rel=1e-6)


def test_maanedlig_historikk_bare_fra_maanedene_foer(con):
    con, df = con
    months = ["2025-01-01", "2025-02-01", "2025-03-01"]
    prefix = build_past_monthly(con, months, rebuild=True)
    for m in months:
        src = df[pd.to_datetime(df.date) < m]
        exp = src.groupby(["line", "stop", "direction", "hour", "daytype"]).y.median()
        got = con.sql(f"SELECT line, stop, direction, hour, daytype, hist_med_lsdhd FROM {prefix}_lsdhd "
                      f"WHERE hkey = DATE '{m}'").df()
        got = got.set_index(["line", "stop", "direction", "hour", "daytype"]).hist_med_lsdhd
        pd.testing.assert_series_equal(got.sort_index(), exp.sort_index().astype("float32"),
                                       check_names=False, check_index_type=False, rtol=1e-6)


def test_baseline_bruker_ikke_radens_maaned(con):
    """Baselinen for en rad i mars endres ikke når y i mars endres."""
    con, _ = con
    prefix = build_past_monthly(con, ["2025-03-01"], rebuild=True)
    q = f"""SELECT {baseline_select('b')} FROM clean c
            {join_sql(prefix, "date_trunc('month', c.date)::DATE", 'b')}
            WHERE c.date >= DATE '2025-03-01' ORDER BY c.date, c.line, c.stop, c.hour"""
    before = con.sql(q).df().baseline.to_numpy()
    con.execute("UPDATE clean SET y = y + 5000 WHERE date >= DATE '2025-03-01'")
    try:
        prefix = build_past_monthly(con, ["2025-03-01"], rebuild=True)
        after = con.sql(q).df().baseline.to_numpy()
    finally:
        con.execute("UPDATE clean SET y = y - 5000 WHERE date >= DATE '2025-03-01'")
    assert np.isfinite(before).all()
    np.testing.assert_array_equal(before, after)


def test_ukefolder_holder_hele_uker_samlet(con):
    """folds="week": alle dagene i en uke (mandag-søndag) er i samme fold, så en treningsrad heller ikke får
    statistikk fra dagene rett før og etter i samme uke."""
    con, df = con
    folds = con.sql(f"SELECT DISTINCT date, {fold_expr('date', 'week')} AS f FROM clean").df()
    folds["week"] = pd.to_datetime(folds.date).dt.to_period("W-SUN")
    assert (folds.groupby("week").f.nunique() == 1).all()
    assert set(folds.f) == {0, 1, 2, 3, 4}
    prefix = build_oof(con, PERIOD_END, rebuild=True, folds="week")
    period = df[pd.to_datetime(df.date) <= PERIOD_END]
    wfold = pd.to_datetime(period.date).map(folds.set_index(pd.to_datetime(folds.date)).f)
    assert wfold.notna().all()
    for k in range(5):
        exp = period[wfold != k].groupby(["line", "stop", "direction"]).y.mean()
        got = con.sql(f"SELECT line, stop, direction, hist_mean_lsd FROM {prefix}_lsd WHERE hkey = {k}").df()
        got = got.set_index(["line", "stop", "direction"]).hist_mean_lsd
        pd.testing.assert_series_equal(got.sort_index(), exp.sort_index().astype("float32"),
                                       check_names=False, check_index_type=False, rtol=1e-6)
