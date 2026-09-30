"""Enhetstester for målene i src/common.py."""
import numpy as np
import pandas as pd
import pytest

from common import bootstrap_mae_gain, metrics


def test_metrics_kjente_verdier():
    y = np.array([0, 0, 0, 0])
    pred = np.array([30, -60, 90, 150])            # feil: 30, 60, 90, 150 sekunder
    m = metrics(y, pred)
    assert m["MAE_s"] == pytest.approx(82.5)
    assert m["RMSE_s"] == pytest.approx(np.sqrt((30**2 + 60**2 + 90**2 + 150**2) / 4))
    assert m["innen_1min_%"] == pytest.approx(50.0)   # 30 og 60 er innenfor
    assert m["innen_2min_%"] == pytest.approx(75.0)   # også 90


def test_metrics_perfekt_prognose():
    y = np.array([10.0, -5.0, 200.0])
    m = metrics(y, y)
    assert m["MAE_s"] == 0 and m["RMSE_s"] == 0 and m["innen_1min_%"] == 100


def _data(seed=1, n=20000, dager=30):
    rng = np.random.default_rng(seed)
    dates = (pd.Timestamp("2025-11-01") + pd.to_timedelta(rng.integers(0, dager, n), "D")).to_numpy()
    y = rng.normal(100, 50, n)
    return dates, y, rng


def test_bootstrap_like_prognoser_gir_null():
    dates, y, rng = _data()
    p = y + rng.normal(0, 60, len(y))
    b = bootstrap_mae_gain(dates, y, p, p)
    assert b["forbedring_%"] == pytest.approx(0) and b["ci95_lav_%"] == pytest.approx(0)
    assert b["dager"] == 30


def test_bootstrap_bedre_modell_har_positivt_intervall():
    dates, y, rng = _data()
    god, svak = y + rng.normal(0, 60, len(y)), y + rng.normal(0, 90, len(y))
    b = bootstrap_mae_gain(dates, y, god, svak)
    punkt = (1 - np.abs(god - y).mean() / np.abs(svak - y).mean()) * 100
    assert b["forbedring_%"] == pytest.approx(punkt)
    assert 0 < b["ci95_lav_%"] < b["forbedring_%"] < b["ci95_høy_%"]


def test_bootstrap_er_reproduserbar():
    dates, y, rng = _data()
    p, q = y + rng.normal(0, 60, len(y)), y + rng.normal(0, 70, len(y))
    assert bootstrap_mae_gain(dates, y, p, q) == bootstrap_mae_gain(dates, y, p, q)
