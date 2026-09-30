"""Enhetstester for tekstformateringen i appen (app/data.py)."""
import datetime as dt

import pytest

from data import fmt_clock, fmt_date, fmt_delay, fmt_num


@pytest.mark.parametrize("sek, tekst", [
    (0, "0 s"), (46, "46 s"), (-46, "-46 s"), (59.6, "1 min 00 s"), (67, "1 min 07 s"),
    (-185, "-3 min 05 s"), (600, "10 min 00 s"),
])
def test_fmt_delay(sek, tekst):
    assert fmt_delay(sek) == tekst


@pytest.mark.parametrize("verdi, kwargs, tekst", [
    (3.64, {}, "3,6"), (0.0, {}, "0,0"), (102.0, {"dec": 0}, "102"),
    (2.3, {"sign": True}, "+2,3"), (-4.2, {"sign": True}, "-4,2"), (74.83, {"dec": 2}, "74,83"),
])
def test_fmt_num_bruker_desimalkomma(verdi, kwargs, tekst):
    assert fmt_num(verdi, **kwargs) == tekst


def test_fmt_clock_og_dato():
    assert fmt_clock(15 * 60 + 19) == "15:19"
    assert fmt_clock(24 * 60 + 5) == "00:05"          # etter midnatt
    assert fmt_date(dt.date(2025, 12, 1)) == "mandag 1. desember 2025"
