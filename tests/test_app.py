"""Røyktest: appen starter med de forhåndsberegnede filene i reports/ og tåler de vanligste valgene."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception[0].value
    return at


def test_appen_starter(app):
    assert app.title[0].value == "Bussforsinkelser i Trondheim"
    assert len(app.metric) >= 4


@pytest.mark.parametrize("valg", ["Dagen før", "60 min før", "10 min før", "30 min før"])
def test_prognose_laget(app, valg):
    app.get("button_group")[0].set_value(valg).run()
    assert not app.exception, app.exception[0].value
    modell = "modell A" if valg == "Dagen før" else "modell B"
    assert any(modell in c.value for c in app.caption)


def test_bytte_linje_og_retning(app):
    for linje in ["1", "11", "3"]:
        app.selectbox[0].set_value(linje).run()
        assert not app.exception, app.exception[0].value
    app.selectbox[1].set_value(app.selectbox[1].options[-1]).run()
    assert not app.exception, app.exception[0].value
