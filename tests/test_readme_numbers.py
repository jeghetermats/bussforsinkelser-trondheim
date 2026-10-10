"""Tallene i README.md og docs/experiments.md skal stemme med resultatfilene i reports/.

Tallene er skrevet inn for hånd i tekstene. Når pipelinen kjøres på nytt og resultatene endres, feiler denne testen
til tekstene er oppdatert (appen leser de samme tallene direkte fra filene).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
README = (ROOT / "README.md").read_text(encoding="utf-8")
EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")


def n(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def ci(b):
    return f"{n(b['forbedring_%'])} % [{n(b['ci95_lav_%'])}-{n(b['ci95_høy_%'])}]"


@pytest.fixture(scope="module")
def final():
    p = REPORTS / "final.json"
    if not p.exists():
        pytest.skip("reports/final.json finnes ikke")
    return json.loads(p.read_text(encoding="utf-8"))


def test_readme_resultattabell(final):
    a, b30 = final["A"], final["B"]["30"]
    assert f"| Historisk median (baseline) | | {n(a['metrics_baseline']['MAE_s'])} s | |" in README
    assert f"| Modell A | Dagen før | {n(a['metrics']['MAE_s'])} s | {ci(a['mot_baseline'])} |" in README
    assert (f"| Modell B | 30 min før avgang, med sanntidsdata | {n(b30['metrics']['MAE_s'])} s | "
            f"{ci(b30['mot_baseline'])} |") in README


def test_readme_hovedresultat(final):
    b = final["B"]["30"]["mot_baseline"]
    assert f"bommer modellen {n(b['forbedring_%'])} % mindre enn den historiske" in README
    assert f"(95 % KI: {n(b['ci95_lav_%'])}-{n(b['ci95_høy_%'])} %)" in README
    assert f"Modell B bommer {n(final['B']['30']['mot_A']['forbedring_%'])} % mindre enn A" in README


def test_readme_strengere_baseline(final):
    a = final["A"]
    if "mot_baseline_8uker" not in a:
        pytest.skip("final.json er laget før baseline_recent fantes")
    recent = (1 - a["metrics_baseline_8uker"]["MAE_s"] / a["metrics_baseline"]["MAE_s"]) * 100
    assert f"siste 8 ukene bommer {n(recent)} % mindre enn den vanlige" in README
    assert f"Mot den er A {n(a['mot_baseline_8uker']['forbedring_%'])} %" in README
    b8 = final["B"]["30"].get("mot_baseline_8uker")
    if b8:
        assert f"og B {n(b8['forbedring_%'])} % bedre (30 min før avgang)" in README


def test_rullende_evaluering():
    r = pd.read_csv(REPORTS / "rolling_eval.csv")
    wins = int((r.MAE_lgbm < r.MAE_baseline).sum())
    avg = (1 - (r.MAE_lgbm * r.n_test).sum() / (r.MAE_baseline * r.n_test).sum()) * 100
    assert f"Den slår baselinen i {wins} av {len(r)} måneder, i snitt med {n(avg)} %" in README
    gain = (1 - r.MAE_lgbm / r.MAE_baseline) * 100
    assert f"bedre i {wins} av {len(r)} måneder, i snitt {n(avg)} %" in EXPERIMENTS
    assert f"fra {n(gain.min())} %" in EXPERIMENTS and f"til +{n(gain.max())} %" in EXPERIMENTS


def test_experiments_modell_a_tabell(final):
    m = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    a = final["A"]
    for name, x in [("Global median", m["Baseline 0 - global median"]),
                    ("Historisk median (baseline)", a["metrics_baseline"]),
                    ("Modell A, én seed", a["metrics_én_seed"]),
                    ("Modell A (snitt av 3 seeds + blanding)", a["metrics"])]:
        row = (f"| {name} | {n(x['MAE_s'])} s | {n(x['RMSE_s'])} s | {n(x['innen_1min_%'])} % | "
               f"{n(x['innen_2min_%'])} % |")
        assert row in EXPERIMENTS, row
    assert (f"({n(a['metrics_baseline']['innen_1min_%'])} % mot {n(a['metrics']['innen_1min_%'])} %)"
            in EXPERIMENTS)
    assert f"({n(final['B']['30']['metrics']['innen_1min_%'])} %)" in EXPERIMENTS
    ens = json.loads((REPORTS / "ensemble.json").read_text(encoding="utf-8"))["A"]
    assert f"{n(ens['w_valgt_på_valid'], 2)} * modell" in EXPERIMENTS and a["w"] == ens["w_valgt_på_valid"]


def test_experiments_modell_b_per_lead(final):
    rt = pd.read_csv(REPORTS / "rt_results.csv").set_index(["metode", "lead_min"]).MAE_s
    leads = [60, 30, 10]
    rows = {"Historisk median": [rt["Baseline", L] for L in leads],
            "Modell A (dagen før)": [final["A"]["metrics"]["MAE_s"]] * 3,
            "Median + linjens avvik siste time (enkel regel)": [rt["Baseline + avvik siste time", L] for L in leads],
            "Modell B, runde 2": [final["B"][str(L)]["metrics"]["MAE_s"] for L in leads]}
    if ("Modell B runde 1", 10) in rt.index:
        rows["Modell B, runde 1"] = [rt["Modell B runde 1", L] for L in leads]
    nort = REPORTS / "rt_results_nort.csv"
    if nort.exists():
        rn = pd.read_csv(nort).set_index(["metode", "lead_min"]).MAE_s
        rows["Kontroll: B uten sanntidsfeatures (runde 1)"] = [rn["Modell B (sanntid)", L] for L in leads]
    for name, vals in rows.items():
        row = f"| {name} | " + " | ".join(n(v) for v in vals) + " |"
        assert row in EXPERIMENTS, row
