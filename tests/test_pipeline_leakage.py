"""Ende-til-ende-test mot datalekkasje: features.py og realtime.py kjøres på små, oppdiktede rådata.

Testen bruker ikke kunnskap om hvordan featurene beregnes. Den endrer fasiten (forsinkelsen) i data som en rad
ikke skal kunne se, bygger datasettene på nytt og sjekker at radens features er nøyaktig de samme:
  * historikk: endres forsinkelsen på radens egen dag, skal treningsradens fold-historikk og baseline ikke endres;
    endres alt i desember, skal testradene i desember ha samme historikk og baseline (de ser bare fortiden).
  * sanntid (modell B): for hver testdag i desember endres alle observasjoner etter et skjæringstidspunkt C.
    Radene med prognosetidspunkt T i timen før C (T = avgang - lead - 2 min) skal ha nøyaktig de samme
    featurene. (Rader tidligere på dagen kan se gårsdagens siste observasjoner, som er endret etter gårsdagens C.)
  * avkortede turer: på én testdag forsvinner resten av turene som er underveis ved C (og de siste stoppene på
    noen turer senere på dagen). Turstrukturen (antall stopp, endeholdeplass) er da ikke kjent før avgang, så de
    gjenværende radene skal ha de samme featurene, og sanntidsradene med T før C likeså.
De tette testdagene (en avgang hvert 5. minutt) gjør at også små lekkasjer (1-2 min etter T) blir oppdaget.

Tar 20-60 sekunder. Krever duckdb (pip install duckdb).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
LINES = {"1": 8, "3": 8, "11": 7}                     # linje -> antall stopp
SHARED = "Felles stopp"                               # samme stopp på alle linjene (features per stopp)
CHECK_DAYS = ["2025-12-02", "2025-12-09", "2025-12-16", "2025-12-23"]   # tette dager for sanntidstesten
CUTOFF = {d: pd.Timestamp(d) + pd.Timedelta(hours=11, minutes=4 + 5 * i) for i, d in enumerate(CHECK_DAYS)}
OWN_DAYS = {"train": "2025-05-06", "valid": "2025-10-06"}   # dager der hele fasiten endres i historikktesten
SHIFT = 400                                           # sekunder lagt til forsinkelsen i data raden ikke skal se
TRUNC_DAY = CHECK_DAYS[-1]                            # dagen med avkortede turer (senere dager kan lovlig se dem)


def make_raw(seed=1):
    rng = np.random.default_rng(seed)
    refs = {}
    days = [pd.Timestamp(y, m, d) for y, m in [(2024, 12)] + [(2025, m) for m in range(1, 13)] for d in (6, 23)]
    days = sorted(set(days) | {pd.Timestamp(d) for d in CHECK_DAYS})
    rows = []
    for day in days:
        dense = str(day.date()) in CHECK_DAYS
        day_eff = rng.normal(0, 40)
        for line, n in LINES.items():
            names = [f"L{line} stopp {i}" for i in range(n)]
            names[n // 2] = SHARED
            for direction in ("Outbound", "Inbound"):
                seq_names = names if direction == "Outbound" else names[::-1]
                step = 5 if dense else 20
                for k, start in enumerate(range(6 * 60, 20 * 60, step)):
                    jid = f"{day:%Y-%m-%d}:ATB:ServiceJourney:{line}_{direction[0]}_{k}"
                    t = day + pd.Timedelta(minutes=start)
                    trip_eff = rng.normal(0, 30)
                    for i, name in enumerate(seq_names):
                        if i:
                            t += pd.Timedelta(seconds=int(rng.integers(60, 150)))
                        delay = day_eff + trip_eff + 10 * i + rng.normal(0, 30)
                        rows.append((jid, day.date(), f"ATB:Line:2_{line}", direction, refs.setdefault(name, f"NSR:Quay:{len(refs) + 1}"),
                                     name, i + 1, t, int(np.clip(round(delay), -400, 600))))
    return pd.DataFrame(rows, columns=["journey_id", "operatingDate", "lineRef", "directionRef", "stopPointRef",
                                       "stopPointName", "seq", "aimed_arrival_local", "arrival_delay_s"])


def write_raw(df, data_dir):
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    out = df.copy()
    out["operatingDate"] = out["operatingDate"].astype(str)
    out["aimed_arrival_local"] = out["aimed_arrival_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv = data_dir / "raw.csv"
    out.to_csv(csv, index=False)
    cols = ("{'journey_id': 'VARCHAR', 'operatingDate': 'DATE', 'lineRef': 'VARCHAR', 'directionRef': 'VARCHAR', "
            "'stopPointRef': 'VARCHAR', 'stopPointName': 'VARCHAR', 'seq': 'BIGINT', "
            "'aimed_arrival_local': 'TIMESTAMP', 'arrival_delay_s': 'BIGINT'}")
    con = duckdb.connect()
    for month in sorted({d[:7] for d in out.operatingDate}):
        con.execute(f"COPY (SELECT * FROM read_csv('{csv.as_posix()}', header=true, columns={cols}, auto_detect=false)"
                    f" WHERE strftime(operatingDate, '%Y-%m') = '{month}') "
                    f"TO '{(raw / f'atb_{month[:4]}_{month[5:]}.parquet').as_posix()}' (FORMAT parquet)")
    con.close()
    csv.unlink()


def run(script, data_dir, **env):
    e = dict(os.environ, DATA_DIR=str(data_dir), PYTHONIOENCODING="utf-8", **env)
    r = subprocess.run([sys.executable, str(ROOT / "src" / script)], env=e, cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, f"{script} feilet:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"


def build(df, data_dir, realtime):
    write_raw(df, data_dir)
    run("features.py", data_dir)
    if realtime:
        run("realtime.py", data_dir, RT_SPLITS="test")
    names = ["train", "valid", "test"] + (["rt_test"] if realtime else [])
    return {n: pd.read_parquet(data_dir / f"{n}.parquet") for n in names}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    raw = make_raw()
    obs_ts = raw.aimed_arrival_local + pd.to_timedelta(raw.arrival_delay_s, unit="s")
    day = raw.operatingDate.astype(str)

    late = pd.Series(False, index=raw.index)       # alt etter skjæringstidspunktet på testdagene i desember
    for d, c in CUTOFF.items():
        late |= (day == d) & (obs_ts > c)
    dec = raw.copy()
    dec.loc[late, "arrival_delay_s"] += SHIFT

    # Avkortede turer på TRUNC_DAY: turer som er underveis ved C mister resten av stoppene, og hver andre tur som
    # starter mer enn 2 timer etter C mister de to siste stoppene
    start = raw.groupby("journey_id").aimed_arrival_local.transform("min")
    n_stops = raw.groupby("journey_id").seq.transform("max")
    c = CUTOFF[TRUNC_DAY]
    on_day = day == TRUNC_DAY
    later = start > c + pd.Timedelta(hours=2)
    odd = raw.journey_id.str.rsplit("_", n=1).str[1].astype(int) % 2 == 1
    cut = on_day & (((start <= c) & (obs_ts > c)) | (later & odd & (raw.seq > n_stops - 2)))
    trunc = raw[~cut]

    tmp = tmp_path_factory.mktemp("lekkasje")
    out = {"base": build(raw, tmp / "base", realtime=True), "dec": build(dec, tmp / "dec", realtime=True),
           "trunc": build(trunc, tmp / "trunc", realtime=True),
           "trip_start": raw.groupby("journey_id").aimed_arrival_local.min()}
    for split, d in OWN_DAYS.items():               # hele fasiten for én dag endres (én kjøring per dag)
        own = raw.copy()
        own.loc[day == d, "arrival_delay_s"] += SHIFT
        out[f"own_{split}"] = build(own, tmp / f"own_{split}", realtime=False)
    return out


def _same_except_y(a, b):
    assert len(a) == len(b) and len(a) > 0
    a, b = a.drop(columns="y").reset_index(drop=True), b.drop(columns="y").reset_index(drop=True)
    diff = [c for c in a.columns if not a[c].equals(b[c])]
    assert not diff, f"features endret seg når data raden ikke skal se ble endret: {diff}"


def _keys_match(a, b):
    for c in ("date", "seq", "line", "stop"):
        assert a[c].equals(b[c]), "radene kom i ulik rekkefølge"


def test_treningsrader_ser_aldri_sin_egen_dag(runs):
    base = runs["base"]
    for split, day in OWN_DAYS.items():
        a, b = base[split], runs[f"own_{split}"][split]
        _keys_match(a, b)
        m = (pd.to_datetime(a.date) == day).to_numpy()
        assert (a.y[m] != b.y[m]).all(), "testoppsettet endret ikke fasiten"
        _same_except_y(a[m], b[m])
    # Kontroll: endringen slår faktisk ut i historikken til andre rader (ellers sier testen ingenting)
    m = (pd.to_datetime(base["train"].date) != OWN_DAYS["train"]).to_numpy()
    assert not base["train"].hist_mean_lsd[m].equals(runs["own_train"]["train"].hist_mean_lsd[m])


def test_testrader_ser_bare_fortiden(runs):
    a, b = runs["base"]["test"], runs["dec"]["test"]
    _keys_match(a, b)
    m = (pd.to_datetime(a.date) >= "2025-12-01").to_numpy()
    assert (a.y[m] != b.y[m]).any()
    _same_except_y(a[m], b[m])      # historikk og baseline er fra før desember


def test_sanntidsfeatures_bruker_bare_data_foer_prognosetidspunktet(runs):
    a, b = runs["base"]["rt_test"], runs["dec"]["rt_test"]
    _keys_match(a, b)
    assert a.journey_id.equals(b.journey_id) and a.lead_min.equals(b.lead_min)
    T = a.journey_id.map(runs["trip_start"]) - pd.to_timedelta(a.lead_min + 2, unit="min")
    day = pd.to_datetime(a.date).dt.strftime("%Y-%m-%d")
    cutoff = day.map(CUTOFF)
    check = (cutoff.notna() & (T <= cutoff) & (T > cutoff - pd.Timedelta(hours=1))).to_numpy()
    near = (check & (T > cutoff - pd.Timedelta(minutes=3))).to_numpy()
    assert check.sum() > 500 and near.sum() > 20, "for få rader nær skjæringstidspunktet"
    _same_except_y(a[check], b[check])
    # Kontroll: radene med T etter skjæringstidspunktet får endrede sanntidsfeatures
    after = (cutoff.notna() & (T > cutoff)).to_numpy()
    rt_cols = [c for c in a.columns if c.startswith("rt_")]
    assert any(not a.loc[after, c].equals(b.loc[after, c]) for c in rt_cols)
    # og featurene er faktisk fylt ut for radene som sjekkes
    assert a.loc[check, "rt_ld_n60"].gt(0).mean() > 0.9
    assert a.loc[check, "rt_in_delay"].notna().mean() > 0.2


def _day(df):
    return pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")


def _same_rows(a, b, keys, skip=()):
    """Radene i b (en delmengde av a) har de samme verdiene som de samme radene i a."""
    m = a.merge(b, on=keys, how="right", suffixes=("", "_b"), indicator=True)
    assert len(m) == len(b) > 0 and (m._merge == "both").all(), "radene finnes ikke i begge kjøringene"
    cols = [c for c in b.columns if c not in keys and c not in skip]
    diff = [c for c in cols if not m[c].reset_index(drop=True).equals(m[c + "_b"].reset_index(drop=True))]
    assert not diff, f"features endret seg når turer ble avkortet etter prognosetidspunktet: {diff}"


def test_avkortede_turer_endrer_ikke_features(runs):
    a, b = runs["base"]["test"], runs["trunc"]["test"]
    a, b = a[_day(a) == TRUNC_DAY], b[_day(b) == TRUNC_DAY]
    keys = ["journey_id", "seq"]
    assert len(b) < len(a), "testoppsettet avkortet ingen turer"
    shortened = b.groupby("journey_id").seq.max() < a.groupby("journey_id").seq.max().reindex(b.journey_id.unique())
    assert shortened.sum() >= 3, "for få avkortede turer med rader igjen"
    _same_rows(a, b, keys, skip=["dest_name"])      # dest_name er den registrerte (bare til visning i appen)


def test_avkortede_turer_endrer_ikke_sanntidsfeatures(runs):
    a, b = runs["base"]["rt_test"], runs["trunc"]["rt_test"]
    T = b.journey_id.map(runs["trip_start"]) - pd.to_timedelta(b.lead_min + 2, unit="min")
    c = CUTOFF[TRUNC_DAY]
    check = ((_day(b) == TRUNC_DAY) & (T <= c) & (T > c - pd.Timedelta(hours=1))).to_numpy()
    assert check.sum() > 100
    _same_rows(a, b[check], ["journey_id", "seq", "lead_min"])
    assert b.loc[check, "rt_in_progress"].notna().mean() > 0.2      # innkommende buss er fylt ut
