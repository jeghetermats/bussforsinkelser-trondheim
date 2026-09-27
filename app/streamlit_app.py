"""
Bussforsinkelser i Trondheim - Streamlit-dashboard.

Kjør lokalt:  streamlit run app/streamlit_app.py
Alle tall og prognoser er forhåndsberegnet (src/train.py og src/train_rt.py) og ligger i reports/.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data import (COLOR_ACTUAL, COLOR_BASELINE, COLOR_MODEL, fmt_clock, fmt_date, fmt_delay, fmt_num, load_final,
                  load_importance, load_metrics, load_oracle, load_predictions, load_rolling, load_rt_predictions,
                  load_rt_results, load_trips)

st.set_page_config(page_title="Bussforsinkelser i Trondheim", page_icon=":material/directions_bus:",
                   layout="wide")

# ---------- Felles grafstil ----------
SERIES = {"Faktisk": COLOR_ACTUAL, "Modell A": COLOR_MODEL, "Modell B": COLOR_MODEL, "Prognose": COLOR_MODEL,
          "Historisk median": COLOR_BASELINE}
DASH = {"Faktisk": [1, 0], "Modell A": [1, 0], "Modell B": [1, 0], "Prognose": [1, 0], "Historisk median": [6, 4]}   # ikke bare farge
COLOR_A = "#8FB3D9"      # modell A i sammenligningen med sanntid (lysere blå)
COLOR_RULE = "#C9A227"   # enkel regel
LEADS = {"Dagen før": None, "60 min før": 60, "30 min før": 30, "10 min før": 10}
COMMA = "replace(datum.label, '.', ',')"   # desimalkomma på aksene
LEGEND = alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2.5, labelFontSize=13)


def line_chart(data, x, y, domain, height, tooltip):
    return (
        alt.Chart(data)
        .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2)
        .encode(
            x=x, y=y, tooltip=tooltip,
            color=alt.Color("serie:N", title=None, legend=LEGEND,
                            scale=alt.Scale(domain=domain, range=[SERIES[d] for d in domain])),
            strokeDash=alt.StrokeDash("serie:N", title=None, legend=LEGEND,
                                      scale=alt.Scale(domain=domain, range=[DASH[d] for d in domain])),
        )
        .properties(height=height)
    )


def _keep_lead():
    """Et nytt klikk på valgt knapp skal ikke fjerne valget: behold forrige verdi."""
    if st.session_state.lead is None:
        st.session_state.lead = st.session_state.get("lead_prev", "30 min før")
    st.session_state.lead_prev = st.session_state.lead


def card_title(text: str):
    st.markdown(f"**{text}**")


df = load_predictions()
trips = load_trips()
metrics = load_metrics()
rolling = load_rolling()
oracle = load_oracle()
importance = load_importance()
rt_pred = load_rt_predictions()
rt_res = load_rt_results()

B1 = "Baseline 1 - historisk median per linje/stopp/time/dagtype"
lgbm, base = metrics["LightGBM"], metrics[B1]
ci = metrics.get("_info", {}).get("mae_forbedring_mot_baseline")
final = load_final()
if final is not None:   # endelig modell A: snitt av 3 seeds + blanding med baselinen
    lgbm, base, ci = final["A"]["metrics"], final["A"]["metrics_baseline"], final["A"]["mot_baseline"]
has_rt = rt_pred is not None and rt_res is not None and final is not None
if has_rt:
    rt30 = final["B"]["30"]["metrics"]
    ci30 = {"B_mot_baseline": final["B"]["30"]["mot_baseline"], "B_mot_A": final["B"]["30"]["mot_A"]}

# ================= Topp =================
st.title("Bussforsinkelser i Trondheim")
st.write(
    "Hvor forsinket blir bussen ved hvert stopp, **før turen har startet**? To LightGBM-modeller er trent på "
    "sanntidsdata fra Entur for AtB (jan-okt 2025) og testet på nov-des 2025, som de aldri har sett: "
    "**modell A** bruker bare det som er kjent dagen før, **modell B** bruker i tillegg hvordan trafikken går de "
    "siste timene før avgang. Begge sammenlignes med en historisk median for samme linje, stopp, retning, time og "
    "dagtype."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Feil dagen før (modell A)", f"{lgbm['MAE_s']:.0f} s",
          delta=f"{fmt_num((lgbm['MAE_s'] / base['MAE_s'] - 1) * 100, sign=True)} % mot baseline",
          delta_color="inverse", border=True,
          help="Gjennomsnittlig avvik fra faktisk forsinkelse. " + (
              f"95 % KI for forbedringen: {fmt_num(ci['ci95_lav_%'])}-{fmt_num(ci['ci95_høy_%'])} % (bootstrap over dager)."
              if ci else ""))
if has_rt:
    k2.metric("Feil 30 min før (modell B)", f"{rt30['MAE_s']:.0f} s",
              delta=f"{fmt_num(-ci30['B_mot_baseline']['forbedring_%'], sign=True)} % mot baseline",
              delta_color="inverse", border=True,
              help="Gjennomsnittlig feil per stopp med prognose laget 30 min før avgang. 95 % KI for forbedringen: "
                   f"{fmt_num(ci30['B_mot_baseline']['ci95_lav_%'])}-{fmt_num(ci30['B_mot_baseline']['ci95_høy_%'])} % "
                   f"mot baseline, {fmt_num(ci30['B_mot_A']['ci95_lav_%'])}-{fmt_num(ci30['B_mot_A']['ci95_høy_%'])} % "
                   "mot modell A.")
    k3.metric("Innenfor +/-2 min (modell B)", f"{fmt_num(rt30['innen_2min_%'])} %",
              delta=f"{fmt_num(rt30['innen_2min_%'] - base['innen_2min_%'], sign=True)} prosentpoeng mot baseline", border=True,
              help="Andel stopp der prognosen 30 min før avgang bommet med 2 minutter eller mindre.")
else:
    k2.metric("Store bom (RMSE)", f"{lgbm['RMSE_s']:.0f} s",
              delta=f"{fmt_num((lgbm['RMSE_s'] / base['RMSE_s'] - 1) * 100, sign=True)} % mot baseline",
              delta_color="inverse", border=True, help="RMSE straffer store feil ekstra hardt.")
    k3.metric("Innenfor +/-2 min (modell A)", f"{fmt_num(lgbm['innen_2min_%'])} %",
              delta=f"{fmt_num(lgbm['innen_2min_%'] - base['innen_2min_%'], sign=True)} prosentpoeng mot baseline", border=True)
if rolling is not None:
    wins = int((rolling.MAE_lgbm < rolling.MAE_baseline).sum())
    k4.metric("Modell A bedre enn baseline", f"{wins} av {len(rolling)} måneder", delta="rullende test mars-des 2025",
              delta_color="off", delta_arrow="off", border=True,
              help="Hver måned mars-des 2025 testet med en modell trent bare på tidligere måneder "
                   "(én seed, uten blanding).")

with st.container(border=True):
    for col, title, text in zip(st.columns(3), [
        "Historisk median: baseline",
        "Modell A: dagen før",
        "Modell B: 10-60 min før avgang",
    ], [
        "Medianforsinkelsen for samme linje, stopp, retning, time og dagtype i månedene før.",
        "Rute, kalender, vær og historisk forsinkelse. Vet ikke hvordan trafikken går i dag.",
        "Alt modell A vet, pluss sanntid om linjen, stoppet, nettet og bussen som skal kjøre turen.",
    ]):
        with col:
            card_title(title)
            st.caption(text)

# ================= Utforsk en tur: kontroller til venstre, graf til høyre =================
st.subheader("Utforsk en tur fra testperioden")
left, right = st.columns([1, 3])

with left:
    with st.container(border=True):
        lines = sorted(trips["line_no"].unique(),
                       key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        line = st.selectbox("Linje", lines, index=lines.index("3") if "3" in lines else 0)

        t_line = trips[trips.line_no == line]
        dests = (t_line.groupby("direction", observed=True)["dest"]
                       .agg(lambda s: s.value_counts().index[0]).to_dict())
        direction = st.selectbox("Retning", sorted(dests), format_func=lambda d: f"Mot {dests[d]}")

        t_dir = t_line[t_line.direction == direction]
        dates = sorted(t_dir["date"].unique())
        date = st.date_input("Dato", value=dates[len(dates) // 2], min_value=dates[0], max_value=dates[-1],
                             format="DD.MM.YYYY")

        t_day = t_dir[t_dir.date == date].sort_values("start")
        trip = None
        if t_day.empty:
            st.selectbox("Avgang", ["Ingen turer denne dagen"], disabled=True)
        else:
            labels = {r.trip: f"{fmt_clock(r.start)} fra {r.origin}" for r in t_day.itertuples()}
            # Standard: turen i ettermiddagsrushet (15-17) med størst faktisk forsinkelse - fast regel
            rush = t_day[(t_day.start >= 15 * 60) & (t_day.start < 17 * 60)]
            default = 0
            if not rush.empty:
                worst = df[df.trip.isin(rush.trip)].groupby("trip")["y"].mean().idxmax()
                default = t_day["trip"].tolist().index(int(worst))
            trip = st.selectbox("Avgang", list(labels), index=default, format_func=labels.get)
        lead_label = "Dagen før"
        if has_rt:
            if "lead" not in st.session_state:
                st.session_state.lead = "30 min før"
            lead_label = st.segmented_control(
                "Prognose laget", list(LEADS), key="lead", on_change=_keep_lead,
                help="'Dagen før' er modell A. De andre er modell B, som også vet hvordan linjen, stoppet og "
                     "hele nettet har gått fram til 2 min før prognosen lages.") or st.session_state.get("lead_prev", "30 min før")
        st.caption("Utvalget inneholder omtrent én av fire turer, så ikke alle avganger er med.")

if trip is None:
    # Dato uten turer: vis beskjed i turpanelet, men la resten av siden stå (ikke st.stop())
    with right:
        st.info("Linjen har ingen turer i datautvalget denne dagen. Velg en annen dato.")
else:
    t = df[df.trip == trip].sort_values("seq").copy()
    info = trips[trips.trip == trip].iloc[0]
    lead = LEADS[lead_label]
    t["pred"] = t["pred_lgbm"]
    if lead is not None:
        r = rt_pred[(rt_pred.trip == trip) & (rt_pred.lead_min == lead)].set_index("seq")["pred_rt"]
        t["pred"] = t["seq"].map(r).fillna(t["pred_lgbm"])
    model_name = "modell A" if lead is None else "modell B"
    mae_model = float(np.mean(np.abs(t.pred - t.y)))
    mae_base = float(np.mean(np.abs(t.pred_baseline - t.y)))

    with right:
        with st.container(border=True):
            card_title(f"Linje {line}, {fmt_date(date)} kl. {fmt_clock(info.start)}: {info.origin} -> {info.dest}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Faktisk forsinkelse (snitt)", fmt_delay(t.y.mean()),
                      help="Gjennomsnitt over alle stoppene på turen.")
            m2.metric(f"Feil, {model_name}", fmt_delay(mae_model),
                      help="Gjennomsnittlig avvik mellom prognose og faktisk forsinkelse per stopp.")
            m3.metric("Feil, historisk median", fmt_delay(mae_base),
                      help="Gjennomsnittlig avvik for baselinen per stopp.")
            diff = mae_base - mae_model
            if abs(diff) < 5:
                st.caption("Modellen og baselinen traff omtrent like godt på denne turen.")
            elif diff > 0:
                st.caption(f"På denne turen traff {model_name} i snitt **{fmt_delay(diff)} bedre** enn baselinen "
                           "per stopp.")
            else:
                st.caption(f"På denne turen traff baselinen i snitt **{fmt_delay(-diff)} bedre** enn {model_name} "
                           "per stopp.")
            long = t.melt(id_vars=["seq", "stop_name", "minute_of_day"],
                          value_vars=["y", "pred", "pred_baseline"], var_name="serie", value_name="sek")
            long["serie"] = long["serie"].map({"y": "Faktisk", "pred": "Prognose",
                                               "pred_baseline": "Historisk median"})
            long["min"] = long["sek"] / 60
            long["planlagt"] = long["minute_of_day"].map(fmt_clock)
            long["stop_name"] = long["stop_name"].astype(str)
            long["tekst"] = long["sek"].map(fmt_delay)
            chart = line_chart(
                long,
                x=alt.X("stop_name:N", sort=t["stop_name"].astype(str).tolist(), title=None,
                        axis=alt.Axis(labelAngle=-40, labelLimit=130, labelOverlap=True)),
                y=alt.Y("min:Q", title="Forsinkelse (minutter)", axis=alt.Axis(labelExpr=COMMA)),
                domain=["Faktisk", "Prognose", "Historisk median"], height=540,
                tooltip=[alt.Tooltip("stop_name:N", title="Stopp"), alt.Tooltip("planlagt:N", title="Planlagt"),
                         alt.Tooltip("serie:N", title="Serie"), alt.Tooltip("tekst:N", title="Forsinkelse")],
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#C9CDD3").encode(y="y:Q")
            st.altair_chart(zero + chart, width="stretch")

    with left:
        with st.container(border=True):
            card_title("Slik leser du grafen")
            st.markdown(
                "- **Svart:** faktisk forsinkelse ved hvert stopp.\n"
                "- **Blå:** prognosen, laget på tidspunktet du har valgt.\n"
                "- **Grå stiplet:** historisk median for samme linje, stopp, time og dagtype."
            )
            st.caption(
                "Modellen har ikke sett de svarte punktene når prognosen lages, og vet ikke hva som skjer på selve turen. "
                "Når forsinkelsen bygger seg opp i trafikken, havner "
                "den ofte under den faktiske linjen. Sanntid hjelper mest på dager der hele linjen går tregt."
            )

    with st.expander("Alle stopp på turen i tabell"):
        table = pd.DataFrame({
            "Stopp": t.stop_name.astype(str).to_numpy(),
            "Planlagt": t.minute_of_day.map(fmt_clock).to_numpy(),
            "Faktisk": t.y.map(fmt_delay).to_numpy(),
            "Modell A (dagen før)": t.pred_lgbm.map(fmt_delay).to_numpy(),
            **({f"Modell B ({lead_label})": t.pred.map(fmt_delay).to_numpy()} if lead is not None else {}),
            "Historisk median": t.pred_baseline.map(fmt_delay).to_numpy(),
            "Prognosens feil": (t.pred - t.y).map(fmt_delay).to_numpy(),
        })
        st.dataframe(table, hide_index=True, height=(len(table) + 1) * 35 + 3)   # ingen egen rullefelt
        st.caption(f"Prognosens feil = {model_name} minus faktisk. Negativ betyr at bussen var mer forsinket "
                   "enn antatt.")

# ================= Mønstre og treffsikkerhet: to kort i bredden =================
st.subheader("Mønstre og treffsikkerhet")
a, b = st.columns(2)

with a:
    with st.container(border=True):
        card_title(f"Linje {line} mot {dests[direction]} gjennom døgnet")
        d = df[(df.line_no == line) & (df.direction == direction)].copy()
        # Følger valget "Prognose laget": dagen før = modell A, ellers modell B med valgt lead
        h_lead = LEADS[lead_label]
        h_model = "Modell A" if h_lead is None else "Modell B"
        d["pred"] = d["pred_lgbm"]
        if h_lead is not None:
            rb = rt_pred[(rt_pred.lead_min == h_lead) & rt_pred.trip.isin(d.trip.unique())]
            d = d.merge(rb[["trip", "seq", "pred_rt"]], on=["trip", "seq"], how="left")
            d["pred"] = d["pred_rt"].fillna(d["pred_lgbm"])
        hourly = (d.assign(time=d.minute_of_day // 60)
                   .groupby("time")[["y", "pred", "pred_baseline"]].median() / 60).reset_index()
        hourly = hourly.rename(columns={"y": "Faktisk", "pred": h_model, "pred_baseline": "Historisk median"})
        hl = hourly.melt(id_vars="time", var_name="serie", value_name="min")
        hl["min_txt"] = hl["min"].map(fmt_num)
        st.altair_chart(line_chart(
            hl, x=alt.X("time:O", title="Time på døgnet", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("min:Q", title="Median forsinkelse (minutter)", axis=alt.Axis(labelExpr=COMMA)),
            domain=["Faktisk", h_model, "Historisk median"], height=300,
            tooltip=[alt.Tooltip("time:O", title="Time"), alt.Tooltip("serie:N", title="Serie"),
                     alt.Tooltip("min_txt:N", title="Minutter")],
        ), width="stretch")
        h_when = "dagen før" if h_lead is None else f"{h_lead} min før avgang"
        st.caption(f"Median over alle turer og stopp i nov-des 2025, med prognosen laget {h_when} (modell {h_model[-1]}), "
                   "som valgt over. Median brukes fordi modellene er trent til å treffe den typiske forsinkelsen. "
                   "Timer uten avganger (natt) er utelatt.")

with b:
    with st.container(border=True):
        if rolling is not None:
            avg = (1 - (rolling.MAE_lgbm * rolling.n_test).sum()
                   / (rolling.MAE_baseline * rolling.n_test).sum()) * 100
            card_title(f"Feil per måned, mars-des 2025: {fmt_num(avg)} % lavere enn baselinen i snitt")
            MND = ["jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"]
            r = rolling.rename(columns={"MAE_baseline": "Historisk median", "MAE_lgbm": "Modell A"})
            r["måned"] = pd.to_datetime(r["month"]).dt.month.map(lambda m: MND[m - 1])
            rl = r.melt(id_vars=["month", "måned"], value_vars=["Modell A", "Historisk median"],
                        var_name="serie", value_name="mae")
            rl["mae_txt"] = rl["mae"].map(fmt_num)
            st.altair_chart(line_chart(
                rl, x=alt.X("måned:N", sort=r["måned"].tolist(), title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("mae:Q", title="Gjennomsnittlig feil (sekunder)", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=COMMA)),
                domain=["Modell A", "Historisk median"], height=300,
                tooltip=[alt.Tooltip("month:N", title="Måned"), alt.Tooltip("serie:N", title="Modell"),
                         alt.Tooltip("mae_txt:N", title="MAE (s)")],
            ), width="stretch")
            st.caption("Hver måned mars-des er testet med en modell trent bare på tidligere måneder, med early "
                       "stopping på måneden før (enkeltmodell A). Gevinsten varierer: størst om vinteren, rundt 1 % i en "
                       "typisk måned. Y-aksen starter ikke på 0.")

st.subheader("Hva hjelper å vite rett før avgang?")
o_col, rt_col = st.columns(2)
if oracle is not None:
    with o_col, st.container(border=True):
        card_title("Hvilken informasjon er verdt mest?")
        o = oracle.rename(columns={"MAE_baseline": "Historisk median", "MAE_modell_a": "Modell A"})
        ol = o.melt(id_vars="kunnskap", var_name="serie", value_name="mae")
        ol["mae_txt"] = ol["mae"].map(fmt_num)
        st.altair_chart(
            alt.Chart(ol).mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("kunnskap:N", sort=o["kunnskap"].tolist(), title=None, axis=alt.Axis(labelLimit=320)),
                yOffset=alt.YOffset("serie:N", sort=["Historisk median", "Modell A"]),
                x=alt.X("mae:Q", title="Gjennomsnittlig feil (sekunder)"),
                color=alt.Color("serie:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["Historisk median", "Modell A"],
                                                range=[COLOR_BASELINE, COLOR_MODEL])),
                tooltip=[alt.Tooltip("kunnskap:N", title="Oraklet vet"), alt.Tooltip("serie:N", title="Modell"),
                         alt.Tooltip("mae_txt:N", title="MAE (s)")],
            )
            .properties(height=300),
            width="stretch",
        )
        st.caption(
            "Et 'orakel' får vite medianfeilen for sin gruppe på selve dagen, noe som først er kjent i ettertid, "
            "så dette er øvre grenser. Informasjon på dagsnivå (vær, hendelser) er verdt lite. Det store potensialet "
            "ligger i forsinkelsen på samme linje de siste timene, som er kjent i sanntid rett før avgang, men ikke "
            "dager i forveien."
        )


if has_rt:
    with rt_col, st.container(border=True):
        card_title("Sanntid slår både historikk og modell A")
        order = ["Baseline", "Modell A (før dagen)", "Baseline + avvik siste time", "Modell B (sanntid)"]
        labels = {"Baseline": "Historisk median", "Modell A (før dagen)": "Modell A (dagen før)",
                  "Baseline + avvik siste time": "Median + linjens avvik", "Modell B (sanntid)": "Modell B (sanntid)"}
        rr = rt_res[rt_res.metode.isin(order)].copy()
        rr.loc[rr.metode == "Modell A (før dagen)", "MAE_s"] = lgbm["MAE_s"]   # endelig A (samme for alle lead)
        rr["serie"] = rr.metode.map(labels)
        rr["når"] = rr.lead_min.map(lambda m: f"{m} min før")
        rr["mae_txt"] = rr["MAE_s"].map(fmt_num)
        st.altair_chart(
            alt.Chart(rr).mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("når:N", sort=["60 min før", "30 min før", "10 min før"], title=None),
                yOffset=alt.YOffset("serie:N", sort=[labels[o] for o in order]),
                x=alt.X("MAE_s:Q", title="Gjennomsnittlig feil (sekunder)", scale=alt.Scale(zero=False)),
                color=alt.Color("serie:N", title=None, sort=[labels[o] for o in order],
                                legend=alt.Legend(orient="top", columns=2),
                                scale=alt.Scale(domain=[labels[o] for o in order],
                                                range=[COLOR_BASELINE, COLOR_A, COLOR_RULE, COLOR_MODEL])),
                tooltip=[alt.Tooltip("når:N", title="Prognose laget"), alt.Tooltip("serie:N", title="Metode"),
                         alt.Tooltip("mae_txt:N", title="MAE (s)")],
            )
            .properties(height=300),
            width="stretch",
        )
        b30 = ci30["B_mot_A"]
        st.markdown(
            "Modell B bruker i tillegg:\n"
            "- Linje og stopp: nylig forsinkelse på samme linje og ved samme stopp\n"
            "- Nettet: forsinkelsesutviklingen på andre linjer og på strekningene videre\n"
            "- Bussen: hvor forsinket bussen som skal kjøre turen er nå"
        )
        st.caption(
            f"30 min før avgang bommer modell B {fmt_num(b30['forbedring_%'])} % mindre enn modell A "
            f"(95 % KI {fmt_num(b30['ci95_lav_%'])}-{fmt_num(b30['ci95_høy_%'])} %). En kontroll trent likt, men uten "
            "sanntidsdata, havnet på nivå med A, så gevinsten kommer fra sanntidsdataene. X-aksen starter ikke på 0."
        )

c, e = st.columns(2)
with c:
    with st.container(border=True):
        card_title("Resultater på testperioden (nov-des 2025)")
        rows = [("Global median", metrics["Baseline 0 - global median"]), ("Historisk median", base),
                ("Modell A (dagen før)", lgbm)]
        if has_rt:
            rows.append(("Modell B (30 min før)", rt30))
        st.dataframe(
            pd.DataFrame([{"Modell": n, "MAE (s)": f"{m['MAE_s']:.0f}", "RMSE (s)": f"{m['RMSE_s']:.0f}",
                           "Innen +/-1 min": f"{fmt_num(m['innen_1min_%'])} %",
                           "Innen +/-2 min": f"{fmt_num(m['innen_2min_%'])} %"}
                          for n, m in rows]),
            hide_index=True,
        )
        n_test = metrics.get("_info", {}).get("n_test")
        if n_test:
            st.caption(f"{n_test:,} stoppanløp. Lavere MAE og RMSE er bedre.".replace(",", " "))

        card_title("Metode")
        st.markdown(
            """
- **Kun informasjon kjent før avgang:** linje, stopp, retning, rutetid, kalender, dagslys, vær og historisk
  forsinkelse. Forsinkelse ved forrige stopp er bevisst utelatt, den ville gjort oppgaven triviell.
- **Tidsbasert splitt:** trening jan-aug 2025, validering sep-okt, test nov-des 2025.
- **Historiske features** bygges fra alle ~43 mill. stoppanløp. Treningsrader får *fold-historikk* (aldri sin
  egen fasit); test- og valideringsrader ser bare fortiden. Oppsettet ble valgt over mars-okt, ikke på testen.
- **Baseline:** historisk median fra alle rader før radens måned, oppdatert månedlig.
- **LightGBM med L1-tap**, som treffer medianen og minimerer gjennomsnittlig absolutt feil. Snitt av tre
  seeds; modell A blandes i tillegg litt med baselinen (vekt 0,85, valgt på sep-okt).
- **Modell B** får i tillegg sanntid fram til 2 min før prognosen: avvik fra baselinen på linjen, ved stoppet og i
  nettet, forrige buss ved samme stopp, forsinkelsesvekst per strekning summert fram til stoppet, og bussen som
  kommer inn til startholdeplassen. Featurene ble valgt på sep-okt; å trene på avviket og roligere læring hjalp ikke.
"""
        )

with e:
    with st.container(border=True):
        if importance is not None:
            card_title("Hva modell A legger vekt på")
            names = {
                "hist_med_lsdhd": "Historisk median (linje, stopp, time, dagtype)", "stop": "Stopp",
                "line": "Linje", "daylight_h": "Timer dagslys", "temp": "Temperatur", "wind": "Vind",
                "hist_mean_lsd": "Historisk snitt (linje, stopp)", "trip_start_min": "Avgangstid",
                "hist_mean_lhd": "Historisk snitt (linje, time)", "minute_of_day": "Klokkeslett ved stoppet",
                "n_stops": "Antall stopp på turen", "hist_mean_sh": "Historisk snitt (stopp, time)",
                "hist_n_lsdhd": "Antall historiske observasjoner", "snow_depth": "Snødybde", "weekday": "Ukedag",
                "seq": "Stoppnummer i ruten", "precip_3h": "Nedbør siste 3 timer", "precip": "Nedbør",
                "sched_min_from_start": "Planlagt tid fra start", "is_school_break": "Skoleferie",
                "hist_std_lsd": "Historisk spredning", "direction": "Retning", "daytype": "Dagtype",
                "route_frac": "Andel av ruten", "hour": "Time", "snowfall": "Snøfall", "is_holiday": "Helligdag",
            }
            top = importance.sort_values("gain_pct", ascending=False).head(10).copy()
            top["navn"] = top["feature"].map(names).fillna(top["feature"])
            top["gain_txt"] = top["gain_pct"].map(fmt_num)
            st.altair_chart(
                alt.Chart(top).mark_bar(color=COLOR_MODEL, cornerRadiusEnd=3)
                .encode(
                    x=alt.X("gain_pct:Q", title="Andel av modellens gain (%)", axis=alt.Axis(tickCount=6)),
                    y=alt.Y("navn:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
                    tooltip=[alt.Tooltip("navn:N", title="Feature"),
                             alt.Tooltip("gain_txt:N", title="%")],
                )
                .properties(height=30 * len(top)),
                width="stretch",
            )
            st.caption(
                "Gain = hvor mye hver feature reduserer feilen i trærne. Stopp og linje har tusenvis av verdier "
                "og kan dele opp dataene på mange måter, så gain overdriver ofte slike kategorier."
            )

        card_title("Begrensninger")
        st.markdown(
            """
- Før avgang er mye av forsinkelsen tilfeldig, så forbedringen over en god historisk regel er moderat.
- Sanntidsdataene er de endelige registrerte tidene. Den ekte strømmen kan komme senere eller bli rettet,
  derav bufferen på 2 min.
- Værdata er målt vær, ikke værvarsel. En ekte tjeneste måtte brukt varsel.
- Innenfor +/-1 minutt treffer baselinen litt oftere enn modell A (49,3 % mot 48,6 %); B treffer oftest
  (50,1 %). Modellenes største styrke er færre store bom.
- Den innkommende bussen finnes bare for ~10 % av turene (samme linje, samme holdeplassnavn); busser som bytter
  linje på endeholdeplassen er ikke med.
- Svakere enn baselinen på nattbusser kl. 01-04, der det er få observasjoner.
"""
        )

st.caption("Data: Entur (NLOD) og Open-Meteo | Kode: [github.com/jeghetermats/bussforsinkelser-trondheim](https://github.com/jeghetermats/bussforsinkelser-trondheim)")
