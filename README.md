# Bussforsinkelser i Trondheim, prediksjon før avgang

Dette prosjektet undersøker hvor godt bussforsinkelser kan predikeres før en tur har startet. Det brukes sanntidsdata fra Entur for AtB, desember 2024 - desember 2025 (~43 mill. målinger etter
rensing, en per buss per stopp), og sammenligner to LightGBM-modeller med en historisk baseline:
medianforsinkelsen for samme linje, stopp, retning, time og dagtype, beregnet fra alle tidligere måneder.

- Modell A bruker bare det som er kjent dagen før: rute, kalender, vær og historikk.
- Modell B bruker i tillegg sanntid: hvordan linjen, stoppet og hele nettet har gått de siste timene,
  10, 30 eller 60 minutter før avgang.

| Test nov-des 2025 | MAE | Forbedring mot baseline (95 % KI) |
|---|---|---|
| Historisk median (baseline) | 106,5 s | |
| Modell A - dagen før | 102,6 s | 3,7 % [2,2-4,9] |
| Modell B - 30 min før avgang | 99,8 s | 6,3 % [5,4-7,2] |

## Modell A: prognose dagen før

Test: november-desember 2025 (1,7 mill. målinger, 61 dager).

| Modell | MAE | RMSE | Innen +/-1 min | Innen +/-2 min |
|---|---|---|---|---|
| Global median | 129,3 s | 211,6 s | 39,7 % | 67,8 % |
| Historisk median (baseline) | 106,5 s | 181,9 s | 49,3 % | 72,5 % |
| Modell A (LightGBM) | 102,6 s | 170,6 s | 47,9 % | 73,1 % |

- MAE: 3,7 % lavere enn baselinen, ca. 4 sekunder per stopp (95 % KI 2,2-4,9 %, bootstrap over dager).
- RMSE: 6,2 % lavere: modellen er best der det betyr mest, på de store bommene.
- Innenfor +/-1 minutt er baselinen litt bedre (49,3 % mot 47,9 %). Medianen er skarpere på "normale" avganger.

### Gjennom året

Rullende evaluering: for hver måned mars-desember trenes modellen bare på tidligere måneder (early stopping på
måneden før) og testes på måneden.

![Rullende evaluering](reports/rolling_eval.png)

LightGBM er bedre i 7 av 10 måneder, men i snitt bare 0,8 %, fra -0,8 % (september) til +3,3 % (november).
Gevinsten er størst om vinteren, en typisk måned gir rundt 1 %.

### Hvor mye er mulig før avgang?

Et "orakel" får vite medianfeilen for sin gruppe på selve dagen, noe som først er kjent i ettertid (`src/oracle.py`).

![Orakel](reports/oracle.png)

Å vite hvor forsinket hele nettet var den dagen hjelper lite (2-10 s). Å vite hvor forsinket hver linje og retning var den aktuelle timen ville derimot senket baselinens feil fra 106,5 til 60 s. Det er ikke kjent dagen før, men delvis kjent i sanntid rett før avgang, og det er derfor modell B finnes.

## Modell B: prognose med sanntid, 10-60 min før avgang

Prognosen lages 'lead' antall minutter før avgang. Modellen får bare ankomster registrert minst 2 minutter før det
(buffer for forsinkelse i sanntidsstrømmen), og turen selv har ikke startet. Features (`src/realtime.py`):

- avvik fra baselinen (faktisk forsinkelse minus historisk median, klippet) på samme linje og retning siste
  20, 60 og 180 min, pluss siste observasjon og hvor gammel den er,
- avvik ved samme stopp (alle linjer) siste 60 min, og i hele nettet siste 15 og 60 min,
- lead og minutter fram til stoppet.

Samme turer, historikk og protokoll som modell A (early stopping sep-okt, retrening jan-okt). Under trening får
hver tur et tilfeldig lead mellom 5 og 90 min; testen kjøres med 10, 30 og 60 min på de samme radene
(`src/train_rt.py`).

![MAE per lead](reports/rt_mae_by_lead.png)

| MAE (s), test nov-des | 60 min før | 30 min før | 10 min før |
|---|---|---|---|
| Historisk median | 106,5 | 106,5 | 106,5 |
| Modell A (dagen før) | 102,6 | 102,6 | 102,6 |
| Median + linjens avvik siste time (enkel regel) | 104,6 | 104,1 | 104,1 |
| Kontroll: modell B uten sanntidsfeatures | 102,7 | 102,7 | 102,7 |
| Modell B | 100,1 | 99,8 | 99,9 |

- B bommer 2,4-2,8 % mindre enn A (95 % KI ved 30 min: 2,0-3,7 %). B er bedre enn A på 54 av 61 testdager.
- Kontrollen er trent helt likt, men uten sanntidsfeaturene, og havner på nivå med A (-0,1 %, KI [-0,5, 0,3]).
  Hele gevinsten kommer altså fra sanntidsdataene, ikke fra en ny treningskjøring.
- Den enkle regelen gir bare ~2 %; modellen trengs for å bruke signalet (-4,2 % mot regelen).
- Nesten like mye verdt 60 som 10 min før. Signalet er hvordan linjen og nettet går de siste timene, ikke
  siste minutt. Gevinsten er minst på de første stoppene (1-2 % mot A) og størst lenger ut i turen (~5 %).

## Oppsett

- Mål: ankomstforsinkelse i sekunder ved hvert stopp (faktisk minus planlagt ankomst).
- Kun informasjon kjent før avgang: linje, stopp, retning, rutetid, kalender (helligdager, skoleferie,
  julaften/nyttårsaften), dagslys, vær og historisk forsinkelse. Forsinkelse ved forrige stopp er bevisst utelatt -
  den forklarer 96 % av variasjonen og gjør oppgaven triviell.
- Tidsbasert splitt: trening jan-aug 2025, validering sep-okt (early stopping), test nov-des 2025.
  Endelig modell retrenes på jan-okt. Desember 2024 brukes bare som historikk.
- Historiske features (median/snitt/spredning per linje, stopp, time ...) beregnes i DuckDB over alle
  ~43 mill. rader med fold-historikk: dagene deles i 5 folder, og hver treningsrad får statistikk fra de andre
  4, så den aldri ser sin egen fasit. Test- og valideringsrader ser bare fortiden.
- Modell: LightGBM med L1-tap (treffer medianen og minimerer gjennomsnittlig absolutt feil).
- Modell B legger sanntidsfeatures oppå de samme featurene (se over).
- Rensing: fjerner kansellerte turer/stopp, ekstraturer, estimerte tider og åpenbare feil (< -30 / > +60 min).
- Data: Entur sitt SIRI-ET-datasett for AtB finnes først fra 3. desember 2024.

## Valg av oppsett

Et tidligere oppsett ga 4,8 % forbedring, men falt til 2,1 % da historikken ble beregnet bare fra fortiden.
En ablasjon med én endring om gangen (`src/ablation.py`, tall i `reports/ablation.csv`) viste at tapet kom fra
to steg: desember 2024 ut av treningen (den eneste vintermåneden som ligner testen) og historikk bare fra
fortiden. Å starte treningen i mars gjorde det verre, mens den nye kalenderen senket feilen på
julaften/nyttårsaften fra 103 til 81 s.

Oppsettet ble valgt uten å se på testperioden. Valideringsvinduet sep-okt klarte ikke å skille variantene, så de
ble sammenlignet over mars-okt, én måned om gangen (`--cv`). Fold-historikk med treningsrader jan-aug (V4) vant
med 0,43 s [0,28-0,58] over "bare fortid" og er grunnlaget for både modell A og B. Fold-historikk i treningen er ikke
lekkasje: testradene ser uansett bare fortiden.

## Streamlit-app

Appen viser prognoser for enkeltturer i testperioden (faktisk vs. prognose vs. historisk median), der du velger
om prognosen er laget dagen før (modell A) eller 60/30/10 min før avgang (modell B), og resultatene.
Den bruker bare forhåndsberegnede filer i `reports/`.

```bash
streamlit run app/streamlit_app.py
```

## Kjør selv

```bash
pip install -r requirements.txt
gcloud auth application-default login
python retrieval.py 2024 2025     # rådata fra BigQuery, måned for måned -> data/raw/ (data finnes fra des 2024)
python src/weather.py             # timesvær fra Open-Meteo
python src/features.py            # DuckDB: rensing, historikk over alle rader, train/valid/test
python src/train.py               # hovedmodell + test nov-des (+ bootstrap-KI)
python src/rolling_eval.py        # rullende evaluering mars-des
python src/oracle.py              # orakel-analyse
python src/realtime.py            # modell B: sanntidsfeatures -> data/rt_*.parquet
python src/train_rt.py            # modell B: trening + test per lead (--no-rt = kontroll uten sanntid)
python src/ablation.py            # ablasjon (valgfritt, flere timer)
python src/ablation.py --cv --seeds 0,1   # valg av oppsett over mars-okt (valgfritt)
```

## Begrensninger

- Før avgang er mye av forsinkelsen tilfeldig, forbedringen over en god historisk median er moderat.
- Værdata er målt vær, ikke værvarsel.
- Svakere enn baselinen innenfor +/-1 minutt, og på nattbusser kl. 01-04 med få observasjoner.
- Modell B bruker de endelige registrerte tidene. Den ekte sanntidsstrømmen kan komme senere eller bli rettet
  i ettertid; bufferen på 2 min dekker bare en del av det.

## Videre arbeid

- Modell B: forsinkelse på pågående turer på samme linje (hvor bussene er nå), og kobling til bussens forrige
  tur (datasettet mangler kjøretøy-id).
- Snitt over flere seeds og blanding med baselinen, valgt over mars-okt.
- Evaluering av det modellen er god på: "blir bussen mer enn 3 min forsinket?" (AUC/Brier) og prediksjonsintervaller.
- Trend-features (siste 7/28 dager, kun fra fortiden), værhendelser, hendelser i byen som kan påvirke tider.

Data: Entur, [data.entur.no](https://data.entur.no) (NLOD) og [Open-Meteo](https://open-meteo.com).
