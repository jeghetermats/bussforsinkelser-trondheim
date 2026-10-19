# Eksperimenter og modellvalg

Detaljer bak resultatene i [README-en](../README.md): hvor mye som er mulig å oppnå, alle features i modell B,
hvordan oppsettene ble valgt, og det som ikke virket. Alle valg er tatt på data før testperioden (nov-des 2025).

## Begreper

- MAE: gjennomsnittlig absolutt feil, hvor mange sekunder prognosen bommer med i snitt.
- RMSE: som MAE, men store bom teller mer.
- Baseline: medianforsinkelsen for samme linje, stopp, retning, time og dagtype, beregnet fra alle tidligere måneder.
- Lead: hvor mange minutter før planlagt avgang prognosen lages (modell B: 10, 30 eller 60).
- Seed: startverdi for tilfeldigheten i treningen. Begge modellene er snitt av tre seeds. Modell A blandes i tillegg
  litt med baselinen (0,80 * modell + 0,20 * baseline, vekten valgt på validering). Modell B blandes ikke: en vekt
  på 0,95 ble valgt for runde 1 av B, men ble aldri brukt i den endelige modellen.
- KI: 95 % konfidensintervall, laget med bootstrap over testdagene.

## Data og metode

- Data: Entur sitt SIRI-ET-datasett for AtB, som finnes fra 3. desember 2024.
- Mål: ankomstforsinkelse i sekunder ved hvert stopp (faktisk minus planlagt ankomst).
- Tidsbasert splitt: trening jan-aug 2025, validering sep-okt (early stopping), test nov-des 2025.
  Endelig modell retrenes på jan-okt. Desember 2024 brukes bare som historikk. Periodene står ett sted
  (`SPLITS` i `src/common.py`); alle andre datoer i skriptene avledes fra dem. Helligdagene beregnes for hvert år,
  mens skoleferiene må legges inn per år i `features.py` (skriptet stopper hvis et år i dataene mangler).
- Utvalg: hele turer trekkes til trening (20 %), validering og test (25 %) med DuckDBs `hash(journey_id)`.
  Den er ikke garantert lik mellom DuckDB-versjoner, derfor er versjonen låst. `SPLIT_HASH=md5` gir en fordeling
  som er lik i alle versjoner (og kan regnes ut i Python), men andre turer enn i resultatene her.
- Historiske features (median/snitt/spredning per linje, stopp, time ...) beregnes i DuckDB over alle
  ~43 mill. rader med fold-historikk: dagene deles i 5 folder, og hver treningsrad får statistikk fra de andre
  4, så den aldri ser sin egen fasit. Test- og valideringsrader ser bare fortiden.
- Turstruktur: antall stopp på turen (`n_stops`, og `route_frac` = stoppnummer / antall stopp) er det planlagte,
  ikke antallet som ble registrert. Det hentes fra samme planlagte tur (linje, retning, dagtype, startholdeplass og
  avgangstid) de siste 4 ukene før dagen (`trip_plan` i `features.py`), og fra linjen hvis avgangstiden er ny.
  Modell B finner den innkommende bussen med den planlagte endeholdeplassen og sluttiden på samme måte.
- Modell: LightGBM med L1-tap (treffer medianen og minimerer gjennomsnittlig absolutt feil), snitt av tre seeds.
- Rensing: fjerner kansellerte turer/stopp, ekstraturer, estimerte tider og åpenbare feil (< -30 / > +60 min).

## Modell A i detalj

| Modell | MAE | RMSE | Innen +/-1 min | Innen +/-2 min |
|---|---|---|---|---|
| Global median | 129,3 s | 211,6 s | 39,7 % | 67,8 % |
| Historisk median (baseline) | 106,5 s | 181,9 s | 49,3 % | 72,5 % |
| Modell A, én seed | 102,7 s | 172,9 s | 48,2 % | 73,4 % |
| Modell A (snitt av 3 seeds + blanding) | 102,3 s | 173,2 s | 48,9 % | 73,6 % |
| Kontroll: median siste 8 uker | 104,6 s | 177,1 s | 49,1 % | 72,6 % |

- RMSE er 5 % lavere enn baselinen: modellen er best på de store bommene.
- Innenfor +/-1 minutt er baselinen litt bedre (49,3 % mot 48,9 %). Medianen er skarpere på "normale" avganger.
  Modell B er litt bedre enn baselinen her (50,3 %).
- Modell A er svakere på nattbusser kl. 01-04 med få observasjoner.
- Hva modellen bruker (`src/importance.py`, økning i MAE når én feature stokkes om på testdata): historisk median
  per linje, stopp, time og dagtype dominerer (+36,0 s), deretter stoppnummer, avgangstid, klokkeslett, linje,
  stopp og planlagte minutter fra turstart (+1,7 til +3,1 s hver). Det planlagte antallet stopp gir +0,8 s (det
  registrerte ga +2,4 s, se kontrollene under). Værfeaturene betyr lite hver for seg (under 0,1 s).

Rullende evaluering: for hver måned mars-desember trenes modellen bare på tidligere måneder og testes på måneden.

![Rullende evaluering](../reports/rolling_eval.png)

Enkeltmodellen er bedre i 7 av 10 måneder, i snitt 0,7 %, fra -1,1 % (september) til +2,4 % (november).
Retrent som hovedmodellen, også med måneden brukt til early stopping og minst 100 trær
(`python src/rolling_eval.py --refit --min-rounds 100`), er den bedre i 10 av 10 måneder, i snitt 2,9 %, fra
+0,1 % (august) til +5,2 % (april).

## Usikkerhet og ytterligere evaluering

`src/evaluate_extra.py`, test nov-des, modell B 30 min før avgang. Alle kalibreringer er gjort på validering.

![Kalibrering og feil per dag](../reports/eval_extra.png)

"Blir bussen mer enn 3 min forsinket?" (32,7 % av målingene i testen). Prognosen gjøres om til en
sannsynlighet med logistisk regresjon på validering.

| | AUC | Brier | Brier-skill mot bare andelen |
|---|---|---|---|
| Historisk median | 0,772 | 0,175 | 0,21 |
| Modell A | 0,794 | 0,166 | 0,25 |
| Modell B | 0,813 | 0,158 | 0,29 |

80 %-prediksjonsintervaller (10.-90. persentil av feilen på validering) dekker 78-80 % av testen, stabilt i november
og desember, med median bredde ~265 s.

De ti dagene der baselinen bommet mest, er nesten alle i andre halvdel av november, med lite nedbør og snø.
Værforholdene ser ikke ut til å forklare disse dagene alene. Det er her modellene tjener mest: A er 4,9 % og B 10,3 %
bedre enn baselinen disse dagene, mot 3,0 % og 6,5 % de øvrige dagene. B er bedre enn A på 60 av 61 dager.

## Modell A: seed-snitt, rullende evaluering og valg av oppsett

### Seed-snitt og blanding

`src/ensemble.py`: tre seeds (42, 1, 2) gir 0,2 % [0,1-0,4], og blanding
0,80 * modell + 0,20 * baseline gir ytterligere 0,2 %. Vekten ble valgt på sep-okt; kontrollen over mars-okt
pekte samme vei (beste vekt 0,65). Antall trær per seed: 479, 517, 541. Valideringskurven er flat, så antallet kan
variere mye mellom seeds (315-766 i en tidligere kjøring), og snittet jevner det ut.

### Rullende evaluering

Early stopping på én måned stopper noen ganger svært tidlig (21-26 trær i april og august), og modellen trenes
aldri på måneden den stoppes på, så de månedlige tallene er pessimistiske. Retrent med den måneden og minst
100 trær slår modellen baselinen hver måned (2,9 % i snitt mot 0,7 %). Retreningen og minstegrensen ble testet
samtidig, så hvor mye hver av dem betyr, er ikke skilt fra hverandre.

### Valg av oppsett og hva som ikke virket

Et tidligere oppsett ga 4,8 % forbedring. Da historikken ble endret til "bare fortiden" og beregnet på alle rader,
falt det til 2,1 %. For å finne ut hvorfor ble overgangen gjort én endring om gangen (`src/ablation.py`,
2 mill. treningsrader, samme validerings- og testrader for alle). Tabellen er fra en tidligere kjøring, med antall
stopp fra de registrerte stoppene; den viser hvorfor oppsettet ble endret, ikke dagens tall:

| Steg | Endring | Valid-MAE | Test-MAE | Diff test |
|---|---|---|---|---|
| V0 | Gammelt oppsett (statistikk fra 12 %-utvalg, fold-historikk) | 95,08 | 102,45 | |
| V1 | + statistikk fra alle rader | 94,39 | 102,31 | -0,14 |
| V2 | + ny kalender (jul/nyttår) | 94,49 | 102,36 | +0,05 |
| V3 | + uten antall-feature | 94,43 | 101,69 | -0,67 |
| V4 | + treningsrader jan-aug (uten des 2024) | 94,24 | 103,12 | +1,43 |
| V5 | + historikk bare fra fortiden, ukentlig | 94,33 | 105,23 | +2,11 |
| V6 | + bare fortiden, månedlig | 94,75 | 105,05 | -0,18 |

- Tapet kom fra to steg: desember 2024 ut av treningen (den eneste vintermåneden som ligner testen) og
  historikk bare fra fortiden. Hvor ofte historikken oppdateres betydde nesten ingenting.
- Oppstartsperiode (trene først fra mars) gjorde det verre (+2,5 s): januar-februar er for verdifulle vinterdata.
- Den nye kalenderen senket feilen på julaften/nyttårsaften fra 103 til 81 s.
- Valideringsvinduet sep-okt klarte ikke å skille variantene (alle innen ~1 s), mens testen spriket 6 s.

Valget av oppsett ble tatt uten å se på testperioden. Variantene ble sammenlignet over mars-okt, én måned om
gangen (trent på alt før, 2 seeds, `python src/ablation.py --cv`):

| Variant | Snitt-MAE mars-okt | Mot "bare fortid" | 95 % KI |
|---|---|---|---|
| Fold-historikk, treningsrader jan-aug (V4) | 91,09 | -0,41 s | [-0,57, -0,26] |
| Fold-historikk, med des 2024 (V3) | 91,27 | -0,22 s | [-0,43, -0,01] |
| Bare fortid + antall måneder historikk (D) | 91,40 | -0,09 s | [-0,20, 0,00] |
| Bare fortid, månedlig (V6) | 91,50 | - | - |

V4 ble valgt og er grunnlaget for både modell A og B. Testresultatet bekreftet valget. Fold-historikk i treningen er ikke
lekkasje: testradene ser uansett bare fortiden.

## Hvor mye er mulig før avgang? (orakel)

Et "orakel" får vite medianfeilen for sin gruppe på selve dagen, noe som først er kjent i ettertid.
Tallene er øvre grenser, ikke oppnåelige mål (`src/oracle.py`).

![Orakel](../reports/oracle.png)

| Hva oraklet vet | Baseline + korreksjon | Modell A + korreksjon |
|---|---|---|
| Ingenting ekstra | 106,5 s | 102,3 s |
| Hele nettets nivå den dagen | 103,9 s | 100,5 s |
| Linjens nivå den dagen | 96,1 s | 92,9 s |
| Linje, retning og time den dagen | 60,0 s | 58,0 s |
| Hver enkelt tur | 46,2 s | 44,1 s |

Informasjon på dagsnivå (vær, hendelser, "kaosdager") er verdt lite, 2-10 s i beste fall. Det store
potensialet ligger innenfor dagen: forsinkelsen på samme linje de siste timene. Den er ikke kjent dager i
forveien, men den er kjent i sanntid rett før avgang, og det er derfor modell B finnes.

## Modell B

### Features

Prognosen lages lead minutter før avgang (lead = 10, 30 eller 60 i testen). Modellen får bare ankomster
registrert minst 2 minutter før det (buffer for forsinkelse i sanntidsstrømmen), og turen selv har ikke startet. Features (`src/realtime.py`):

- Runde 1, hvordan dagen går: avvik fra baselinen (faktisk forsinkelse minus historisk median, klippet) på
  samme linje og retning siste 20, 60 og 180 min og siste observasjon; ved samme stopp siste 60 min; i hele
  nettet siste 15 og 60 min.
- Runde 2, hvor på ruten og hvilken buss:
  - forrige buss ved samme stopp: avviket for siste buss på samme linje og retning ved stopp k, og hvor lenge siden,
  - forsinkelsesvekst per strekning (stopp k-1 -> k, alle linjer) siste 30/90 min, summert langs turen fram til
    stopp k,
  - innkommende buss: siste planlagte tur på samme linje som ender ved startholdeplassen: hvor forsinket den er
    nå, hvor langt den har kommet og pausen som er igjen. Datasettet har ingen kjøretøy-id, så dette er en
    tilnærming; den finnes for ~10 % av turene.
- lead og minutter fram til stoppet.

Samme turer, historikk og protokoll som modell A (early stopping sep-okt, retrening jan-okt). Under trening får
hver tur et tilfeldig lead mellom 5 og 90 min; testen kjøres med 10, 30 og 60 min på de samme radene
(`src/train_rt.py`, snitt av tre seeds).

### Resultater per lead

| MAE (s), test nov-des | 60 min før | 30 min før | 10 min før |
|---|---|---|---|
| Historisk median | 106,5 | 106,5 | 106,5 |
| Modell A (dagen før) | 102,3 | 102,3 | 102,3 |
| Median + linjens avvik siste time (enkel regel) | 104,6 | 104,1 | 104,1 |
| Kontroll: B uten sanntidsfeatures (runde 1) | 102,5 | 102,5 | 102,5 |
| Modell B, runde 1 | 100,4 | 99,9 | 100,0 |
| Modell B, runde 2 | 99,1 | 98,7 | 98,5 |

- B bommer 3,1-3,7 % mindre enn A (95 % KI ved 30 min: 2,7-4,4 %) og 6,9-7,5 % mindre enn baselinen.
- Mot medianen fra bare de siste 8 ukene (se kontrollene under) er B 5,7 % [4,5-6,9] bedre 30 min før avgang.
- Kontrollen er trent helt likt som B, men uten sanntidsfeaturene. Den gir nøyaktig de samme modellene som A
  (samme rader i samme rekkefølge, samme antall trær per seed: 479, 517, 541) og havner på 102,5 s, det samme
  som seed-snittet av A. Gevinsten kommer altså fra sanntidsdataene, ikke fra en ny treningskjøring.
- Den enkle regelen gir bare ~2 %; modellen trengs for å bruke signalet.
- Runde 2 mot runde 1: 1,2-1,5 % [1,0-1,7], hvorav 0,7-0,9 % fra featurene (B med én seed mot runde 1) og resten
  fra seed-snittet. Runde 1 var én seed (42). Sammenligningen lages med `SEEDS=42 python src/train_rt.py --r1`
  før `python src/train_rt.py`; uten `SEEDS=42` får runde 1 de samme tre seedene som B, og forskjellen skyldes bare
  featurene.

### Hvor hjelper sanntid?

![B mot A](../reports/rt_breakdown.png)

Gevinsten over modell A er minst på de første stoppene (2,0 % de første 5 minuttene av turen 10 min før avgang,
mot 3,1-5,9 % lenger ut) og bare litt lavere 60 enn 10 min før avgang: featurene fanger hvordan dagen går, mer enn det
som er i ferd med å skje. Runde 2 (forrige buss, strekninger, innkommende buss) skulle rette på det, men
gevinsten mot runde 1 er ikke størst på de første stoppene:

| Runde 2 mot runde 1 (%) | 0-5 min fra turstart | 5-15 | 15-30 | 30-60 | 60+ |
|---|---|---|---|---|---|
| 10 min før | 1,5 | 1,3 | 1,8 | 1,4 | 1,6 |
| 60 min før | 1,3 | 1,1 | 1,5 | 1,1 | 1,2 |

Tabellen lages av `src/rt_breakdown.py` (kolonnen `B_mot_runde1_%` i `reports/rt_breakdown.csv`), med runde 1
trent på de samme datasettene (`SEEDS=42 python src/train_rt.py --r1`). En tidligere versjon av tabellen
sammenlignet med prognoser fra en eldre kjøring og viste størst gevinst på de første stoppene (2,6 % ved 10 min);
det holdt ikke da runde 1 ble trent på nytt på de samme dataene.

Runde 2 hjelper litt mer ved kort lead, men mesteparten av signalet er fortsatt hvordan dagen går: 98,5 s ved
10 min mot 99,1 s ved 60 min. At den innkommende bussen bare finnes for ~10 % av turene, er trolig en del av
forklaringen (se `RT_INCOMING=alle` under).

### Valg av oppsett for B

`src/select_rt.py`, snitt av 2 seeds, parvis bootstrap over dager mot runde 1:

| Variant | Valid-MAE | Mot runde 1 (95 % KI) |
|---|---|---|
| Runde 1 | 92,79 | |
| Runde 2 | 92,21 | 0,63 % [0,50, 0,75] |
| Runde 2, trent på avviket y - baseline | 92,73 | 0,06 % [-0,14, 0,28] |
| Runde 2, avvik + learning_rate 0,05, min_data_in_leaf 500 | 92,65 | 0,15 % [-0,06, 0,39] |

Å trene på avviket og roligere læring hjalp ikke og ble forkastet. Det finnes ingen måned-for-måned-kjøring over
mars-okt for B (sanntidsfeaturene er bare bygget for de faste periodene), så her er valget tatt på sep-okt.

## Kontroller av evalueringen

Noen valg i evalueringen kan favorisere modellen eller baselinen. Hvert av dem kan testes med et eget valg i
pipelinen; standardoppsettet (tallene over) er uendret. Kontrollene som ikke bygger datasettene på nytt, er
kjørt; resten er ikke kjørt på de ekte dataene ennå.

| Spørsmål | Slik testes det | Resultat |
|---|---|---|
| Baselinen oppdateres hver måned, men modellens historikk er frosset ved slutten av treningsperioden. Er det urettferdig mot modellen? | `HIST_AFTER=monthly python src/features.py` (og resten av pipelinen): valid- og testrader får historikk fra alt før radens måned, som baselinen. | Ikke kjørt ennå. |
| Forsinkelsene stiger gjennom høsten (snittforsinkelse 139-148 s aug-okt og 172 s i november). Er gevinsten bare å følge nivået? | `train.py` og `finalize.py` sammenligner også med "Baseline 2": medianen fra bare de siste 8 ukene før radens måned (`baseline_recent`, ikke en feature). | 8-ukersmedianen er 1,8 % [0,6-2,7] bedre enn baselinen. Mot den er A 2,3 % [1,6-3,0] og B (30 min) 5,7 % [4,5-6,9] bedre: gevinsten krymper, men holder. |
| Dagene er ikke uavhengige (de verste dagene kommer samlet i andre halvdel av november). Er KI for smale? | `final.json` og `metrics.json` har også KI med hele uker som trekkenhet (`*_ukeblokker`). Testen dekker 10 uker (to av dem delvise), så de er bredere og grovere. | A mot baseline: [1,8-5,3] mot [3,0-4,8] med dager. B (30 min): [6,0-8,6] mot [6,6-8,1]. Bredere, men godt over 0. |
| Dagfolder gir treningsrader statistikk fra dagene rett før og etter, så en ukelang hendelse er med i treningsfeaturene, men aldri i testfeaturene. | `HIST_FOLDS=week python src/features.py`, og varianten V4u i `python src/ablation.py --cv` (folder av hele uker). | Ablasjonen over mars-okt: ukefolder (V4u) gir 91,19 s mot 91,09 s med dagfolder, og slår fortsatt "bare fortid" (-0,31 s [-0,45, -0,16]). Dagfoldene beholdes; features.py med `HIST_FOLDS=week` er ikke kjørt. |
| Været er målt, ikke varslet, så modell A vet litt mer enn den ville gjort dagen før. | `python src/weather.py --forecast` henter varselet laget 24 timer før (Open-Meteo Previous Runs API), deretter `features.py` på nytt. | Ikke kjørt ennå. Værfeaturene betyr lite i permutasjonsviktigheten (under 0,1 s hver), så forskjellen er trolig liten. |
| n_stops, route_frac, trip_start_min og start/endeholdeplass ble beregnet fra stoppene som ble registrert. Avslører avkortede turer noe? | `python src/check_trip_structure.py`: andel turer der første eller siste stopp mangler, og om de er mer forsinket. `python src/train.py --without n_stops,route_frac`: modell A uten de to featurene. | 6,5 % av turene mangler de siste stoppene (sammenlignet med samme planlagte tur andre dager i måneden), nesten ingen mangler starten. De avkortede turene er klart mer forsinket allerede på de tre første stoppene (133 s mot 70 s, +66 s innen samme linje), så det registrerte antallet stopp røper noe om forsinkelsen. Med det ble modell A (én seed) 0,58 % [0,29-0,85] dårligere uten n_stops og route_frac (3,3 % mot 3,8 % bedre enn baselinen). Derfor er antall stopp, endeholdeplass og sluttid nå de planlagte, fra tidligere dager (se Data og metode), og `tests/test_pipeline_leakage.py` sjekker at avkortede turer ikke endrer featurene. Med det planlagte antallet er gevinsten mot baselinen 4,0 % for A (før: 4,3 %) og 7,3 % for B 30 min før avgang (før: 7,8 %). Uten de to featurene blir A (én seed) nå bare 0,29 % [0,08-0,53] dårligere (3,3 % mot 3,5 % bedre enn baselinen), så det planlagte antallet stopp er litt, men ærlig, nyttig. |
| Den rullende evalueringen retrener aldri med early stopping-måneden og stopper noen ganger etter ~20 trær. | `python src/rolling_eval.py --refit --min-rounds 100` (egen fil, `reports/rolling_eval_refit_min100.csv`). | Bedre enn baselinen i 10 av 10 måneder, 2,9 % i snitt (mot 7 av 10 og 0,7 %). Se rullende evaluering over. |
| Innkommende buss finnes bare for ~10 % av turene. | `RT_INCOMING=alle python src/realtime.py`: også en annen linje som ender ved startholdeplassen, med flagget `rt_in_same_line`. | Ikke kjørt ennå. |

## Flere begrensninger

- Valget av oppsett for modell B er bare gjort på sep-okt, ikke måned for måned.
- Den planlagte turstrukturen (antall stopp, endeholdeplass, sluttid) er det vanlige for samme tur de siste 4
  ukene, ikke rutetabellen. Etter en ruteendring kan den henge etter i et par uker, og for nye avgangstider brukes
  det vanlige for linjen. Rutetabellen (NeTEx) ville vært mer presis. Turens start (avgangstid og startholdeplass)
  er fortsatt fra det første registrerte stoppet, men nesten ingen turer mangler starten.
- Første stopp på turen har bare avgangstid og er ikke med i dataene (spørringen krever ankomsttid). Turens
  start, og dermed prognosetidspunktet for modell B, regnes derfor fra planlagt ankomst ved andre stopp, noen
  minutter etter avgang fra første stopp.
- Modell B holder turen selv utenfor sanntidsfeaturene bare via tidspunktet: observasjoner må komme før
  planlagt avgang minus lead minus 2 min. En buss som går over 12 min for tidlig fra første stopp, kan i
  sjeldne tilfeller se sin egen observasjon (ved 10 min lead). Effekten er trolig ubetydelig.

## Videre arbeid

- Måned-for-måned-valg over mars-okt også for modell B.
- Egne kvantilmodeller for intervallene i stedet for feil fra validering, og en egen klassifikator for > 3 min.
- Trend-features (siste 7/28 dager, kun fra fortiden), værhendelser, hendelser i byen som kan påvirke tider.
