"""
Henter AtB-sanntidsdata fra Entur (BigQuery) måned for måned til data/raw/.

Kjør:  python retrieval.py 2024          (henter alle måneder i 2024)
       python retrieval.py 2024 2025     (flere år)

Spørringene faktureres til ditt eget Google Cloud-prosjekt: sett miljøvariabelen GCP_PROJECT,
eller velg et standardprosjekt med  gcloud config set project <prosjekt-id>.

Måneder som allerede er lastet ned hoppes over, så skriptet kan avbrytes og startes på nytt.
Måneder uten data lagres ikke (tomme filer ville sett ut som ferdige måneder), og måneden som pågår (og framtidige
måneder) hentes ikke, så en halvferdig måned aldri blir liggende som ferdig. Filer fra tidligere kjøringer som er
tomme eller ser ufullstendige ut, meldes; slett dem for å hente måneden på nytt.
Har du allerede atb_2025.parquet i prosjektmappen, trenger du bare å hente 2024.
"""
import calendar
import datetime as dt
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")   # None = standardprosjektet fra gcloud
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

years = [int(y) for y in sys.argv[1:]] or [2024]
template = (ROOT / "query.sql").read_text(encoding="utf-8")
client = bigquery.Client(project=PROJECT)
today = dt.date.today()


def check_existing(path, year, month, last):
    """Melder om en eksisterende fil er tom eller slutter før måneden er over."""
    meta = pq.read_metadata(path)
    if meta.num_rows == 0:
        print(f"{path.name}: tom fil fra en tidligere kjøring (ingen data denne måneden) - kan slettes")
        return
    dates = pq.read_table(path, columns=["operatingDate"]).column(0)
    newest = max(d for d in dates.to_pylist() if d is not None)
    if newest < dt.date(year, month, last) - dt.timedelta(days=1):
        print(f"{path.name}: siste dag er {newest} - kan være ufullstendig (slett filen for å hente måneden på nytt)")
    else:
        print(f"{path.name} finnes allerede - hopper over")


for year in years:
    for month in range(1, 13):
        out = RAW / f"atb_{year}_{month:02d}.parquet"
        last = calendar.monthrange(year, month)[1]
        if out.exists():
            check_existing(out, year, month, last)
            continue
        if dt.date(year, month, last) >= today:
            print(f"{year}-{month:02d} er ikke ferdig ennå - hopper over")
            continue
        sql = (template.replace("{start}", f"{year}-{month:02d}-01")
                       .replace("{end}", f"{year}-{month:02d}-{last}"))
        rows = client.query(sql).result()
        if rows.total_rows == 0:
            print(f"{year}-{month:02d}: ingen rader - lagres ikke")
            continue
        print(f"{year}-{month:02d}: {rows.total_rows:,} rader - laster ned ...")
        table = rows.to_arrow(progress_bar_type="tqdm")
        tmp = out.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(out)            # skriv ferdig fil først når nedlastingen er komplett
