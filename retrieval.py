"""
Henter AtB-sanntidsdata fra Entur (BigQuery) måned for måned til data/raw/.

Kjør:  python retrieval.py 2024          (henter alle måneder i 2024)
       python retrieval.py 2024 2025     (flere år)

Måneder som allerede er lastet ned hoppes over, så skriptet kan avbrytes og startes på nytt.
Har du allerede atb_2025.parquet i prosjektmappen, trenger du bare å hente 2024.
"""
import calendar
import sys
from pathlib import Path

import pyarrow.parquet as pq
from google.cloud import bigquery

PROJECT = "tough-totem-259120"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

years = [int(y) for y in sys.argv[1:]] or [2024]
template = (ROOT / "query.sql").read_text(encoding="utf-8")
client = bigquery.Client(project=PROJECT)

for year in years:
    for month in range(1, 13):
        out = RAW / f"atb_{year}_{month:02d}.parquet"
        if out.exists():
            print(f"{out.name} finnes allerede - hopper over")
            continue
        last = calendar.monthrange(year, month)[1]
        sql = (template.replace("{start}", f"{year}-{month:02d}-01")
                       .replace("{end}", f"{year}-{month:02d}-{last}"))
        rows = client.query(sql).result()
        print(f"{year}-{month:02d}: {rows.total_rows:,} rader - laster ned ...")
        table = rows.to_arrow(progress_bar_type="tqdm")
        tmp = out.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(out)            # skriv ferdig fil først når nedlastingen er komplett
