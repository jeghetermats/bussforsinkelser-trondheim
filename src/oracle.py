"""
Orakel: hvor mye bedre kunne prognosen blitt hvis vi visste mer om selve dagen?

Hver prognose får en korreksjon lik medianfeilen i sin gruppe (f.eks. linje per dag).
Korreksjonen er bare kjent i ettertid, så dette er ØVRE GRENSER, ikke oppnåelige mål.
Finere grupper (time, tur) har få rader og inkluderer raden selv, så de er ekstra optimistiske.

Kjør:  python src/oracle.py   (bruker reports/test_predictions.parquet fra train.py)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import REPORTS

d = pd.read_parquet(REPORTS / "test_predictions.parquet")
d["hour"] = d.minute_of_day // 60
y = d.y.astype(float)
LEVELS = [
    ("Ingen (dagens modeller)", None),
    ("Hele nettets nivå den dagen", ["date"]),
    ("Linjens nivå den dagen", ["line", "date"]),
    ("Linje, retning og time den dagen", ["line", "direction", "date", "hour"]),
    ("Hver enkelt tur", ["trip"]),
]
rows = []
for label, keys in LEVELS:
    r = {"kunnskap": label}
    for name, col in [("baseline", "pred_baseline"), ("lightgbm", "pred_lgbm")]:
        p = d[col].astype(float)
        if keys:
            p = p + (y - p).groupby([d[k] for k in keys], observed=True).transform("median")
        r[f"MAE_{name}"] = float(np.mean(np.abs(p - y)))
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(REPORTS / "oracle.csv", index=False)
print(res.round(1).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(res))
ax.bar(x - 0.2, res.MAE_baseline, 0.4, color="#999999", label="Historisk median + korreksjon")
ax.bar(x + 0.2, res.MAE_lightgbm, 0.4, color="#1F4E79", label="LightGBM + korreksjon")
ax.set_xticks(x, [l.replace(" den dagen", "\nden dagen") for l in res.kunnskap], fontsize=8)
ax.set_ylabel("MAE (sekunder)"); ax.legend(fontsize=8)
ax.set_title("Hvor mye kunne vi tjent på å vite mer om dagen? (øvre grenser)")
plt.tight_layout(); plt.savefig(REPORTS / "oracle.png", dpi=130); plt.close()
print("Lagret reports/oracle.csv og reports/oracle.png")
