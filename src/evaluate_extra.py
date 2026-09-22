"""
Evalueringer utover MAE, for baseline, modell A og modell B (30 min før avgang), test nov-des 2025.

  1. 'Blir bussen mer enn 3 min forsinket?'  AUC (rangering) og Brier-score (sannsynlighet).
     Prognosen gjøres om til en sannsynlighet med logistisk regresjon på valid (sep-okt), aldri på test.
  2. Prediksjonsintervaller (80 %): empiriske 10.- og 90.-persentiler av feilen på valid, per tidel av
     prognosen (split-conformal). Dekning og bredde på test, totalt og per måned.
  3. De verste dagene: MAE per dag, de 10 dagene der baselinen bommet mest, med vær.

Valid-prognosene kommer fra early stopping-modellene (trent jan-aug), testprognosene fra de retrente
(jan-okt), så kalibreringen er litt konservativ.

Resultat: reports/eval_extra.json, reports/eval_days.csv, reports/eval_extra.png
Kjør:     python src/evaluate_extra.py      (etter finalize.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, REPORTS

LIMIT = 180                     # 'mer enn 3 min forsinket'
LEAD = 30                       # modell B vurderes 30 min før avgang
Q_LO, Q_HI = 0.10, 0.90         # 80 %-intervall


def auc(score, label):
    """AUC via rangsum (Mann-Whitney), med snitt-rang ved like verdier."""
    r = pd.Series(score).rank(method="average").to_numpy()
    n1 = label.sum()
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(x, label, iters=25):
    """P(y > LIMIT) = sigmoid(a + b*x), x i minutter. Newtons metode (2 parametre)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (label - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        w += np.linalg.solve(H + 1e-9 * np.eye(2), g)
    return w


def prob(w, x):
    return 1 / (1 + np.exp(-(w[0] + w[1] * x)))


def interval_table(pred_va, y_va):
    """Kvantiler av feilen per tidel av prognosen på valid."""
    edges = np.unique(np.quantile(pred_va, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(pred_va, edges[1:-1])
    res = y_va - pred_va
    lo = np.array([np.quantile(res[b == i], Q_LO) for i in range(len(edges) - 1)])
    hi = np.array([np.quantile(res[b == i], Q_HI) for i in range(len(edges) - 1)])
    return edges, lo, hi


def apply_interval(tab, pred):
    edges, lo, hi = tab
    b = np.digitize(pred, edges[1:-1])
    return pred + lo[b], pred + hi[b]


# ---------- Valid (sep-okt) ----------
ens = json.loads((REPORTS / "ensemble.json").read_text(encoding="utf-8"))["A"]
w_blend, seeds = ens["w_valgt_på_valid"], ens["seeds"]
va = pd.read_parquet(DATA / "valid.parquet", columns=["y", "baseline"])
a_va = w_blend * np.mean([np.load(DATA / "ens" / f"A_s{s}.npz")["valid"] for s in seeds], axis=0) \
    + (1 - w_blend) * va.baseline.to_numpy("float64")
rv = pd.read_parquet(DATA / "rt_valid.parquet", columns=["y", "baseline", "lead_min"])
fp = int((rv.y.to_numpy("int64") * (np.arange(len(rv)) % 9973 + 1)).sum())
sel = [np.load(DATA / "ens" / f"sel_R2_s{s}.npz") for s in (42, 1)]
assert all(int(r["fp"]) == fp for r in sel), "Valid-prognosene for B passer ikke til data/rt_valid.parquet"
b_va = np.mean([r["pred"] for r in sel], axis=0).astype("float64")
near = (rv.lead_min.between(20, 40)).to_numpy()          # valid har tilfeldig lead; bruk 20-40 min for B
valid = {"Historisk median": (va.baseline.to_numpy("float64"), va.y.to_numpy("float64")),
         "Modell A": (a_va, va.y.to_numpy("float64")),
         "Modell B": (b_va[near], rv.y.to_numpy("float64")[near])}

# ---------- Test (nov-des) ----------
t = pd.read_parquet(REPORTS / "test_predictions.parquet",
                    columns=["trip", "seq", "date", "y", "pred_lgbm", "pred_baseline", "line"])
rt = pd.read_parquet(REPORTS / "rt_test_predictions.parquet", columns=["trip", "seq", "lead_min", "pred_rt"])
t = t.merge(rt[rt.lead_min == LEAD].drop(columns="lead_min"), on=["trip", "seq"], how="left")
assert t.pred_rt.notna().all()
y = t.y.to_numpy("float64")
test = {"Historisk median": t.pred_baseline.to_numpy("float64"), "Modell A": t.pred_lgbm.to_numpy("float64"),
        "Modell B": t.pred_rt.to_numpy("float64")}
label = y > LIMIT
month = pd.to_datetime(t.date).dt.month.map({11: "nov", 12: "des"}).to_numpy()

out = {"andel_over_3min_test_%": float(label.mean() * 100), "klassifisering": {}, "intervaller": {}}
climate = float((valid["Historisk median"][1] > LIMIT).mean())
brier_clim = float(np.mean((climate - label) ** 2))
calib = {}
for name, p in test.items():
    pv, yv = valid[name]
    w = fit_logistic(pv / 60, yv > LIMIT)
    pr = prob(w, p / 60)
    brier = float(np.mean((pr - label) ** 2))
    out["klassifisering"][name] = {"AUC": auc(p, label), "Brier": brier,
                                   "Brier_skill_mot_andel": 1 - brier / brier_clim}
    calib[name] = pr
    lo, hi = apply_interval(interval_table(pv, yv), p)
    inside = (y >= lo) & (y <= hi)
    out["intervaller"][name] = {"dekning_%": float(inside.mean() * 100), "median_bredde_s": float(np.median(hi - lo)),
                                **{f"dekning_{m}_%": float(inside[month == m].mean() * 100) for m in ("nov", "des")}}
out["klassifisering"]["_Brier_bare_andel"] = brier_clim

# ---------- De verste dagene ----------
wx = pd.read_parquet(DATA / "test.parquet", columns=["snowfall", "precip", "temp"])
d = pd.DataFrame({"date": pd.to_datetime(t.date).dt.date, "eb": np.abs(test["Historisk median"] - y),
                  "ea": np.abs(test["Modell A"] - y), "eB": np.abs(test["Modell B"] - y),
                  "y": y, "snø_cm": wx.snowfall.to_numpy(), "nedbør_mm": wx.precip.to_numpy(), "temp": wx.temp.to_numpy()})
days = d.groupby("date").agg(MAE_baseline=("eb", "mean"), MAE_A=("ea", "mean"), MAE_B=("eB", "mean"),
                             snitt_forsinkelse=("y", "mean"), snø_cm=("snø_cm", "mean"),
                             nedbør_mm=("nedbør_mm", "mean"), temp=("temp", "mean"))
days["A_mot_baseline_%"] = (1 - days.MAE_A / days.MAE_baseline) * 100
days["B_mot_baseline_%"] = (1 - days.MAE_B / days.MAE_baseline) * 100
days.sort_values("MAE_baseline", ascending=False).to_csv(REPORTS / "eval_days.csv")
worst = days.sort_values("MAE_baseline", ascending=False).head(10)
rest = days.drop(worst.index)
out["verste_10_dager"] = {"A_mot_baseline_%": float((1 - worst.MAE_A.sum() / worst.MAE_baseline.sum()) * 100),
                          "B_mot_baseline_%": float((1 - worst.MAE_B.sum() / worst.MAE_baseline.sum()) * 100),
                          "øvrige_A_%": float((1 - rest.MAE_A.sum() / rest.MAE_baseline.sum()) * 100),
                          "øvrige_B_%": float((1 - rest.MAE_B.sum() / rest.MAE_baseline.sum()) * 100)}
out["dager_B_bedre_enn_A"] = int((days.MAE_B < days.MAE_A).sum())
out["dager"] = int(len(days))
(REPORTS / "eval_extra.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------- Figur: kalibrering og dager ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
cols = {"Historisk median": "#8A8F98", "Modell A": "#6B8FB5", "Modell B": "#1F4E79"}
for name, pr in calib.items():
    bins = pd.qcut(pr, 10, duplicates="drop")
    g = pd.DataFrame({"p": pr, "l": label}).groupby(bins, observed=True).mean()
    ax1.plot(g.p, g.l, marker="o", color=cols[name], label=f"{name} (AUC {out['klassifisering'][name]['AUC']:.2f})")
m = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
ax1.plot([0, m], [0, m], color="#C9CDD3", ls="--", lw=1)
ax1.set_xlabel("Predikert sannsynlighet for > 3 min"); ax1.set_ylabel("Faktisk andel")
ax1.set_title("Kalibrering, test nov-des (tideler)"); ax1.legend(fontsize=8)
ds = days.sort_index()
x = pd.to_datetime(ds.index)
ax2.plot(x, ds.MAE_baseline, color=cols["Historisk median"], ls="--", label="Historisk median")
ax2.plot(x, ds.MAE_A, color=cols["Modell A"], label="Modell A")
ax2.plot(x, ds.MAE_B, color=cols["Modell B"], label="Modell B (30 min før)")
ax2.set_ylabel("MAE per dag (sekunder)"); ax2.set_title("Feil per dag"); ax2.legend(fontsize=8)
fig.autofmt_xdate()
plt.tight_layout(); plt.savefig(REPORTS / "eval_extra.png", dpi=130)

# ---------- Utskrift ----------
print(f"Andel stopp > 3 min forsinket (test): {out['andel_over_3min_test_%']:.1f} %")
print(pd.DataFrame(out["klassifisering"]).drop(columns="_Brier_bare_andel").T.round(4).to_string())
print(f"Brier med bare andelen fra valid: {brier_clim:.4f}")
print("\n80 %-intervaller:")
print(pd.DataFrame(out["intervaller"]).T.round(1).to_string())
print("\nDe 10 verste dagene (etter baselinens feil):")
print(worst.round(1).to_string())
print(json.dumps(out["verste_10_dager"], ensure_ascii=False))
print(f"B bedre enn A på {out['dager_B_bedre_enn_A']} av {out['dager']} dager")
