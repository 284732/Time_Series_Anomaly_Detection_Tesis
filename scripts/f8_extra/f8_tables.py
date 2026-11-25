"""Tabelle per docs/f8_protocol_switch.md, in sola lettura da reports_local/f8_protocol_switch/ e dallo store.

**Non è coperto da test**: script di supporto in sola lettura per ``docs/f8_protocol_switch.md`` (F8 passo 6),
copiato dalla cartella di lavoro di Claude con i soli percorsi d'ingresso resi argomenti. Si lancia dalla radice del repository.
Uso: python scripts/f8_extra/f8_tables.py --report reports_local/f8_protocol_switch --out <file.md> [--store store]
"""
import json, glob, collections
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import argparse

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--store", default="store", help="radice dello store (sola lettura)")
_ap.add_argument("--report", required=True, help="cartella delle uscite di protocol_switch_report.py")
_ap.add_argument("--out", required=True, help="file Markdown di uscita")
_ARGS = _ap.parse_args()
STORE = _ARGS.store.rstrip("/\\")
REPORT = _ARGS.report.rstrip("/\\")

R = REPORT + "/"
OUT = _ARGS.out
MODELS = ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "lstm_forecaster",
          "conv_autoencoder_untrained", "lstm_autoencoder_untrained", "input_norm", "rolling_zscore", "random_baseline"]
NATIVE = {"rolling_zscore", "random_baseline"}
SCOR = [("mean", "none"), ("mean", "std"), ("max", "std"), ("top5", "std")]
LINKS = ["V0->V3", "V3->V2", "V2->V1"]
out = []


def f(x, sign=True, nd=4):
    if pd.isna(x):
        return "—"
    if x == 0:
        s = "0"
    else:
        s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
        if float(s) == 0:
            s = ("+" if x > 0 else "−") + "0,0000" if sign else "0,0000"
            return s + "…"
    return s.replace("-", "−").replace(".", ",")


def cell(med, lo, hi):
    return f"{f(med)} [{f(lo)}; {f(hi)}]"


ds = pd.read_csv(R + "delta_summary.csv", dtype={"pa": str})
d = pd.read_csv(R + "deltas.csv", dtype={"pa": str})
base = (ds.scenario == "primary") & (ds.window_time_agg.isin(["last", "none"])) & (ds.ts_agg == "last") & (ds.table == "free")


def link_table(link, metric):
    s = ds[base & (ds.link == link) & (ds.metric == metric)]
    rows = ["| modello | " + " | ".join(f"{a}/{n}" for a, n in SCOR) + " |", "|---|" + "---|" * 4]
    for m in MODELS:
        sm = s[s.model == m]
        if m in NATIVE:
            r = sm.iloc[0]
            cells = [cell(r["median"], r["min"], r["max"]) + " (native)"] + ["·"] * 3
        else:
            cells = []
            for a, n in SCOR:
                r = sm[(sm.sensor_agg == a) & (sm.normalizer == n)]
                cells.append(cell(*r[["median", "min", "max"]].iloc[0]) if len(r) else "—")
        assert all(sm.n_machines == 6), (link, metric, m)
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def robustness(link):
    s = d[(d.link == link) & (d.metric == "ap_norm") & (d.scenario == "primary") & (d.table == "free")]
    rows = ["| modello | celle | mediana | min | max | Δ > 0 | Δ < 0 | \\|Δ\\| ≥ 0,01 |", "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        x = s[s.model == m].delta
        rows.append(f"| `{m}` | {len(x)} | {f(x.median())} | {f(x.min())} | {f(x.max())} | {int((x > 0).sum())} | "
                    f"{int((x < 0).sum())} | {int((x.abs() >= 0.01).sum())} |")
    return "\n".join(rows)


for link in LINKS:
    out.append(f"\n### {link} ap_norm (primary, wta last, ts_agg last)\n\n" + link_table(link, "ap_norm"))
    out.append(f"\n### {link} roc_auc\n\n" + link_table(link, "roc_auc"))
    out.append(f"\n### {link} robustezza ap_norm (tutte le celle: wta × ts_agg × scoring × macchine)\n\n" + robustness(link))

# --- contrasti (fattori della letteratura)
c = pd.read_csv(R + "contrasts.csv", dtype={"pa": str, "threshold_level": str},
                usecols=["variant", "factor", "machine", "model", "scenario", "window_time_agg", "sensor_agg", "normalizer",
                         "table", "threshold_policy", "threshold_level", "pa", "metric", "contrast", "ts_agg"])
c = c[c.contrast.notna()]
print("contrasti non NaN:", len(c), c.groupby(["variant", "factor"]).size().to_dict())
print("valori pa:", c.pa.unique()[:10], "thr policy:", c.threshold_policy.unique(), "levels:", c.threshold_level.unique())


def pick(df, m):
    a, n = ("native", "none") if m in NATIVE else ("mean", "std")
    return df[(df.model == m) & (df.sensor_agg == a) & (df.normalizer == n) & (df.window_time_agg.isin(["last", "none"]))]


FACT = [("ts_agg_mean_vs_last", "ap_norm", "free", None), ("ts_agg_max_vs_last", "ap_norm", "free", None),
        ("pa", "f1", "thr", "static_pct"), ("pa", "f1", "thr", "oracle"), ("oracle_vs_static", "f1", "thr", None),
        ("context_vs_primary", "ap_norm", "free", None)]


def fsel(df, fac, metric, table, pol):
    x = df[(df.factor == fac) & (df.metric == metric) & (df.table == table)]
    if fac == "pa":
        x = x[x.threshold_policy == pol]
    if fac == "context_vs_primary":
        x = x[x.ts_agg == "last"]
    if fac == "pa" and pol == "static_pct":
        x = x[x.threshold_level.astype(float) == 98.0]
    if fac in ("pa", "oracle_vs_static"):
        x = x[x.ts_agg == "last"]
    return x


for fac, metric, table, pol in FACT:
    name = fac + (f" ({pol})" if pol else "")
    variants = ["V1"] if fac == "context_vs_primary" else ["V0", "V3", "V2", "V1"]
    rows = ["| modello | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for m in MODELS:
        cells = []
        for v in variants:
            x = pick(fsel(c[c.variant == v], fac, metric, table, pol), m)
            if len(x) and fac != "context_vs_primary" and m == "random_baseline" and fac.startswith("ts_agg"):
                pass
            if len(x) == 0:
                cells.append("—")
                continue
            assert len(x) == 6, (fac, pol, v, m, len(x))
            cells.append(cell(x.contrast.median(), x.contrast.min(), x.contrast.max()))
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    out.append(f"\n### contrasto {name}, metrica {metric} (scoring mean/std o native, wta last, ts_agg last per le soglie)\n\n"
               + "\n".join(rows))

# V1 − V0 dei contrasti
cv = pd.read_csv(R + "contrasts_v1_vs_v0.csv", dtype={"pa": str, "threshold_level": str},
                 usecols=["factor", "machine", "model", "window_time_agg", "sensor_agg", "normalizer", "table",
                          "threshold_policy", "threshold_level", "pa", "metric", "contrast_v0", "contrast_v1", "delta", "ts_agg"])
cv = cv[cv.delta.notna()]
cv["contrast"] = cv["delta"]
rows = ["| modello | " + " | ".join(n + (f" ({p})" if p else "") for n, _, _, p in FACT[:5]) + " |", "|---|" + "---|" * 5]
for m in MODELS:
    cells = []
    for fac, metric, table, pol in FACT[:5]:
        x = pick(fsel(cv, fac, metric, table, pol), m)
        cells.append(cell(x.delta.median(), x.delta.min(), x.delta.max()) if len(x) == 6 else f"— ({len(x)})")
    rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
out.append("\n### V1 − V0 dei contrasti\n\n" + "\n".join(rows))

# --- fattore «metrica»: concordanza dei ranking ap_norm vs roc_auc (calcolo di Claude, non dello script)
S = STORE + "/runs/exploratory/"
A = {"V0": "protocol_switch_dev_v0_legacy/analyses/07ff59c9b83f", "V3": "protocol_switch_dev_v3_split3/analyses/c53662adf11b",
     "V2": "protocol_switch_dev_v2_epochs/analyses/eadfaab3dca3", "V1": "protocol_switch_dev_v1_canonical/analyses/d538a64944dd"}
rows = ["| variante | τ di Kendall, mediana [min; max] | macchine con lo stesso modello al primo posto |", "|---|---|---|"]
for v, p in A.items():
    t = pd.read_csv(f"{S}{p}/tables/free.csv.gz")
    t = t[(t.scenario == "primary") & (t.ts_agg == "last") & (t.metric.isin(["ap_norm", "roc_auc"]))
          & (t.model != "random_baseline")]
    t = t[((t.sensor_agg == "mean") & (t.normalizer == "std") & (t.window_time_agg == "last"))
          | (t.model == "rolling_zscore")]
    g = t.groupby(["machine", "model", "metric"]).value.mean().unstack("metric")
    taus, same = [], 0
    for mach, gm in g.groupby(level=0):
        assert len(gm) == 9, (v, mach, len(gm))
        taus.append(kendalltau(gm.ap_norm, gm.roc_auc).statistic)
        same += int(gm.ap_norm.idxmax() == gm.roc_auc.idxmax())
    taus = np.array(taus)
    rows.append(f"| {v} | {f(np.median(taus), False, 2)} [{f(taus.min(), False, 2)}; {f(taus.max(), False, 2)}] | {same} su 6 |")
out.append("\n### fattore metrica: ranking dei 9 modelli per ap_norm e per roc_auc (mean/std, wta last, ts_agg last)\n\n" + "\n".join(rows))

# --- D-40: PCA puntuale V2->V1 per macchina
pw = d[(d.link == "V2->V1") & (d.model == "pca_pointwise")]
grp = json.load(open(R + "report.json", encoding="utf-8"))["zero_range"]
rows = ["| macchina | gruppo | sensori a range nullo variabili nel test | Δ ap_norm mean/none | mean/std | max/std | top5/std | max \\|Δ\\| su tutte le righe | righe con Δ ≠ 0 |",
        "|---|---|---|---|---|---|---|---|---|"]
for mach in sorted(pw.machine.unique()):
    x = pw[pw.machine == mach]
    b = x[(x.metric == "ap_norm") & (x.scenario == "primary") & (x.window_time_agg == "last") & (x.ts_agg == "last") & (x.table == "free")]
    cells = [f(b[(b.sensor_agg == a) & (b.normalizer == n)].delta.iloc[0]) for a, n in SCOR]
    gz = grp[mach]
    rows.append(f"| {mach} | {'variabili' if gz['zero_range_varying_in_test'] else 'costanti'} | {gz['zero_range_varying_in_test'] or '—'} | "
                + " | ".join(cells) + f" | {x.delta.abs().max():.3g} | {int((x.delta != 0).sum())} / {len(x)} |")
out.append("\n### D-40 PCA puntuale V2->V1 per macchina\n\n" + "\n".join(rows))
# segno dei Δ di ap_norm della PCA puntuale nel gruppo «variabili», tutte le celle
xv = pw[(pw.metric == "ap_norm") & pw.machine.isin([m for m, g in grp.items() if g["zero_range_varying_in_test"]])]
out.append(f"\nPCA puntuale, gruppo variabili, ap_norm tutte le celle: {len(xv)} celle, Δ>0 {int((xv.delta>0).sum())}, "
           f"Δ<0 {int((xv.delta<0).sum())} (min {xv.delta.min():.3g}), Δ=0 {int((xv.delta==0).sum())}, |Δ|<1e-12 {int((xv.delta.abs()<1e-12).sum())}")
for mach in sorted(xv.machine.unique()):
    y = xv[xv.machine == mach].delta
    out.append(f"  {mach}: mediana {y.median():.4f} min {y.min():.3g} max {y.max():.4f}")
# i sensori a range nullo variabili: altri modelli sul gruppo costanti (per contrasto)
xc = d[(d.link == "V2->V1") & d.machine.isin(["machine-1-7", "machine-2-6"])]
out.append("\nV2->V1 max|Δ| per modello su 1-7 e 2-6: " + str(xc.groupby(["machine", "model"]).delta.apply(lambda s: float(s.abs().max())).round(6).to_dict()))

# --- punti di valutazione ed eventi per macchina (V1, primary)
t = pd.read_csv(f"{S}{A['V1']}/tables/free.csv.gz")
ev = t[(t.scenario == "primary")].groupby("machine")[["n_eval_points", "n_events", "prevalence", "n_cal_points"]].agg(["min", "max"])
out.append("\n### punti di valutazione\n\n" + ev.to_string())
tc = pd.read_csv(f"{S}protocol_switch_dev_v1_canonical/analyses/523248b05cea/tables/free.csv.gz")
out.append(tc.groupby(["scenario", "machine"])[["n_eval_points", "n_events"]].agg(["min", "max"]).to_string())

# --- spazio per run di creazione
size = collections.Counter()
import os
for p in glob.glob(STORE + "/fits/*/meta.json"):
    m = json.load(open(p, encoding="utf-8"))
    dirp = os.path.dirname(p)
    b = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fs in os.walk(dirp) for fn in fs)
    size[(m["created_by"]["experiment"], m["created_by"]["run_id"])] += b
out.append("\n### spazio per run\n" + "\n".join(f"{k}: {v/1e9:.3f} GB" for k, v in sorted(size.items())))

open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("scritto", OUT)
