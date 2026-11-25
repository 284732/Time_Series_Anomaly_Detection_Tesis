"""F8 passo 6, revisione A (2026-11-25): mediane di |Δ| della catena e confronto a parità di metrica con i contrasti.

Sola lettura dei CSV di ``scripts/protocol_switch_report.py`` (``deltas.csv``, ``contrasts.csv``); scrive solo il file
Markdown indicato con ``--out``. **Non è coperto da test**: script di supporto al report ``docs/f8_protocol_switch.md``.

Uscite:
  * tabelle di §4.1–4.4 (``ap_norm`` e ROC-AUC, ``primary``, ``last``/``last``) con, per ogni cella, mediana [min; max]
    dei Δ sulle 6 macchine e tra graffe la mediana di |Δ|;
  * tabelle di robustezza (tutte le celle ``free``) con la colonna «mediana |Δ|»;
  * confronto a parità di metrica: |Δ| della catena contro |contrasto| delle proiezioni offline (``ap_norm``) e contro
    i contrasti di PA con soglia statica (F1 senza PA, ``static_pct`` 98).

Uso, dalla radice del repository:
  python scripts/f8_extra/f8_abs_deltas.py --report reports_local/f8_protocol_switch --out <file.md>
"""
import argparse
from pathlib import Path

import pandas as pd

MODELS = ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "lstm_forecaster",
          "conv_autoencoder_untrained", "lstm_autoencoder_untrained", "input_norm", "rolling_zscore", "random_baseline"]
NATIVE = {"rolling_zscore", "random_baseline"}
SCOR = [("mean", "none"), ("mean", "std"), ("max", "std"), ("top5", "std")]
LINKS = ["V0->V3", "V3->V2", "V2->V1"]
VARIANTS = ["V0", "V3", "V2", "V1"]
NO_RANDOM = [m for m in MODELS if m != "random_baseline"]
NO_BASELINES = [m for m in NO_RANDOM if m != "rolling_zscore"]


def f(x, sign=True, nd=4):
    """Formato del report: «0» = esattamente zero; «±0,0000…» = non nullo ma |x| < 0,00005; virgola decimale."""
    if pd.isna(x):
        return "—"
    if x == 0:
        return "0"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if float(s) == 0:
        return (("+" if x > 0 else "−") if sign else "") + "0,0000…"
    return s.replace("-", "−").replace(".", ",")


def cell(x):
    return f"{f(x.median())} [{f(x.min())}; {f(x.max())}] {{{f(x.abs().median(), sign=False)}}}"


def causal(d):
    """Celle ``primary``, ``window_time_agg`` last (o none per i modelli native), ``ts_agg`` last."""
    return d[(d.scenario == "primary") & d.window_time_agg.isin(["last", "none"]) & (d.ts_agg == "last")]


def link_table(d, link, metric, models=MODELS):
    s = causal(d[(d.link == link) & (d.metric == metric) & (d.table == "free")])
    rows = ["| modello | " + " | ".join(f"{a}/{n}" for a, n in SCOR) + " |", "|---|" + "---|" * 4]
    for m in models:
        sm = s[s.model == m]
        if m in NATIVE:
            assert len(sm) == 6, (link, metric, m, len(sm))
            cells = [cell(sm.delta) + " (native)"] + ["·"] * 3
        else:
            cells = []
            for a, n in SCOR:
                x = sm[(sm.sensor_agg == a) & (sm.normalizer == n)].delta
                assert len(x) == 6, (link, metric, m, a, n, len(x))
                cells.append(cell(x))
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def robustness(d, link, models=MODELS):
    s = d[(d.link == link) & (d.metric == "ap_norm") & (d.scenario == "primary") & (d.table == "free")]
    rows = ["| modello | celle | mediana | mediana \\|Δ\\| | min | max | Δ > 0 | Δ < 0 | \\|Δ\\| ≥ 0,01 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        x = s[s.model == m].delta
        rows.append(f"| `{m}` | {len(x)} | {f(x.median())} | {f(x.abs().median(), sign=False)} | {f(x.min())} | "
                    f"{f(x.max())} | {int((x > 0).sum())} | {int((x < 0).sum())} | {int((x.abs() >= 0.01).sum())} |")
    return "\n".join(rows)


def f1_static(d, all_cells):
    """F1 con soglia statica 98, senza PA (tabella thr)."""
    x = d[(d.table == "thr") & (d.metric == "f1") & (d.threshold_policy == "static_pct")
          & (d.threshold_level.astype(float) == 98.0) & (d.pa.astype(str) == "False") & (d.scenario == "primary")]
    return x if all_cells else causal(x)


def summary_row(label, x):
    a = x.abs()
    return f"| {label} | {len(x)} | {f(a.median(), sign=False)} | {f(a.quantile(0.9), sign=False)} | {f(a.max(), sign=False)} |"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True, help="cartella delle uscite di protocol_switch_report.py")
    ap.add_argument("--out", required=True, help="file Markdown di uscita (nuovo)")
    a = ap.parse_args(argv)
    rep, out = Path(a.report), Path(a.out)
    d = pd.read_csv(rep / "deltas.csv", dtype={"pa": str})
    c = pd.read_csv(rep / "contrasts.csv", dtype={"pa": str},
                    usecols=["variant", "factor", "machine", "model", "scenario", "window_time_agg", "sensor_agg",
                             "normalizer", "table", "threshold_policy", "threshold_level", "pa", "metric", "contrast",
                             "ts_agg"])
    c = c[c.contrast.notna()]
    res = []
    for link in LINKS:
        res.append(f"\n### {link}: ap_norm (primary, last/last)\n\n" + link_table(d, link, "ap_norm"))
        res.append(f"\n### {link}: roc_auc\n\n" + link_table(d, link, "roc_auc"))
        res.append(f"\n### {link}: robustezza ap_norm\n\n" + robustness(d, link))

    for models, tag in ((NO_RANDOM, "9 modelli (senza random_baseline)"), (NO_BASELINES, "8 modelli (senza random_baseline e rolling_zscore)")):
        # ap_norm: catena contro proiezioni offline
        rows = ["| grandezza | celle | mediana | 90° percentile | massimo |", "|---|---|---|---|---|"]
        for link in LINKS:
            x = d[(d.link == link) & (d.metric == "ap_norm") & (d.table == "free") & d.model.isin(models)]
            rows.append(summary_row(f"\\|Δ\\| {link}, last/last", causal(x).delta))
        for link in LINKS:
            x = d[(d.link == link) & (d.metric == "ap_norm") & (d.table == "free") & d.model.isin(models)
                  & (d.scenario == "primary")]
            rows.append(summary_row(f"\\|Δ\\| {link}, tutte le celle", x.delta))
        for fac, name in (("ts_agg_mean_vs_last", "mean − last"), ("ts_agg_max_vs_last", "max − last")):
            for v in VARIANTS:
                x = c[(c.variant == v) & (c.factor == fac) & (c.metric == "ap_norm") & c.model.isin(models)
                      & (c.scenario == "primary")]
                xl = x[x.window_time_agg.isin(["last", "none"])]
                rows.append(summary_row(f"\\|contrasto\\| {name}, {v}, wta last", xl.contrast))
                rows.append(summary_row(f"\\|contrasto\\| {name}, {v}, tutte le wta", x.contrast))
        res.append(f"\n### ap_norm: catena contro proiezioni offline, {tag}\n\n" + "\n".join(rows))
        # F1 statico senza PA: catena contro PA
        rows = ["| grandezza | celle | mediana | 90° percentile | massimo |", "|---|---|---|---|---|"]
        for link in LINKS:
            x = f1_static(d[(d.link == link) & d.model.isin(models)], all_cells=False)
            rows.append(summary_row(f"\\|Δ\\| {link}, last/last", x.delta))
        for link in LINKS:
            x = f1_static(d[(d.link == link) & d.model.isin(models)], all_cells=True)
            rows.append(summary_row(f"\\|Δ\\| {link}, tutte le celle", x.delta))
        for v in VARIANTS:
            x = c[(c.variant == v) & (c.factor == "pa") & (c.metric == "f1") & (c.threshold_policy == "static_pct")
                  & (c.threshold_level.astype(float) == 98.0) & c.model.isin(models) & (c.scenario == "primary")]
            rows.append(summary_row(f"PA statica, {v}, last/last", causal(x).contrast))
            rows.append(summary_row(f"PA statica, {v}, tutte le celle", x.contrast))
        assert (c[(c.factor == "pa") & (c.metric == "f1")].contrast >= 0).all()   # PA non abbassa l'F1: |contrasto| = contrasto
        res.append(f"\n### F1 statico senza PA: catena contro PA, {tag}\n\n" + "\n".join(rows))
    if out.exists():
        raise SystemExit(f"{out} esiste già: scegliere un file nuovo")
    out.write_text("\n".join(res), encoding="utf-8")
    print("scritto", out)


if __name__ == "__main__":
    main()
