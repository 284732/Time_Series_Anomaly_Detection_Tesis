"""
Aggregazione dei risultati di main.py.

Gerarchia corretta delle medie:
  1. per ogni (macchina, modello, variante) si media sui SEED  -> un numero per macchina;
  2. si calcolano media e deviazione standard sulle MACCHINE.
La std sulle macchine misura quanto i dati sono eterogenei; la std sui seed (colonna AP_seed_std)
misura la variabilita' dovuta all'inizializzazione: solo i modelli stocastici ne hanno una.

Confronto appaiato: per ogni modello e variante si confronta con il riferimento (PCA) macchina per
macchina, a parita' di variante (es. Conv-AE z_top5 contro PCA z_top5); i modelli con un solo score
(Isolation Forest, z-score mobile) sono confrontati con la variante raw_mean di PCA. Si riportano
differenza media, vittorie/pareggi/sconfitte e p-value di Wilcoxon (signed-rank, bilaterale).
ATTENZIONE: si fanno molti confronti; i p-value non sono corretti per confronti multipli.

Uso:
    python aggregate_results.py [--results-dir results] [--ref pca]
Output: results/aggregate/{summary.md, summary_by_model.csv, paired_vs_<ref>.csv, per_machine_AP.csv}
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["ROC_AUC", "AP", "BestF1_oracle", "F1", "EventRecall", "FalseAlarmSeg", "AlertRate"]
PAIRED_METRICS = ["AP", "ROC_AUC"]
MODEL_ORDER = ["random_baseline", "rolling_zscore", "isolation_forest", "one_class_svm",
               "pca", "conv_autoencoder", "lstm_autoencoder", "vae"]
VARIANT_ORDER = ["random", "raw", "raw_mean", "z_mean", "z_max", "z_top5"]


def load_rows(results_dir: Path) -> pd.DataFrame:
    csvs = sorted((results_dir / "per_machine").glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nessun CSV in {results_dir / 'per_machine'}: esegui prima main.py")
    return pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)


def _sort_key(model: str, variant: str):
    m = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    v = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
    return (m, v)


def per_machine_means(df: pd.DataFrame) -> pd.DataFrame:
    """Media sui seed per (macchina, modello, variante), con std sui seed dell'AP."""
    keys = ["Machine", "Model", "Variant"]
    num_cols = [c for c in df.columns if c not in keys + ["TsAgg", "Seed"]]
    pm = df.groupby(keys, as_index=False)[num_cols].mean(numeric_only=True)
    seeds = df.groupby(keys)["Seed"].nunique().rename("n_seeds")
    ap_std = df.groupby(keys)["AP"].std().rename("AP_seed_std")   # NaN con un solo seed
    return pm.merge(seeds.reset_index(), on=keys).merge(ap_std.reset_index(), on=keys)


def summary_by_model(pm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant), sub in pm.groupby(["Model", "Variant"]):
        r = {"Model": model, "Variant": variant, "N_machines": len(sub),
             "Seeds": int(sub["n_seeds"].max())}
        for m in METRICS:
            r[m] = f"{sub[m].mean():.3f} ± {sub[m].std():.3f}"
        r["AP_lift"] = f"{(sub['AP'] / sub['Prevalence']).mean():.2f}x"
        s = sub["AP_seed_std"].mean()
        r["AP_seed_std"] = "—" if np.isnan(s) else f"{s:.3f}"
        r["FitSec"] = "—" if sub["FitSec"].isna().all() else f"{sub['FitSec'].mean():.1f}"
        r["InferSec"] = "—" if sub["InferSec"].isna().all() else f"{sub['InferSec'].mean():.1f}"
        r["Epochs"] = "—" if sub["Epochs"].isna().all() else f"{sub['Epochs'].mean():.1f}"
        r["_key"] = _sort_key(model, variant)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("_key").drop(columns="_key")
    return out.reset_index(drop=True)


def paired_vs_reference(pm: pd.DataFrame, ref: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        piv = pm.pivot_table(index="Machine", columns=["Model", "Variant"], values=metric)
        for (model, variant) in piv.columns:
            if model == ref or model == "random_baseline":
                continue
            ref_col = (ref, variant) if (ref, variant) in piv.columns else (ref, "raw_mean")
            if ref_col not in piv.columns:
                continue
            pair = pd.concat([piv[(model, variant)], piv[ref_col]], axis=1, keys=["m", "r"]).dropna()
            if pair.empty:
                continue
            delta = pair["m"] - pair["r"]
            tol = 1e-4
            try:
                p = wilcoxon(delta, zero_method="wilcox").pvalue if (delta.abs() > 0).any() and len(delta) >= 6 else np.nan
            except ValueError:
                p = np.nan
            rows.append({
                "Metric": metric, "Model": model, "Variant": variant, "Ref": f"{ref_col[0]}/{ref_col[1]}",
                "N_machines": len(pair), "MeanDelta": delta.mean(),
                "Wins": int((delta > tol).sum()), "Ties": int((delta.abs() <= tol).sum()),
                "Losses": int((delta < -tol).sum()), "Wilcoxon_p": p, "_key": _sort_key(model, variant),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Metric", "_key"]).drop(columns="_key")
    return out.reset_index(drop=True)


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else float_fmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(results_dir: str = "results", ref: str = "pca", out_dir: Optional[str] = None) -> None:
    results_path = Path(results_dir)
    out_path = Path(out_dir) if out_dir else results_path / "aggregate"
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_rows(results_path)
    pm = per_machine_means(df)
    table = summary_by_model(pm)
    paired = paired_vs_reference(pm, ref, PAIRED_METRICS)

    table.to_csv(out_path / "summary_by_model.csv", index=False)
    paired.to_csv(out_path / f"paired_vs_{ref}.csv", index=False)
    pm.pivot_table(index="Machine", columns=["Model", "Variant"], values="AP").to_csv(out_path / "per_machine_AP.csv")

    n_mach = df["Machine"].nunique()
    n_seeds = df["Seed"].nunique()
    ts_aggs = ", ".join(sorted(df["TsAgg"].astype(str).unique()))
    md = [
        "# Risultati aggregati",
        "",
        f"- Macchine: {n_mach} | seed: {n_seeds} | proiezione score: {ts_aggs}",
        "- Valori = media ± std **sulle macchine** (dopo aver mediato i seed); `AP_seed_std` = std sui seed dell'AP "
        "(— per i modelli deterministici). `AP_lift` = AP / prevalenza (1x = casuale).",
        "- `FitSec`/`InferSec` in secondi per macchina (CPU); `Epochs` = epoche effettive dopo l'early stopping.",
        "",
        "## Tabella per modello e variante",
        "",
        to_markdown(table),
        "",
        f"## Confronto appaiato con `{ref}` (a parita' di variante)",
        "",
        "`MeanDelta` > 0: il modello supera il riferimento. Wins/Ties/Losses contati sulle macchine "
        "(tolleranza 1e-4). p-value di Wilcoxon bilaterale, NON corretto per confronti multipli.",
        "",
        to_markdown(paired, "{:.4f}") if not paired.empty else "(nessun confronto disponibile)",
        "",
    ]
    (out_path / "summary.md").write_text("\n".join(md), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"[INFO] {n_mach} macchine, {n_seeds} seed, proiezione: {ts_aggs}")
    print(table.to_string(index=False))
    if not paired.empty:
        print(f"\n--- Confronto appaiato con {ref} ---")
        print(paired.round(4).to_string(index=False))
    print(f"\n[INFO] Report salvato in {out_path.resolve() / 'summary.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--ref", default="pca", help="modello di riferimento per il confronto appaiato")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    main(args.results_dir, args.ref, args.out_dir)
