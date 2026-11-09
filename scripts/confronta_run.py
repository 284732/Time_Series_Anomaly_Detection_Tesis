"""
Confronta due cartelle di risultati (stessa config, stessi seed): score .npz e metriche del riepilogo.
Uso:  python confronta_run.py <cartella_A> <cartella_B>
Esempio: python scripts/confronta_run.py results/results_tests/results_repro_last results/results_tests/results_repro_last_01
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows, missing = [], []
for fa in sorted((a_dir / "scores").glob("machine-*/*_scores.npz")):
    fb = b_dir / "scores" / fa.parent.name / fa.name
    if not fb.exists():
        missing.append(str(fb))
        continue
    A, B = np.load(fa), np.load(fb)
    m = A["test_mask"] & B["test_mask"]
    ta, tb = A["test_ts"][m].astype(np.float64), B["test_ts"][m].astype(np.float64)
    rel = np.abs(ta - tb) / np.maximum(np.abs(ta), 1e-12)
    rows.append({
        "machine": fa.parent.name,
        "model__variant": fa.name.replace("__s42_scores.npz", ""),
        "mask_equal": bool(np.array_equal(A["test_mask"], B["test_mask"])),
        "max_abs_diff": float(np.abs(ta - tb).max()),
        "max_rel_diff": float(rel.max()),
        "thr_rel_diff": float(abs(float(A["threshold"]) - float(B["threshold"])) / max(abs(float(A["threshold"])), 1e-12)),
        "bit_identical": bool(np.array_equal(ta, tb)),
    })
d = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(d.groupby("model__variant")[["max_abs_diff", "max_rel_diff", "thr_rel_diff"]].max().to_string(float_format=lambda x: f"{x:.2e}"))
print("\nfile identici bit a bit:", int(d.bit_identical.sum()), "/", len(d), "| maschere diverse:", int((~d.mask_equal).sum()))
if missing:
    print("file mancanti in B:", len(missing))

sa, sb = (pd.read_csv(p / "summary_timestamp_metrics.csv") for p in (a_dir, b_dir))
key = ["Machine", "Model", "Variant", "Seed"]
j = sa.merge(sb, on=key, suffixes=("_A", "_B"))
for c in ["AP", "ROC_AUC", "F1", "EventRecall"]:
    j[f"d{c}"] = (j[f"{c}_A"] - j[f"{c}_B"]).abs()
print("\nDifferenza massima nelle metriche (valore assoluto):")
print(j.groupby("Model")[["dAP", "dROC_AUC", "dF1", "dEventRecall"]].max().to_string(float_format=lambda x: f"{x:.2e}"))