"""Controlli puntuali di F8 passo 6 fatti prima con comandi singoli, raccolti qui per citarli nel report.

**Non è coperto da test**: script di supporto in sola lettura per ``docs/f8_protocol_switch.md`` (F8 passo 6). Legge
solo lo store e le uscite di ``scripts/protocol_switch_report.py``; stampa a schermo, non scrive file.

Controlli:
  1. ``run.log`` dei 4 fit di variante: righe «fit completato», «saltato», WARNING / ERROR / «ambiente», «failed»;
  2. ``random_baseline``, V0 contro V3 su machine-1-7: valori per seed identici, ordine delle righe nelle tabelle;
  3. PCA puntuale V2 contro V1: rapporto delle std interne, differenza delle componenti, ``n_components_``;
  4. PCA puntuale V2 contro V1: ordine degli score mean/none nella regione primaria (Spearman);
  5. contrasto PA con soglia statica (F1, V1): quota di celle con contrasto esattamente 0, per proiezione;
  6. PCA puntuale V2 -> V1 (``deltas.csv``): righe con Δ ≠ 0 per macchina e metrica, minimo e massimo;
  7. PCA puntuale V2 e V1, residui di test (agg last): sensore con il residuo massimo e valore.

Uso, dalla radice del repository:
  python scripts/f8_extra/f8_inline_checks.py [--store store] --report reports_local/f8_protocol_switch
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUNS = {"V0": "protocol_switch_dev_v0_legacy/20261124T170125-7c9a833d",
        "V3": "protocol_switch_dev_v3_split3/20261124T180508-914894ed",
        "V2": "protocol_switch_dev_v2_epochs/20261124T190607-b3ad329a",
        "V1": "protocol_switch_dev_v1_canonical/20261124T212456-534d69d0"}
ANALYSES = {"V0": "protocol_switch_dev_v0_legacy/analyses/07ff59c9b83f",
            "V3": "protocol_switch_dev_v3_split3/analyses/c53662adf11b",
            "V2": "protocol_switch_dev_v2_epochs/analyses/eadfaab3dca3",
            "V1": "protocol_switch_dev_v1_canonical/analyses/d538a64944dd"}
W = 100


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default="store", help="radice dello store (sola lettura)")
    ap.add_argument("--report", required=True, help="cartella delle uscite di protocol_switch_report.py")
    a = ap.parse_args(argv)
    runs = Path(a.store) / "runs" / "exploratory"
    fits_dir = Path(a.store) / "fits"

    print("== 1. run.log")
    for v, r in RUNS.items():
        lines = (runs / r / "run.log").read_text(encoding="utf-8").splitlines()
        cnt = lambda *k: sum(any(x in ln for x in k) for ln in lines)
        print(f"  {v}: completati {cnt('fit completato')}, saltati {cnt('saltato')}, "
              f"WARNING/ERROR/ambiente {cnt('WARNING', 'ERROR', 'ambiente')}, failed {cnt('failed', 'fallit')}")

    print("== 2. random_baseline V0 contro V3, machine-1-7, ap_norm, ts_agg last")
    sel = {}
    for v in ("V0", "V3"):
        t = pd.read_csv(runs / ANALYSES[v] / "tables" / "free.csv.gz")
        x = t[(t.model == "random_baseline") & (t.metric == "ap_norm") & (t.machine == "machine-1-7")
              & (t.ts_agg == "last")]
        sel[v] = x
        print(f"  {v}: ordine dei seed nelle righe {x.seed.tolist()}")
    a0, a3 = (sel[v].sort_values("seed").value.to_numpy() for v in ("V0", "V3"))
    print(f"  valori per seed identici: {np.array_equal(a0, a3)}")

    fk = {}
    for v in ("V2", "V1"):
        t = pd.read_csv(runs / ANALYSES[v] / "tables" / "fits.csv.gz")
        fk[v] = t[t.model == "pca_pointwise"].set_index("machine").fit_key.to_dict()
    print("== 3. PCA puntuale, checkpoint V2 contro V1")
    for m in sorted(fk["V1"]):
        c2 = np.load(fits_dir / fk["V2"][m] / "checkpoint" / "state.npz")
        c1 = np.load(fits_dir / fk["V1"][m] / "checkpoint" / "state.npz")
        ratio = c1["std_"] / c2["std_"]
        diff = {i: round(float(ratio[i]), 5) for i in range(len(ratio)) if abs(ratio[i] - 1) > 1e-6}
        k2, k1 = c2["pca.components_"], c1["pca.components_"]
        dc = float(np.abs(np.abs(k2) - np.abs(k1)).max()) if k2.shape == k1.shape else None
        print(f"  {m}: n_components {k2.shape[0]} / {k1.shape[0]}; std V1/V2 ≠ 1: {diff}; max |Δ componenti| {dc:.3g}")

    print("== 4. PCA puntuale, ordine degli score mean/none (test, agg last, regione primaria)")
    for m in sorted(fk["V1"]):
        r2 = np.load(fits_dir / fk["V2"][m] / "residuals.npz")["test__last"]
        r1 = np.load(fits_dir / fk["V1"][m] / "residuals.npz")["test__last"]
        n = r2.shape[0] - (W - 1)
        s2, s1 = r2[:n].mean(1), r1[:n].mean(1)
        same = np.array_equal(np.argsort(s2, kind="stable"), np.argsort(s1, kind="stable"))
        print(f"  {m}: stesso ordine {same}; Spearman {spearmanr(s2, s1).statistic:.10f}")

    print("== 5. PA con soglia statica 98 (F1), V1: quota di contrasti esattamente 0")
    c = pd.read_csv(Path(a.report) / "contrasts.csv", dtype={"pa": str},
                    usecols=["variant", "factor", "model", "window_time_agg", "threshold_policy", "metric", "contrast",
                             "ts_agg"])
    c = c[c.contrast.notna() & (c.factor == "pa") & (c.metric == "f1") & (c.threshold_policy == "static_pct")
          & (c.variant == "V1") & (c.model != "random_baseline")]
    g = c.groupby(["window_time_agg", "ts_agg"]).contrast
    print(pd.DataFrame({"celle": g.size(), "mediana": g.median(), "quota zero": g.apply(lambda s: (s == 0).mean())})
          .round(4).to_string())

    print("== 6. PCA puntuale V2 -> V1: righe con Δ ≠ 0 per macchina e metrica")
    d = pd.read_csv(Path(a.report) / "deltas.csv", dtype={"pa": str})
    x = d[(d.link == "V2->V1") & (d.model == "pca_pointwise")]
    nz = x[x.delta != 0]
    print("  righe per macchina:", x.groupby("machine").size().to_dict())
    print(nz.groupby(["machine", "metric"]).delta.agg(["size", "min", "max"]).to_string())

    print("== 7. PCA puntuale, residuo di test massimo (agg last)")
    for m in sorted(fk["V1"]):
        for v in ("V2", "V1"):
            r = np.load(fits_dir / fk[v][m] / "residuals.npz")["test__last"]
            j = int(np.argmax(r.max(0)))
            print(f"  {m} {v}: sensore {j}, residuo massimo {r[:, j].max():.4g}")


if __name__ == "__main__":
    main()
