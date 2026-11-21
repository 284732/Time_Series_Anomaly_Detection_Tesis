"""
Verifica di D-40 (F8 passo 7; nota del 2026-11-20 su D-40): intervalli per sensore e amplificazione dello scaler storico.

Sola lettura: legge solo ``train/`` e ``test/`` delle **macchine di sviluppo** (``configs/dev_machines.yaml``, D-51),
mai le label né le macchine di valutazione; nessun training. Scrive solo nella cartella ``--out``, che deve non
esistere (mai sovrascritta).

Per macchina e sensore:
  * range su fit (split3, frazioni D-05) e sul train storico (legacy2, ``val_ratio``), con i confini di
    ``src/data/segments.py::segment_bounds`` (gli stessi della pipeline);
  * range nel test;
  * massimo assoluto nel test dopo lo scaler storico (``SMDScaler`` minmax, ``min_range = 0``, nessun clip), fittato
    a parte su fit e sul train storico; con range esattamente nullo lo scaler usa scala 1 (come sklearn);
  * sensori con range nullo e con range quasi nullo sul fit (``range < tau``, soglia dichiarata in ``meta.json``);
  * conteggio per macchina dei sensori con massimo assoluto > ``--maxabs-threshold`` (2) nel test, casi peggiori.
Nessuna conclusione sulla decisione: solo numeri.

Uso (dalla radice): python scripts/verify_d40.py --out reports_local/d40
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.segments import segment_bounds  # noqa: E402
from src.preprocessing.scaler import SMDScaler  # noqa: E402
from src.utils.schema import SplitConfig  # noqa: E402

TAGS = ("fit60", "train80")


def _read(path: Path) -> np.ndarray:
    # stesso formato e tipo di src/data/loader.py, senza leggere le label
    return pd.read_csv(path, header=None, dtype=np.float32).values


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001 - provenance facoltativa
        return None


def analyse_machine(name: str, train: np.ndarray, test: np.ndarray, args) -> tuple:
    s3 = segment_bounds(len(train), len(test), SplitConfig("split3", None, tuple(args.fractions)), args.window_size)
    l2 = segment_bounds(len(train), len(test), SplitConfig("legacy2", args.val_ratio, None), args.window_size)
    segs = {"fit60": train[s3.fit.slice], "train80": train[l2.fit.slice]}
    rows = {"machine": name, "sensor": np.arange(train.shape[1])}
    test64 = test.astype(np.float64)
    rows["range_test"] = test64.max(0) - test64.min(0)
    for tag, seg in segs.items():
        sc = SMDScaler(scaler_type="minmax", min_range=0.0, clip=None).fit(seg)
        rng_ = sc.fit_max_.astype(np.float64) - sc.fit_min_.astype(np.float64)
        rows[f"range_{tag}"] = rng_
        rows[f"zero_range_{tag}"] = rng_ == 0
        rows[f"near_zero_{tag}"] = rng_ < args.tau
        rows[f"scale_{tag}"] = sc.scale_                   # 1/range (1 se range nullo)
        z = sc.transform(test)
        rows[f"maxabs_test_{tag}"] = np.abs(z).max(0).astype(np.float64)
        rows[f"argmax_test_{tag}"] = np.abs(z).argmax(0)
    bounds = {"fit_split3": [s3.fit.start, s3.fit.stop], "train_legacy2": [l2.fit.start, l2.fit.stop],
              "n_train": len(train), "n_test": len(test)}
    return pd.DataFrame(rows), bounds


def per_machine(s: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = []
    for m, g in s.groupby("machine", sort=True):
        r = {"machine": m, "n_sensors": len(g)}
        for tag in TAGS:
            r[f"n_zero_range_{tag}"] = int(g[f"zero_range_{tag}"].sum())
            r[f"n_near_zero_{tag}"] = int(g[f"near_zero_{tag}"].sum())
            r[f"n_maxabs_gt2_{tag}"] = int((g[f"maxabs_test_{tag}"] > thr).sum())
            r[f"max_maxabs_{tag}"] = float(g[f"maxabs_test_{tag}"].max())
            r[f"worst_sensor_{tag}"] = int(g.loc[g[f"maxabs_test_{tag}"].idxmax(), "sensor"])
        r["n_range_fit60_lt_train80"] = int((g["range_fit60"] < g["range_train80"]).sum())
        out.append(r)
    return pd.DataFrame(out)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data-dir", default=str(ROOT / "dataset"))
    p.add_argument("--dev-split", default=str(ROOT / "configs" / "dev_machines.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2])   # D-05
    p.add_argument("--val-ratio", type=float, default=0.2)                       # configs/base.yaml, legacy2
    p.add_argument("--window-size", type=int, default=100)                       # configs/base.yaml
    p.add_argument("--tau", type=float, default=1e-2)                            # soglia «quasi nullo»
    p.add_argument("--maxabs-threshold", type=float, default=2.0)                # nota su D-40
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        p.error(f"la cartella di uscita esiste già: {out} (mai sovrascritta)")
    machines: List[str] = list(yaml.safe_load(Path(args.dev_split).read_text(encoding="utf-8"))["dev"])
    data = Path(args.data_dir)

    frames, bounds = [], {}
    for m in machines:
        df, b = analyse_machine(m, _read(data / "train" / f"{m}.txt"), _read(data / "test" / f"{m}.txt"), args)
        frames.append(df)
        bounds[m] = b
    s = pd.concat(frames, ignore_index=True)
    pm = per_machine(s, args.maxabs_threshold)
    worst = []
    for tag in TAGS:
        w = s.sort_values(f"maxabs_test_{tag}", ascending=False).groupby("machine", sort=True).head(args.top)
        w = w[["machine", "sensor", f"range_{tag}", f"maxabs_test_{tag}", f"argmax_test_{tag}", "range_test"]]
        worst.append(w.rename(columns={f"range_{tag}": "range_fit_segment", f"maxabs_test_{tag}": "maxabs_test",
                                       f"argmax_test_{tag}": "argmax_test"}).assign(scaler_fit=tag))
    worst = pd.concat(worst, ignore_index=True).sort_values(["scaler_fit", "machine", "maxabs_test"],
                                                             ascending=[True, True, False])

    out.mkdir(parents=True)
    s.to_csv(out / "per_sensor.csv", index=False)
    pm.to_csv(out / "per_machine.csv", index=False)
    worst.to_csv(out / "worst_cases.csv", index=False)
    meta = {"script": "scripts/verify_d40.py", "commit": _commit(), "data_dir": str(data),
            "dev_split": str(args.dev_split), "machines": machines, "labels_read": False,
            "fractions_split3": args.fractions, "val_ratio_legacy2": args.val_ratio, "window_size": args.window_size,
            "scaler": {"scaler_type": "minmax", "min_range": 0.0, "clip": None},
            "near_zero_tau": args.tau, "maxabs_threshold": args.maxabs_threshold, "top": args.top,
            "bounds": bounds}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    summ = [f"# Verifica di D-40 (sola lettura, {len(machines)} macchine di sviluppo)", "",
            f"Soglia «quasi nullo»: range sul segmento di fit dello scaler < {args.tau}; "
            f"massimo assoluto > {args.maxabs_threshold}. Nessuna label letta.", "", _md_table(pm), "",
            f"## Casi peggiori (top {args.top} per macchina)", "", _md_table(worst)]
    (out / "summary.md").write_text("\n".join(summ) + "\n", encoding="utf-8")
    print(f"[INFO] scritto {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
