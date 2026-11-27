"""
Stima del costo dei fit di tuning dal pilota sui tempi e regola K = 8 / 6 (F9 passo 3, D-41; ``src/tuning/pilot.py``).

    python scripts/f9_pilot_estimate.py --out reports_local/f9_pilot

Legge solo ``meta.json``: i fit dello slot 1 (= V1 di F8) dallo store di tuning e i fit del pilota dallo store del
pilota. Nessuna metrica, nessuna tabella di analisi; il manifest dei dati serve solo per i ``data_version`` dei
``fit_key`` (come ``status``). Prima verifica che gli esperimenti di tuning e del pilota siano quelli generati dalla
griglia. Scrive ``estimate.json`` e ``fits.csv`` in una cartella nuova (rifiuta una cartella esistente).
Codici di uscita: 0 stima fatta; 1 fit mancanti o incoerenti (``EstimateError``) o file generati non aggiornati;
2 config o dati non validi o cartella di uscita esistente.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import ManifestError, build_manifest, load_dataset_spec, spec_path_for  # noqa: E402
from src.pipeline.store import Store  # noqa: E402
from src.tuning import grid as tg  # noqa: E402
from src.tuning import pilot as tp  # noqa: E402
from src.utils.config import load_experiment  # noqa: E402
from src.utils.schema import ConfigError  # noqa: E402


def _up_to_date(files, out: Path) -> list:
    return [n for n, text in files.items()
            if not (out / n).is_file() or (out / n).read_text(encoding="utf-8") != text]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", required=True, help="cartella nuova per estimate.json e fits.csv")
    ap.add_argument("--grid", default=str(tg.DEFAULT_GRID))
    ap.add_argument("--base", default=str(tg.DEFAULT_BASE))
    ap.add_argument("--dev", default=str(tg.DEFAULT_DEV))
    ap.add_argument("--tuning-dir", default=str(tg.DEFAULT_OUT))
    ap.add_argument("--pilot-dir", default=str(tg.DEFAULT_PILOT_OUT))
    ap.add_argument("--datasets-dir", default=str(ROOT / "configs" / "datasets"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"errore: la cartella di uscita {out} esiste già (mai sovrascritta)", file=sys.stderr)
        return 2
    tdir, pdir = Path(args.tuning_dir), Path(args.pilot_dir)
    try:
        grid = tg.load_grid(args.grid)
        stale = (_up_to_date(tg.generate(args.grid, args.base, args.dev), tdir)
                 + _up_to_date(tg.generate_pilot(args.grid, args.base, args.dev), pdir))
        if stale:
            print(f"errore: esperimenti non uguali alla griglia {stale}: rigenerare con "
                  "scripts/make_tuning_experiments.py", file=sys.stderr)
            return 1
        ref = load_experiment(tdir / tg.SLOT_FILE.format(1))
        targets = {s: load_experiment(tdir / tg.SLOT_FILE.format(s)) for s in grid.pilot_slots()}
        pilots = {s: load_experiment(pdir / tg.SLOT_FILE.format(s)) for s in grid.pilot_slots()}
        manifest = build_manifest(load_dataset_spec(spec_path_for(ref.data.dataset, args.datasets_dir)),
                                  ref.data.data_dir)
        machines = manifest.resolve(ref.data.machines)
        dvs = {m: manifest.data_version(m) for m in machines}
    except (ConfigError, ManifestError, OSError) as exc:
        print(f"errore: config o dati non validi: {exc}", file=sys.stderr)
        return 2
    pilot_store = {Path(c.store_root) for c in pilots.values()}
    if len(pilot_store) != 1:
        print(f"errore: store del pilota non unico {sorted(map(str, pilot_store))}", file=sys.stderr)
        return 2
    try:
        res = tp.run_estimate(grid, ref, pilots, targets, dvs, Store(ref.store_root).read_meta,
                              Store(pilot_store.pop()).read_meta)
    except tp.EstimateError as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 1

    out.mkdir(parents=True)
    rows = res.pop("rows")
    with open(out / "estimate.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    with open(out / "fits.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"fit nuovi stimati: {res['n_fits']} (slot 2..{res['k']})")
    for m, d in res["per_model"].items():
        print(f"  {m}: {d['n_fits']} fit, {d['seconds'] / tp.HOUR:.2f} h")
    print(f"stima con K = {res['k']}: {res['estimate_hours_k']:.2f} h a muro; con K = {res['k_reduced']}: "
          f"{res['estimate_hours_k_reduced']:.2f} h")
    print(f"regola: {res['rule']} -> K = {res['chosen_k']}")
    print(f"scritto: {out / 'estimate.json'}, {out / 'fits.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
