"""
Estrazione riproducibile delle 6 macchine di sviluppo (D-51).

Procedura (D-51):
  * fisse: machine-1-1 e machine-3-8 (gia' osservate, D-52);
  * estratte: 1 macchina dal gruppo 1 (esclusa 1-1), 2 dal gruppo 2, 1 dal gruppo 3 (esclusa 3-8);
  * candidati di ogni gruppo ordinati per nome (ordinamento lessicografico, `sorted`);
  * numpy.random.default_rng(20261110), estrazione senza reinserimento con `Generator.choice`,
    nell'ordine gruppo 1, 2, 3;
  * le altre 22 macchine sono di valutazione.

Usa solo l'elenco dei nomi (`SMD_MACHINES` in src/data/loader.py): non legge dataset, label, metriche o risultati.

Uso (dalla radice del progetto):
    python scripts/make_dev_split.py           # scrive configs/dev_machines.yaml
    python scripts/make_dev_split.py --check   # verifica che il file esistente coincida, senza scrivere
Se il file esiste con un contenuto diverso, lo script si ferma: per sovrascriverlo serve --force.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SMD_MACHINES  # noqa: E402  (solo l'elenco dei nomi)

SEED = 20261110
FIXED = ("machine-1-1", "machine-3-8")
DRAWS = {1: 1, 2: 2, 3: 1}          # macchine da estrarre per gruppo, in quest'ordine
OUT_PATH = ROOT / "configs" / "dev_machines.yaml"

PROCEDURE = (
    "fisse machine-1-1 e machine-3-8 (D-52); estratte 1 dal gruppo 1 (esclusa 1-1), 2 dal gruppo 2, "
    "1 dal gruppo 3 (esclusa 3-8); candidati di ogni gruppo ordinati per nome (ordinamento lessicografico); "
    "numpy.random.default_rng(seed), Generator.choice senza reinserimento, nell'ordine gruppo 1, 2, 3; "
    "nessuna lettura di dataset, label, metriche o risultati"
)


def group_of(machine: str) -> int:
    """Gruppo server dal nome 'machine-<gruppo>-<indice>'."""
    return int(machine.split("-")[1])


def make_dev_split(machines: List[str] = SMD_MACHINES, seed: int = SEED) -> Dict[str, object]:
    """Restituisce {'dev': [...], 'eval': [...], 'seed': ..., 'procedure': ...} secondo D-51."""
    rng = np.random.default_rng(seed)
    dev = list(FIXED)
    for group, k in DRAWS.items():
        candidates = sorted(m for m in machines if group_of(m) == group and m not in FIXED)
        dev += [str(m) for m in rng.choice(candidates, size=k, replace=False)]
    dev_set = set(dev)
    return {
        "seed": seed,
        "procedure": PROCEDURE,
        "numpy_version": np.__version__,
        "dev": sorted(dev),
        "eval": [m for m in machines if m not in dev_set],
    }


def render(split: Dict[str, object]) -> str:
    header = (
        "# Generato da scripts/make_dev_split.py (D-51). Non modificare a mano:\n"
        "# rigenerare con `python scripts/make_dev_split.py`.\n"
    )
    return header + yaml.safe_dump(split, sort_keys=False, allow_unicode=True, width=1000)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verifica il file esistente senza scrivere")
    ap.add_argument("--force", action="store_true", help="sovrascrive un file esistente diverso")
    args = ap.parse_args(argv)

    split = make_dev_split()
    text = render(split)
    if OUT_PATH.exists():
        current = yaml.safe_load(OUT_PATH.read_text(encoding="utf-8")) or {}
        # si confrontano solo i contenuti: la versione di numpy e' informativa
        if all(current.get(k) == split[k] for k in ("seed", "procedure", "dev", "eval")):
            print(f"[OK] {OUT_PATH.relative_to(ROOT)} coincide con la procedura di D-51")
            return 0
        if args.check or not args.force:
            print(f"[ERRORE] {OUT_PATH.relative_to(ROOT)} esiste e differisce dalla procedura; "
                  "usa --force solo se il cambiamento e' stato approvato")
            return 1
    elif args.check:
        print(f"[ERRORE] {OUT_PATH.relative_to(ROOT)} non esiste")
        return 1

    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"[INFO] scritto {OUT_PATH.relative_to(ROOT)}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
