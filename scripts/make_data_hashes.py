"""
Riferimento degli hash dei file del dataset (D-61): scrive ``hashes_file`` dello spec.

Legge i file di ``--data-dir`` solo in modalità binaria (sha256 dei byte dopo CRLF -> LF, D-59): non li modifica.
Non lancia training, non legge risultati.

Uso (dalla radice del progetto; per SMD lo lancia l'utente, passo 7 di F3):
    python scripts/make_data_hashes.py            # scrive configs/datasets/smd_sha256.yaml
    python scripts/make_data_hashes.py --check    # verifica che il file esistente coincida, senza scrivere
Opzioni: --spec <spec.yaml> (default configs/datasets/smd.yaml), --data-dir <cartella> (default dataset),
--out <file> (default: hashes_file dello spec), --force (sovrascrive un file esistente diverso).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import (  # noqa: E402
    HASH_RULE, MANIFEST_FORMAT, compute_file_hashes, load_dataset_spec, load_reference_hashes,
)

DEFAULT_SPEC = ROOT / "configs" / "datasets" / "smd.yaml"
DEFAULT_DATA_DIR = ROOT / "dataset"


def render(dataset: str, files: Dict[str, str]) -> str:
    header = ("# Generato da scripts/make_data_hashes.py (D-61). Non modificare a mano:\n"
              "# rigenerare con `python scripts/make_data_hashes.py` (con --force se il contenuto cambia).\n")
    body = {"format": MANIFEST_FORMAT, "dataset": dataset, "rule": HASH_RULE,
            "files": {k: files[k] for k in sorted(files)}}
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    spec = load_dataset_spec(args.spec)
    out = args.out or spec.hashes_file
    if out is None:
        print("lo spec non prevede hashes_file e manca --out", file=sys.stderr)
        return 2
    files = compute_file_hashes(spec, args.data_dir)
    text = render(spec.name, files)

    if args.check:
        if not out.exists():
            print(f"[CHECK] {out} non esiste", file=sys.stderr)
            return 1
        ref = load_reference_hashes(out, spec.name)
        diff = sorted(k for k in set(ref) | set(files) if ref.get(k) != files.get(k))
        if diff:
            print(f"[CHECK] {len(diff)} file diversi dal riferimento, es. {diff[:5]}", file=sys.stderr)
            return 1
        print(f"[CHECK] {out}: {len(files)} file, tutti coincidenti")
        return 0

    if out.exists() and out.read_text(encoding="utf-8") != text and not args.force:
        print(f"{out} esiste con un contenuto diverso: usa --check per vedere le differenze, --force per "
              "sovrascrivere (annota il motivo in docs/decisioni.md)", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"scritto {out}: {len(files)} file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
