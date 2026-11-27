"""
Genera gli esperimenti di tuning di F9 (D-41, passo 2): ``configs/experiments/tuning/slot_<k>.yaml``, k = 1..K, e
quelli del pilota sui tempi (passo 3): ``configs/experiments/tuning_pilot/slot_<k>.yaml``, k = 2..K, da
``configs/tuning/grid.yaml``, ``configs/base.yaml`` e ``configs/dev_machines.yaml`` (``src/tuning/grid.py``).

    python scripts/make_tuning_experiments.py            # scrive i file che mancano o differiscono
    python scripts/make_tuning_experiments.py --check    # nessuna scrittura; codice 1 se un file manca o differisce

I file generati si committano. Il confronto ignora la differenza CRLF / LF (lettura in modo testo, nessun hash sui
byte), quindi non dipende da ``core.autocrlf``. Un ``slot_<n>.yaml`` non previsto in una delle due cartelle è un
errore, e allora non si scrive nulla: va tolto a mano, mai in silenzio. Gli altri file delle cartelle (esperimenti del
passo 5) non si toccano. Nessuna lettura di dati, label o risultati.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tuning import grid as tg  # noqa: E402

_SLOT_RE = re.compile(r"slot_(\d+)\.yaml")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verifica senza scrivere (codice 1 se diversi)")
    ap.add_argument("--grid", default=str(tg.DEFAULT_GRID))
    ap.add_argument("--base", default=str(tg.DEFAULT_BASE))
    ap.add_argument("--dev", default=str(tg.DEFAULT_DEV))
    ap.add_argument("--out", default=str(tg.DEFAULT_OUT))
    ap.add_argument("--pilot-out", default=str(tg.DEFAULT_PILOT_OUT))
    args = ap.parse_args(argv)

    plan = [(tg.generate(args.grid, args.base, args.dev), Path(args.out)),
            (tg.generate_pilot(args.grid, args.base, args.dev), Path(args.pilot_out))]
    for files, out in plan:                             # nessuna scrittura se una cartella ha slot in più
        if _extra(files, out):
            print(f"errore: file di slot non previsti dalla griglia in {out}: {_extra(files, out)}", file=sys.stderr)
            return 1
    differ = [(files, out, n) for files, out in plan for n in _differ(files, out)]
    if args.check:
        for _, out, n in differ:
            print(f"diverso o mancante: {out / n}")
        print("ok: file generati uguali alla griglia" if not differ else f"{len(differ)} file da rigenerare")
        return 1 if differ else 0
    for files, out, n in differ:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / n, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(files[n])
        print(f"scritto: {out / n}")
    total = sum(len(f) for f, _ in plan)
    print(f"{total} file, {len(differ)} scritti, {total - len(differ)} invariati")
    return 0


def _extra(files, out: Path):
    present = {p.name for p in out.glob("slot_*.yaml")} if out.is_dir() else set()
    return sorted(n for n in present if _SLOT_RE.fullmatch(n) and n not in files)


def _differ(files, out: Path):
    return [n for n, text in files.items()
            if not (out / n).is_file() or (out / n).read_text(encoding="utf-8") != text]


if __name__ == "__main__":
    sys.exit(main())
