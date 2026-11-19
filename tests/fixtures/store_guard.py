"""Guardia di ``store/`` nella radice del repository (D-16, nota del 2026-11-19).

Dopo M1 ``store/`` esiste e contiene i run dell'utente: il controllo non è più «``store/`` assente» ma «``store/`` non
modificato dai test». L'impronta (percorsi relativi, tipo, dimensione e ``mtime_ns`` dei file) si prende a inizio
sessione e si confronta alla fine dei test che la richiedono; se ``store/`` mancava, deve mancare ancora.
La guardia legge soltanto: non crea, non sposta e non scrive nulla.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

Entry = Tuple[str, Optional[int], Optional[int]]


def fingerprint(root: Path) -> Optional[Dict[str, Entry]]:
    """None se ``root`` non esiste; altrimenti {percorso relativo: (tipo, dimensione, mtime_ns)}.

    Le cartelle compaiono con tipo «d» (una cartella vuota creata o tolta cambia l'impronta); i file con dimensione e
    ``mtime_ns``, così anche una riscrittura con lo stesso contenuto viene rilevata.
    """
    root = Path(root)
    if not root.exists():
        return None
    out: Dict[str, Entry] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for d in dirnames:
            out[(base / d).relative_to(root).as_posix()] = ("d", None, None)
        for f in sorted(filenames):
            p = base / f
            st = p.stat()
            out[p.relative_to(root).as_posix()] = ("f", st.st_size, st.st_mtime_ns)
    return out


def describe(before: Optional[Dict[str, Entry]], after: Optional[Dict[str, Entry]]) -> str:
    """Differenze leggibili tra due impronte (vuota se uguali)."""
    if before == after:
        return ""
    if before is None:
        return "store/ creato (mancava a inizio sessione)"
    if after is None:
        return "store/ rimosso (esisteva a inizio sessione)"
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    parts = []
    for name, items in (("aggiunti", added), ("rimossi", removed), ("modificati", changed)):
        if items:
            parts.append(f"{name} {len(items)}: {items[:5]}")
    return "; ".join(parts)


class StoreGuard:
    """Impronta di ``root`` presa alla costruzione; ``check()`` fallisce se ``root`` è cambiata."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.before = fingerprint(self.root)

    def check(self) -> None:
        diff = describe(self.before, fingerprint(self.root))
        assert not diff, f"un test ha modificato {self.root}: {diff}"
