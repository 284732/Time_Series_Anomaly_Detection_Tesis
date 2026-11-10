"""
Test di caratterizzazione della pipeline storica (F1, D-14, D-16, D-17).

Esegue ``main.run_experiment`` sul dataset sintetico (2 macchine, 6 sensori) con due config
(``window_time_agg`` last / mean, seed [42, 43]) e confronta il risultato con lo snapshot
``tests/golden/legacy_synth.json``, a tolleranza zero. Tutti gli output stanno in ``tmp_path``.

Lo snapshot è legato all'ambiente (lock D-55) e a main.py (sha256 del contenuto): se cambiano le
versioni delle librerie, il backend BLAS o main.py, il test fallisce (non viene saltato).

Procedura di rigenerazione (comando esplicito, mai eseguito da pytest):

    python tests/test_legacy_characterization.py --write-golden

- se lo snapshot esiste, prima confronta vecchio e nuovo e stampa le differenze
  (file/chiave, scarto delle statistiche);
- rifiuta di scrivere se lo sha256 di main.py è cambiato, salvo ``--force-main-changed``
  (main.py è congelato, D-13: un cambiamento va prima motivato e approvato);
- si rigenera quando cambia il lock dell'ambiente (atteso prima di M1, D-55); la differenza
  riportata dal comparatore va annotata in docs/decisioni.md.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # esecuzione diretta con --write-golden
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tests.fixtures.legacy_snapshot import (  # noqa: E402
    build_snapshot, compare_snapshots, dump_snapshot, environment, expected_rows,
    load_snapshot, main_py_sha256, single_thread,
)
from tests.fixtures.synth_smd import write_synth_smd  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "legacy_synth.json"

# Cartelle o file che main.py creerebbe con i default: non devono comparire.
DEFAULT_OUTPUTS = ["results/scores", "results/metrics", "results/plots", "store", "configs/config.yaml"]


def _git_status() -> list:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if "__pycache__" not in line)


def _normalize(snap: dict) -> dict:
    """Stessi tipi dello snapshot caricato da disco (liste, float, NaN)."""
    return json.loads(json.dumps(snap, allow_nan=True))


@pytest.mark.regression
def test_legacy_pipeline_matches_snapshot(synth_smd_dir, tmp_path):
    assert GOLDEN.exists(), f"snapshot mancante: {GOLDEN}. Generalo con --write-golden."
    golden = load_snapshot(GOLDEN)

    # Ambiente e main.py: un cambio fa fallire con un messaggio esplicito
    assert golden["meta"]["environment"] == environment(), (
        "ambiente diverso da quello dello snapshot: "
        f"{golden['meta']['environment']} != {environment()}. Se il lock è cambiato, rigenera lo "
        "snapshot (--write-golden) e annota la differenza in docs/decisioni.md.")
    assert golden["meta"]["main_py_sha256"] == main_py_sha256(), (
        "main.py è cambiato rispetto allo snapshot (congelato, D-13).")

    status_before = _git_status()
    defaults_before = {p: (ROOT / p).exists() for p in DEFAULT_OUTPUTS}

    snap = _normalize(build_snapshot(synth_smd_dir, tmp_path))

    assert _git_status() == status_before, "il run ha modificato file del repository"
    assert {p: (ROOT / p).exists() for p in DEFAULT_OUTPUTS} == defaults_before, \
        "il run ha creato cartelle di default nel repository"
    assert snap["meta"]["rows"] == expected_rows(), "righe del summary diverse da quelle attese"

    problems = compare_snapshots(golden, snap)
    assert not problems, f"{len(problems)} differenze rispetto allo snapshot:\n" + "\n".join(problems[:50])


def _write_golden(force_main_changed: bool) -> int:
    old = load_snapshot(GOLDEN) if GOLDEN.exists() else None
    if old is not None and old["meta"]["main_py_sha256"] != main_py_sha256() and not force_main_changed:
        print("[ERRORE] lo sha256 di main.py è diverso da quello dello snapshot esistente. "
              "Usa --force-main-changed solo dopo aver approvato e registrato la modifica.")
        return 2
    with tempfile.TemporaryDirectory() as tmp, single_thread():
        tmp = Path(tmp)
        data_dir = write_synth_smd(tmp / "synth_smd")
        snap = _normalize(build_snapshot(data_dir, tmp / "work"))
    if snap["meta"]["rows"] != expected_rows():
        print("[ERRORE] righe del summary diverse da quelle attese: snapshot non scritto.")
        return 3
    if old is not None:
        problems = compare_snapshots(old, snap)
        print(f"\n[CONFRONTO] {len(problems)} differenze tra lo snapshot esistente e il nuovo")
        for p in problems:
            print("  " + p)
    dump_snapshot(snap, GOLDEN)
    print(f"[OK] snapshot scritto: {GOLDEN} ({GOLDEN.stat().st_size} byte)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigenera lo snapshot della pipeline storica.")
    parser.add_argument("--write-golden", action="store_true", required=True)
    parser.add_argument("--force-main-changed", action="store_true")
    args = parser.parse_args()
    sys.exit(_write_golden(args.force_main_changed))
