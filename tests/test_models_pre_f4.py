"""
Golden pre-F4 (T0): output di ``score_anomalies`` e ``score_features(last|mean)`` dei modelli del nucleo,
generato su d2d8478 prima di qualsiasi modifica a ``src/`` (F4, vincolo "solo aggiunte").

Lo snapshot F1 non basta: ``main.py`` usa ``score_features`` per PCA, PCA puntuale, AE e ``input_norm``
e non chiama mai il loro ``score_anomalies``. Questo golden fissa entrambi, a tolleranza zero (D-17),
sul dataset sintetico (``machine-9-1``, seed 42, un thread di torch e di BLAS).

Il golden è legato all'ambiente (lock D-55): se cambia, il test fallisce (non viene saltato).

Rigenerazione (comando esplicito, mai eseguito da pytest):

    python tests/test_models_pre_f4.py --write-golden

Rifiuta di sovrascrivere un golden esistente, salvo ``--force``: dopo F4 il golden serve proprio a
dimostrare che le aggiunte non hanno cambiato i metodi storici, quindi si rigenera solo quando cambia
il lock dell'ambiente (prima di M1, D-55), annotando la differenza in docs/decisioni.md.
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

from tests.fixtures.core_models import (  # noqa: E402
    CORE_MODELS, MACHINE, SEED, fit_like_main, model_config, prepare, score_outputs,
)
from tests.fixtures.legacy_snapshot import (  # noqa: E402
    array_summary, compare_arrays, environment, single_thread,
)
from tests.fixtures.synth_smd import GENERATOR_VERSION, write_synth_smd  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "models_pre_f4.json"
GOLDEN_FORMAT = 1


def _key(model: str, method: str, agg: str, seg: str) -> str:
    return f"{model}|{method}|{agg}|{seg}"


# Ingressi registrati: dati scalati e finestre passate a ciascun modello (fit, early stopping, scoring).
DATA_KEYS = ("train_scaled", "val_scaled", "test_scaled")


def model_inputs(model_key: str, data: dict) -> dict:
    """Array passati al modello: stessa regola di ``fit_like_main`` e di ``score_outputs``."""
    import main
    if getattr(main.MODEL_REGISTRY[model_key], "fit_on_points", False):
        fit_x, fit_x_val = data["train_scaled"], None
    else:
        fit_x, fit_x_val = data["train_windows"], data["val_for_early_stop"]
    out = {"fit_X": fit_x, "score_val": data["val_windows"], "score_test": data["test_windows"]}
    if fit_x_val is not None:
        out["fit_X_val"] = fit_x_val
    return out


def build_inputs(data: dict) -> dict:
    inputs = {f"data|{k}": array_summary(data[k]) for k in DATA_KEYS}
    for model_key in CORE_MODELS:
        for role, a in model_inputs(model_key, data).items():
            inputs[f"{model_key}|{role}"] = array_summary(a)
    return inputs


def _diff(ref: dict, cand: dict) -> list:
    if sorted(ref) != sorted(cand):
        return [f"chiavi diverse: mancanti {sorted(set(ref) - set(cand))}, nuove {sorted(set(cand) - set(ref))}"]
    out = []
    for k, r in ref.items():
        d = compare_arrays(r, cand[k])
        if not d.equal:
            out.append(f"{k}: {'; '.join(d.reasons)}")
    return out


def build_golden(data_dir: Path) -> dict:
    import torch
    data = prepare(data_dir)
    inputs = build_inputs(data)
    arrays = {}
    for model_key in CORE_MODELS:
        model = fit_like_main(model_key, data)
        for (method, agg, seg), a in score_outputs(model, data).items():
            arrays[_key(model_key, method, agg, seg)] = array_summary(a)
    meta = {
        "golden_format": GOLDEN_FORMAT,
        "environment": environment(),
        "torch_num_threads": torch.get_num_threads(),
        "generator_version": GENERATOR_VERSION,
        "machine": MACHINE, "seed": SEED,
        "models": {m: model_config(m) for m in CORE_MODELS},
    }
    return json.loads(json.dumps({"meta": meta, "inputs": inputs, "arrays": arrays}, allow_nan=True))


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _load_golden() -> dict:
    assert GOLDEN.exists(), f"golden mancante: {GOLDEN}. Generalo con --write-golden."
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["meta"]["environment"] == environment(), (
        "ambiente diverso da quello del golden: se il lock è cambiato, rigenera con --write-golden --force "
        "e annota la differenza in docs/decisioni.md.")
    return golden


def test_inputs_match_pre_f4_golden(synth_smd_dir):
    """Ingressi (dati e finestre per modello): nessun fit, solo preparazione dei dati."""
    golden = _load_golden()
    problems = _diff(golden["inputs"], json.loads(json.dumps(build_inputs(prepare(synth_smd_dir)))))
    assert not problems, (f"INGRESSI CAMBIATI ({len(problems)}): dati sintetici o preparazione delle finestre "
                          "diversi dal golden; le uscite non sono confrontabili.\n" + "\n".join(problems))


@pytest.mark.regression
def test_historical_scoring_methods_match_pre_f4_golden(synth_smd_dir):
    golden = _load_golden()
    cand = build_golden(synth_smd_dir)
    assert cand["meta"]["torch_num_threads"] == golden["meta"]["torch_num_threads"] == 1
    in_problems = _diff(golden["inputs"], cand["inputs"])
    assert not in_problems, (f"INGRESSI CAMBIATI ({len(in_problems)}): le uscite non sono confrontabili "
                             "(vedi test_inputs_match_pre_f4_golden).\n" + "\n".join(in_problems))
    problems = _diff(golden["arrays"], cand["arrays"])
    assert not problems, (f"USCITE CAMBIATE con ingressi identici ({len(problems)}): i metodi di scoring "
                          "storici non sono più invariati.\n" + "\n".join(problems))


def _write_golden(force: bool) -> int:
    if GOLDEN.exists() and not force:
        print(f"[ERRORE] {GOLDEN} esiste: si rigenera solo con --force (cambio del lock, D-55).")
        return 2
    with tempfile.TemporaryDirectory() as tmp, single_thread():
        snap = build_golden(write_synth_smd(Path(tmp) / "synth_smd"))
    snap["meta"]["generated_at_commit"] = _git_head()
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(f"[OK] golden scritto: {GOLDEN} ({len(snap['arrays'])} array, commit {snap['meta']['generated_at_commit']})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera il golden pre-F4 dei metodi di scoring storici.")
    parser.add_argument("--write-golden", action="store_true", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(_write_golden(args.force))
