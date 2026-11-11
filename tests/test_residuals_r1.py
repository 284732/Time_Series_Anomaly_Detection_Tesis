"""
Golden R1 (F4, D-14, D-67): hash dei residui per sensore e dello stato fittato dei 6 modelli del nucleo sul dataset
sintetico, legato a ``FIT_CODE_VERSION``. Fallisce se cambia la numerica del fit o il contratto degli artefatti, e così
ricorda di aggiornare ``FIT_CODE_VERSION``.

Contenuto:
  * ingressi: stessi hash del golden T0 (dati scalati e array passati a ogni modello) più i lotti di F3;
  * residui di ``compute_residuals`` (aggregazioni dichiarate, val = cal di ``legacy2`` e test); per il rolling
    z-score anche il top-k dei residui, da confrontare con ``score_anomalies`` di T0;
  * stato fittato: tensori di ``model.pt``, array di ``state.npz`` e ``checkpoint.json`` senza i campi di tempo
    (``checkpoint_hashable_meta``), serializzato in modo canonico;
  * ``FIT_CODE_VERSION``, ``checkpoint_format`` e ambiente (un cambio di ambiente fa fallire, non saltare).

Rigenerazione (comando esplicito, mai eseguito da pytest; rifiuta di sovrascrivere senza ``--force``):

    python tests/test_residuals_r1.py --write-golden [--force]

Si rigenera quando cambia ``FIT_CODE_VERSION`` (con la motivazione in docs/decisioni.md) o il lock (D-55).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # esecuzione diretta con --write-golden
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "residuals_r1.json"
GOLDEN_T0 = ROOT / "tests" / "golden" / "models_pre_f4.json"
GOLDEN_FORMAT = 1
SEGS = ("val", "test")


def _canonical_sha256(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_hashes(ckpt_dir: Path) -> dict:
    """Hash dello stato fittato di un checkpoint: tensori, array e meta senza tempi."""
    import torch
    from src.models.base import CHECKPOINT_JSON, STATE_NPZ, TORCH_STATE, checkpoint_hashable_meta
    from tests.fixtures.legacy_snapshot import array_summary
    meta = json.loads((ckpt_dir / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    out = {"checkpoint_json": _canonical_sha256(checkpoint_hashable_meta(meta))}
    if (ckpt_dir / STATE_NPZ).exists():
        with np.load(ckpt_dir / STATE_NPZ, allow_pickle=False) as z:
            out["state_npz"] = {k: array_summary(z[k])["sha256"] for k in sorted(z.files)}
    if (ckpt_dir / TORCH_STATE).exists():
        sd = torch.load(ckpt_dir / TORCH_STATE, map_location="cpu", weights_only=True)
        out["model_pt"] = {k: array_summary(v.numpy())["sha256"] for k, v in sorted(sd.items())}
    return out


def build_r1(work: Path) -> dict:
    import torch
    from src.data.manifest import build_manifest, load_dataset_spec
    from src.data.segments import fit_scaler, segment_bounds, slice_segments
    from src.data.windows import score_test_windows, score_windows
    from src.evaluation.scoring import aggregate_scores
    from src.models import registry
    from src.models.base import CHECKPOINT_FORMAT
    from src.models.batching import compute_residuals, legacy_layout_batch
    from src.utils import schema
    from src.utils.schema import PreprocConfig, SplitConfig
    from tests.fixtures.core_models import MACHINE, SEED, fit_like_main, prepare
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.legacy_snapshot import BASE_CONFIG, array_summary, environment
    from tests.fixtures.synth_smd import GENERATOR_VERSION
    from tests.test_models_pre_f4 import build_inputs

    ds = BASE_CONFIG["dataset"]
    w, step = ds["window_size"], ds["step"]
    paths = write_synth_dataset(work / "synth")
    data = prepare(paths["data"])
    man = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    train, test, labels = man.load(MACHINE)
    segs = segment_bounds(len(train), len(test), SplitConfig("legacy2", val_ratio=ds["val_ratio"]), w)
    _, sd = fit_scaler(slice_segments(train, test, labels, segs), PreprocConfig(ds["scaler_type"], 0.0, None))
    ws = {"val": score_windows(sd.cal, w, step, "cal"), "test": score_test_windows(sd, w, step, "primary")}

    inputs = build_inputs(data)
    for seg in SEGS:
        inputs[f"f3|score_{seg}"] = array_summary(legacy_layout_batch(ws[seg], 0, len(ws[seg])))
    residuals, state = {}, {}
    for name in registry.NUCLEUS:
        model = fit_like_main(name, data)
        for seg in SEGS:
            res = compute_residuals(model, ws[seg], model.residual_aggs, len(ws[seg]), "legacy2")
            for agg, a in res.items():
                residuals[f"{name}|{agg}|{seg}"] = array_summary(a)
            if model.native_only:
                top = aggregate_scores(res["last"], None, "topk", k=model.top_k)
                residuals[f"{name}|topk_last|{seg}"] = array_summary(top)
        model.save(work / "ckpt" / name)
        state[name] = state_hashes(work / "ckpt" / name)
    meta = {"golden_format": GOLDEN_FORMAT, "fit_code_version": schema.FIT_CODE_VERSION,
            "checkpoint_format": CHECKPOINT_FORMAT, "environment": environment(),
            "torch_num_threads": torch.get_num_threads(), "generator_version": GENERATOR_VERSION,
            "machine": MACHINE, "seed": SEED, "batch": "un solo lotto per segmento (legacy2)"}
    return json.loads(json.dumps({"meta": meta, "inputs": inputs, "residuals": residuals, "state": state}))


def version_problems(golden: dict) -> list:
    from src.models.base import CHECKPOINT_FORMAT
    from src.utils import schema
    out = []
    if golden["meta"]["fit_code_version"] != schema.FIT_CODE_VERSION:
        out.append(f"FIT_CODE_VERSION {schema.FIT_CODE_VERSION} diversa da quella del golden R1 "
                   f"{golden['meta']['fit_code_version']}: rigenera R1 (--write-golden --force) e motiva il cambio "
                   "in docs/decisioni.md")
    if golden["meta"]["checkpoint_format"] != CHECKPOINT_FORMAT:
        out.append(f"checkpoint_format {CHECKPOINT_FORMAT} diverso da quello del golden R1 "
                   f"{golden['meta']['checkpoint_format']}")
    return out


def _diff(ref, cand, prefix="") -> list:
    if isinstance(ref, dict) and isinstance(cand, dict):
        out = []
        for k in sorted(set(ref) | set(cand)):
            if k not in ref or k not in cand:
                out.append(f"{prefix}{k}: presente solo in {'golden' if k in ref else 'candidato'}")
            else:
                out += _diff(ref[k], cand[k], f"{prefix}{k}.")
        return out
    return [] if ref == cand else [f"{prefix[:-1]}: {ref!r} -> {cand!r}"]


def _load(path: Path) -> dict:
    assert path.exists(), f"golden mancante: {path}. Generalo con --write-golden."
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def r1(tmp_path_factory):
    return build_r1(tmp_path_factory.mktemp("r1"))


@pytest.mark.regression
def test_r1_matches_golden(r1):
    from tests.fixtures.legacy_snapshot import environment
    golden = _load(GOLDEN)
    assert golden["meta"]["environment"] == environment(), "ambiente diverso da quello del golden R1 (D-55)"
    problems = version_problems(golden)
    assert not problems, "\n".join(problems)
    for part in ("inputs", "residuals", "state"):
        d = _diff(golden[part], r1[part])
        assert not d, f"{part.upper()} CAMBIATI ({len(d)}):\n" + "\n".join(d[:40])
    assert r1["meta"] == golden["meta"]


def test_r1_fails_when_fit_code_version_changes_without_regeneration(monkeypatch):
    from src.utils import schema
    golden = _load(GOLDEN)
    assert version_problems(golden) == []
    monkeypatch.setattr(schema, "FIT_CODE_VERSION", schema.FIT_CODE_VERSION + 1)
    problems = version_problems(golden)
    assert len(problems) == 1 and "rigenera R1" in problems[0]


def test_golden_r1_agrees_with_golden_t0():
    """Incrocio statico dei due golden: stessi ingressi; residui = score_features di T0; top-k del rolling z-score
    = score_anomalies di T0."""
    r1g, t0 = _load(GOLDEN), _load(GOLDEN_T0)
    assert {k: r1g["inputs"][k] for k in t0["inputs"]} == t0["inputs"]
    for seg in SEGS:
        assert r1g["inputs"][f"f3|score_{seg}"]["sha256"] == t0["inputs"][f"pca|score_{seg}"]["sha256"]
    checked = 0
    for key, summary in r1g["residuals"].items():
        name, agg, seg = key.split("|")
        if agg == "topk_last":
            ref = t0["arrays"][f"{name}|score_anomalies|-|{seg}"]
        else:
            ref = t0["arrays"].get(f"{name}|score_features|{agg}|{seg}")
            if ref is None:
                assert name == "rolling_zscore", key      # nessun score_features storico
                continue
        assert summary == ref, key
        checked += 1
    assert checked == 5 * 2 * 2 + 2            # 5 modelli con score_features × last/mean × val/test + top-k


def test_fit_seconds_does_not_change_any_hash(r1, tmp_path):
    """Due checkpoint dello stesso modello che differiscono solo in fit_seconds: stessi hash di stato."""
    import dataclasses
    from src.models.base import CHECKPOINT_JSON
    from tests.fixtures.core_models import fit_like_main, prepare
    from tests.fixtures.synth_smd import write_synth_smd
    data = prepare(write_synth_smd(tmp_path / "synth"))
    for name in ("pca", "conv_autoencoder"):
        model = fit_like_main(name, data)
        model.save(tmp_path / f"{name}_a")
        model.fit_info_ = dataclasses.replace(model.fit_info_, fit_seconds=model.fit_info_.fit_seconds + 1000.0)
        model.save(tmp_path / f"{name}_b")
        raw = [(tmp_path / f"{name}_{s}" / CHECKPOINT_JSON).read_text(encoding="utf-8") for s in "ab"]
        assert raw[0] != raw[1]
        ha, hb = state_hashes(tmp_path / f"{name}_a"), state_hashes(tmp_path / f"{name}_b")
        assert ha == hb == r1["state"][name], name


def test_write_golden_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    target = tmp_path / "residuals_r1.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "GOLDEN", target)
    assert _write_golden(force=False) == 2
    assert target.read_text(encoding="utf-8") == "{}"


def _write_golden(force: bool) -> int:
    from tests.fixtures.legacy_snapshot import single_thread
    if GOLDEN.exists() and not force:
        print(f"[ERRORE] {GOLDEN} esiste: si rigenera solo con --force (FIT_CODE_VERSION o lock cambiati).")
        return 2
    with tempfile.TemporaryDirectory() as tmp, single_thread():
        snap = build_r1(Path(tmp))
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] golden R1 scritto: {GOLDEN} ({len(snap['residuals'])} residui, {len(snap['state'])} modelli, "
          f"FIT_CODE_VERSION {snap['meta']['fit_code_version']})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera il golden R1 dei residui e dello stato fittato.")
    parser.add_argument("--write-golden", action="store_true", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(_write_golden(args.force))
