"""
F5 passo 2: ``residual_cache.compute_all`` e ``array_record`` (D-70; D-04, D-33, D-64, D-66).

Parità: golden R1 anticipato e ``score_features`` storico sulle finestre di ``main.py``. Invarianti: unica via
``compute_residuals``, regola dei lotti, tipo nativo, ``test_ctx_head`` separato, es solo in split3, generatori
invariati, formato ``np.savez`` senza pickle. Dati sintetici in ``tmp_path`` (D-16).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.manifest import build_manifest, load_dataset_spec
from src.models import batching, registry
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.utils.schema import model_kwargs
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, rng_state, same_rng_state, synth_experiment
from tests.fixtures.legacy_snapshot import array_summary

ROOT = Path(__file__).resolve().parents[1]
MACHINE = "machine-9-1"
SEED = 42
D66 = {"pca": "<f8", "pca_pointwise": "<f8", "input_norm": "<f8",
       "conv_autoencoder": "<f4", "lstm_autoencoder": "<f4", "rolling_zscore": "<f4"}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_rc")
    paths = write_synth_dataset(root)
    man = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    return man, paths["data"], root


def _cfg(synth, mode="legacy2", step=None):
    return synth_experiment(str(synth[1]), str(synth[2] / "store"), mode=mode, step=step)


def _prepare(synth, cfg):
    train, test, labels = synth[0].load(MACHINE)
    return fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)


@pytest.fixture(scope="module")
def legacy(synth):
    """Fit dei 6 modelli in legacy2 (seed 42) e residui di compute_all."""
    cfg = _cfg(synth)
    prepared = _prepare(synth, cfg)
    models = {n: fitmod.fit_one(n, model_kwargs(cfg, n, SEED), prepared, SEED) for n in CORE}
    res = {n: rc.compute_all(m, prepared, "legacy2") for n, m in models.items()}
    return cfg, prepared, models, res


@pytest.fixture(scope="module")
def split3(synth):
    """split3 con step = 1 (test_ctx_head) e step = 2; modelli economici più un AE."""
    out = {}
    for step in (1, 2):
        cfg = _cfg(synth, "split3", step)
        prepared = _prepare(synth, cfg)
        models = {n: fitmod.fit_one(n, model_kwargs(cfg, n, SEED), prepared, SEED)
                  for n in ("pca", "input_norm", "rolling_zscore", "conv_autoencoder")}
        out[step] = (cfg, prepared, models)
    return out


# ------------------------------------------------------------------ 1. golden R1 anticipato
def test_residuals_match_golden_r1(legacy):
    golden = json.loads((ROOT / "tests" / "golden" / "residuals_r1.json").read_text(encoding="utf-8"))["residuals"]
    _, _, _, res = legacy
    checked = 0
    for name in CORE:
        for rname, a in res[name].arrays.items():
            seg, agg = rname.split(rc.SEP)
            key = f"{name}|{agg}|{'val' if seg == 'cal' else seg}"
            assert array_summary(a) == golden[key], key
            checked += 1
    assert checked == 5 * 2 * 2 + 2                         # 5 modelli × last/mean × cal/test + rolling z-score


# ------------------------------------------------------------------ 2. parità con la pipeline storica
def test_residuals_equal_historical_score_features(legacy):
    from tests.fixtures.core_models import fit_like_main, prepare
    cfg, _, _, res = legacy
    data = prepare(Path(cfg.data.data_dir), MACHINE)
    for name in CORE:
        hist = fit_like_main(name, data, seed=SEED)
        for seg, key in (("cal", "val_windows"), ("test", "test_windows")):
            for agg in rc.residual_aggs_for(hist):
                new = res[name].arrays[rc.residual_name(seg, agg)]
                if hist.native_only:          # rolling z-score: nessun score_features storico, top-k = score_anomalies
                    from src.evaluation.scoring import aggregate_scores
                    ref = hist.score_anomalies(data[key])
                    new = aggregate_scores(new, None, "topk", k=hist.top_k)
                else:
                    ref = hist.score_features(data[key], time_agg=agg)
                assert ref.dtype == new.dtype and ref.shape == new.shape, (name, seg, agg)
                assert ref.tobytes() == new.tobytes(), (name, seg, agg)


# ------------------------------------------------------------------ 3. unica via
def test_only_compute_residuals_is_used(legacy, split3, monkeypatch):
    import src.data.windows as windows

    def boom(*a, **k):
        raise AssertionError("iter_batches usato per lo scoring dei residui")

    monkeypatch.setattr(windows, "iter_batches", boom)
    calls = []
    real = batching.compute_residuals

    def spy(model, ws, aggs, batch_size, mode):
        calls.append((ws.segment, ws.context, tuple(aggs)))
        return real(model, ws, aggs, batch_size, mode)

    monkeypatch.setattr(rc, "compute_residuals", spy)
    _, prepared, models, _ = legacy
    rc.compute_all(models["pca"], prepared, "legacy2")
    assert calls == [("cal", 0, ("last", "mean")), ("test", 0, ("last", "mean"))]
    calls.clear()
    cfg, prepared3, models3 = split3[1]
    rc.compute_all(models3["pca"], prepared3, "split3")
    w = cfg.windows.window_size
    assert calls == [("es", 0, ("last", "mean")), ("cal", 0, ("last", "mean")), ("test", 0, ("last", "mean")),
                     ("test", w - 1, ("last", "mean"))]       # test_ctx_head: chiamata separata


# ------------------------------------------------------------------ 4. lotti
@pytest.mark.parametrize("name", ["conv_autoencoder", "lstm_autoencoder"])
def test_multiblock_batch_equals_single_batch(legacy, name):
    _, prepared, models, res = legacy
    block = models[name].legacy_score_block["last"]
    n = len(prepared.score["test"])
    assert block < n                                        # il lotto da 2 blocchi divide davvero il segmento
    multi = rc.compute_all(models[name], prepared, "legacy2", score_batch=2 * block)
    for rname, rec in res[name].records.items():
        assert multi.records[rname]["sha256"] == rec["sha256"], rname


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
@pytest.mark.parametrize("name,batch", [("conv_autoencoder", 50), ("input_norm", 10), ("pca", 100)])
def test_incompatible_batch_is_an_error(legacy, split3, mode, name, batch):
    if mode == "legacy2":
        _, prepared, models, _ = legacy
    else:
        _, prepared, models = split3[1]
    if name not in models:
        pytest.skip("modello non fittato in questo scenario")
    with pytest.raises(ValueError, match="incompatibile"):
        rc.compute_all(models[name], prepared, mode, score_batch=batch)


# ------------------------------------------------------------------ 5. dtype (D-66) e rolling z-score
def test_native_dtypes_and_rolling_zscore_last_only(legacy):
    _, _, _, res = legacy
    for name in CORE:
        for rname, a in res[name].arrays.items():
            assert a.dtype.str == D66[name] == res[name].records[rname]["dtype"], (name, rname)
    assert set(res["rolling_zscore"].arrays) == {"cal__last", "test__last"}
    assert set(res["pca"].arrays) == {"cal__last", "cal__mean", "test__last", "test__mean"}


# ------------------------------------------------------------------ 6. test_ctx_head
def test_test_ctx_head_only_in_split3_step1(legacy, split3):
    _, _, _, res = legacy
    assert not any(k.startswith(fitmod.TEST_CTX_HEAD) for r in res.values() for k in r.arrays)
    cfg, prepared, models = split3[1]
    w = cfg.windows.window_size
    for name, model in models.items():
        out = rc.compute_all(model, prepared, "split3")
        heads = [k for k in out.arrays if k.startswith(fitmod.TEST_CTX_HEAD + rc.SEP)]
        assert heads, name
        for k in heads:
            rec = out.records[k]
            assert out.arrays[k].shape[0] == w - 1 == rec["n_windows"] == rec["context"]
            starts = np.arange(rec["n_windows"]) * rec["step"] - rec["context"]
            assert np.array_equal(starts, np.arange(-(w - 1), 0))
        # i residui del test non dipendono dal calcolo di test_ctx_head
        no_head = dataclasses.replace(prepared, score={k: v for k, v in prepared.score.items()
                                                       if k != fitmod.TEST_CTX_HEAD})
        out2 = rc.compute_all(model, no_head, "split3")
        for k in out2.arrays:
            assert out2.records[k]["sha256"] == out.records[k]["sha256"], (name, k)
    _, prepared2, models2 = split3[2]
    out = rc.compute_all(models2["pca"], prepared2, "split3")
    assert not any(k.startswith(fitmod.TEST_CTX_HEAD) for k in out.arrays)


# ------------------------------------------------------------------ 7. es solo in split3, diagnostica
def test_es_only_in_split3_and_diagnostic(legacy, split3):
    _, _, _, res = legacy
    assert not any(k.startswith("es" + rc.SEP) for r in res.values() for k in r.arrays)
    assert not any(rec["diagnostic_only"] for r in res.values() for rec in r.records.values())
    _, prepared, models = split3[1]
    out = rc.compute_all(models["pca"], prepared, "split3")
    for k, rec in out.records.items():
        assert rec["diagnostic_only"] == k.startswith("es" + rc.SEP), k
    assert {"es__last", "es__mean"} <= set(out.arrays)


# ------------------------------------------------------------------ 8. generatori
@pytest.mark.parametrize("name", CORE)
def test_compute_all_leaves_rng_untouched(legacy, name):
    _, prepared, models, _ = legacy
    np.random.rand(2)
    before = rng_state()
    rc.compute_all(models[name], prepared, "legacy2")
    diff = same_rng_state(before, rng_state())
    assert diff is None, f"{name}: compute_all consuma il generatore {diff}"


# ------------------------------------------------------------------ 9. formato
def test_npz_roundtrip_without_pickle(legacy, tmp_path):
    _, _, _, res = legacy
    for name in CORE:
        path = tmp_path / f"{name}.npz"
        path.write_bytes(rc.npz_bytes(res[name].arrays))
        back = rc.read_npz(path)
        assert set(back) == set(res[name].arrays)
        for k, a in back.items():
            rec = rc.array_record(a)
            assert {f: rec[f] for f in ("shape", "dtype", "sha256")} == \
                   {f: res[name].records[k][f] for f in ("shape", "dtype", "sha256")}, (name, k)


# ------------------------------------------------------------------ 10. array_record
def test_array_record_hash_depends_on_dtype_and_shape():
    a = np.arange(12, dtype=np.float32)
    h = rc.array_record(a)["sha256"]
    assert rc.array_record(a.reshape(3, 4))["sha256"] != h
    assert rc.array_record(a.view(np.int32))["sha256"] != h
    assert rc.array_record(a.astype(">f4"))["sha256"] != h           # stessi valori, altro ordine dei byte
    assert rc.array_record(a.copy())["sha256"] == h
    assert rc.array_record(a.reshape(3, 4).T)["sha256"] == rc.array_record(np.ascontiguousarray(a.reshape(3, 4).T))["sha256"]


def test_array_record_counts_nonfinite():
    a = np.zeros((4, 3))
    a[0, 0], a[1, 2], a[3, 1] = np.nan, np.inf, -np.inf
    rec = rc.array_record(a)
    assert rec["n_nonfinite"] == 3 and rec["shape"] == [4, 3] and rec["dtype"] == "<f8"
    assert rc.array_record(np.ones(5))["n_nonfinite"] == 0
