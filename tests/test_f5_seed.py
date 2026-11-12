"""
F5 passo 1: generatori casuali, seed e ``load`` nello stadio di fit (D-70; nota su D-67, C7).

  * all'ingresso di ``fit`` lo stato di torch, numpy globale e ``random`` coincide con quello subito dopo un
    ``set_seed(seed)`` nuovo: lo stadio di fit replica solo la chiamata di ``main.py:330``;
  * ``prepare_machine`` non consuma generatori e non modifica i dati di ingresso;
  * i modelli deterministici danno residui e stato identici con ``set_seed(42)`` e ``set_seed(7)``;
  * solo la PCA riceve e legge ``random_state``;
  * nessun ``load`` durante ``prepare_machine`` + ``fit_one``.
"""

import inspect
import re

import numpy as np
import pytest

from src.data.manifest import build_manifest, load_dataset_spec
from src.models import registry
from src.models.batching import compute_residuals
from src.pipeline import fit as fitmod
from src.utils import schema
from src.utils.schema import model_kwargs
from src.utils.seed import set_seed
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, rng_state, same_rng_state, synth_experiment

MACHINE = "machine-9-1"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_seed")
    paths = write_synth_dataset(root)
    man = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    return man, paths["data"], root


def _cfg(synth, mode="legacy2"):
    return synth_experiment(str(synth[1]), str(synth[2] / "store"), mode=mode)


def _prepare(synth, cfg):
    train, test, labels = synth[0].load(MACHINE)
    return fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("name", CORE)
def test_rng_state_at_fit_entry_equals_fresh_set_seed(synth, monkeypatch, name, seed):
    cfg = _cfg(synth)
    prepared = _prepare(synth, cfg)
    captured = []

    def spy(self, X, y=None, X_val=None):
        captured.append(rng_state())
        return self

    monkeypatch.setattr(registry.get(name), "fit", spy)
    np.random.rand(3)                                   # stato "sporco" prima della chiamata
    fitmod.fit_one(name, model_kwargs(cfg, name, seed), prepared, seed)
    set_seed(seed)
    fresh = rng_state()
    assert len(captured) == 1
    diff = same_rng_state(captured[0], fresh)
    assert diff is None, f"{name}, seed {seed}: generatore {diff} consumato tra set_seed e fit"


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
def test_prepare_machine_consumes_no_rng(synth, mode):
    cfg = _cfg(synth, mode)
    train, test, labels = synth[0].load(MACHINE)
    set_seed(123)
    before = rng_state()
    fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)
    diff = same_rng_state(before, rng_state())
    assert diff is None, f"prepare_machine consuma il generatore {diff}"


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
def test_prepare_machine_does_not_modify_inputs(synth, mode):
    cfg = _cfg(synth, mode)
    train, test, labels = synth[0].load(MACHINE)
    copies = [a.copy() for a in (train, test, labels)]
    p = fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)
    for a, c in zip((train, test, labels), copies):
        assert a.dtype == c.dtype and a.shape == c.shape and a.tobytes() == c.tobytes()
    for seg in ("fit", "es", "cal", "test", "labels"):
        assert not getattr(p.data, seg).flags.writeable, seg
    assert not p.train_scaled.flags.writeable
    for k, ws in p.score.items():
        assert not ws.view.flags.writeable, k


def _deterministic():
    return [n for n in CORE if registry.get(n).deterministic]


def test_deterministic_models_are_the_expected_ones():
    assert _deterministic() == ["pca", "pca_pointwise", "rolling_zscore", "input_norm"]


@pytest.mark.parametrize("name", _deterministic())
def test_deterministic_models_do_not_depend_on_set_seed(synth, tmp_path, name):
    """Residui (lotto unico) e stato identici con set_seed(42) e set_seed(7): giustifica il seed assente dal
    fit_key. Il fit gira davvero (nessuna spia).

    Oggi è banalmente vero: le due PCA usano il solver ``full`` perché ``n_components`` è un float, e rolling z-score
    e ``input_norm`` non apprendono nulla, quindi nessun modello deterministico consuma un generatore. Resta come
    protezione per D-57: con ``n_components`` intero (solver randomizzato) il test diventerebbe significativo."""
    from tests.test_residuals_r1 import state_hashes
    cfg = _cfg(synth)
    prepared = _prepare(synth, cfg)
    out = {}
    for seed in (42, 7):
        model = fitmod.fit_one(name, model_kwargs(cfg, name, 42), prepared, seed)
        res = {}
        for seg in ("cal", "test"):
            ws = prepared.score[seg]
            for agg, a in compute_residuals(model, ws, model.residual_aggs, len(ws), "legacy2").items():
                res[(seg, agg)] = a
        model.save(tmp_path / f"{name}_{seed}")
        out[seed] = (res, state_hashes(tmp_path / f"{name}_{seed}"))
    (r42, s42), (r7, s7) = out[42], out[7]
    assert set(r42) == set(r7)
    for k in r42:
        assert r42[k].dtype == r7[k].dtype and r42[k].tobytes() == r7[k].tobytes(), (name, k)
    assert s42 == s7, name


def test_only_pca_receives_random_state_from_schema(synth):
    cfg = _cfg(synth)
    for name in CORE:
        kw = model_kwargs(cfg, name, 42)
        assert ("random_state" in kw) == (name == "pca"), name


def test_only_pca_reads_random_state_in_nucleus_source():
    """Ricerca nel sorgente dei modelli del nucleo (più base e trainer): solo PCADetector legge random_state."""
    import src.models.base as base
    import src.training.trainer as trainer
    readers = set()
    pattern = re.compile(r"""random_state["']""")
    modules = {schema.MODEL_SCHEMAS[n].class_path.split(":")[0] for n in CORE}
    for modname in sorted(modules) + [base.__name__, trainer.__name__]:
        src = inspect.getsource(__import__(modname, fromlist=["_"]))
        if pattern.search(src):
            readers.add(modname)
    assert readers == {"src.models.traditional.pca_detector"}, readers


def test_no_load_during_prepare_and_fit(synth, monkeypatch):
    calls = []

    def make(cls):
        def spy(klass, *a, **k):
            calls.append(klass.__name__)
            raise AssertionError(f"load di {klass.__name__} durante lo stadio di fit")
        return classmethod(spy)

    for name in CORE:
        monkeypatch.setattr(registry.get(name), "load", make(registry.get(name)))
    monkeypatch.setattr(registry, "load_checkpoint",
                        lambda *a, **k: calls.append("load_checkpoint") or (_ for _ in ()).throw(AssertionError()))
    cfg = _cfg(synth)
    prepared = _prepare(synth, cfg)
    for name in CORE:
        fitmod.fit_one(name, model_kwargs(cfg, name, 42), prepared, 42)   # fit veri (modelli piccolissimi)
    assert calls == []
