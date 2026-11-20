"""
F8 passo 3: AE non addestrati (D-69, nota del 2026-11-20; D-73, nota del 2026-11-20).

``conv_autoencoder_untrained`` e ``lstm_autoencoder_untrained`` sono l'AE addestrato con **zero epoche**: stessa rete,
inizializzazione standard di PyTorch, BatchNorm con le statistiche iniziali (γ = 1, β = 0, media 0, varianza 1).
Chiavi di schema: quelle dell'AE addestrato più ``score_batch_size`` (lotto di scoring, blocco interno di D-64), che
entra nel ``fit_key`` come blocco ``scoring``; seed nel ``fit_key``; training fuori. Validatore di appaiamento quando
l'AE e il suo ``*_untrained`` sono entrambi in ``run_models``. In ``plan.py`` un ``*_untrained`` compare solo in
``descriptive_baselines`` (e nelle righe descrittive di RQ2 che ne derivano).

Dati sintetici in memoria o in ``tmp_path`` (D-16): nessuna lettura di ``dataset/``, nessuna scrittura in ``store/``.
"""

import copy

import numpy as np
import pytest
import torch

from src.models import registry
from src.models.deep.conv_ae import ConvAutoencoderDetector
from src.models.deep.lstm_ae import LSTMAutoencoderDetector
from src.training import trainer as trainer_mod
from src.utils import config as cfgmod
from src.utils import schema
from src.utils.schema import ConfigError, model_kwargs, validate_experiment
from src.utils.seed import set_seed
from tests.fixtures.config_factory import raw_config

PAIRS = {"conv_autoencoder_untrained": "conv_autoencoder", "lstm_autoencoder_untrained": "lstm_autoencoder"}
SMALL = {"conv_autoencoder": {"channels": [4, 8], "latent_dim": 3},
         "lstm_autoencoder": {"hidden_dim": 6, "latent_dim": 3}}
TRAINED_CLS = {"conv_autoencoder": ConvAutoencoderDetector, "lstm_autoencoder": LSTMAutoencoderDetector}


def _raw(untrained=("conv_autoencoder_untrained", "lstm_autoencoder_untrained"), with_trained=True, sbs=None):
    """Config grezza con gli AE non addestrati appaiati (params = quelli dell'AE, lotto = training.batch_size)."""
    raw = raw_config()
    b = raw["training"]["batch_size"] if sbs is None else sbs
    for u in untrained:
        raw["models"][u] = dict(copy.deepcopy(raw["models"][PAIRS[u]]), score_batch_size=b)
    raw["run_models"] = ([PAIRS[u] for u in untrained] if with_trained else []) + list(untrained)
    return raw


def _key(raw, model, seed=42):
    return cfgmod.fit_key(validate_experiment(raw), "machine-9-1", model, seed, "dv1-x")


def _windows(seed=0, n=40, w=16, f=5):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, w, f)).astype(np.float32)


def _kwargs(name):
    """kwargs del modello addestrato piccolo e del suo non addestrato (come li costruisce model_kwargs)."""
    trained = dict(SMALL[name], epochs=1, early_stopping_patience=1, early_stopping_min_delta=1e-6,
                   learning_rate=1e-3, weight_decay=1e-5, batch_size=8, device="cpu")
    untrained = dict(SMALL[name], batch_size=8, epochs=0, device="cpu")
    return trained, untrained


# ---------------------------------------------------------------- schema
def test_schema_declares_the_untrained_models_with_the_trained_keys_plus_the_scoring_batch():
    for u, t in PAIRS.items():
        ms = schema.MODEL_SCHEMAS[u]
        assert ms.untrained_of == t and ms.trained is False
        assert set(ms.params) == set(schema.MODEL_SCHEMAS[t].params) | {"score_batch_size"}
        assert issubclass(schema.model_class(u), TRAINED_CLS[t])
        assert schema.model_class(u).deterministic is False
        assert schema.model_class(u).residual_aggs == TRAINED_CLS[t].residual_aggs
    assert {n for n, s in schema.MODEL_SCHEMAS.items() if s.untrained_of} == set(PAIRS)
    assert set(PAIRS) <= set(registry.NUCLEUS) and set(PAIRS) <= set(registry.NEW_PIPELINE_ONLY)


@pytest.mark.parametrize("missing", ["score_batch_size", "latent_dim"])
def test_no_default_for_untrained_keys(missing):
    raw = _raw()
    del raw["models"]["conv_autoencoder_untrained"][missing]
    with pytest.raises(ConfigError, match=missing):
        validate_experiment(raw)


# ---------------------------------------------------------------- fit_key
@pytest.mark.parametrize("u", sorted(PAIRS))
def test_fit_key_has_seed_and_scoring_batch_but_not_training(u):
    cfg = validate_experiment(_raw())
    p = cfgmod.fit_payload(cfg, "machine-9-1", u, 42, "dv1-x")
    assert p["seed"] == 42 and p["scoring"] == {"batch_size": 64} and "training" not in p
    # params completi: config_id si ricalcola da qui in analysis_grid.fits_row con la formula di config.config_id
    assert p["model"]["params"] == dict(cfg.models[u].params)
    assert cfgmod.canonical_hash({"model": u, "params": p["model"]["params"]})[:12] == cfgmod.config_id(cfg.models[u])
    assert schema.untrained_architecture(cfg.models[u]) == dict(cfg.models[PAIRS[u]].params)
    assert _key(_raw(), u, seed=42) != _key(_raw(), u, seed=43)
    assert _key(_raw(), u) != _key(_raw(), PAIRS[u])


@pytest.mark.parametrize("u", sorted(PAIRS))
@pytest.mark.parametrize("field,value", [("learning_rate", 5e-4), ("max_epochs", 100), ("patience", 10),
                                         ("weight_decay", 0.0), ("min_delta", 1e-3)])
def test_changing_only_training_settings_keeps_the_untrained_fit_key(u, field, value):
    """Cambia solo il blocco training (batch_size escluso: è appaiato al lotto di scoring) → stesso fit_key."""
    raw = _raw(with_trained=False)
    changed = copy.deepcopy(raw)
    changed["training"][field] = value
    assert _key(changed, u) == _key(raw, u)


@pytest.mark.parametrize("u", sorted(PAIRS))
def test_changing_the_scoring_batch_changes_the_fit_key(u):
    assert _key(_raw(with_trained=False, sbs=32), u) != _key(_raw(with_trained=False, sbs=64), u)


def test_model_kwargs_of_untrained_models_are_zero_epochs_and_the_scoring_batch():
    cfg = validate_experiment(_raw())
    for u, t in PAIRS.items():
        kw = model_kwargs(cfg, u, 42)
        expected = dict(cfg.models[t].params, batch_size=64, epochs=0, device="cpu")
        if "channels" in expected:
            expected["channels"] = list(expected["channels"])
        assert kw == expected


# ---------------------------------------------------------------- validatore di appaiamento
def test_paired_untrained_models_validate():
    cfg = validate_experiment(_raw())
    assert set(PAIRS) <= set(cfg.run_models)


@pytest.mark.parametrize("u", sorted(PAIRS))
def test_unpaired_params_are_an_error(u):
    raw = _raw()
    raw["models"][u]["latent_dim"] = 7
    with pytest.raises(ConfigError, match="params"):
        validate_experiment(raw)


@pytest.mark.parametrize("u", sorted(PAIRS))
def test_scoring_batch_different_from_training_batch_is_an_error(u):
    raw = _raw(sbs=32)
    with pytest.raises(ConfigError, match="score_batch_size"):
        validate_experiment(raw)


def test_pairing_is_checked_only_when_both_are_in_run_models():
    raw = _raw(with_trained=False, sbs=32)
    raw["models"]["conv_autoencoder_untrained"]["latent_dim"] = 7
    validate_experiment(raw)                       # l'AE addestrato non è in run_models: nessun appaiamento


# ---------------------------------------------------------------- pesi iniziali e zero epoche
@pytest.mark.parametrize("t", sorted(SMALL))
def test_untrained_weights_equal_the_initial_weights_of_the_trained_ae(t, monkeypatch):
    """Controllo di parità dedicato (D-69): stesso seed → pesi del non addestrato = pesi dell'AE addestrato
    all'ingresso di PyTorchTrainer.fit, prima della prima epoca (buffer di BatchNorm compresi)."""
    u = [k for k, v in PAIRS.items() if v == t][0]
    x = _windows()
    captured = {}
    original = trainer_mod.PyTorchTrainer.fit

    def spy(self, train_loader, val_loader=None):
        captured.setdefault("state", {k: v.detach().clone() for k, v in self.model.state_dict().items()})
        return original(self, train_loader, val_loader)
    monkeypatch.setattr(trainer_mod.PyTorchTrainer, "fit", spy)
    kw_t, kw_u = _kwargs(t)
    set_seed(42)
    TRAINED_CLS[t](name=t, config=kw_t).fit(x, X_val=x[:8])
    initial = captured.pop("state")
    set_seed(42)
    model = schema.model_class(u)(name=u, config=kw_u).fit(x, X_val=x[:8])
    got = model.model.state_dict()
    assert list(got) == list(initial)
    for k in initial:
        assert torch.equal(got[k], initial[k]), k
    set_seed(43)                                   # seed diverso → pesi diversi (il seed conta)
    other = schema.model_class(u)(name=u, config=kw_u).fit(x).model.state_dict()
    assert any(not torch.equal(other[k], got[k]) for k in got if got[k].is_floating_point())


def test_conv_untrained_batchnorm_has_the_initial_statistics():
    _, kw_u = _kwargs("conv_autoencoder")
    set_seed(42)
    model = schema.model_class("conv_autoencoder_untrained")(config=kw_u).fit(_windows())
    bns = [m for m in model.model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert bns
    for bn in bns:
        assert torch.equal(bn.weight, torch.ones_like(bn.weight)) and torch.equal(bn.bias, torch.zeros_like(bn.bias))
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))
        assert int(bn.num_batches_tracked) == 0


@pytest.mark.parametrize("u", sorted(PAIRS))
def test_zero_epoch_path(u, monkeypatch):
    """Storia vuota, best_epoch None, stopped_epoch 0, nessuna valutazione su es, nessun ripristino, rete in eval."""
    calls = {"evaluate": 0, "load_state_dict": 0, "train_epoch": 0}
    for name in ("evaluate", "train_epoch"):
        orig = getattr(trainer_mod.PyTorchTrainer, name)

        def counted(self, *a, _orig=orig, _name=name, **k):
            calls[_name] += 1
            return _orig(self, *a, **k)
        monkeypatch.setattr(trainer_mod.PyTorchTrainer, name, counted)
    orig_load = torch.nn.Module.load_state_dict

    def counted_load(self, *a, **k):
        calls["load_state_dict"] += 1
        return orig_load(self, *a, **k)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", counted_load)
    _, kw_u = _kwargs(PAIRS[u])
    x = _windows()
    set_seed(42)
    model = schema.model_class(u)(config=dict(kw_u, early_stopping_patience=3)).fit(x, X_val=x[:8])
    assert calls == {"evaluate": 0, "load_state_dict": 0, "train_epoch": 0}
    assert model.history == {"train_loss": [], "val_loss": [], "best_epoch": None, "stopped_epoch": 0}
    fi = model.fit_info_
    assert fi.train_loss == () and fi.val_loss == () and fi.best_epoch is None and fi.stopped_epoch == 0
    assert fi.has_nonfinite_loss() is False
    assert model.model.training is False
    res = model.residuals(x, model.residual_aggs)
    assert set(res) == {"last", "mean"} and all(a.shape == (len(x), x.shape[2]) for a in res.values())


@pytest.mark.parametrize("u", sorted(PAIRS))
@pytest.mark.parametrize("key", ["epochs", "max_epochs"])
def test_nonzero_epochs_are_refused(u, key):
    _, kw_u = _kwargs(PAIRS[u])
    with pytest.raises(ValueError, match="zero epoche"):
        schema.model_class(u)(config=dict(kw_u, **{key: 3}))


@pytest.mark.parametrize("u", sorted(PAIRS))
def test_checkpoint_roundtrip_through_the_registry(u, tmp_path):
    _, kw_u = _kwargs(PAIRS[u])
    x = _windows()
    set_seed(42)
    model = schema.model_class(u)(name=u, config=kw_u).fit(x)
    model.save(tmp_path / "ckpt")
    loaded = registry.load_checkpoint(tmp_path / "ckpt")
    assert type(loaded) is schema.model_class(u)
    a, b = model.residuals(x, model.residual_aggs), loaded.residuals(x, loaded.residual_aggs)
    for agg in model.residual_aggs:
        assert a[agg].dtype == b[agg].dtype and np.array_equal(a[agg], b[agg]), agg
    assert loaded.fit_info_.stopped_epoch == 0 and loaded.fit_info_.train_loss == ()


# ---------------------------------------------------------------- stadio di fit e di analisi
def test_fit_and_analysis_with_untrained_models(tmp_path, repo_store_guard):
    """Integrazione su dati sintetici: fit, meta.json e righe dell'analisi per gli AE non addestrati."""
    import json
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from src.pipeline.store import Store
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    paths = write_synth_dataset(tmp_path / "synth")
    repo = make_repo(tmp_path / "repo")
    spec = synth_spec(str(paths["data"]), str(tmp_path / "store"), mode="split3", step=1, seeds=[42],
                      run_models=("conv_autoencoder", "conv_autoencoder_untrained", "lstm_autoencoder_untrained"))
    b = spec["training"]["batch_size"]
    for u in ("conv_autoencoder_untrained", "lstm_autoencoder_untrained"):
        spec["models"][u] = dict(spec["models"][PAIRS[u]], score_batch_size=b)
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo)
    store = Store(tmp_path / "store")
    cfg = validate_experiment(copy.deepcopy(spec))
    plan = {"scenarios": ["primary"], "window_time_aggs": ["last", "mean"],
            "scorings": [{"sensor_agg": "mean", "normalizer": "std"}], "ts_aggs": ["last"],
            "thresholds": {"static_pct": [99.0], "roll_quantile": [], "roll_robust_z": [], "oracle": False},
            "pa": []}
    rep = an.run_analysis(spec, plan, datasets_dir=paths["spec"].parent, repo_root=repo)
    free = store.read_table(rep.analysis_dir, "free")
    fits = store.read_table(rep.analysis_dir, "fits")
    for u in ("conv_autoencoder_untrained", "lstm_autoencoder_untrained"):
        assert (free["model"] == u).any(), u
        row = fits[fits["model"] == u].iloc[0]
        assert row["status"] == "complete" and int(row["stopped_epoch"]) == 0
        assert str(row["config_id"]) == cfgmod.config_id(cfg.models[u])
    for key in fits["fit_key"]:
        meta = json.loads((store.root / "fits" / key / "meta.json").read_text(encoding="utf-8"))
        if meta["model"]["name"] in PAIRS:
            assert meta["model"]["kwargs"]["epochs"] == 0 and "learning_rate" not in meta["model"]["kwargs"]
            assert meta["fit_payload"]["scoring"] == {"batch_size": b}
            assert meta["fit_info"]["train_loss"] == [] and meta["fit_info"]["stopped_epoch"] == 0
    repo_store_guard.check()


# ---------------------------------------------------------------- piano statistico (D-73, nota del 2026-11-20)
CID_U = "5123456789ab"


def _plan_with_untrained():
    from tests.test_f7_plan import _plan
    p = _plan()
    p["descriptive_baselines"] = [{"model": "conv_autoencoder_untrained", "config_id": CID_U}]
    p["rq2"]["baselines"] = [{"model": "conv_autoencoder_untrained", "config_id": CID_U, "window_time_agg": "last",
                              "sensor_agg": "top5", "normalizer": "std"}]
    return p


def _untr_scored():
    return {"model": "conv_autoencoder_untrained", "config_id": "6123456789ab", "window_time_agg": "last",
            "sensor_agg": "mean", "normalizer": "std"}


def test_untrained_models_are_allowed_as_descriptive_baselines():
    from src.stats import plan as sp
    sp.validate_stats_plan(_plan_with_untrained())


def test_native_models_are_still_allowed_as_descriptive_baselines():
    from src.stats import plan as sp
    p = _plan_with_untrained()
    p["descriptive_baselines"].append({"model": "rolling_zscore", "config_id": "7123456789ab"})
    sp.validate_stats_plan(p)


def _mutations():
    """Un *_untrained con un config_id NON dichiarato baseline: il rifiuto vale per nome, non per coppia."""
    def family(p):
        p["rq2"]["family"].append(_untr_scored())

    def reference(p):
        p["rq2"]["reference"] = _untr_scored()

    def rq1a_models(p):
        p["rq1a"]["blocks"][1]["models"].append(_untr_scored())

    def rq1a_delta(p):
        p["rq1a"]["blocks"][1]["models"].append(_untr_scored())
        p["rq1a"]["blocks"][1]["delta_reference"] = _untr_scored()

    def rq1b_models(p):
        p["rq1b"]["blocks"][0]["models"].append({"model": "conv_autoencoder_untrained", "config_id": "6123456789ab"})

    def rq4_models(p):
        p["rq4"]["models"].append(_untr_scored())
    return [family, reference, rq1a_models, rq1a_delta, rq1b_models, rq4_models]


@pytest.mark.parametrize("mut", _mutations(), ids=lambda f: f.__name__)
def test_untrained_models_are_refused_outside_descriptive_baselines(mut):
    from src.stats import plan as sp
    p = _plan_with_untrained()
    mut(p)
    with pytest.raises(ConfigError, match="solo in descriptive_baselines"):
        sp.validate_stats_plan(p)
