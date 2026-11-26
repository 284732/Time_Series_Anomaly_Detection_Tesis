"""
F8 passo 4: modello di previsione ``lstm_forecaster`` e ``ForecastTrainer`` (D-74, nota del 2026-11-20; D-71).

Forma (D-74 a): dai primi W−1 punti della finestra si prevede l'ultimo; un residuo per finestra (vettore per sensore),
quindi ``residual_aggs = ("last",)`` (b) e solo ``ts_agg = last`` (c). Rete: un solo strato LSTM sui primi W−1 punti più
una testa lineare verso gli F sensori. Chiavi: solo ``hidden_dim`` (stesso nome e significato dell'LSTM-AE), nessun
valore predefinito (i valori di questo file sono provvisori, solo per i test sintetici: D-41 aperta). Blocco
``training`` condiviso con gli AE. ``ForecastTrainer`` è una sottoclasse di ``PyTorchTrainer``, che resta intatto.

Analisi: righe offline (``ts_agg`` ``mean``/``max``) del modello assenti, non NaN, contate in ``analysis.json``
(``absent_rows``); stesso trattamento per ``window_time_agg`` ``mean``, degenerato (D-74 b). L'esportazione legacy
rifiuta il modello (ammessi solo i modelli di ``registry.LEGACY_NUCLEUS``: anche gli AE non addestrati sono
rifiutati). Le regole di ``src/stats/plan.py`` sono il passo 5 e qui non si provano.

Dati sintetici in memoria o in ``tmp_path`` (D-16): nessuna lettura di ``dataset/``, nessuna scrittura in ``store/``.
"""

import copy
import json

import numpy as np
import pytest
import torch

from src.models import registry
from src.training.trainer import PyTorchTrainer
from src.utils import config as cfgmod
from src.utils import schema
from src.utils.schema import ConfigError, model_kwargs, validate_experiment
from src.utils.seed import set_seed
from tests.fixtures.config_factory import raw_config

NAME = "lstm_forecaster"
HIDDEN = 6                                   # provvisorio, solo test sintetici (D-41 aperta)
W, F = 12, 5


def _kwargs(**over):
    """kwargs come li costruisce model_kwargs per un modello addestrato (valori piccoli, provvisori)."""
    kw = dict(hidden_dim=HIDDEN, epochs=2, early_stopping_patience=1, early_stopping_min_delta=1e-6,
              learning_rate=1e-3, weight_decay=1e-5, batch_size=8, device="cpu")
    kw.update(over)
    return kw


def _windows(seed=0, n=40, w=W, f=F):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, w, f)).astype(np.float32)


def _cls():
    return schema.model_class(NAME)


def _fitted(seed=42, x=None, x_val=None, **over):
    set_seed(seed)
    return _cls()(name=NAME, config=_kwargs(**over)).fit(_windows() if x is None else x, X_val=x_val)


def _raw(**params):
    raw = raw_config()
    raw["models"][NAME] = dict({"hidden_dim": HIDDEN}, **params)
    raw["run_models"] = ["lstm_autoencoder", NAME]
    return raw


def _key(raw, seed=42):
    return cfgmod.fit_key(validate_experiment(raw), "machine-9-1", NAME, seed, "dv1-x")


# ---------------------------------------------------------------- schema e registro
def test_schema_declares_only_hidden_dim_and_the_training_block():
    ms = schema.MODEL_SCHEMAS[NAME]
    assert ms.class_path == "src.models.deep.lstm_forecaster:LSTMForecasterDetector"
    assert ms.trained is True and ms.untrained_of is None
    assert set(ms.params) == {"hidden_dim"}
    assert set(schema.MODEL_SCHEMAS["lstm_autoencoder"].params) >= set(ms.params)    # stesso nome dell'LSTM-AE


def test_registry_lists_the_forecaster_as_new_pipeline_only():
    import main
    assert NAME in registry.NUCLEUS and NAME in registry.NEW_PIPELINE_ONLY
    assert NAME not in registry.LEGACY_NUCLEUS and NAME not in main.MODEL_REGISTRY
    assert registry.get(NAME) is _cls()


def test_no_default_for_hidden_dim():
    raw = _raw()
    del raw["models"][NAME]["hidden_dim"]
    with pytest.raises(ConfigError, match="hidden_dim"):
        validate_experiment(raw)


@pytest.mark.parametrize("extra", [{"num_layers": 2}, {"latent_dim": 4}, {"horizon": 1}])
def test_keys_other_than_hidden_dim_are_refused(extra):
    """Un solo strato e orizzonte 1 fissi; nessuna chiave dell'LSTM-AE oltre hidden_dim (D-74, revisione passo 0)."""
    with pytest.raises(ConfigError, match="chiavi sconosciute"):
        validate_experiment(_raw(**extra))


def test_model_kwargs_are_hidden_dim_plus_the_shared_training_block():
    cfg = validate_experiment(_raw())
    t = cfg.training
    assert model_kwargs(cfg, NAME, 42) == {
        "hidden_dim": HIDDEN, "epochs": t.max_epochs, "early_stopping_patience": t.patience,
        "early_stopping_min_delta": t.min_delta, "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay, "batch_size": t.batch_size, "device": "cpu"}


def test_fit_key_contains_training_and_seed():
    raw = _raw()
    p = cfgmod.fit_payload(validate_experiment(raw), "machine-9-1", NAME, 42, "dv1-x")
    assert p["model"] == {"name": NAME, "params": {"hidden_dim": HIDDEN}}
    assert "training" in p and p["seed"] == 42 and "scoring" not in p
    assert _key(raw, 42) != _key(raw, 43)
    lr = copy.deepcopy(raw)
    lr["training"]["learning_rate"] = 5e-4
    assert _key(lr) != _key(raw)
    hd = _raw(hidden_dim=HIDDEN + 1)
    assert _key(hd) != _key(raw)


def test_detector_has_no_default_for_its_keys():
    """Nessun valore predefinito nel codice del modello: hidden_dim, batch_size e training obbligatori."""
    x = _windows()
    for missing in ("hidden_dim", "batch_size", "learning_rate", "weight_decay", "epochs",
                    "early_stopping_patience", "early_stopping_min_delta"):
        kw = _kwargs()
        del kw[missing]
        with pytest.raises((KeyError, ValueError), match=missing):
            _cls()(name=NAME, config=kw).fit(x)


def test_max_epochs_is_accepted_as_the_alias_of_epochs():
    kw = _kwargs()
    kw["max_epochs"] = kw.pop("epochs")
    set_seed(1)
    model = _cls()(name=NAME, config=kw).fit(_windows())
    assert model.fit_info_.stopped_epoch == 2


# ---------------------------------------------------------------- architettura
def test_one_lstm_layer_on_w_minus_1_points_and_a_linear_head():
    model = _fitted()
    net = model.model
    lstms = [m for m in net.modules() if isinstance(m, torch.nn.LSTM)]
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(lstms) == 1 and lstms[0].num_layers == 1
    assert lstms[0].input_size == F and lstms[0].hidden_size == HIDDEN and lstms[0].batch_first
    assert len(linears) == 1 and (linears[0].in_features, linears[0].out_features) == (HIDDEN, F)
    assert model.arch_ == {"window_size": W, "num_features": F, "hidden_dim": HIDDEN}


def test_windows_need_at_least_two_points():
    with pytest.raises(ValueError, match="W"):
        _fitted(x=_windows(w=1))


# ---------------------------------------------------------------- leakage (obbligatorio, D-74)
def test_last_point_does_not_change_the_prediction_and_the_first_w_minus_1_do():
    """Verso 1: l'ultimo punto della finestra (il bersaglio) non entra nella previsione.
    Verso 2: ogni punto tra i primi W−1 la cambia (il contesto si usa davvero)."""
    model = _fitted()
    x = _windows(seed=3, n=6)
    base = model.predict(x)
    assert base.shape == (6, F)
    last = x.copy()
    last[:, -1, :] += 10.0
    assert np.array_equal(model.predict(last), base)
    for t in (0, W // 2, W - 2):
        y = x.copy()
        y[:, t, :] += 1.0
        assert not np.allclose(model.predict(y), base), t


def test_last_point_changes_only_the_target_of_the_residual():
    model = _fitted()
    x = _windows(seed=4, n=6)
    r = model.residuals(x, ["last"])["last"]
    assert np.allclose(r, (model.predict(x) - x[:, -1, :]) ** 2, rtol=1e-6, atol=0)
    y = x.copy()
    y[:, -1, :] += 1.0
    assert np.array_equal(model.predict(y), model.predict(x))
    assert not np.allclose(model.residuals(y, ["last"])["last"], r)


def test_training_and_validation_feed_only_the_first_w_minus_1_points(monkeypatch):
    """La rete riceve sempre (B, W−1, F), in train_epoch, evaluate e scoring; la loss confronta con l'ultimo punto."""
    shapes = []
    cls = _cls()
    orig = cls.net_class.forward

    def spy(self, ctx):
        shapes.append(tuple(ctx.shape[1:]))
        return orig(self, ctx)

    monkeypatch.setattr(cls.net_class, "forward", spy)
    model = _fitted(x_val=_windows(seed=9, n=16))
    model.residuals(_windows(seed=5, n=4), ["last"])
    assert shapes and set(shapes) == {(W - 1, F)}


def test_validation_loss_is_the_forecast_mse_of_the_last_point():
    model = _fitted()
    x = _windows(seed=6, n=16)
    loader = torch.utils.data.DataLoader(torch.from_numpy(x), batch_size=8, shuffle=False)
    got = model.trainer.evaluate(loader)
    with torch.no_grad():
        pred = model.model(torch.from_numpy(x[:, :-1, :]))
        exp = torch.nn.functional.mse_loss(pred, torch.from_numpy(x[:, -1, :])).item()
    assert got == pytest.approx(exp, rel=1e-6)


# ---------------------------------------------------------------- residui
def test_residual_aggs_are_last_only():
    assert _cls().residual_aggs == ("last",)
    assert _cls().allowed_ts_aggs == ("last",)
    assert _cls().native_only is False and _cls().deterministic is False
    model = _fitted()
    with pytest.raises(ValueError, match="D-33"):
        model.residuals(_windows(n=4), ["mean"])
    with pytest.raises(ValueError, match="mean"):
        model.score_features(_windows(n=4), time_agg="mean")


def test_residual_shapes_and_dtype():
    model = _fitted()
    x = _windows(seed=7, n=13)
    r = model.residuals(x, ["last"])
    assert set(r) == {"last"} and r["last"].shape == (13, F) and r["last"].dtype == np.float32
    s = model.score_anomalies(x)
    assert s.shape == (13,) and np.allclose(s, r["last"].mean(axis=1), rtol=1e-6)


def test_residuals_are_independent_of_the_scoring_batch_rule():
    """Regola di D-64: lotto esterno multiplo del blocco interno (batch_size) → residui identici bit a bit."""
    from src.data.windows import score_windows
    from src.models.batching import compute_residuals
    model = _fitted()
    assert model.legacy_score_block == {"last": 8}
    series = np.random.default_rng(8).normal(size=(60, F)).astype(np.float32)
    ws = score_windows(series, W, 1, segment="cal")
    one = compute_residuals(model, ws, ["last"], len(ws), "split3")["last"]
    many = compute_residuals(model, ws, ["last"], 16, "split3")["last"]
    assert np.array_equal(one, many)
    with pytest.raises(ValueError, match="D-64"):
        compute_residuals(model, ws, ["last"], 5, "split3")


def test_residuals_do_not_touch_the_global_torch_generator():
    model = _fitted()
    state = torch.get_rng_state()
    model.residuals(_windows(n=10), ["last"])
    assert torch.equal(torch.get_rng_state(), state)


def test_score_window_size_must_match_the_fitted_one():
    model = _fitted()
    with pytest.raises(ValueError, match="window_size"):
        model.residuals(_windows(n=4, w=W + 1), ["last"])


# ---------------------------------------------------------------- determinismo
def test_same_seed_gives_identical_weights_and_residuals():
    x, xv = _windows(), _windows(seed=11, n=16)
    a, b = _fitted(seed=42, x=x, x_val=xv), _fitted(seed=42, x=x, x_val=xv)
    for (ka, va), (kb, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka
    xs = _windows(seed=12, n=9)
    assert np.array_equal(a.residuals(xs, ["last"])["last"], b.residuals(xs, ["last"])["last"])
    assert a.fit_info_.train_loss == b.fit_info_.train_loss and a.fit_info_.val_loss == b.fit_info_.val_loss
    c = _fitted(seed=43, x=x, x_val=xv)
    assert not np.array_equal(a.residuals(xs, ["last"])["last"], c.residuals(xs, ["last"])["last"])


def test_early_stopping_restores_the_best_epoch():
    """Il ciclo di fit è quello ereditato da PyTorchTrainer (early stopping su es, ripristino dell'epoca migliore)."""
    model = _fitted(x_val=_windows(seed=13, n=16), epochs=6, early_stopping_patience=2)
    fi = model.fit_info_
    assert len(fi.train_loss) == fi.stopped_epoch == len(fi.val_loss)
    assert fi.best_epoch is not None and 1 <= fi.best_epoch <= fi.stopped_epoch
    assert fi.val_loss[fi.best_epoch - 1] == min(fi.val_loss)


def test_checkpoint_roundtrip_through_the_registry(tmp_path):
    from src.training.forecast_trainer import ForecastTrainer
    model = _fitted(x_val=_windows(seed=14, n=16))
    model.save(tmp_path / "ckpt")
    loaded = registry.load_checkpoint(tmp_path / "ckpt")
    assert type(loaded) is _cls() and type(loaded.trainer) is ForecastTrainer
    assert loaded.arch_ == model.arch_
    x = _windows(seed=15, n=11)
    a, b = model.residuals(x, ["last"])["last"], loaded.residuals(x, ["last"])["last"]
    assert a.dtype == b.dtype and np.array_equal(a, b)
    assert loaded.fit_info_.hashable_dict() == model.fit_info_.hashable_dict()


# ---------------------------------------------------------------- ForecastTrainer
def test_forecast_trainer_is_a_subclass_and_inherits_the_fit_loop():
    from src.training.forecast_trainer import ForecastTrainer
    assert issubclass(ForecastTrainer, PyTorchTrainer)
    assert ForecastTrainer.fit is PyTorchTrainer.fit
    for name in ("train_epoch", "evaluate", "get_error_matrix", "get_reconstruction_errors"):
        assert getattr(ForecastTrainer, name) is not getattr(PyTorchTrainer, name), name


def test_forecast_trainer_refuses_missing_training_keys():
    from src.training.forecast_trainer import ForecastTrainer
    net = _cls().net_class(num_features=F, hidden_dim=HIDDEN)
    cfg = {k: v for k, v in _kwargs().items() if k not in ("hidden_dim", "batch_size")}
    ForecastTrainer(net, cfg, device="cpu")
    for k in ("learning_rate", "weight_decay", "epochs", "early_stopping_patience", "early_stopping_min_delta"):
        bad = {kk: v for kk, v in cfg.items() if kk != k}
        with pytest.raises(ValueError, match=k):
            ForecastTrainer(net, bad, device="cpu")


# ---------------------------------------------------------------- stadio di fit e di analisi
def _plan(**over):
    p = {"scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
         "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"}],
         "ts_aggs": ["last", "mean", "max"],
         "thresholds": {"static_pct": [99.0], "roll_quantile": [{"level": 99.0, "n_past": 50}],
                        "roll_robust_z": [], "oracle": True},
         "pa": ["static_pct", "oracle"]}
    p.update(over)
    return p


def _env(tmp_path, *, mode="split3", run_models=("conv_autoencoder", NAME), seeds=(42,)):
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    paths = write_synth_dataset(tmp_path / "synth")
    repo = make_repo(tmp_path / "repo")
    spec = synth_spec(str(paths["data"]), str(tmp_path / "store"), mode=mode, step=1, seeds=list(seeds),
                      run_models=run_models)
    spec["models"][NAME] = {"hidden_dim": 4}                    # provvisorio, solo test sintetici
    return paths, repo, spec


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    """Un fit split3 (Conv-AE e previsione, un seed) e un'analisi con livelli offline e window_time_agg mean."""
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from src.pipeline.store import Store
    tmp = tmp_path_factory.mktemp("f8_forecaster")
    paths, repo, spec = _env(tmp)
    fitmod.run_fit(copy.deepcopy(spec), datasets_dir=paths["spec"].parent, repo_root=repo, allow_eval_machines=True)
    rep = an.run_analysis(copy.deepcopy(spec), _plan(), datasets_dir=paths["spec"].parent, repo_root=repo,
                          allow_eval_machines=True)
    store = Store(tmp / "store")
    return {"store": store, "rep": rep, "spec": spec, "paths": paths, "repo": repo,
            "free": store.read_table(rep.analysis_dir, "free"), "thr": store.read_table(rep.analysis_dir, "thr"),
            "fits": store.read_table(rep.analysis_dir, "fits"),
            "aj": json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))}


def test_fit_stage_saves_only_last_residuals(analysed, repo_store_guard):
    store, fits = analysed["store"], analysed["fits"]
    rows = fits[fits["model"] == NAME]
    assert len(rows) == 2 and set(rows["status"]) == {"complete"}          # due macchine, un seed
    for key in rows["fit_key"]:
        meta = json.loads((store.root / "fits" / key / "meta.json").read_text(encoding="utf-8"))
        assert meta["model"]["residual_aggs"] == ["last"] and meta["model"]["native_only"] is False
        assert set(meta["residuals"]) == {"es__last", "cal__last", "test__last", "test_ctx_head__last"}
        assert meta["fit_payload"]["model"]["params"] == {"hidden_dim": 4}
        assert "training" in meta["fit_payload"] and meta["fit_payload"]["seed"] == 42
        assert meta["model"]["kwargs"]["hidden_dim"] == 4 and meta["fit_info"]["stopped_epoch"] >= 1
    repo_store_guard.check()


def test_forecaster_rows_are_only_last_last(analysed):
    for t in ("free", "thr"):
        df = analysed[t]
        mine = df[df["model"] == NAME]
        assert len(mine) > 0, t
        assert set(mine["window_time_agg"]) == {"last"} and set(mine["ts_agg"]) == {"last"}, t
        assert mine["nan_reason"].isna().all(), t
        ae = df[df["model"] == "conv_autoencoder"]
        assert set(ae["window_time_agg"]) == {"last", "mean"} and set(ae["ts_agg"]) == {"last", "mean", "max"}, t
        # stesse righe dell'AE per window_time_agg = ts_agg = last (stesse combinazioni di scenario, scoring, soglie)
        cols = ["machine", "scenario", "sensor_agg", "normalizer", "metric"] + \
            (["threshold_policy", "threshold_level", "n_past", "pa"] if t == "thr" else ["diagnostic"])
        ae_ll = ae[(ae["window_time_agg"] == "last") & (ae["ts_agg"] == "last")]
        key = lambda d: sorted(map(tuple, d[cols].astype(str).values.tolist()))  # noqa: E731
        assert key(mine) == key(ae_ll), t


def test_absent_rows_are_counted_in_analysis_json(analysed):
    """Le righe assenti del modello di previsione sono quelle che l'AE ha in più, per motivo (stesso piano, stessi
    fit per macchina e seed)."""
    aj = analysed["aj"]
    assert set(aj["absent_rows"]) == {NAME}
    got = aj["absent_rows"][NAME]
    assert set(got) == {"offline_ts_agg", "window_time_agg_degenerate"}
    for t in ("free", "thr"):
        ae = analysed[t][analysed[t]["model"] == "conv_autoencoder"]
        offline = int((ae["ts_agg"] != "last").sum())
        degenerate = int(((ae["ts_agg"] == "last") & (ae["window_time_agg"] == "mean")).sum())
        assert got["offline_ts_agg"][t] == offline > 0, t
        assert got["window_time_agg_degenerate"][t] == degenerate > 0, t
    assert aj["analysis_code_version"] == 2 and aj["components"]["analysis_code_version"] == 2


def test_analysis_identity_reads_only_the_last_residuals(analysed):
    aj = analysed["aj"]
    store = analysed["store"]
    for fit in aj["components"]["fits"]:
        meta = json.loads((store.root / "fits" / fit["fit_key"] / "meta.json").read_text(encoding="utf-8"))
        names = set(fit["residual_sha256"])
        if meta["model"]["name"] == NAME:
            assert names == {"cal__last", "test__last", "test_ctx_head__last"}
        else:
            assert {"cal__mean", "test__mean"} <= names


def test_no_absent_rows_without_the_forecaster(tmp_path, repo_store_guard):
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    paths, repo, spec = _env(tmp_path, run_models=("pca", "rolling_zscore"))
    fitmod.run_fit(copy.deepcopy(spec), datasets_dir=paths["spec"].parent, repo_root=repo, allow_eval_machines=True)
    rep = an.run_analysis(copy.deepcopy(spec), _plan(), datasets_dir=paths["spec"].parent, repo_root=repo,
                          allow_eval_machines=True)
    aj = json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    assert aj["absent_rows"] == {}
    repo_store_guard.check()


@pytest.mark.parametrize("over,what", [({"window_time_aggs": ["mean"]}, "window_time_agg"),
                                       ({"ts_aggs": ["mean", "max"],
                                         "thresholds": {"static_pct": [99.0], "roll_quantile": [],
                                                        "roll_robust_z": [], "oracle": False},
                                         "pa": []}, "ts_agg")])
def test_plan_without_last_is_refused_before_any_write(analysed, over, what):
    """Senza il livello last il modello di previsione non avrebbe righe: errore nei controlli preliminari."""
    from src.pipeline import analyze as an
    runs_before = sorted(p.name for p in (analysed["store"].root / "runs").rglob("*"))
    with pytest.raises(ConfigError, match=what):
        an.run_analysis(copy.deepcopy(analysed["spec"]), _plan(**over),
                        datasets_dir=analysed["paths"]["spec"].parent, repo_root=analysed["repo"],
                        allow_eval_machines=True)
    assert sorted(p.name for p in (analysed["store"].root / "runs").rglob("*")) == runs_before


UNTRAINED_PAIRS = {"conv_autoencoder_untrained": "conv_autoencoder", "lstm_autoencoder_untrained": "lstm_autoencoder"}
LEGACY_PLAN = {"scenarios": ["primary"],
               "legacy_export": {"threshold_percentile": 99.0, "timestamp_agg": "last", "window_time_agg": "last"}}


@pytest.mark.parametrize("model", [NAME] + sorted(UNTRAINED_PAIRS))
def test_legacy_export_refuses_models_outside_the_legacy_nucleus(tmp_path, repo_store_guard, model):
    """legacy2 con export legacy: elenco di ammessi = registry.LEGACY_NUCLEUS; il modello di previsione (D-74) e gli AE
    non addestrati (D-69) sono rifiutati nei controlli preliminari, prima di ogni scrittura."""
    from src.pipeline import analyze as an
    paths, repo, spec = _env(tmp_path, mode="legacy2", run_models=("pca", model))
    if model in UNTRAINED_PAIRS:
        spec["models"][model] = dict(spec["models"][UNTRAINED_PAIRS[model]],
                                     score_batch_size=spec["training"]["batch_size"])
    with pytest.raises(ConfigError, match=f"{model}.*LEGACY_NUCLEUS"):
        an.run_analysis(copy.deepcopy(spec), _plan(**LEGACY_PLAN), datasets_dir=paths["spec"].parent,
                        repo_root=repo, allow_eval_machines=True)
    assert not (tmp_path / "store").exists()
    repo_store_guard.check()


def test_legacy_export_accepts_every_historical_model(tmp_path):
    from src.pipeline import analyze as an
    from src.utils.analysis_plan import validate_plan
    paths, repo, spec = _env(tmp_path, mode="legacy2", run_models=registry.LEGACY_NUCLEUS)
    cfg = validate_experiment(copy.deepcopy(spec))
    assert cfg.run_models == registry.LEGACY_NUCLEUS
    an.check_models_for_plan(validate_plan(_plan(**LEGACY_PLAN)), cfg)          # nessun errore


def test_refused_models_is_an_allow_list():
    """Ammessi solo i modelli storici: ogni nome nuovo, anche futuro, è rifiutato per costruzione."""
    from src.pipeline.legacy_export import refused_models
    assert refused_models(registry.LEGACY_NUCLEUS) == []
    assert refused_models(registry.NEW_PIPELINE_ONLY) == list(registry.NEW_PIPELINE_ONLY)
    assert refused_models(["un_modello_futuro"]) == ["un_modello_futuro"]


# ---------------------------------------------------------------- funzioni pure di analysis_grid
def _ap(**over):
    from src.utils.analysis_plan import validate_plan
    return validate_plan(_plan(**over))


def test_model_levels_and_absent_counts_match_metric_rows():
    """Conteggio analitico delle righe assenti = righe che metric_rows produrrebbe per le combinazioni tolte."""
    from src.pipeline import analysis_grid as ag
    plan = _ap()
    lv = ag.model_levels(plan, residual_aggs=("last",), allowed_ts_aggs=("last",))
    assert lv.window_time_aggs == ("last",) and lv.ts_aggs == ("last",)
    full = ag.model_levels(plan, residual_aggs=("last", "mean"), allowed_ts_aggs=None)
    assert full.window_time_aggs == plan.window_time_aggs and full.ts_aggs == plan.ts_aggs
    assert full.absent == {}
    geom = ag.Geometry(window_size=10, step=1, t_cal=200, t_test=300, mode="split3")
    rng = np.random.default_rng(0)
    res = {f"{s}__{a}": rng.random((n, 3)).astype(np.float32) for a in ("last", "mean")
           for s, n in (("cal", 191), ("test", 291), ("test_ctx_head", 9))}
    labels = np.zeros(300, dtype=int)
    labels[120:140] = 1
    ident = {"model": "x"}
    rows_full = ag.metric_rows(ident, ag.fit_scores(res, plan, False, None, geom), labels, geom, plan)
    rows_lv = ag.metric_rows(ident, ag.fit_scores(res, plan, False, None, geom, levels=lv), labels, geom, plan)
    for t in ("free", "thr"):
        n_full, n_lv = len(getattr(rows_full, t)), len(getattr(rows_lv, t))
        assert sum(c[t] for c in lv.absent.values()) == n_full - n_lv > 0, t
    off = sum(1 for r in rows_full.free if r["ts_agg"] != "last")
    assert lv.absent["offline_ts_agg"]["free"] == off


def test_residual_names_follow_the_saved_aggregations():
    from src.pipeline import analysis_grid as ag
    plan = _ap()
    assert ag.residual_names(plan, False, residual_aggs=("last",)) == (
        "cal__last", "test__last", "test_ctx_head__last")
    assert ag.residual_names(plan, False) == ag.residual_names(plan, False, residual_aggs=("last", "mean"))
