"""
F8 passo 2 (fase vera e propria): anti-leakage end-to-end con la config CANONICA (CLAUDE.md §5; D-03, D-04, D-05,
D-06, D-23, D-40, D-65).

Config: esperimento scritto su file e caricato sopra il ``configs/base.yaml`` reale (split3 60/20/20, minmax su fit
con ``min_range`` 0,02, ``train_step`` 5, epoche 100 / patience 10); il livello dell'esperimento aggiunge solo i dati
sintetici, W = 20 (serie sintetiche corte) e modelli piccolissimi. Tutti i modelli: PCA a finestre, PCA puntuale,
Conv-AE, LSTM-AE, i due AE non addestrati (D-69), il modello di previsione (D-74), rolling z-score, input norm.

Metodo (come ``tests/test_f5_leakage.py``): si perturba un segmento dei dati grezzi (es, cal o test) e si ripete il
percorso di ``run_fit`` (``prepare_machine`` → ``fit_one`` → ``compute_all``), poi quello di ``analyze`` sulle stesse
funzioni dell'analisi (``analysis_grid.fit_normalizer``, ``fit_scores``, ``thresholds.static_pct``,
``roll_quantile``, ``roll_robust_z`` con il supporto ``cal_mask``). Ciò che non deve vedere quel segmento resta
identico bit a bit; un controllo positivo verifica ogni volta che la perturbazione ha effetto. Dati sintetici in
``tmp_path`` (D-16), seed 42.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import segment_bounds
from src.data.windows import cal_mask
from src.evaluation import thresholds as th
from src.models import registry
from src.pipeline import analysis_grid as ag
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.utils import config as cfgmod
from src.utils.analysis_plan import validate_plan
from src.utils.schema import MODEL_SCHEMAS, model_kwargs
from tests.fixtures.datasets import write_synth_dataset

pytestmark = pytest.mark.slow

MACHINE = "machine-9-1"
SEED = 42
TAU = 0.02
W = 20
BATCH = 64
PCAS = ("pca", "pca_pointwise")
TRAINED = ("conv_autoencoder", "lstm_autoencoder", "lstm_forecaster")
UNTRAINED = ("conv_autoencoder_untrained", "lstm_autoencoder_untrained")
NATIVE = ("rolling_zscore", "input_norm")
MODELS = PCAS + TRAINED + UNTRAINED + NATIVE
NOT_TRAINED = PCAS + UNTRAINED + NATIVE
N_PAST = 20
PLAN = {"scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "mean", "normalizer": "std"},
                     {"sensor_agg": "max", "normalizer": "std"}, {"sensor_agg": "top5", "normalizer": "robust"}],
        "ts_aggs": ["last", "mean", "max"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": N_PAST}],
                       "roll_robust_z": [{"level": 99.0, "n_past": N_PAST}], "oracle": True},
        "pa": ["static_pct"]}


def _sha(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def _experiment(root: Path, data_dir: Path) -> Path:
    small = {"conv_autoencoder": {"channels": [8, 16], "latent_dim": 4},
             "lstm_autoencoder": {"hidden_dim": 8, "latent_dim": 4}}
    models = dict(small, lstm_forecaster={"hidden_dim": 8}, pca={"n_components": 0.9, "random_state": 42},
                  pca_pointwise={"n_components": 0.9})
    for u in UNTRAINED:
        models[u] = dict(small[MODEL_SCHEMAS[u].untrained_of], score_batch_size=BATCH)   # appaiato (D-69)
    raw = {"experiment": {"name": "f8_leak", "kind": "exploratory", "seeds": [SEED]},
           "paths": {"store_root": str(root / "store_unused")},
           "data": {"dataset": "synth", "data_dir": str(data_dir), "machines": [MACHINE]},
           "windows": {"window_size": W},                  # step e train_step dal base.yaml canonico
           "models": models, "run_models": list(MODELS)}
    p = root / "exp.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f8_leak")
    paths = write_synth_dataset(root / "synth")
    cfg = cfgmod.load_experiment(_experiment(root, paths["data"]))
    # premessa: il protocollo è quello canonico del base.yaml reale (F8 passo 1)
    assert (cfg.split.mode, cfg.split.fractions) == ("split3", (0.6, 0.2, 0.2))
    assert (cfg.preprocessing.scaler_type, cfg.preprocessing.min_range, cfg.preprocessing.clip) == ("minmax", TAU, None)
    assert (cfg.windows.step, cfg.windows.train_step) == (1, 5)
    assert (cfg.training.max_epochs, cfg.training.patience, cfg.training.batch_size) == (100, 10, BATCH)
    manifest = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    return {"root": root, "cfg": cfg, "raw": manifest.load(MACHINE)}


def _perturb(env, segment):
    """Copie dei dati grezzi con rumore nel solo segmento indicato (``None`` = nessuna perturbazione)."""
    train, test, labels = (np.array(a, copy=True) for a in env["raw"])
    if segment is None:
        return train, test, labels
    cfg = env["cfg"]
    rng = np.random.default_rng(123)
    if segment == "test":
        test += 0.2 * rng.standard_normal(test.shape)
    else:
        s = getattr(segment_bounds(len(train), len(test), cfg.split, cfg.windows.window_size), segment)
        train[s.start:s.stop] += 0.2 * rng.standard_normal((s.stop - s.start, train.shape[1]))
    return train, test, labels


def _analysis(cfg, plan, prepared, model, arrays):
    """Stesse funzioni di ``analyze._tables`` per un fit: normalizer su cal, score per timestamp, soglie."""
    native = bool(type(model).native_only)
    geom = ag.Geometry(cfg.windows.window_size, cfg.windows.step, len(prepared.data.cal), len(prepared.data.test),
                       prepared.mode)
    m_cal = cal_mask(geom.t_cal, geom.window_size, geom.step, geom.mode)
    levels = None if native else ag.model_levels(plan, rc.residual_aggs_for(model), type(model).allowed_ts_aggs)
    norms = {}
    if not native:
        for wagg in levels.window_time_aggs:
            for kind in ("std", "robust"):
                nz = ag.fit_normalizer(kind, arrays[f"cal{rc.SEP}{wagg}"])
                norms[(wagg, kind)] = (_sha(nz.mean_ if kind == "std" else nz.center_), _sha(nz.scale_))
    fs = ag.fit_scores(arrays, plan, native, model.config.get("top_k") if native else None, geom, levels=levels)
    assert not fs.errors and fs.scores
    cal_ts, test_ts, static, roll_q_first, roll_z_tau = {}, {}, {}, {}, {}
    for combo, s in fs.scores.items():
        cal_ts[combo], test_ts[combo] = _sha(s.cal_ts), _sha(s.test_ts)
        static[combo] = th.static_pct(s.cal_ts, m_cal, 98.0, test=s.test_ts).tau
        if combo.ts_agg == "last":                                    # soglie mobili solo con last (D-24)
            rq = th.roll_quantile(s.cal_ts, s.test_ts, n_past=N_PAST, level=99.0, ts_agg="last")
            first = int(np.flatnonzero(~np.isnan(s.test_ts))[0])      # primo score valido del test
            roll_q_first[combo] = rq.tau[first]                       # storia: solo i N_PAST score di cal
            roll_z_tau[combo] = th.roll_robust_z(s.cal_ts, s.test_ts, m_cal, n_past=N_PAST, level=99.0,
                                                 ts_agg="last").tau   # calibrazione su cal (D-23)
    return {"norms": norms, "cal_ts": cal_ts, "test_ts": test_ts, "static": static, "roll_q_first": roll_q_first,
            "roll_z_tau": roll_z_tau}


def _artifacts(env, data, workdir: Path):
    from tests.test_residuals_r1 import state_hashes
    cfg = env["cfg"]
    plan = validate_plan(PLAN)
    prepared = fitmod.prepare_machine(*data, cfg.split, cfg.preprocessing, cfg.windows)
    out = {"scaler": {k: _sha(a) for k, a in fitmod._scaler_arrays(prepared.scaler).items()},
           "scale": np.array(prepared.scaler.scale_, copy=True)}
    for name in MODELS:
        assert registry.get(name) is not None
        model = fitmod.fit_one(name, model_kwargs(cfg, name, SEED), prepared, SEED)
        d = workdir / name
        model.save(d)
        state = state_hashes(d)
        fi = getattr(model, "fit_info_", None)
        res = rc.compute_all(model, prepared, prepared.mode)
        out[name] = {"state": {k: v for k, v in state.items() if k != "checkpoint_json"},
                     "checkpoint_json": state["checkpoint_json"],
                     "train_loss": () if fi is None else tuple(fi.train_loss),
                     "val_loss": () if fi is None else tuple(fi.val_loss),
                     "best_epoch": None if fi is None else fi.best_epoch,
                     "res": {k: _sha(a) for k, a in res.arrays.items()},
                     "analysis": _analysis(cfg, plan, prepared, model, res.arrays)}
    return out


@pytest.fixture(scope="module")
def runs(env, tmp_path_factory):
    w = tmp_path_factory.mktemp("f8_leak_runs")
    return {seg or "base": _artifacts(env, _perturb(env, seg), w / (seg or "base")) for seg in (None, "es", "cal", "test")}


def _res(art, name, segment):
    return {k: v for k, v in art[name]["res"].items() if k.startswith(segment + rc.SEP)}


# ---------------------------------------------------------------- premessa: il pavimento è attivo
def test_canonical_floor_is_active_on_the_synthetic_constant_sensor(runs):
    # Fatto: il pavimento si applica in float32 (range su fit float32), quindi il denominatore è float32(τ) =
    # 0,0199999996 e la scala 1/float32(τ) = 50,0000011: tolleranza relativa dichiarata 1e−6.
    scale = runs["base"]["scale"]
    assert np.isclose(scale, 1.0 / TAU, rtol=1e-6, atol=0).any()   # sensore costante su fit: 1/τ = 50, non 1 (D-40)
    assert scale.max() == pytest.approx(1.0 / float(np.float32(TAU)), rel=1e-12)   # float(): niente float32 (NEP 50)


# ---------------------------------------------------------------- es, cal, test → scaler e PCA (D-06, D-65)
@pytest.mark.parametrize("segment", ["es", "cal", "test"])
def test_perturbing_any_non_fit_segment_leaves_scaler_and_both_pcas_unchanged(runs, segment):
    base, pert = runs["base"], runs[segment]
    assert pert["scaler"] == base["scaler"]
    for name in PCAS:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
    probe = {"es": "es", "cal": "cal", "test": "test"}[segment]           # controllo positivo
    assert _res(pert, "pca", probe) != _res(base, "pca", probe)


@pytest.mark.parametrize("segment", ["cal", "test"])
def test_cal_and_test_reach_no_fitted_model(runs, segment):
    """cal e test non entrano in alcun fit né nell'early stopping: stato identico per tutti i modelli."""
    base, pert = runs["base"], runs[segment]
    for name in MODELS:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
        assert pert[name]["val_loss"] == base[name]["val_loss"], name
        assert _res(pert, name, "es") == _res(base, name, "es") and _res(base, name, "es"), name


# ---------------------------------------------------------------- test → normalizer, soglie, storia di cal (D-23)
def test_perturbing_the_test_leaves_normalizers_static_thresholds_and_cal_history_unchanged(runs):
    base, pert = runs["base"], runs["test"]
    changed = 0
    for name in MODELS:
        assert _res(pert, name, "cal") == _res(base, name, "cal"), name
        a, b = base[name]["analysis"], pert[name]["analysis"]
        assert b["norms"] == a["norms"], name                               # normalizer su cal
        assert b["cal_ts"] == a["cal_ts"], name                             # storia di cal di ogni combinazione
        assert b["static"] == a["static"], name                             # soglie statiche su M_cal
        assert b["roll_q_first"] == a["roll_q_first"] and a["roll_q_first"], name   # 1ª soglia mobile: solo cal
        assert b["roll_z_tau"] == a["roll_z_tau"] and a["roll_z_tau"], name         # calibrazione di robust_z
        changed += sum(b["test_ts"][c] != a["test_ts"][c] for c in a["test_ts"])
    assert changed > 0                                                       # la perturbazione ha effetto
    assert base["lstm_autoencoder"]["analysis"]["norms"]                    # normalizer presenti (non native)


def test_perturbing_cal_changes_normalizers_and_thresholds(runs):
    """Controllo positivo dell'analisi: le grandezze sopra dipendono davvero da cal."""
    a, b = runs["base"]["pca"]["analysis"], runs["cal"]["pca"]["analysis"]
    assert b["norms"] != a["norms"] and b["cal_ts"] != a["cal_ts"] and b["static"] != a["static"]


# ---------------------------------------------------------------- es → solo l'early stopping (D-04, D-05)
def test_perturbing_es_changes_only_the_early_stopping(runs):
    base, pert = runs["base"], runs["es"]
    for name in NOT_TRAINED:                     # nessun early stopping: tutto identico su cal e test
        assert pert[name]["state"] == base[name]["state"], name
        for seg in ("cal", "test", fitmod.TEST_CTX_HEAD):
            assert _res(pert, name, seg) == _res(base, name, seg) and _res(base, name, seg), (name, seg)
    for name in TRAINED:
        a, b = base[name], pert[name]
        assert b["val_loss"] != a["val_loss"], name                        # es cambia la loss di early stopping
        n = min(len(a["train_loss"]), len(b["train_loss"]))
        assert n >= 1 and b["train_loss"][:n] == a["train_loss"][:n], name  # ma non la traiettoria di training
        if b["best_epoch"] == a["best_epoch"] and len(b["train_loss"]) == len(a["train_loss"]):
            assert b["state"] == a["state"], name                          # stessa epoca → stessi pesi
