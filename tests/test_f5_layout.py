"""
F5 passo 1: layout di memoria degli array passati a ``model.fit`` dallo stadio di fit (D-70, C8; nota su D-64).

``fit_X`` / ``fit_X_val`` / ``train_scaled`` della nuova pipeline devono avere forma, dtype, passi, contiguità e valori
identici a ``train_windows`` / ``val_for_early_stop`` / ``train_scaled`` di ``main.prepare_machine_data`` (legacy2).
In split3 il riferimento è ``create_sliding_windows`` applicato alle stesse righe, scalate con uno scaler fittato solo
su fit (D-65). Una spia sostituisce ``fit`` (nessun training): registra gli argomenti e restituisce il modello.
"""

import numpy as np
import pytest

from src.data.dataset import create_sliding_windows
from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import make_scaler, segment_bounds
from src.data.windows import es_loss_windows
from src.models import registry
from src.pipeline import fit as fitmod
from src.utils.schema import model_kwargs
from src.utils.seed import set_seed
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, array_layout, synth_experiment
from tests.fixtures.legacy_snapshot import BASE_CONFIG
from tests.fixtures.parity import legacy_dataset_cfg, legacy_prepare

MACHINES = ("machine-9-1", "machine-9-2")
STEPS = [(1, 1), (5, 1), (1, 2), (5, 2)]          # (train_step, step)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_layout")
    paths = write_synth_dataset(root)
    man = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    return man, paths["data"], root


@pytest.fixture
def fit_spy(monkeypatch):
    """Sostituisce ``fit`` di ogni classe del nucleo: registra (X, X_val) senza addestrare."""
    calls = []

    def make(cls):
        def spy(self, X, y=None, X_val=None):
            calls.append({"model": self.name, "X": X, "X_val": X_val})
            return self
        return spy

    for name in CORE:
        cls = registry.get(name)
        monkeypatch.setattr(cls, "fit", make(cls))
    return calls


def _cfg(synth, mode, train_step, step):
    _, data_dir, root = synth
    return synth_experiment(str(data_dir), str(root / "store"), mode=mode, step=step, train_step=train_step)


def _prepare(synth, cfg, machine):
    man = synth[0]
    train, test, labels = man.load(machine)
    return fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)


def _writeable(a):
    return None if a is None else bool(a.flags.writeable)


def _check_same(ref, new, what):
    """Layout e valori identici; il flag writeable è solo un'informazione nel messaggio (non fa fallire)."""
    flags = f"writeable: nuovo {_writeable(new)}, main.py {_writeable(ref)}"
    assert array_layout(new) == array_layout(ref),         f"{what}: layout {array_layout(new)} != {array_layout(ref)} ({flags})"
    if ref is not None:
        assert np.array_equal(new, ref), f"{what}: stessi passi ma valori diversi (errore di confini?) ({flags})"


def _expected_legacy(synth, cfg, machine):
    w = cfg.windows
    ds = legacy_dataset_cfg(w.window_size, w.step, w.train_step, cfg.split.val_ratio, cfg.preprocessing)
    return legacy_prepare(machine, synth[1], ds)


def _expected_split3(synth, cfg, machine):
    """Riferimento indipendente da fit_scaler: SMDScaler su fit, poi create_sliding_windows (main.py:228-231)."""
    man, w = synth[0], cfg.windows
    train, test, _ = man.load(machine)
    segs = segment_bounds(len(train), len(test), cfg.split, w.window_size)
    scaler = make_scaler(cfg.preprocessing)
    fit_scaled = scaler.fit_transform(train[segs.fit.slice])
    es_scaled = scaler.transform(train[segs.es.slice])
    es_windows = create_sliding_windows(es_scaled, window_size=w.window_size, step=w.step)
    return {"train_scaled": fit_scaled,
            "train_windows": create_sliding_windows(fit_scaled, window_size=w.window_size, step=w.train_step),
            "val_for_early_stop": es_windows[::max(1, w.train_step // w.step)]}


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
@pytest.mark.parametrize("train_step,step", STEPS, ids=[f"ts{a}_s{b}" for a, b in STEPS])
@pytest.mark.parametrize("machine", MACHINES)
def test_fit_inputs_have_main_layout(synth, fit_spy, mode, train_step, step, machine):
    cfg = _cfg(synth, mode, train_step, step)
    ref = (_expected_legacy if mode == "legacy2" else _expected_split3)(synth, cfg, machine)
    prepared = _prepare(synth, cfg, machine)
    for name in CORE:
        fit_spy.clear()
        fitmod.fit_one(name, model_kwargs(cfg, name, 42), prepared, 42)
        assert len(fit_spy) == 1, name
        call = fit_spy[0]
        if getattr(registry.get(name), "fit_on_points", False):
            _check_same(ref["train_scaled"], call["X"], f"{name} X (train_scaled)")
            assert call["X_val"] is None, name                       # main.py:335
        else:
            _check_same(ref["train_windows"], call["X"], f"{name} X (train_windows)")
            _check_same(ref["val_for_early_stop"], call["X_val"], f"{name} X_val (val_for_early_stop)")


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
@pytest.mark.parametrize("train_step,step", STEPS, ids=[f"ts{a}_s{b}" for a, b in STEPS])
def test_fit_x_val_is_inside_es_with_d63_rule(synth, mode, train_step, step):
    """Stessi valori delle finestre della loss di early stopping di F3 (solo indici di es, regola di D-63)."""
    cfg = _cfg(synth, mode, train_step, step)
    p = _prepare(synth, cfg, "machine-9-1")
    ref = es_loss_windows(p.data.es, cfg.windows.window_size, step, train_step)
    assert np.array_equal(p.fit_X_val, ref.view)
    assert p.es_loss_factor == max(1, train_step // step)


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
def test_score_windowsets(synth, mode):
    """Viste di scoring per segmento: legacy2 cal e test; split3 anche es e, con step = 1, test_ctx_head."""
    cfg = _cfg(synth, mode, 5, 1)
    p = _prepare(synth, cfg, "machine-9-1")
    w = cfg.windows.window_size
    expected = {"cal", "test"} if mode == "legacy2" else {"es", "cal", "test", "test_ctx_head"}
    assert set(p.score) == expected
    for name, ws in p.score.items():
        assert ws.purpose == "score" and not ws.view.flags.writeable, name
    assert p.score["test"].context == 0 and p.score["cal"].context == 0
    if mode == "split3":
        head = p.score["test_ctx_head"]
        assert len(head) == w - 1 and head.context == w - 1
        assert np.array_equal(head.starts, np.arange(-(w - 1), 0))
        assert np.array_equal(head.ends, np.arange(0, w - 1))
        # la prima finestra inizia W−1 punti prima della fine di cal; l'ultima finisce al punto W−2 del test
        assert np.array_equal(head.view[0], np.concatenate([p.data.cal[-(w - 1):], p.data.test[:1]]))
        assert np.array_equal(head.view[-1], np.concatenate([p.data.cal[-1:], p.data.test[:w - 1]]))
    # step > 1 in split3: nessun test_ctx_head (D-60)
    if mode == "split3":
        p2 = _prepare(synth, _cfg(synth, mode, 5, 2), "machine-9-1")
        assert "test_ctx_head" not in p2.score


@pytest.mark.parametrize("train_step", [1, 5], ids=["ts1", "ts5"])
@pytest.mark.parametrize("name", CORE)
def test_fit_state_equals_fit_on_main_inputs(synth, tmp_path, name, train_step):
    """Equivalenza con main.py: stesso modello fittato (a) sugli ingressi di ``main.prepare_machine_data``
    (train_windows, val_for_early_stop, train_scaled, scrivibili) e (b) su quelli di ``prepare_machine`` (via
    ``fit_one``), con lo stesso ``set_seed(42)``: stato bit a bit identico (state.npz, tensori di model.pt e
    ``checkpoint.json`` senza tempi, standardizzatore della PCA puntuale compreso). legacy2, step = 1."""
    from tests.test_residuals_r1 import state_hashes
    cfg = _cfg(synth, "legacy2", train_step, 1)
    ref = _expected_legacy(synth, cfg, "machine-9-1")
    prepared = _prepare(synth, cfg, "machine-9-1")
    kw = model_kwargs(cfg, name, 42)
    cls = registry.get(name)

    model_a = cls(name=name, config=dict(kw))
    if getattr(cls, "fit_on_points", False):
        xa, xva = ref["train_scaled"], None
    else:
        xa, xva = ref["train_windows"], ref["val_for_early_stop"]
    set_seed(42)
    model_a.fit(xa, X_val=xva)
    model_a.save(tmp_path / "main_inputs")

    model_b = fitmod.fit_one(name, kw, prepared, 42)
    model_b.save(tmp_path / "new_inputs")

    ha, hb = state_hashes(tmp_path / "main_inputs"), state_hashes(tmp_path / "new_inputs")
    assert ha == hb, f"{name}: stato diverso con gli ingressi di main.py e di prepare_machine"
    if name == "pca_pointwise":
        assert {"mean_", "std_"} <= set(ha["state_npz"])


def test_fixture_config_matches_f1_snapshot(tmp_path):
    """La config di tests/fixtures/f5.py coincide campo per campo con quella che main.py usa davvero nello snapshot
    F1 (BASE_CONFIG con i default impliciti di main.py e del trainer, verificati sul sorgente in
    test_legacy_config_equivalence)."""
    from src.models.traditional.pca_detector import PCADetector
    from src.models.traditional.pca_pointwise import PointwisePCADetector
    from src.models.traditional.rolling_zscore import RollingZScoreDetector
    from tests.test_legacy_config_equivalence import DATASET_DEFAULTS, READ_KEYS, TRAINER_DEFAULTS

    cfg = synth_experiment(str(tmp_path / "data"), str(tmp_path / "store"))
    ds = {**DATASET_DEFAULTS, **BASE_CONFIG["dataset"]}
    ds.setdefault("train_step", ds["step"])
    assert ds["train_fraction"] == 1.0
    main_data = {"window_size": ds["window_size"], "step": ds["step"], "train_step": ds["train_step"],
                 "val_ratio": ds["val_ratio"], "scaler_type": ds["scaler_type"], "min_range": ds["min_range"],
                 "clip": tuple(ds["clip_range"]) if ds["clip_enabled"] else None,
                 "seeds": tuple(BASE_CONFIG["seeds"]), "machines": tuple(ds["machines"]),
                 "run_models": tuple(BASE_CONFIG["run_models"]), "device": BASE_CONFIG["device"]}
    new_data = {"window_size": cfg.windows.window_size, "step": cfg.windows.step,
                "train_step": cfg.windows.train_step, "val_ratio": cfg.split.val_ratio,
                "scaler_type": cfg.preprocessing.scaler_type, "min_range": cfg.preprocessing.min_range,
                "clip": cfg.preprocessing.clip, "seeds": cfg.seeds, "machines": cfg.data.machines,
                "run_models": cfg.run_models, "device": cfg.device}
    for k in main_data:
        assert new_data[k] == main_data[k], f"{k}: fixture {new_data[k]!r} != snapshot F1 {main_data[k]!r}"
    assert cfg.split.mode == "legacy2"

    seed0 = BASE_CONFIG["seeds"][0]
    for name in cfg.run_models:
        mc = dict(BASE_CONFIG["models"].get(name) or {})          # main.py:325-327
        mc["device"] = BASE_CONFIG["device"]
        mc.setdefault("random_state", seed0)
        kw = model_kwargs(cfg, name, seed0)
        if name in ("conv_autoencoder", "lstm_autoencoder"):
            eff = {**TRAINER_DEFAULTS, **mc}
            for k in READ_KEYS[name]:
                exp = list(eff[k]) if k == "channels" else eff[k]
                assert kw[k] == exp, f"{name}.{k}: fixture {kw[k]!r} != snapshot F1 {exp!r}"
            assert set(kw) <= set(READ_KEYS[name]), name
        elif name == "pca":
            assert PCADetector(config=kw).model.get_params() == PCADetector(config=mc).model.get_params()
        elif name == "pca_pointwise":
            assert (PointwisePCADetector(config=kw).model.get_params()
                    == PointwisePCADetector(config=mc).model.get_params())
        elif name == "rolling_zscore":
            a, b = RollingZScoreDetector(config=kw), RollingZScoreDetector(config=mc)
            assert (a.top_k, a.eps) == (b.top_k, b.eps), name
        elif name == "input_norm":
            assert kw == {"device": "cpu"}
        else:
            raise AssertionError(f"modello senza confronto: {name}")
