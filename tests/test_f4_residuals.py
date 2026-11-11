"""
F4: residui per sensore e checkpoint dei modelli del nucleo (T1, T2, T4, voce 11; D-33, D-64, D-66, D-67, D-68).

Dati: dataset sintetico (D-16), ``machine-9-1``, config dello snapshot F1 (W = 20). Ogni modello è addestrato una
volta per modulo con la sequenza di ``main.run_machine``. Il riferimento di T1 è ``score_features`` (o, per il
rolling z-score, ``score_anomalies``) sulle finestre prodotte da ``main.prepare_machine_data``; i residui nuovi si
calcolano con ``compute_residuals`` sulle finestre di F3 (manifest + segmenti ``legacy2`` + scaler + ``iter_batches``).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import fit_scaler, segment_bounds, slice_segments
from src.data.windows import score_test_windows, score_windows
from src.evaluation.scoring import aggregate_scores
from src.models.base import CHECKPOINT_JSON, STATE_NPZ, TORCH_STATE, checkpoint_versions
from src.models.batching import compute_residuals
from src.utils import schema
from src.utils.schema import PreprocConfig, SplitConfig
from tests.fixtures.core_models import MACHINE, fit_like_main, model_config, prepare
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.legacy_snapshot import BASE_CONFIG, array_summary

GOLDEN_T0 = Path(__file__).resolve().parent / "golden" / "models_pre_f4.json"
DS = BASE_CONFIG["dataset"]
W = DS["window_size"]
SEGS = ("val", "test")                       # val di main.py = cal di legacy2
AE_MODELS = ("conv_autoencoder", "lstm_autoencoder")
FEATURE_MODELS = ("pca", "pca_pointwise", "input_norm") + AE_MODELS
MODELS = FEATURE_MODELS + ("rolling_zscore",)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    paths = write_synth_dataset(tmp_path_factory.mktemp("f4_residuals"))
    data = prepare(paths["data"])                                       # main.prepare_machine_data
    man = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    train, test, labels = man.load(MACHINE)
    segs = segment_bounds(len(train), len(test), SplitConfig("legacy2", val_ratio=DS["val_ratio"]), W)
    _, sd = fit_scaler(slice_segments(train, test, labels, segs), PreprocConfig(DS["scaler_type"], 0.0, None))
    ws = {"val": score_windows(sd.cal, W, DS["step"], "cal"), "test": score_test_windows(sd, W, DS["step"], "primary")}
    models = {k: fit_like_main(k, data) for k in MODELS}
    golden = json.loads(GOLDEN_T0.read_text(encoding="utf-8"))["arrays"]
    return {"data": data, "ws": ws, "models": models, "golden": golden}


def _main_windows(env, seg):
    return env["data"][f"{seg}_windows"]


# --- T1: residui = metodi storici sulle finestre di main.prepare_machine_data -----------------------
def test_f3_scoring_windows_equal_main_windows(env):
    for seg in SEGS:
        assert np.array_equal(np.asarray(env["ws"][seg].view), _main_windows(env, seg))


@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize("model_key", FEATURE_MODELS)
@pytest.mark.parametrize("batch", ["one_batch", 4096])
def test_residuals_equal_score_features_and_t0_golden(env, model_key, seg, batch):
    """T1 (decisivo): stesso valore, tipo e forma di score_features storico e degli hash del golden T0."""
    model, ws = env["models"][model_key], env["ws"][seg]
    b = len(ws) if batch == "one_batch" else batch
    res = compute_residuals(model, ws, model.residual_aggs, b, "legacy2")
    for agg in model.residual_aggs:
        ref = model.score_features(_main_windows(env, seg), time_agg=agg)
        assert res[agg].dtype == ref.dtype and res[agg].shape == ref.shape
        assert np.array_equal(res[agg], ref), (model_key, seg, agg)
        assert array_summary(res[agg])["sha256"] == env["golden"][f"{model_key}|score_features|{agg}|{seg}"]["sha256"]


@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize("model_key,agg,chunk,batch", [
    ("pca", "last", 64, 64), ("pca", "mean", 64, 128), ("pca", "mean", 64, 192),
    ("pca_pointwise", "mean", 64, 64), ("pca_pointwise", "mean", 64, 128),
])
def test_residuals_with_several_inner_blocks(env, monkeypatch, model_key, agg, chunk, batch, seg):
    """Più blocchi interni (``_CHUNK`` piccolo sull'istanza), ultimo blocco parziale: identità bit a bit."""
    model, ws = env["models"][model_key], env["ws"][seg]
    monkeypatch.setattr(model, "_CHUNK", chunk)
    assert model.legacy_score_block[agg] == chunk and len(ws) % chunk != 0
    res = compute_residuals(model, ws, (agg,), batch, "legacy2")[agg]
    ref = model.score_features(_main_windows(env, seg), time_agg=agg)
    assert res.dtype == ref.dtype and np.array_equal(res, ref)


@pytest.mark.parametrize("seg", SEGS)
@pytest.mark.parametrize("model_key", AE_MODELS)
@pytest.mark.parametrize("batch", [64, 128, 192])
def test_ae_residuals_with_several_dataloader_batches(env, model_key, batch, seg):
    """AE: lotti multipli di batch_size (64), ultimo lotto del DataLoader parziale; lotti di legacy_layout_batch."""
    model, ws = env["models"][model_key], env["ws"][seg]
    assert model.legacy_score_block == {"last": 64, "mean": 64} and len(ws) % 64 != 0
    res = compute_residuals(model, ws, ("last", "mean"), batch, "legacy2")
    for agg in ("last", "mean"):
        ref = model.score_features(_main_windows(env, seg), time_agg=agg)
        assert res[agg].dtype == ref.dtype == np.float32 and np.array_equal(res[agg], ref), agg


@pytest.mark.parametrize("model_key,aggs,batch", [
    ("pca", ("last",), 64), ("pca_pointwise", ("last",), 64), ("input_norm", ("mean",), 64),
    ("rolling_zscore", ("last",), 64), ("conv_autoencoder", ("last",), 100), ("lstm_autoencoder", ("mean",), 96)])
def test_incompatible_batch_is_rejected(env, model_key, aggs, batch):
    model, ws = env["models"][model_key], env["ws"]["test"]
    with pytest.raises(ValueError, match="D-64"):
        compute_residuals(model, ws, aggs, batch, "legacy2")


@pytest.mark.parametrize("seg", SEGS)
def test_rolling_zscore_residuals_rebuild_score_anomalies(env, seg):
    """Rolling z-score: top-k dei residui `last` = score_anomalies storico, bit a bit (nota su D-33)."""
    model, ws = env["models"]["rolling_zscore"], env["ws"][seg]
    res = compute_residuals(model, ws, ("last",), len(ws), "legacy2")["last"]
    ref = model.score_anomalies(_main_windows(env, seg))
    rebuilt = aggregate_scores(res, None, "topk", k=model.top_k)
    assert rebuilt.dtype == ref.dtype and np.array_equal(rebuilt, ref)
    assert array_summary(ref)["sha256"] == env["golden"][f"rolling_zscore|score_anomalies|-|{seg}"]["sha256"]
    assert res.dtype == np.float32 and res.shape == (len(ws), _main_windows(env, seg).shape[2])
    with pytest.raises(ValueError, match="D-33"):
        model.residuals(_main_windows(env, seg), ("mean",))


def test_legacy_score_blocks(env):
    m = env["models"]
    assert m["pca"].legacy_score_block == {"last": 4096, "mean": 4096}
    assert m["pca_pointwise"].legacy_score_block == {"last": None, "mean": 2048}
    assert m["input_norm"].legacy_score_block == {"last": None, "mean": None}
    assert m["rolling_zscore"].legacy_score_block == {"last": None}
    for key in AE_MODELS:
        assert m[key].legacy_score_block == {"last": 64, "mean": 64}     # batch_size della config storica


# --- T2: calcolo manuale indipendente (semantica); l'identità bit a bit con la pipeline storica è di T1 ---------------------------------------------------------------------------
def test_input_norm_residuals_are_squared_inputs_exactly(env):
    x = _main_windows(env, "test")
    res = env["models"]["input_norm"].residuals(x, ("last", "mean"))
    sq = x.astype(np.float64) ** 2
    assert np.array_equal(res["last"], sq[:, -1, :]) and np.array_equal(res["mean"], sq.mean(axis=1))


def test_pca_residuals_match_manual_projection(env):
    """Proiezione scritta a mano in float64: tolleranza esplicita (l'ordine delle operazioni di sklearn è diverso)."""
    model, x = env["models"]["pca"], _main_windows(env, "test")
    n, w, f = x.shape
    c, mu = model.model.components_.astype(np.float64), model.model.mean_.astype(np.float64)
    x2 = x.reshape(n, w * f).astype(np.float64)
    sq = ((x2 - ((x2 - mu) @ c.T @ c + mu)) ** 2).reshape(n, w, f)
    res = model.residuals(x, ("last", "mean"))
    # sklearn proietta in float32 (dati e componenti float32, eps ~1.2e-7); il calcolo manuale è in float64.
    # Errore relativo atteso ~1e-6 sulla ricostruzione, amplificato dalla differenza x - x_hat piccola: rtol 1e-4;
    # atol 1e-7 per i residui vicini a zero (sensore costante).
    np.testing.assert_allclose(res["last"], sq[:, -1, :], rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(res["mean"], sq.mean(axis=1), rtol=1e-4, atol=1e-7)


def test_pca_pointwise_residuals_match_manual_projection(env):
    model, x = env["models"]["pca_pointwise"], _main_windows(env, "test")
    c, mu = model.model.components_, model.model.mean_
    z = (x.astype(np.float64) - model.mean_) / model.std_
    sq = (z - ((z - mu) @ c.T @ c + mu)) ** 2
    res = model.residuals(x, ("last", "mean"))
    # Tutto in float64 (eps ~2.2e-16) ma con un ordine delle operazioni diverso da sklearn (centratura, prodotti):
    # rtol 1e-9 lascia margine per la cancellazione in z - z_hat; atol 1e-12 per i residui vicini a zero.
    np.testing.assert_allclose(res["last"], sq[:, -1, :], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res["mean"], sq.mean(axis=1), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("model_key", AE_MODELS)
def test_ae_residuals_match_explicit_forward(env, model_key):
    model, x = env["models"][model_key], _main_windows(env, "test")
    net = model.model
    net.eval()
    with torch.no_grad():
        recon = net(torch.from_numpy(np.ascontiguousarray(x))).numpy()      # un solo forward su tutto il segmento
    sq = (x - recon) ** 2
    res = model.residuals(x, ("last", "mean"))
    # Forward in float32 su tutto il segmento invece che a lotti da 64: i kernel di torch possono sommare in un
    # altro ordine (errore relativo ~1e-6 in float32); rtol 1e-5, atol 1e-8 per gli errori vicini a zero.
    np.testing.assert_allclose(res["last"], sq[:, -1, :], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(res["mean"], sq.mean(axis=1), rtol=1e-5, atol=1e-8)


def test_rolling_zscore_residuals_match_manual_loop(env):
    model, x = env["models"]["rolling_zscore"], _main_windows(env, "val")
    res = model.residuals(x, ("last",))["last"]
    for i in (0, 17, len(x) - 1):
        h = x[i, :-1, :].astype(np.float64)
        z = np.abs(x[i, -1, :] - h.mean(axis=0)) / np.maximum(h.std(axis=0), model.eps)
        # Il modello lavora in float32 (media e deviazione su W-1 = 19 punti), il ciclo manuale in float64:
        # errore relativo atteso ~1e-6, rtol 1e-5; atol 1e-6 per |z| vicini a zero.
        np.testing.assert_allclose(res[i], z, rtol=1e-5, atol=1e-6)


# --- voce 11: residuals non cambia nulla -----------------------------------------------------------
def _state(model):
    """Copia profonda degli attributi numerici del modello (sklearn compreso)."""
    out = {}
    for k, v in vars(model).items():
        if isinstance(v, torch.nn.Module):               # rete degli AE: state_dict con i buffer di BatchNorm
            out.update({f"{k}.{a}": b.detach().clone().numpy() for a, b in v.state_dict().items()})
            out[f"{k}.training"] = np.array(v.training)
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif hasattr(v, "get_params"):
            out.update({f"{k}.{a}": np.array(b, copy=True) for a, b in vars(v).items()
                        if isinstance(b, (np.ndarray, np.generic))})
    return out


@pytest.mark.parametrize("model_key", MODELS)
def test_residuals_twice_identical_and_side_effect_free(env, model_key):
    model, x = env["models"][model_key], _main_windows(env, "val")
    before, torch_rng, np_rng = _state(model), torch.get_rng_state(), np.random.get_state()
    a = model.residuals(x, model.residual_aggs)
    b = model.residuals(x, model.residual_aggs)
    for agg in model.residual_aggs:
        assert a[agg] is not b[agg] and np.array_equal(a[agg], b[agg])
    after = _state(model)
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k]) for k in before)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert all(np.array_equal(p, q) for p, q in zip(np_rng[1:2], np.random.get_state()[1:2]))


@pytest.mark.parametrize("model_key", AE_MODELS)
def test_historical_ae_score_features_advances_torch_rng(env, model_key):
    """Fatto che motiva fork_rng in residuals: il DataLoader di score_features (invariato) consuma il generatore
    globale di torch anche con shuffle=False e num_workers=0."""
    model, x = env["models"][model_key], _main_windows(env, "val")
    before = torch.get_rng_state()
    model.score_features(x, "last")
    assert not torch.equal(before, torch.get_rng_state())


# --- FitInfo ---------------------------------------------------------------------------------------
def test_fit_info_of_traditional_models(env):
    m, d = env["models"], env["data"]
    for key, n in (("pca", len(d["train_windows"])), ("pca_pointwise", len(d["train_scaled"]))):
        fi = m[key].fit_info_
        assert fi.n_components == m[key].model.n_components_ and fi.n_fit_samples == n
        assert fi.svd_solver == fi.svd_solver_effective == "full"           # n_components float (D-57)
        assert fi.explained_variance == pytest.approx(float(m[key].model.explained_variance_ratio_.sum()))
        assert fi.best_epoch is None and fi.fit_seconds >= 0
    for key in ("rolling_zscore", "input_norm"):
        assert m[key].fit_info_.n_fit_samples == len(d["train_windows"])


def _expected_best_epoch(val_loss, min_delta):
    """Regola di PyTorchTrainer.fit: nuova migliore solo se val < migliore - min_delta (epoche da 1)."""
    best, best_epoch = float("inf"), 0
    for e, v in enumerate(val_loss, start=1):
        if v < best - min_delta:
            best, best_epoch = v, e
    return best_epoch


@pytest.mark.parametrize("model_key", AE_MODELS)
def test_fit_info_of_autoencoders(env, model_key):
    model, d = env["models"][model_key], env["data"]
    fi, h = model.fit_info_, model.history
    assert (fi.best_epoch, fi.stopped_epoch) == (h["best_epoch"], h["stopped_epoch"])
    assert fi.train_loss == tuple(h["train_loss"]) and fi.val_loss == tuple(h["val_loss"])
    assert 1 <= fi.best_epoch <= fi.stopped_epoch == len(fi.train_loss) == len(fi.val_loss)
    assert fi.best_epoch == _expected_best_epoch(fi.val_loss, model.config.get("early_stopping_min_delta", 1e-6))
    assert fi.val_loss[fi.best_epoch - 1] == min(fi.val_loss)
    assert fi.n_fit_samples == len(d["train_windows"]) and fi.n_val_samples == len(d["val_for_early_stop"])
    assert fi.n_components is None and fi.fit_seconds > 0


# --- T4: checkpoint --------------------------------------------------------------------------------
def _cls(model):
    return type(model)


@pytest.mark.parametrize("model_key", MODELS)
def test_checkpoint_roundtrip_gives_identical_residuals(env, tmp_path, model_key):
    model = env["models"][model_key]
    model.save(tmp_path / "ckpt")
    loaded = _cls(model).load(tmp_path / "ckpt")
    assert loaded.is_fitted and loaded.loaded_with_version_mismatch is False
    assert loaded.fit_info_ == model.fit_info_ and loaded.config == model.config and loaded.name == model.name
    for seg in SEGS:
        x = _main_windows(env, seg)
        a, b = model.residuals(x, model.residual_aggs), loaded.residuals(x, loaded.residual_aggs)
        for agg in model.residual_aggs:
            assert a[agg].dtype == b[agg].dtype and np.array_equal(a[agg], b[agg]), (seg, agg)
        assert np.array_equal(model.score_anomalies(x), loaded.score_anomalies(x))
    meta = json.loads((tmp_path / "ckpt" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    assert meta["checkpoint_format"] == 1 and meta["fit_code_version"] == schema.FIT_CODE_VERSION == 1
    assert meta["model"] == model_key and meta["versions"] == checkpoint_versions()
    assert meta["class_path"] == f"{_cls(model).__module__}:{_cls(model).__qualname__}"


@pytest.mark.parametrize("model_key", ["pca", "pca_pointwise"])
def test_pca_checkpoint_restores_every_fitted_attribute(env, tmp_path, model_key):
    model = env["models"][model_key]
    model.save(tmp_path / "ckpt")
    loaded = _cls(model).load(tmp_path / "ckpt")
    va, vb = vars(model.model), vars(loaded.model)
    assert va.keys() == vb.keys()
    for k, v in va.items():
        assert type(v) is type(vb[k]), k
        assert np.array_equal(v, vb[k]) if isinstance(v, (np.ndarray, np.generic)) else v == vb[k], k
    meta = json.loads((tmp_path / "ckpt" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    assert meta["state"]["pca"]["private_attrs"] == ["_fit_svd_solver"]      # annotato (D-67)
    with np.load(tmp_path / "ckpt" / STATE_NPZ, allow_pickle=False) as z:
        assert all(z[k].dtype != object for k in z.files)
    if model_key == "pca_pointwise":
        assert np.array_equal(loaded.mean_, model.mean_) and np.array_equal(loaded.std_, model.std_)


def test_pca_with_integer_components_roundtrip(env, tmp_path):
    """D-57 aperta: con n_components intero il checkpoint salva le componenti, quindi i residui non dipendono
    dal solver al momento del load."""
    cfg = {"n_components": 3, "random_state": 7}
    model = fit_like_main("pca", env["data"], config=cfg)
    assert model.fit_info_.svd_solver == "auto" and model.fit_info_.n_components == 3
    model.save(tmp_path / "ckpt")
    loaded = _cls(model).load(tmp_path / "ckpt")
    x = _main_windows(env, "test")
    assert np.array_equal(model.residuals(x, ("last",))["last"], loaded.residuals(x, ("last",))["last"])


# --- AE: pesi dopo l'early stopping, state_dict, caricamento solo dal checkpoint ------------------
@pytest.mark.parametrize("model_key", AE_MODELS)
def test_early_stopping_restores_best_weights_and_checkpoint_saves_them(env, tmp_path, monkeypatch, model_key):
    """Scenario forzato: val loss [1.0, 0.5, 0.6, 0.7, 0.8], patience 2 -> best 2, stop 4. Dopo fit restano i pesi
    dell'epoca 2 (ricaricati), ed esattamente quelli finiscono nel checkpoint e in score_features."""
    from src.training.trainer import PyTorchTrainer
    seq = iter([1.0, 0.5, 0.6, 0.7, 0.8])
    per_epoch = []
    orig_train_epoch = PyTorchTrainer.train_epoch

    def train_epoch(self, loader):
        loss = orig_train_epoch(self, loader)
        per_epoch.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})
        return loss

    monkeypatch.setattr(PyTorchTrainer, "train_epoch", train_epoch)
    monkeypatch.setattr(PyTorchTrainer, "evaluate", lambda self, loader: next(seq))
    cfg = dict(model_config(model_key), epochs=5, early_stopping_patience=2)
    model = fit_like_main(model_key, env["data"], config=cfg)
    fi = model.fit_info_
    assert (fi.best_epoch, fi.stopped_epoch, fi.val_loss) == (2, 4, (1.0, 0.5, 0.6, 0.7))
    final = model.model.state_dict()
    assert all(torch.equal(final[k], per_epoch[1][k]) for k in final)                  # epoca 2
    assert not all(torch.equal(final[k], per_epoch[3][k]) for k in final)              # non l'ultima
    model.save(tmp_path / "ckpt")
    sd = torch.load(tmp_path / "ckpt" / TORCH_STATE, map_location="cpu", weights_only=True)
    assert sd.keys() == final.keys() and all(torch.equal(sd[k], final[k]) for k in final)
    monkeypatch.undo()
    loaded = type(model).load(tmp_path / "ckpt")
    x = _main_windows(env, "test")
    assert np.array_equal(loaded.residuals(x, ("mean",))["mean"], model.score_features(x, "mean"))


@pytest.mark.parametrize("model_key", AE_MODELS)
def test_ae_state_dict_in_checkpoint_has_every_buffer(env, tmp_path, model_key):
    model = env["models"][model_key]
    model.save(tmp_path / "ckpt")
    sd = torch.load(tmp_path / "ckpt" / TORCH_STATE, map_location="cpu", weights_only=True)
    ref = model.model.state_dict()
    assert list(sd) == list(ref)
    for k in ref:
        assert sd[k].dtype == ref[k].dtype and torch.equal(sd[k], ref[k]), k
    if model_key == "conv_autoencoder":
        bn = [k for k in ref if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        assert len(bn) == 3 * 3 and all(ref[k].dtype == torch.int64 for k in bn if "num_batches" in k)
        assert all(int(ref[k]) > 0 for k in bn if "num_batches" in k)           # aggiornati dal training
    meta = json.loads((tmp_path / "ckpt" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    assert meta["state"]["arch"] == model.arch_ and meta["config"]["batch_size"] == 64
    assert meta["config"]["device"] == "cpu" and meta["torch_num_threads"] == 1


@pytest.mark.parametrize("model_key", AE_MODELS)
def test_ae_load_refuses_a_different_device_unless_forced(env, tmp_path, model_key):
    """Il checkpoint registra il device effettivo del fit; load lo confronta con quello scelto dalla regola di fit."""
    model = env["models"][model_key]
    model.save(tmp_path / "ckpt")
    meta = json.loads((tmp_path / "ckpt" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    assert meta["state"]["device"] == str(model.trainer.device) == "cpu"
    _edit_meta(tmp_path / "ckpt", lambda m: m["state"].update(device="cuda:0"))
    with pytest.raises(ValueError, match="device"):
        type(model).load(tmp_path / "ckpt")
    with pytest.warns(RuntimeWarning, match="device"):
        loaded = type(model).load(tmp_path / "ckpt", allow_version_mismatch=True)
    assert loaded.loaded_with_version_mismatch is True and str(loaded.trainer.device) == "cpu"
    x = _main_windows(env, "val")
    assert np.array_equal(loaded.residuals(x, ("last",))["last"], model.residuals(x, ("last",))["last"])


LOAD_SCRIPT = """
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
torch.set_num_threads(1)
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from src.models.deep.conv_ae import ConvAutoencoderDetector
from src.models.deep.lstm_ae import LSTMAutoencoderDetector
classes = {"conv_autoencoder": ConvAutoencoderDetector, "lstm_autoencoder": LSTMAutoencoderDetector}
x = np.load(sys.argv[2])
for key, ckpt, out in zip(sys.argv[3::3], sys.argv[4::3], sys.argv[5::3]):
    r = classes[key].load(ckpt).residuals(x, ("last", "mean"))
    np.savez(out, last=r["last"], mean=r["mean"])
"""


def test_ae_checkpoint_loads_in_a_fresh_process_from_the_checkpoint_only(env, tmp_path):
    """Istanza costruita solo dal checkpoint (nessuna config, nessun dato di training), in un processo nuovo
    (uno solo per entrambi gli AE: l'import di torch domina il tempo)."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    x = _main_windows(env, "test")
    np.save(tmp_path / "x.npy", x)
    script = tmp_path / "load.py"
    script.write_text(LOAD_SCRIPT, encoding="utf-8")
    args = []
    for key in AE_MODELS:
        env["models"][key].save(tmp_path / key)
        args += [key, str(tmp_path / key), str(tmp_path / f"{key}_res.npz")]
    subprocess.run([sys.executable, "-I", str(script), str(root), str(tmp_path / "x.npy"), *args],
                   check=True, capture_output=True, cwd=tmp_path)
    for key in AE_MODELS:
        with np.load(tmp_path / f"{key}_res.npz") as z:
            for agg in ("last", "mean"):
                ref = env["models"][key].score_features(x, agg)
                assert z[agg].dtype == ref.dtype and np.array_equal(z[agg], ref), (key, agg)


def _saved(env, tmp_path, key="pca"):
    model = env["models"][key]
    model.save(tmp_path / "ckpt")
    return model, tmp_path / "ckpt"


def _edit_meta(path, fn):
    p = path / CHECKPOINT_JSON
    meta = json.loads(p.read_text(encoding="utf-8"))
    fn(meta)
    p.write_text(json.dumps(meta), encoding="utf-8")


def test_save_refuses_non_empty_dir_and_unfitted_model(env, tmp_path):
    model, path = _saved(env, tmp_path)
    with pytest.raises(FileExistsError, match="D-67"):
        model.save(path)
    from src.models.traditional.pca_detector import PCADetector
    with pytest.raises(RuntimeError):
        PCADetector(config=model_config("pca")).save(tmp_path / "other")


def test_load_refuses_other_fit_code_version(env, tmp_path, monkeypatch):
    model, path = _saved(env, tmp_path)
    monkeypatch.setattr(schema, "FIT_CODE_VERSION", schema.FIT_CODE_VERSION + 1)
    with pytest.raises(ValueError, match="fit_code_version"):
        _cls(model).load(path)


def test_load_refuses_unknown_format_and_other_class(env, tmp_path):
    from src.models.traditional.pca_pointwise import PointwisePCADetector
    model, path = _saved(env, tmp_path)
    with pytest.raises(ValueError, match="non di"):
        PointwisePCADetector.load(path)
    _edit_meta(path, lambda m: m.update(checkpoint_format=2))
    with pytest.raises(ValueError, match="formato"):
        _cls(model).load(path)


@pytest.mark.parametrize("lib", ["python", "numpy", "torch", "scikit-learn"])
def test_load_refuses_other_library_versions_unless_forced(env, tmp_path, lib):
    model, path = _saved(env, tmp_path)
    _edit_meta(path, lambda m: m["versions"].update({lib: "0.0.0"}))
    with pytest.raises(ValueError, match="versioni diverse"):
        _cls(model).load(path)
    with pytest.warns(RuntimeWarning, match=lib):
        loaded = _cls(model).load(path, allow_version_mismatch=True)
    assert loaded.loaded_with_version_mismatch is True
    x = _main_windows(env, "val")
    assert np.array_equal(loaded.residuals(x, ("last",))["last"], model.residuals(x, ("last",))["last"])


def test_checkpoints_differ_only_in_timing_when_fit_seconds_changes(env, tmp_path):
    """Due salvataggi dello stesso modello con fit_seconds diversi: checkpoint.json cambia solo in quel campo."""
    import dataclasses
    model = env["models"]["pca"]
    model.save(tmp_path / "a")
    orig = model.fit_info_
    try:
        model.fit_info_ = dataclasses.replace(orig, fit_seconds=orig.fit_seconds + 123.0)
        model.save(tmp_path / "b")
    finally:
        model.fit_info_ = orig
    a, b = (json.loads((tmp_path / d / CHECKPOINT_JSON).read_text(encoding="utf-8")) for d in "ab")
    assert a["fit_info"]["fit_seconds"] != b["fit_info"]["fit_seconds"]
    for m in (a, b):
        m["fit_info"].pop("fit_seconds")
    assert a == b
