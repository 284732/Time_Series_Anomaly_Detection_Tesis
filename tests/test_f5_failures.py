"""
F5 passo 4: fallimenti isolati per ``fit_key``, controlli preliminari bloccanti, interruzione, loss non finita (D-70,
D-58, nota su D-44). Modelli economici (pca, input_norm) dove basta; Conv-AE solo dove serve un AE. Dati sintetici in
``tmp_path``, nessuna lettura di ``dataset/``, nessuna scrittura nel repository (D-16).
"""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from src.models import registry
from src.models.base import FitInfo
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

MACHINES = ("machine-9-1", "machine-9-2")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_fail")
    paths = write_synth_dataset(root / "synth")
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent,
            "repo": make_repo(root / "repo")}


def _spec(env, store_name, data=None, **kw):
    kw.setdefault("seeds", (42,))
    return synth_spec(str(data or env["data"]), str(env["root"] / store_name), **kw)


def _root(spec):
    """store_root della specifica grezza (run_fit riceve la specifica, non la config validata)."""
    return spec["paths"]["store_root"]


def _run(env, spec, datasets_dir=None):
    return fitmod.run_fit(spec, datasets_dir=datasets_dir or env["datasets_dir"], repo_root=env["repo"])


def _metas(spec):
    fits = Path(_root(spec)) / "fits"
    if not fits.exists():
        return {}
    return {d.name: json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in sorted(fits.iterdir())}


def _by_task(spec):
    return {(m["machine"], m["model"]["name"]): (k, m) for k, m in _metas(spec).items()}


def _tmp_entries(spec):
    tmp = Path(_root(spec)) / "tmp"
    return sorted(p.name for p in tmp.iterdir()) if tmp.exists() else []


def _first_call_raises(real, exc):
    """Sostituto che solleva ``exc`` alla prima chiamata, poi delega."""
    calls = {"n": 0}

    def wrapper(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        return real(*a, **k)
    return wrapper, calls


def _strict_json(path):
    """json.load che rifiuta NaN/Infinity: prova che il file è stato scritto con allow_nan=False."""
    def bad(token):
        raise AssertionError(f"{path}: token non finito {token!r} nel JSON")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


# ------------------------------------------------------------------ 1. errore nel fit di un modello
def test_model_error_is_isolated_and_rerun_redoes_only_failed(env, monkeypatch):
    spec = _spec(env, "s_iso", run_models=("pca", "conv_autoencoder", "input_norm"))
    cls = registry.get("conv_autoencoder")
    boom, _ = _first_call_raises(cls.fit, RuntimeError("errore iniettato nel fit del Conv-AE"))
    monkeypatch.setattr(cls, "fit", lambda self, *a, **k: boom(self, *a, **k))
    report = _run(env, spec)

    assert report.counts == {"fitted": 5, "skipped": 0, "failed": 1}
    failed = [r for r in report.rows if r["outcome"] == "failed"]
    assert [(r["machine"], r["model"], r["phase"]) for r in failed] == [("machine-9-1", "conv_autoencoder", "fit")]
    tasks = _by_task(spec)
    key, meta = tasks[("machine-9-1", "conv_autoencoder")]
    assert meta["status"] == "failed"
    err = meta["error"]
    assert err["phase"] == "fit" and err["type"] == "RuntimeError"
    assert "errore iniettato" in err["message"] and "Traceback" in err["traceback"]
    assert "errore iniettato" in err["traceback"]
    assert sorted(p.name for p in (Path(_root(spec)) / "fits" / key).iterdir()) == ["meta.json"]
    assert all(m["status"] == "complete" for t, (_, m) in tasks.items() if t != ("machine-9-1", "conv_autoencoder"))
    assert [r["status"] for r in Store(_root(spec)).read_registry()] == ["started", "completed"]
    summary = (report.run_dir / "fit_summary.csv").read_text(encoding="utf-8").splitlines()
    assert sum(",failed," in line for line in summary) == 1

    # rilancio senza errore: si rifà solo la chiave failed, gli altri meta restano identici byte per byte
    monkeypatch.undo()
    fits = Path(_root(spec)) / "fits"
    before = {d.name: (d / "meta.json").read_bytes() for d in fits.iterdir() if d.name != key}
    calls = []
    for name in ("pca", "conv_autoencoder", "input_norm"):
        c = registry.get(name)
        monkeypatch.setattr(c, "fit", lambda self, *a, _r=c.fit, **k: (calls.append(self.name), _r(self, *a, **k))[1])
    second = _run(env, spec)
    assert calls == ["conv_autoencoder"]
    assert second.counts == {"fitted": 1, "skipped": 5, "failed": 0}
    assert {d.name: (d / "meta.json").read_bytes() for d in fits.iterdir() if d.name != key} == before
    assert _metas(spec)[key]["status"] == "complete" and "error" not in _metas(spec)[key]
    assert _tmp_entries(spec) == [] and Store(_root(spec)).orphan_tmp() == []


# ------------------------------------------------------------------ 2. errore in ciascuna fase
def _inject(phase, monkeypatch):
    if phase == "save":
        cls = registry.get("pca")
        boom, _ = _first_call_raises(cls.save, OSError("save iniettato"))
        monkeypatch.setattr(cls, "save", lambda self, *a, **k: boom(self, *a, **k))
    elif phase == "residuals":
        boom, _ = _first_call_raises(rc.compute_all, RuntimeError("residui iniettati"))
        monkeypatch.setattr(rc, "compute_all", boom)
    elif phase == "serialize":
        boom, _ = _first_call_raises(rc.npz_bytes, ValueError("serializzazione iniettata"))
        monkeypatch.setattr(rc, "npz_bytes", boom)


@pytest.mark.parametrize("phase", ["save", "residuals", "serialize"])
def test_error_phase_is_recorded(env, monkeypatch, phase):
    spec = _spec(env, f"s_phase_{phase}", run_models=("pca", "input_norm"))
    _inject(phase, monkeypatch)
    report = _run(env, spec)
    assert report.counts == {"fitted": 3, "skipped": 0, "failed": 1}
    key, meta = _by_task(spec)[("machine-9-1", "pca")]
    assert meta["status"] == "failed" and meta["error"]["phase"] == phase, meta.get("error")
    assert [r["phase"] for r in report.rows if r["outcome"] == "failed"] == [phase]
    assert sorted(p.name for p in (Path(_root(spec)) / "fits" / key).iterdir()) == ["meta.json"]
    assert not Store(_root(spec)).exists_complete(key)
    assert _tmp_entries(spec) == []


# ------------------------------------------------------------------ 3. DeviceMismatch isolato
def test_device_mismatch_is_isolated(env, monkeypatch):
    spec = _spec(env, "s_dev", run_models=("pca", "input_norm"))
    real = fitmod.effective_device
    monkeypatch.setattr(fitmod, "effective_device", lambda m: "cuda" if m.name == "pca" else real(m))
    report = _run(env, spec)
    assert report.counts == {"fitted": 2, "skipped": 0, "failed": 2}
    for machine in MACHINES:
        key, meta = _by_task(spec)[(machine, "pca")]
        assert meta["status"] == "failed"
        assert meta["error"]["phase"] == "device_check" and meta["error"]["type"] == "DeviceMismatch"
        assert "cuda" in meta["error"]["message"]
        assert sorted(p.name for p in (Path(_root(spec)) / "fits" / key).iterdir()) == ["meta.json"]
        assert _by_task(spec)[(machine, "input_norm")][1]["status"] == "complete"


# ------------------------------------------------------------------ 4. controlli preliminari bloccanti
def _bad_dataset(root, kind):
    paths = write_synth_dataset(root)
    train = paths["data"] / "train" / "machine-9-2.txt"
    if kind == "manifest":                               # valore fuori da value_range = [0, 1]
        lines = train.read_text(encoding="utf-8").splitlines()
        vals = lines[3].split(",")
        vals[0] = "5.000000"
        lines[3] = ",".join(vals)
        train.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:                                                # segmento troppo corto per W = 20 (val 20% di 60 = 12)
        for sub in ("train", "test", "test_label"):
            f = paths["data"] / sub / "machine-9-2.txt"
            f.write_text("\n".join(f.read_text(encoding="utf-8").splitlines()[:60]) + "\n", encoding="utf-8")
    return paths


@pytest.mark.parametrize("kind", ["manifest", "short_segment"])
def test_preflight_blocks_before_any_write(env, monkeypatch, tmp_path, kind):
    paths = _bad_dataset(tmp_path / "bad", kind)
    spec = _spec(env, f"s_pre_{kind}", data=paths["data"], run_models=("pca", "input_norm"))
    calls = []
    monkeypatch.setattr(fitmod, "fit_one", lambda *a, **k: calls.append(a[0]))
    with pytest.raises(fitmod.PreflightError) as ei:
        _run(env, spec, datasets_dir=paths["spec"].parent)
    msg = str(ei.value)
    assert "machine-9-2" in msg
    assert ("fuori da" in msg) if kind == "manifest" else ("troppo corti" in msg)
    assert calls == []
    assert not Path(_root(spec)).exists()            # nessun lock, registro, run, fit, tmp


# ------------------------------------------------------------------ 4-bis. validazione dentro run_fit (C5)
INVALID_SPECS = {
    "device_tpu": lambda s: s.__setitem__("device", "tpu"),
    "seeds_empty": lambda s: s["experiment"].__setitem__("seeds", []),
    "seed_str": lambda s: s["experiment"].__setitem__("seeds", ["42"]),
    "kind_typo": lambda s: s["experiment"].__setitem__("kind", "esplorativo"),
    "run_model_missing": lambda s: s.__setitem__("run_models", ["pca", "modello_assente"]),
    "name_with_slash": lambda s: s["experiment"].__setitem__("name", "a/b"),
}
NAME_GAP = pytest.mark.xfail(strict=True, reason="lacuna di F2: validate_experiment accetta experiment.name con '/' "
                             "(schema._str controlla solo stringa non vuota); begin_run lo rifiuta dopo il lock. "
                             "Da decidere a parte, lo schema non si cambia al passo 4")


@pytest.mark.parametrize("case", [pytest.param(c, marks=NAME_GAP) if c == "name_with_slash" else c
                                  for c in INVALID_SPECS])
def test_invalid_spec_is_refused_before_store(env, monkeypatch, case):
    from src.utils.schema import ConfigError
    spec = _spec(env, f"s_inv_{case}", run_models=("pca", "input_norm"))
    INVALID_SPECS[case](spec)
    calls = []
    monkeypatch.setattr(fitmod, "fit_one", lambda *a, **k: calls.append(a[0]))
    with pytest.raises(ConfigError):
        _run(env, spec)
    assert calls == []
    assert not Path(_root(spec)).exists()               # nessun lock, registro, run, fit, tmp


def test_validated_config_is_not_accepted(env):
    """La validazione non si salta: una ExperimentConfig già costruita è rifiutata prima di toccare lo store."""
    from src.utils.schema import validate_experiment
    spec = _spec(env, "s_cfg_obj", run_models=("pca",))
    with pytest.raises(TypeError, match="specifica grezza"):
        _run(env, validate_experiment(spec))
    assert not Path(_root(spec)).exists()


def test_run_fit_accepts_yaml_path(env, tmp_path):
    import yaml
    spec = _spec(env, "s_yaml", run_models=("pca",))
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    report = _run(env, path)
    assert report.counts == {"fitted": 2, "skipped": 0, "failed": 0}
    assert all(m["status"] == "complete" for m in _metas(spec).values())


# ------------------------------------------------------------------ 5. interruzione
def _interrupt(where, monkeypatch, spec):
    """KeyboardInterrupt sul compito (machine-9-1, pca), il secondo del run."""
    seen = {}
    if where == "fit":
        cls = registry.get("pca")
        boom, _ = _first_call_raises(cls.fit, KeyboardInterrupt())
        monkeypatch.setattr(cls, "fit", lambda self, *a, **k: boom(self, *a, **k))
    else:                                                # dentro il writer, dopo save (scaler.npz)
        real = fitmod._scaler_arrays
        calls = {"n": 0}

        def ki(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                tmp = Path(_root(spec)) / "tmp"
                seen["checkpoint_in_tmp"] = any((d / "checkpoint" / "checkpoint.json").exists()
                                                for d in tmp.iterdir())
                raise KeyboardInterrupt()
            return real(*a, **k)
        monkeypatch.setattr(fitmod, "_scaler_arrays", ki)
    return seen


def _hashes(spec):
    from tests.test_residuals_r1 import state_hashes
    out = {}
    for key, m in _metas(spec).items():
        out[key] = ({k: v["sha256"] for k, v in m["residuals"].items()},
                    state_hashes(Path(_root(spec)) / "fits" / key / "checkpoint"))
    return out


@pytest.fixture(scope="module")
def uninterrupted(env):
    spec = _spec(env, "s_ref_uninterrupted", run_models=("input_norm", "pca"))
    _run(env, spec)
    return _hashes(spec)


@pytest.mark.parametrize("where", ["fit", "writer"])
def test_interrupt_propagates_and_cleans_up(env, monkeypatch, uninterrupted, where):
    spec = _spec(env, f"s_int_{where}", run_models=("input_norm", "pca"))
    store = Store(_root(spec))
    seen = _interrupt(where, monkeypatch, spec)
    with pytest.raises(KeyboardInterrupt):
        _run(env, spec)
    if where == "writer":
        assert seen.get("checkpoint_in_tmp") is True     # l'interruzione è arrivata dopo save
    tasks = _by_task(spec)
    assert ("machine-9-1", "pca") not in tasks                                  # nessun fits/<key> in corso
    assert set(tasks) == {("machine-9-1", "input_norm")} and tasks[("machine-9-1", "input_norm")][1]["status"] == \
        "complete"
    assert _tmp_entries(spec) == []
    assert [r["status"] for r in store.read_registry()] == ["started", "interrupted"]
    assert store.read_lock() is None and not store.lock_path.exists()

    monkeypatch.undo()
    report = _run(env, spec)
    assert report.counts == {"fitted": 3, "skipped": 1, "failed": 0}
    assert _hashes(spec) == uninterrupted
    assert [r["status"] for r in store.read_registry()] == ["started", "interrupted", "started", "completed"]


# ------------------------------------------------------------------ 6. loss non finita
def test_nonfinite_loss_is_complete_with_flag(env, monkeypatch):
    from src.training.trainer import PyTorchTrainer
    spec = _spec(env, "s_nan", run_models=("conv_autoencoder",))
    real = PyTorchTrainer.fit

    def nan_fit(self, *a, **k):
        h = real(self, *a, **k)
        h["val_loss"][0] = float("nan")
        h["train_loss"][-1] = float("inf")
        return h
    monkeypatch.setattr(PyTorchTrainer, "fit", nan_fit)
    report = _run(env, spec)
    assert report.counts == {"fitted": 2, "skipped": 0, "failed": 0}
    for key, meta in _metas(spec).items():
        d = Path(_root(spec)) / "fits" / key
        assert meta["status"] == "complete" and meta["nonfinite_loss"] is True
        strict = _strict_json(d / "meta.json")
        assert strict["fit_info"]["val_loss"][0] is None and strict["fit_info"]["train_loss"][-1] is None
        ck = _strict_json(d / "checkpoint" / "checkpoint.json")
        assert ck["fit_info"]["val_loss"][0] is None
        fi = FitInfo.from_dict(ck["fit_info"])
        assert math.isnan(fi.val_loss[0]) and math.isnan(fi.train_loss[-1])
        assert all("n_nonfinite" in r for r in meta["residuals"].values())


def test_finite_loss_has_flag_false(env):
    spec = _spec(env, "s_finite", run_models=("conv_autoencoder", "pca", "input_norm"))
    _run(env, spec)
    for key, meta in _metas(spec).items():
        assert meta["nonfinite_loss"] is False
        assert all(isinstance(v, float) for v in (meta["fit_info"] or {}).get("val_loss", []))
        ck = json.loads((Path(_root(spec)) / "fits" / key / "checkpoint" / "checkpoint.json").read_text("utf-8"))
        for fi in (meta["fit_info"], ck["fit_info"]):     # modelli reali: round-trip identico, nessuna chiave in più
            assert "nonfinite_fields" not in fi
            assert FitInfo.from_dict(fi).to_dict() == fi, meta["model"]["name"]


def test_fitinfo_serialization_unchanged_for_finite_values():
    fi = FitInfo(best_epoch=1, stopped_epoch=2, train_loss=(0.5, 0.25), val_loss=(0.75, 0.5),
                 explained_variance=0.9, fit_seconds=1.5)
    d = fi.to_dict()
    assert d["train_loss"] == [0.5, 0.25] and d["val_loss"] == [0.75, 0.5]
    assert d["explained_variance"] == 0.9 and d["fit_seconds"] == 1.5
    assert FitInfo.from_dict(d) == fi
    assert fi.has_nonfinite_loss() is False
    assert FitInfo(train_loss=(float("nan"),)).has_nonfinite_loss() is True
    assert json.dumps(FitInfo(val_loss=(float("inf"), 1.0)).to_dict(), allow_nan=False)


FIELDS = tuple(f.name for f in dataclasses.fields(FitInfo))

# FitInfo come la costruiscono i modelli del nucleo: campi assenti (None) per AE, input_norm, rolling_zscore
MODEL_LIKE_FITINFO = {
    "ae": FitInfo.from_history({"train_loss": [0.5, 0.25], "val_loss": [0.75, 0.5], "best_epoch": 2,
                                "stopped_epoch": 2}, n_fit_samples=10, n_val_samples=4, fit_seconds=0.3),
    "input_norm": FitInfo(n_fit_samples=10, fit_seconds=0.0),
    "pca": FitInfo(n_components=3, explained_variance=0.91, svd_solver="full", svd_solver_effective="full",
                   n_fit_samples=10, fit_seconds=0.1),
    "empty": FitInfo(),
}


@pytest.mark.parametrize("name", list(MODEL_LIKE_FITINFO))
def test_fitinfo_finite_roundtrip_and_format_identical(name):
    """Valori finiti: nessuna chiave in più, None resta None (campo assente), round-trip identico."""
    fi = MODEL_LIKE_FITINFO[name]
    d = json.loads(json.dumps(fi.to_dict(), allow_nan=False))
    assert set(d) == set(FIELDS)           # formato invariato, niente nonfinite_fields
    back = FitInfo.from_dict(d)
    assert back == fi
    for k in FitInfo.FLOAT_FIELDS:
        assert (getattr(back, k) is None) == (getattr(fi, k) is None), k


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("field", ["train_loss", "val_loss", "explained_variance", "fit_seconds"])
def test_fitinfo_nonfinite_roundtrip(field, value):
    """Non finito in un campo → null (+ nonfinite_fields per gli scalari) → NaN; gli altri campi identici;
    un campo scalare assente resta None. La distinzione NaN/±inf non si conserva."""
    base = MODEL_LIKE_FITINFO["ae"] if field.endswith("loss") else MODEL_LIKE_FITINFO["pca"]
    if field.endswith("loss"):
        seq = list(getattr(base, field))
        seq[0] = value
        fi = FitInfo(**{**{f: getattr(base, f) for f in FIELDS}, field: tuple(seq)})
    else:
        fi = FitInfo(**{**{f: getattr(base, f) for f in FIELDS}, field: value})
    text = json.dumps(fi.to_dict(), allow_nan=False)
    d = json.loads(text)
    if field.endswith("loss"):
        assert d[field][0] is None and "nonfinite_fields" not in d
    else:
        assert d[field] is None and d["nonfinite_fields"] == [field]
    back = FitInfo.from_dict(d)
    got = getattr(back, field)[0] if field.endswith("loss") else getattr(back, field)
    assert math.isnan(got)
    for f in FIELDS:
        if f != field:
            assert getattr(back, f) == getattr(fi, f), f
    if field == "fit_seconds":                       # explained_variance era finita: resta tale
        assert back.explained_variance == 0.91
    absent = FitInfo.from_dict(json.loads(json.dumps(MODEL_LIKE_FITINFO["ae"].to_dict())))
    assert absent.explained_variance is None         # null senza nonfinite_fields = assente


def test_no_store_in_repository():
    assert not (Path(__file__).resolve().parents[1] / "store").exists()

