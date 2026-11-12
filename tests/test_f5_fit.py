"""
F5 passo 3: ``plan_fits``, ``run_fit``, ``meta.json``, parità legacy2 (D-70; D-04, D-33, D-66, D-67).

Fixture di modulo: ``run_fit`` una volta sul dataset sintetico in ``tmp_path`` con la config di
``tests/fixtures/f5.py`` (2 macchine, 6 modelli, seed [42, 43] → 16 fit). Il riferimento storico è calcolato dal vivo
con la sequenza di ``main.run_machine`` (``fit_like_main`` + ``score_features``). Nessuna lettura di ``dataset/``,
nessuna scrittura nel repository (D-16).
"""

import copy
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.data.manifest import build_manifest, load_dataset_spec
from src.models import registry
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store, StoreLocked
from src.utils import config as cfgmod
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, synth_experiment
from tests.fixtures.git_repo import make_repo
from tests.fixtures.legacy_snapshot import array_summary

ROOT = Path(__file__).resolve().parents[1]
MACHINES = ("machine-9-1", "machine-9-2")
SEEDS = (42, 43)
D66 = {"pca": "<f8", "pca_pointwise": "<f8", "input_norm": "<f8",
       "conv_autoencoder": "<f4", "lstm_autoencoder": "<f4", "rolling_zscore": "<f4"}
EXCLUDED_META = {"timings", "created_by", "provenance"}


# ------------------------------------------------------------------ fixture
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_fit")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent, "repo": repo,
            "manifest": build_manifest(load_dataset_spec(paths["spec"]), paths["data"])}


def _cfg(env, store_name, **kw):
    return synth_experiment(str(env["data"]), str(env["root"] / store_name), **kw)


def _run(env, cfg, **kw):
    return fitmod.run_fit(cfg, datasets_dir=env["datasets_dir"], repo_root=env["repo"], **kw)


@pytest.fixture(scope="module")
def run_a(env):
    cfg = _cfg(env, "store_a")
    t0 = time.perf_counter()
    report = _run(env, cfg)
    seconds = time.perf_counter() - t0
    return cfg, report, seconds


def _fits(cfg):
    store = Store(cfg.store_root)
    out = {}
    for d in sorted((store.root / "fits").iterdir()):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        out[d.name] = {"meta": meta, "dir": d, "res": rc.read_npz(d / "residuals.npz")}
    return out


def _state_hashes(fit_dir):
    from tests.test_residuals_r1 import state_hashes
    return state_hashes(fit_dir / "checkpoint")


# ------------------------------------------------------------------ 1. parità legacy2
@pytest.fixture(scope="module")
def historical(env):
    """Residui storici dal vivo: per macchina, modello e seed la sequenza di main.run_machine."""
    from tests.fixtures.core_models import fit_like_main, prepare
    out = {}
    for machine in MACHINES:
        data = prepare(env["data"], machine)
        for name in CORE:
            for seed in (SEEDS[:1] if registry.get(name).deterministic else SEEDS):
                model = fit_like_main(name, data, seed=seed)
                ref = {}
                for seg, key in (("cal", "val_windows"), ("test", "test_windows")):
                    if model.native_only:
                        ref[(seg, "topk")] = model.score_anomalies(data[key])
                    else:
                        for agg in ("last", "mean"):
                            ref[(seg, agg)] = model.score_features(data[key], time_agg=agg)
                out[(machine, name, None if model.deterministic else seed)] = ref
    return out


def test_parity_with_historical_pipeline(run_a, historical):
    from src.evaluation.scoring import aggregate_scores
    cfg, _, _ = run_a
    fits = _fits(cfg)
    assert len(fits) == 16
    seen = set()
    for key, f in fits.items():
        m = f["meta"]
        ident = (m["machine"], m["model"]["name"], m["seed"]["key_seed"])
        seen.add(ident)
        ref = historical[ident]
        for (seg, agg), r in ref.items():
            if agg == "topk":
                new = aggregate_scores(f["res"][f"{seg}__last"], None, "topk", k=5)
            else:
                new = f["res"][f"{seg}__{agg}"]
            assert r.dtype == new.dtype and r.shape == new.shape and r.tobytes() == new.tobytes(), (ident, seg, agg)
    assert len(seen) == 16


def test_parity_with_golden_r1(run_a):
    cfg, _, _ = run_a
    golden = json.loads((ROOT / "tests" / "golden" / "residuals_r1.json").read_text(encoding="utf-8"))["residuals"]
    checked = 0
    for f in _fits(cfg).values():
        m = f["meta"]
        if m["machine"] != "machine-9-1" or m["seed"]["set_seed_value"] != 42:
            continue
        for rname, a in f["res"].items():
            seg, agg = rname.split(rc.SEP)
            assert array_summary(a) == golden[f"{m['model']['name']}|{agg}|{'val' if seg == 'cal' else seg}"], rname
            checked += 1
    assert checked == 5 * 2 * 2 + 2


# ------------------------------------------------------------------ 2. pianificazione
def test_plan_fits(env):
    cfg = _cfg(env, "store_plan")
    tasks = fitmod.plan_fits(cfg, env["manifest"])
    assert len(tasks) == 16 and len({t.fit_key for t in tasks}) == 16
    for machine in MACHINES:
        for name in CORE:
            ts = [t for t in tasks if t.machine == machine and t.model == name]
            if registry.get(name).deterministic:
                assert len(ts) == 1 and ts[0].seed is None and ts[0].set_seed_value == 42, (machine, name)
                assert "seed" not in ts[0].payload
            else:
                assert [t.seed for t in ts] == [42, 43] and [t.set_seed_value for t in ts] == [42, 43]
                assert [t.payload["seed"] for t in ts] == [42, 43]
            for t in ts:
                assert t.fit_key == cfgmod.fit_key(cfg, machine, name, 42 if t.seed is None else t.seed,
                                                   env["manifest"].data_version(machine))
    assert not Path(cfg.store_root).exists()                       # pianificare non scrive nulla


# ------------------------------------------------------------------ 3. rilancio
def test_rerun_does_not_refit(env, run_a, monkeypatch):
    cfg, first, _ = run_a
    before = {k: ((f["dir"] / "meta.json").read_bytes(), (f["dir"] / "residuals.npz").read_bytes(),
                  (f["dir"] / "residuals.json").read_bytes()) for k, f in _fits(cfg).items()}
    calls = []

    def no_fit(self, *a, **k):
        calls.append(self.name)
        raise AssertionError(f"fit di {self.name} chiamato al rilancio: il riuso non funziona")

    for name in CORE:
        monkeypatch.setattr(registry.get(name), "fit", no_fit)
    second = _run(env, cfg)
    assert calls == []
    assert [r["outcome"] for r in second.rows] == ["skipped"] * 16
    after = {k: ((f["dir"] / "meta.json").read_bytes(), (f["dir"] / "residuals.npz").read_bytes(),
                 (f["dir"] / "residuals.json").read_bytes()) for k, f in _fits(cfg).items()}
    assert after == before
    for f in _fits(cfg).values():
        assert f["meta"]["created_by"]["run_id"] == first.run_id
    summary = (second.run_dir / "fit_summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(summary) == 17 and all(",skipped," in line for line in summary[1:])


# ------------------------------------------------------------------ 4 e 5. sequenza e ordine dei modelli
@pytest.fixture(scope="module")
def run_reversed(env, tmp_path_factory):
    """Store nuovo, ordine dei modelli invertito, spie su costruzione, set_seed, fit e load."""
    mp = pytest.MonkeyPatch()
    events = []
    real_set_seed = fitmod.set_seed
    mp.setattr(fitmod, "set_seed", lambda s: (events.append(("set_seed", s)), real_set_seed(s))[1])
    for name in CORE:
        cls = registry.get(name)
        real_init, real_fit = cls.__init__, cls.fit

        def init(self, *a, _r=real_init, **k):
            _r(self, *a, **k)
            events.append(("init", self.name))

        def fit(self, *a, _r=real_fit, **k):
            events.append(("fit", self.name))
            return _r(self, *a, **k)

        def load(klass, *a, **k):
            events.append(("load", klass.__name__))
            raise AssertionError("load durante lo stadio di fit")

        mp.setattr(cls, "__init__", init)
        mp.setattr(cls, "fit", fit)
        mp.setattr(cls, "load", classmethod(load))
    mp.setattr(registry, "load_checkpoint", lambda *a, **k: events.append(("load_checkpoint",)))
    try:
        cfg = _cfg(env, "store_rev", run_models=tuple(reversed(CORE)))
        report = _run(env, cfg)
        n_first = len(events)
        report2 = _run(env, cfg)
        rerun_events = events[n_first:]
    finally:
        mp.undo()
    return cfg, report, events[:n_first], rerun_events, report2


def test_sequence_init_set_seed_fit_and_no_load(run_reversed):
    _, report, events, rerun_events, _ = run_reversed
    assert not [e for e in events if e[0].startswith("load")]
    fitted = [r for r in report.rows if r["outcome"] == "fitted"]
    assert len(fitted) == 16
    triples = [events[i:i + 3] for i in range(0, len(events), 3)]
    assert len(triples) == 16 and len(events) == 48
    for (e1, e2, e3), row in zip(triples, fitted):
        assert e1 == ("init", row["model"]) and e2[0] == "set_seed" and e3 == ("fit", row["model"]), (e1, e2, e3)
        assert e2[1] == row["set_seed_value"]
    assert [e for e in rerun_events if e[0] in ("set_seed", "fit") or e[0].startswith("load")] == []


def test_model_order_does_not_change_artifacts(run_a, run_reversed):
    fa, fb = _fits(run_a[0]), _fits(run_reversed[0])
    assert set(fa) == set(fb)
    for k in fa:
        assert fa[k]["meta"]["residuals"] == fb[k]["meta"]["residuals"], k
        assert _state_hashes(fa[k]["dir"]) == _state_hashes(fb[k]["dir"]), k


# ------------------------------------------------------------------ 6. tempi fuori dagli hash
@pytest.fixture(scope="module")
def run_c(env):
    cfg = _cfg(env, "store_c")
    return cfg, _run(env, cfg)


def _diff_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(prefix + k)
            else:
                out |= _diff_paths(a[k], b[k], prefix + k + ".")
        return out
    return set() if a == b else {prefix[:-1]}


def test_timings_outside_hashes(run_a, run_c):
    fa, fc = _fits(run_a[0]), _fits(run_c[0])
    assert set(fa) == set(fc)
    for k in fa:
        ma, mc = fa[k]["meta"], fc[k]["meta"]
        assert fitmod.meta_hashable(ma) == fitmod.meta_hashable(mc), k
        assert _state_hashes(fa[k]["dir"]) == _state_hashes(fc[k]["dir"]), k
        for name, a in fa[k]["res"].items():
            assert a.tobytes() == fc[k]["res"][name].tobytes(), (k, name)
        diff = _diff_paths(ma, mc)
        allowed = {p for p in diff if p.split(".")[0] in EXCLUDED_META or p == "fit_info.fit_seconds"}
        assert diff == allowed, (k, sorted(diff - allowed))
        assert "created_by.run_id" in diff
    m = copy.deepcopy(next(iter(fa.values()))["meta"])
    h = fitmod.meta_hashable(m)
    m["fit_info"]["fit_seconds"] = (m["fit_info"]["fit_seconds"] or 0.0) + 1000.0
    m["timings"] = {k: v + 1.0 for k, v in m["timings"].items()}
    assert fitmod.meta_hashable(m) == h


# ------------------------------------------------------------------ 7. dtype e convenzioni
def test_dtypes_and_conventions(run_a):
    for f in _fits(run_a[0]).values():
        m, name = f["meta"], f["meta"]["model"]["name"]
        assert set(m["residuals"]) == set(f["res"])
        for rname, a in f["res"].items():
            assert a.dtype.str == m["residuals"][rname]["dtype"] == D66[name], (name, rname)
            assert rc.array_record(a)["sha256"] == m["residuals"][rname]["sha256"]
        assert not any(r.startswith("es__") for r in f["res"])
        assert m["segments"]["es_is_cal"] is True and m["loaded_with_version_mismatch"] is False
        if name == "rolling_zscore":
            assert set(f["res"]) == {"cal__last", "test__last"}
            assert m["model"]["native_only"] is True
            assert m["model"]["table_convention"] == {"window_time_agg": "none", "sensor_agg": "native"}
        else:
            assert m["model"]["native_only"] is False and m["model"]["table_convention"] is None
        rj = json.loads((f["dir"] / "residuals.json").read_text(encoding="utf-8"))
        assert rj["origin"] == "fit" and rj["loaded_with_version_mismatch"] is False
        assert rj["environment_signature"] == m["provenance"]["environment_signature"]


def test_meta_contents(run_a):
    """Campi di §3 del piano F5 (D-70)."""
    from src.models.base import CHECKPOINT_JSON, checkpoint_hashable_meta
    for f in _fits(run_a[0]).values():
        m = f["meta"]
        assert m["status"] == "complete" and m["meta_format"] == 1
        assert m["fit_key"] == f["dir"].name == cfgmod.canonical_hash(m["fit_payload"])[:cfgmod.FIT_KEY_LEN]
        assert m["fit_code_version"] == m["fit_payload"]["fit_code_version"]
        assert m["data_version"] == m["fit_payload"]["data_version"] and m["group"] == "9"
        for k in ("name", "class_path", "config_id", "kwargs", "deterministic", "native_only", "residual_aggs"):
            assert k in m["model"], k
        assert set(m["seed"]) == {"key_seed", "set_seed_value"}
        assert set(m["segments"]) >= {"mode", "fit", "es", "cal", "test", "es_is_cal"}
        w = m["windows"]
        assert (w["window_size"], w["step"], w["train_step"], w["es_loss_factor"]) == (20, 1, 5, 5)
        assert set(w["fit_inputs"]) == {"X", "X_val"} and "strides" in w["fit_inputs"]["X"]
        ck = json.loads((f["dir"] / "checkpoint" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
        assert m["checkpoint"]["hashable_sha256"] == cfgmod.canonical_hash(checkpoint_hashable_meta(ck))
        with np.load(f["dir"] / "scaler.npz", allow_pickle=False) as z:
            assert {k: rc.array_record(z[k])["sha256"] for k in z.files} == m["scaler"]["arrays"]
        assert m["device"] == {"requested": "cpu", "effective": "cpu"}
        assert set(m["timings"]) >= {"fit_seconds", "residual_seconds", "total_seconds"}
        assert set(m["created_by"]) == {"experiment", "kind", "run_id"}


# ------------------------------------------------------------------ 8. lock
def test_lock_held_during_run_and_released(env, monkeypatch):
    cfg = _cfg(env, "store_lock", run_models=("pca", "input_norm"), seeds=(42,))
    store = Store(cfg.store_root)
    seen = []
    real = fitmod.fit_one

    def spy(*a, **k):
        seen.append(store.read_lock())
        return real(*a, **k)

    monkeypatch.setattr(fitmod, "fit_one", spy)
    report = _run(env, cfg)
    assert seen and all(s is not None and s["run_id"] == report.run_id for s in seen)
    assert store.read_lock() is None and not store.lock_path.exists()
    assert {r["run_id"] for r in store.read_registry()} == {report.run_id}


def test_run_refused_when_locked_without_touching_store(env):
    cfg = _cfg(env, "store_locked", run_models=("pca",), seeds=(42,))
    store = Store(cfg.store_root)
    with store.lock("altro-run"):
        before = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*"))
        with pytest.raises(StoreLocked):
            _run(env, cfg)
        after = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*"))
        assert after == before == [".lock"]
        assert store.read_lock()["run_id"] == "altro-run"


# ------------------------------------------------------------------ 9. fit_key e device
def test_fit_key_uses_requested_device_not_effective(env):
    """Comportamento attuale documentato (nota su D-70): nel fit_key entra cfg.device (richiesto); il device
    effettivo, scelto dal trainer, non entra. Oggi coincidono: schema.DEVICES = ("cpu",)."""
    from src.utils import schema
    cfg = _cfg(env, "store_dev")
    assert schema.DEVICES == ("cpu",)
    for name in CORE:
        p = cfgmod.fit_payload(cfg, "machine-9-1", name, 42, "dv1-x")
        assert p["device"] == cfg.device == "cpu"
        assert set(p) == {"fit_code_version", "data_version", "dataset", "machine", "split", "preprocessing",
                          "windows", "model", "residual_aggs", "device"} | \
            ({"training"} if schema.MODEL_SCHEMAS[name].trained else set()) | \
            (set() if registry.get(name).deterministic else {"seed"})


@pytest.mark.parametrize("name", ["conv_autoencoder", "pca"])
def test_device_mismatch_is_refused_before_write(env, monkeypatch, name):
    cfg = _cfg(env, f"store_devmis_{name}", run_models=(name,), seeds=(42,))
    monkeypatch.setattr(fitmod, "effective_device", lambda model: "cuda")
    with pytest.raises(fitmod.DeviceMismatch, match="cuda"):
        _run(env, cfg)
    fits = Path(cfg.store_root) / "fits"
    assert not fits.exists() or not any(fits.iterdir())


def test_effective_device_of_models(env, run_a):
    """Device effettivo: quello del trainer per gli AE (registrato nel checkpoint), cpu per gli altri."""
    for f in _fits(run_a[0]).values():
        ck = json.loads((f["dir"] / "checkpoint" / "checkpoint.json").read_text(encoding="utf-8"))
        if "device" in ck["state"]:
            assert ck["state"]["device"] == f["meta"]["device"]["effective"] == "cpu"


# ------------------------------------------------------------------ varie
def test_official_kind_refused(env):
    cfg = dataclasses.replace(_cfg(env, "store_off", run_models=("pca",), seeds=(42,)), kind="official")
    with pytest.raises(fitmod.FitRefused, match="F10"):
        _run(env, cfg)
    assert not Path(cfg.store_root).exists()


def test_fit_summary_and_report(run_a):
    _, report, seconds = run_a
    assert [r["outcome"] for r in report.rows] == ["fitted"] * 16
    lines = (report.run_dir / "fit_summary.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[:5] == ["machine", "model", "seed", "set_seed_value", "fit_key"]
    assert len(lines) == 17
    print(f"\n[F5] run_fit sintetico completo (16 fit): {seconds:.2f} s")
    assert seconds < 60


def test_no_store_in_repository():
    assert not (ROOT / "store").exists()
