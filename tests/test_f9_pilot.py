"""
F9 passo 3: esperimenti del pilota sui tempi e stima del costo con la regola K = 8 / 6 (D-41).

  * config del pilota (``configs/experiments/tuning_pilot/slot_2..8.yaml``): tre famiglie addestrate, seed [42],
    max_epochs / patience 3 / 3, store separato escluso da git, 6 macchine dev; stesso ``fit_payload`` degli slot di
    tuning salvo seed ed epoche, quindi nessun ``fit_key`` in comune;
  * formula di D-41 verificata a mano su ``meta.json`` sintetici (mediane, non medie; residui di V1, non del pilota;
    ``total_seconds`` di V1 per i deterministici; slot 1 escluso; stima a K = 6 solo sugli slot 2..6);
  * confine: 40 h esatte → K = 8, appena sopra → K = 6;
  * prove di difetto: fit mancanti, falliti, con epoche o architettura diverse, riferimento mancante.
Nessun dato reale e nessuno store reale: i ``meta.json`` sono costruiti in memoria dai compiti delle config.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline.fit_plan import plan_tasks
from src.tuning import grid as tg
from src.tuning import pilot as tp
from src.utils import schema
from src.utils.config import load_experiment, load_yaml_strict
from src.utils.schema import ConfigError

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs" / "tuning" / "grid.yaml"
TRAINED = ("conv_autoencoder", "lstm_autoencoder", "lstm_forecaster")
DET = ("pca", "pca_pointwise", "rolling_zscore")
K, K_RED = 8, 6
N_MACHINES = 6


@pytest.fixture(scope="module")
def grid():
    return tg.load_grid(GRID)


@pytest.fixture(scope="module")
def cfgs(grid):
    tdir, pdir = tg.DEFAULT_OUT, tg.DEFAULT_PILOT_OUT
    return {"ref": load_experiment(tdir / "slot_1.yaml"),
            "targets": {s: load_experiment(tdir / tg.SLOT_FILE.format(s)) for s in grid.pilot_slots()},
            "pilots": {s: load_experiment(pdir / tg.SLOT_FILE.format(s)) for s in grid.pilot_slots()}}


@pytest.fixture(scope="module")
def dvs(cfgs):
    return {m: f"dv-{m}" for m in cfgs["ref"].data.machines}


# ------------------------------------------------------------------ config del pilota
def test_pilot_experiments_as_planned(grid, cfgs):
    dev = load_yaml_strict(ROOT / "configs" / "dev_machines.yaml")["dev"]
    assert grid.pilot_slots() == tuple(range(2, K + 1)) and sorted(cfgs["pilots"]) == list(range(2, K + 1))
    for s, c in cfgs["pilots"].items():
        assert c.name == f"tuning_pilot_slot_{s}" and c.kind == "exploratory"
        assert c.run_models == TRAINED and c.seeds == (42,)
        assert (c.training.max_epochs, c.training.patience) == (3, 3)
        assert c.store_root == "store_pilot" and c.data.machines == tuple(dev)
        t = cfgs["targets"][s]
        assert {m: dict(c.models[m].params) for m in TRAINED} == {m: dict(t.models[m].params) for m in TRAINED}
        assert (c.split, c.preprocessing, c.windows) == (t.split, t.preprocessing, t.windows)
        assert (c.training.min_delta, c.training.learning_rate, c.training.weight_decay, c.training.batch_size) == \
            (t.training.min_delta, t.training.learning_rate, t.training.weight_decay, t.training.batch_size)


def test_pilot_payload_equals_tuning_except_seed_and_epochs_and_keys_are_new(cfgs, dvs):
    machines = cfgs["ref"].data.machines
    for s, c in cfgs["pilots"].items():
        pil = {(t.model, t.machine): t for t in plan_tasks(c, machines, dvs)}
        tgt = [t for t in plan_tasks(cfgs["targets"][s], machines, dvs) if t.model in TRAINED]
        assert len(pil) == len(TRAINED) * N_MACHINES and len(tgt) == 3 * len(pil)
        for t in tgt:
            p = pil[(t.model, t.machine)]
            assert tp._comparable(p.payload) == tp._comparable(t.payload)
            assert p.payload["training"]["max_epochs"] == 3 and t.payload["training"]["max_epochs"] == 100
            assert p.fit_key != t.fit_key
    keys = [t.fit_key for c in cfgs["pilots"].values() for t in plan_tasks(c, machines, dvs)]
    assert len(keys) == len(set(keys)) == 7 * 3 * N_MACHINES          # 126 fit nel pilota


def test_pilot_files_equal_generator_output():
    files = tg.generate_pilot(GRID, tg.DEFAULT_BASE, tg.DEFAULT_DEV)
    assert sorted(files) == sorted(tg.SLOT_FILE.format(s) for s in range(2, K + 1))
    for n, text in files.items():
        assert (tg.DEFAULT_PILOT_OUT / n).read_text(encoding="utf-8") == text
    assert sorted(p.name for p in tg.DEFAULT_PILOT_OUT.glob("slot_*.yaml")) == sorted(files)


def test_pilot_store_is_ignored_by_git():
    r = subprocess.run(["git", "check-ignore", "-q", "store_pilot/fits/x/meta.json"], cwd=ROOT)
    assert r.returncode == 0


@pytest.mark.parametrize("path,value,match", [
    (["pilot", "families"], ["conv_autoencoder", "lstm_autoencoder"], "famiglie addestrate"),
    (["pilot", "families"], ["conv_autoencoder", "lstm_autoencoder", "lstm_forecaster", "pca"], "famiglie addestrate"),
    (["pilot", "seeds"], [7], "seed distinti"),
    (["pilot", "store_root"], "store", "store separato"),
    (["pilot", "experiment_prefix"], "tuning_slot", "diverso"),
    (["pilot", "threshold_hours"], 0, "fuori intervallo"),
    (["pilot", "max_epochs"], 0, ">= 1"),
])
def test_defect_pilot_block_is_validated(path, value, match):
    raw = load_yaml_strict(GRID)
    raw[path[0]][path[1]] = value
    with pytest.raises(ConfigError, match=match):
        tg.validate_grid(raw)


# ------------------------------------------------------------------ meta.json sintetici
REF_STOP = {"conv_autoencoder": [10, 20, 30, 1000] * 4 + [20, 20],     # mediana 20 (la media sarebbe ~236)
            "lstm_autoencoder": [50] * 18, "lstm_forecaster": [100] * 9 + [60] * 9}   # mediana 80
REF_RES = {"conv_autoencoder": [4.0] * 17 + [400.0], "lstm_autoencoder": [12.0] * 18,
           "lstm_forecaster": [3.0] * 18}
DET_TOTAL = {"pca": 13.0, "pca_pointwise": 5.0, "rolling_zscore": 1.0}
PILOT_FIT = {"conv_autoencoder": 3.0, "lstm_autoencoder": 9.0, "lstm_forecaster": 6.0}   # secondi su 3 epoche


def _meta(task, *, fit=0.0, res=0.0, total=0.0, stop=None, status="complete", payload=None):
    return {"fit_key": task.fit_key, "status": status, "fit_payload": copy.deepcopy(payload or task.payload),
            "timings": {"fit_seconds": fit, "residual_seconds": res, "total_seconds": total},
            "fit_info": {"stopped_epoch": stop}}


def _inputs(cfgs, dvs, slot_factor=lambda s: 1.0):
    machines = cfgs["ref"].data.machines
    ref, count = [], {}
    for t in plan_tasks(cfgs["ref"], machines, dvs):
        if t.model in TRAINED:
            i = count.setdefault(t.model, 0)
            count[t.model] += 1
            ref.append((t, _meta(t, stop=REF_STOP[t.model][i], res=REF_RES[t.model][i])))
        else:
            ref.append((t, _meta(t, total=DET_TOTAL.get(t.model, 99.0))))
    pilot = {s: [(t, _meta(t, fit=PILOT_FIT[t.model] * slot_factor(s), stop=3, res=1e6))
                 for t in plan_tasks(c, machines, dvs)] for s, c in cfgs["pilots"].items()}
    targets = {s: plan_tasks(c, machines, dvs) for s, c in cfgs["targets"].items()}
    return ref, pilot, targets


def _est(ref, pilot, targets, threshold=40.0):
    return tp.estimate(reference=ref, pilot=pilot, targets=targets, pilot_max_epochs=3, pilot_patience=3,
                       threshold_hours=threshold, k=K, k_reduced=K_RED)


# per fit: secondi per epoca del pilota × mediana stopped_epoch di V1 + mediana residual_seconds di V1
PER_FIT = {"conv_autoencoder": 1.0 * 20 + 4.0, "lstm_autoencoder": 3.0 * 50 + 12.0,
           "lstm_forecaster": 2.0 * 80 + 3.0}


def test_formula_of_d41_by_hand(cfgs, dvs):
    res = _est(*_inputs(cfgs, dvs))
    per_slot = N_MACHINES * (3 * sum(PER_FIT.values()) + sum(DET_TOTAL.values()))
    assert res["n_fits"] == 7 * N_MACHINES * (3 * 3 + 3)                  # 504: slot 1 escluso (riuso di V1)
    assert {r["slot"] for r in res["rows"]} == set(range(2, K + 1))
    assert res["estimate_hours_k"] == pytest.approx(7 * per_slot / 3600)
    assert res["estimate_hours_k_reduced"] == pytest.approx(5 * per_slot / 3600)     # slot 2..6
    for m, v in PER_FIT.items():
        assert res["per_model"][m]["seconds"] == pytest.approx(7 * N_MACHINES * 3 * v)
    assert res["per_model"]["conv_autoencoder"]["v1_median_stopped_epoch"] == 20             # mediana, non media
    assert res["per_model"]["conv_autoencoder"]["v1_median_residual_seconds"] == 4.0
    assert res["per_model"]["pca"]["seconds"] == pytest.approx(7 * N_MACHINES * 13.0)       # total_seconds di V1
    assert sum(res["per_slot_hours"].values()) == pytest.approx(res["estimate_hours_k"])
    assert res["chosen_k"] == K


def test_boundary_exactly_40_hours_keeps_k_8(cfgs, dvs):
    ref, pilot, targets = _inputs(cfgs, dvs)
    hours = _est(ref, pilot, targets)["estimate_hours_k"]
    assert _est(ref, pilot, targets, threshold=hours)["chosen_k"] == K                # uguale: K = 8
    assert _est(ref, pilot, targets, threshold=hours * (1 - 1e-12))["chosen_k"] == K_RED   # appena sopra: K = 6


def test_rule_with_the_40_hour_threshold(cfgs, dvs):
    # fattore per slot che porta la stima oltre 40 h (secondi per epoca 60 volte quelli sopra)
    res = _est(*_inputs(cfgs, dvs, slot_factor=lambda s: 60.0))
    assert res["estimate_hours_k"] > 40 and res["chosen_k"] == K_RED
    assert res["threshold_hours"] == 40.0


def test_run_estimate_through_the_configs_reads_only_meta(grid, cfgs, dvs):
    ref, pilot, _ = _inputs(cfgs, dvs)
    ref_by = {t.fit_key: m for t, m in ref}
    pil_by = {t.fit_key: m for items in pilot.values() for t, m in items}
    read = []
    def reader(d):
        return lambda key: read.append(key) or d.get(key)
    res = tp.run_estimate(grid, cfgs["ref"], cfgs["pilots"], cfgs["targets"], dvs, reader(ref_by), reader(pil_by))
    assert res["chosen_k"] == K and res["n_fits"] == 504
    assert sorted(read) == sorted(list(ref_by) + list(pil_by))        # solo i meta.json di V1 e del pilota


# ------------------------------------------------------------------ prove di difetto
def _drop(items, model, machine):
    return [(t, m) for t, m in items if not (t.model == model and t.machine == machine)]


def test_defect_missing_or_failed_pilot_fit(cfgs, dvs):
    ref, pilot, targets = _inputs(cfgs, dvs)
    m0 = cfgs["ref"].data.machines[0]
    bad = {**pilot, 5: _drop(pilot[5], "lstm_autoencoder", m0)}
    with pytest.raises(tp.EstimateError, match="manca il fit di lstm_autoencoder"):
        _est(ref, bad, targets)
    bad = {**pilot, 4: [(t, None if t.model == "conv_autoencoder" else m) for t, m in pilot[4]]}
    with pytest.raises(tp.EstimateError, match="fit mancante"):
        _est(ref, bad, targets)
    bad = {**pilot, 3: [(t, {**m, "status": "failed"}) for t, m in pilot[3]]}
    with pytest.raises(tp.EstimateError, match="'failed'"):
        _est(ref, bad, targets)


def test_defect_pilot_with_other_epochs_or_architecture(cfgs, dvs):
    ref, pilot, targets = _inputs(cfgs, dvs)
    def with_payload(items, fn):
        out = []
        for t, m in items:
            m = copy.deepcopy(m)
            fn(m["fit_payload"])
            out.append((t, m))
        return out
    bad = {**pilot, 2: with_payload(pilot[2], lambda p: p["training"].__setitem__("max_epochs", 100))}
    with pytest.raises(tp.EstimateError, match="max_epochs / patience"):
        _est(ref, bad, targets)
    bad = {**pilot, 6: with_payload(pilot[6], lambda p: p["model"]["params"].__setitem__("hidden_dim", 999)
                                    if "hidden_dim" in p["model"]["params"] else None)}
    with pytest.raises(tp.EstimateError, match="stesso fit_payload"):
        _est(ref, bad, targets)
    swapped = {**pilot, 7: pilot[8], 8: pilot[7]}                      # pilota di uno slot attribuito a un altro
    with pytest.raises(tp.EstimateError, match="stesso fit_payload"):
        _est(ref, swapped, targets)


def test_defect_reference_missing_failed_or_without_epochs(cfgs, dvs):
    ref, pilot, targets = _inputs(cfgs, dvs)
    m0 = cfgs["ref"].data.machines[0]
    with pytest.raises(tp.EstimateError, match="manca il fit di pca_pointwise"):
        _est(_drop(ref, "pca_pointwise", m0), pilot, targets)
    with pytest.raises(tp.EstimateError, match="nessun fit di lstm_forecaster"):
        _est([(t, m) for t, m in ref if t.model != "lstm_forecaster"], pilot, targets)
    with pytest.raises(tp.EstimateError, match="'failed'"):
        _est([(t, {**m, "status": "failed"} if t.model == "pca" else m) for t, m in ref], pilot, targets)
    no_ep = [(t, {**m, "fit_info": {"stopped_epoch": 0}} if t.model == "conv_autoencoder" else m) for t, m in ref]
    with pytest.raises(tp.EstimateError, match="stopped_epoch"):
        _est(no_ep, pilot, targets)


def test_defect_wrong_slots(cfgs, dvs):
    ref, pilot, targets = _inputs(cfgs, dvs)
    with pytest.raises(tp.EstimateError, match="attesi 2..8"):
        _est(ref, pilot, {s: v for s, v in targets.items() if s != 8})
    with pytest.raises(tp.EstimateError, match="attesi 2..8"):
        _est(ref, pilot, {1: targets[2], **targets})


def _script(*args):
    """Lo script in un sottoprocesso, con stdout e stderr in UTF-8 in entrambi i versi (senza, su Windows la codifica
    del figlio dipende dall'ambiente e «già» arriva come «giÃ »)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "f9_pilot_estimate.py"), *map(str, args)],
                          capture_output=True, encoding="utf-8", env=env)


def test_script_refuses_existing_out_and_stale_configs(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    r = _script("--out", out)
    assert r.returncode == 2 and "esiste già" in r.stderr, (r.returncode, r.stdout, r.stderr)
    r = _script("--out", tmp_path / "new", "--pilot-dir", tmp_path / "vuota")
    assert r.returncode == 1 and "non uguali alla griglia" in r.stderr, (r.returncode, r.stdout, r.stderr)
    assert not (tmp_path / "new").exists()
