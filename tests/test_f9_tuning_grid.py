"""
F9 passo 2: griglia di tuning, esperimenti per slot e piano di analisi del tuning (D-41, D-57, D-74, D-08).

Test «budget pari per costruzione» (piano F9, sezione Test), calcolato dagli esperimenti generati e non solo dalla
griglia:
  * stesso K = 8 per le 6 famiglie tarate, ``input_norm`` K = 1 (solo slot 1), AE non addestrati assenti;
  * la regola K = 8 / 6 dà esattamente gli slot 1–6 per tutte le famiglie;
  * slot 1 = V1 di F8, anche nel ``fit_payload`` (stessi ``fit_key``: riuso dei fit di V1);
  * livelli di scoring per famiglia dal piano di tuning: 14, 7 per ``lstm_forecaster`` (D-74), 1 per
    ``rolling_zscore`` (native), con le stesse regole di ``analyze`` e su un'analisi sintetica vera;
  * file generati uguali all'uscita del generatore; solo macchine dev.
Le prove di difetto (``test_defect_*``) mostrano che ogni controllo fallisce su una griglia o un esperimento difettosi.
Nessun dato reale: le config si leggono, i fit e l'analisi sintetici girano in ``tmp_path`` (D-16).
"""

import copy
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline.fit_plan import plan_tasks
from src.tuning import grid as tg
from src.utils import schema
from src.utils.analysis_plan import load_analysis_plan
from src.utils.config import canonical_json, load_experiment, load_yaml_strict
from src.utils.schema import ConfigError

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "configs" / "tuning" / "grid.yaml"
PLAN = ROOT / "configs" / "analysis" / "tuning.yaml"
SLOTS_DIR = ROOT / "configs" / "experiments" / "tuning"
V1 = ROOT / "configs" / "experiments" / "protocol_switch_dev" / "v1_canonical.yaml"
BASE = ROOT / "configs" / "base.yaml"
DEV_FILE = ROOT / "configs" / "dev_machines.yaml"

TUNED = ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "lstm_forecaster", "rolling_zscore")
UNTUNED = ("input_norm",)
UNTRAINED = tuple(n for n, s in schema.MODEL_SCHEMAS.items() if s.untrained_of is not None)
K, K_REDUCED = 8, 6
EXPECTED_SCORING_LEVELS = {"conv_autoencoder": 14, "lstm_autoencoder": 14, "pca": 14, "pca_pointwise": 14,
                           "input_norm": 14, "lstm_forecaster": 7, "rolling_zscore": 1}

# Trascrizione indipendente delle griglie di D-41 (docs/decisioni.md), in ordine di slot
_FR = (0.9, 0.5, 0.7, 0.8, 0.95, 0.99, 0.6, 0.97)
D41 = {
    "pca": [{"n_components": f, "random_state": 42} for f in _FR],
    "pca_pointwise": [{"n_components": f} for f in _FR],
    "conv_autoencoder": [{"channels": c, "latent_dim": d} for c, d in [
        ([32, 64], 16), ([16, 32], 8), ([16, 32], 16), ([16, 32], 32), ([32, 64], 8), ([32, 64], 32), ([16, 32], 4),
        ([32, 64], 4)]],
    "lstm_autoencoder": [{"hidden_dim": h, "latent_dim": d} for h, d in [
        (64, 16), (32, 8), (32, 16), (32, 32), (64, 8), (64, 32), (32, 4), (64, 4)]],
    "lstm_forecaster": [{"hidden_dim": h} for h in (64, 8, 16, 32, 48, 128, 24, 96)],
    "rolling_zscore": [{"top_k": t, "eps": e} for t, e in [
        (5, 0.01), (1, 0.01), (1, 0.1), (5, 0.1), (10, 0.01), (10, 0.1), (3, 0.01), (3, 0.1)]],
}
D41_SCORINGS = [("mean", "none"), ("mean", "std"), ("max", "std"), ("top5", "std"), ("mean", "robust"),
                ("max", "robust"), ("top5", "robust")]


def _plain(p):
    return {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()}


def _slot_path(s):
    return SLOTS_DIR / tg.SLOT_FILE.format(s)


@pytest.fixture(scope="module")
def grid():
    return tg.load_grid(GRID)


@pytest.fixture(scope="module")
def slots():
    """Esperimenti generati e committati, caricati come li carica la CLI (base + modelli + esperimento)."""
    return {s: load_experiment(_slot_path(s)) for s in range(1, K + 1)}


# ------------------------------------------------------------------ budget dagli esperimenti generati
def budget_of(cfgs):
    """{modello: configurazioni di iperparametri distinte} sugli esperimenti dati."""
    out = {}
    for cfg in cfgs:
        for name in cfg.run_models:
            out.setdefault(name, set()).add(canonical_json(_plain(cfg.models[name].params)))
    return {m: len(v) for m, v in out.items()}


def check_even_budget(cfgs, k):
    """Errori del budget (lista vuota = budget pari): K = k per le famiglie tarate, 1 per input_norm, nessun AE non
    addestrato, ogni famiglia tarata in ogni slot."""
    errs = []
    b = budget_of(cfgs)
    for m in TUNED:
        if b.get(m) != k:
            errs.append(f"{m}: {b.get(m)} configurazioni, attese {k}")
        missing = [cfg.name for cfg in cfgs if m not in cfg.run_models]
        if missing:
            errs.append(f"{m}: assente da {missing}")
    for m in UNTUNED:
        if b.get(m) != 1:
            errs.append(f"{m}: {b.get(m)} configurazioni, attesa 1")
    for m in sorted(set(b) & set(UNTRAINED)):
        errs.append(f"{m}: AE non addestrato in un esperimento di tuning (D-41, D-69)")
    for m in sorted(set(b) - set(TUNED) - set(UNTUNED) - set(UNTRAINED)):
        errs.append(f"{m}: modello non previsto da D-41")
    return errs


def test_budget_is_even_by_construction(slots):
    assert check_even_budget(list(slots.values()), K) == []


def test_k_reduced_rule_gives_exactly_slots_1_to_6_for_every_family(grid, slots):
    assert grid.slots(reduced=True) == tuple(range(1, K_REDUCED + 1))
    reduced = [slots[s] for s in grid.slots(reduced=True)]
    assert check_even_budget(reduced, K_REDUCED) == []
    for m in TUNED:                                     # i primi 6 punti di ogni famiglia, nell'ordine
        assert [_plain(c.models[m].params) for c in reduced] == D41[m][:K_REDUCED]
    assert grid.budget(reduced=True) == {**{m: K_REDUCED for m in TUNED}, "input_norm": 1}
    assert grid.budget() == {**{m: K for m in TUNED}, "input_norm": 1}


def test_grid_is_the_d41_grid_in_slot_order(grid):
    assert (grid.k, grid.k_reduced, grid.seeds, grid.machines) == (K, K_REDUCED, (42, 43, 44), "dev")
    assert tuple(grid.families) == TUNED and tuple(grid.untuned) == UNTUNED
    for m in TUNED:
        assert [_plain(p) for p in grid.families[m]] == D41[m], m
    tb = grid.tie_break
    assert tb["order"] == ("complexity", "slot", "scoring")
    assert tb["complexity"] == {"n_features": 38, "window_size": 100, "by_family": {
        "conv_autoencoder": "trainable_params", "lstm_autoencoder": "trainable_params",
        "lstm_forecaster": "trainable_params", "pca": "n_components", "pca_pointwise": "n_components"}}
    assert tb["window_time_aggs"] == ("last", "mean") and list(tb["scorings"]) == D41_SCORINGS


def test_slot_k_contains_the_kth_point_of_every_family(slots):
    for s, cfg in slots.items():
        assert cfg.name == f"tuning_slot_{s}" and cfg.kind == "exploratory" and cfg.seeds == (42, 43, 44)
        assert cfg.run_models == TUNED + (UNTUNED if s == 1 else ())
        for m in TUNED:
            assert _plain(cfg.models[m].params) == D41[m][s - 1], (s, m)
        assert all(isinstance(cfg.models[m].params["n_components"], float) for m in ("pca", "pca_pointwise"))


def test_only_the_models_differ_between_slots_and_the_rest_is_canonical(slots):
    v1 = load_experiment(V1)
    def rest(cfg):
        d = dataclasses.asdict(cfg)
        for k in ("name", "models", "run_models"):
            d.pop(k)
        return d
    for cfg in slots.values():
        assert rest(cfg) == rest(v1)                    # stesso protocollo, seed, macchine e store di V1 di F8
    base = load_yaml_strict(BASE)
    raw1 = load_yaml_strict(_slot_path(1))
    for b in tg.BASE_BLOCKS:                            # valori del protocollo canonico, scritti per esteso
        want = base[b] if b != "data" else {k: base[b][k] for k in tg.BASE_DATA_KEYS}
        got = raw1[b] if b != "data" else {k: raw1[b][k] for k in tg.BASE_DATA_KEYS}
        assert got == want, b


def test_slot_1_is_v1_of_f8_also_in_the_fit_payload(slots):
    v1 = load_experiment(V1)
    dvs = {m: f"dv-{m}" for m in v1.data.machines}      # data_version fittizia, uguale per i due esperimenti
    t1 = {(t.machine, t.model, t.seed): t for t in plan_tasks(slots[1], slots[1].data.machines, dvs)}
    tv = {(t.machine, t.model, t.seed): t for t in plan_tasks(v1, v1.data.machines, dvs)}
    assert set(t1) <= set(tv)
    assert {m for _, m, _ in set(tv) - set(t1)} == set(UNTRAINED)   # V1 ha in più solo gli AE non addestrati
    for key, t in t1.items():
        assert t.payload == tv[key].payload and t.fit_key == tv[key].fit_key, key
    assert len(t1) == 6 * (4 + 3 * 3)                   # 78 fit, come `status` sullo store di V1


def test_slots_2_to_8_have_only_new_fit_keys(slots):
    dvs = {m: f"dv-{m}" for m in slots[1].data.machines}
    keys = [t.fit_key for s in slots for t in plan_tasks(slots[s], slots[s].data.machines, dvs)]
    assert len(keys) == len(set(keys))                  # nessun fit condiviso tra slot
    assert len(keys) == 78 + 7 * 6 * (3 + 3 * 3)


def test_only_dev_machines(slots):
    dev = load_yaml_strict(DEV_FILE)
    for cfg in slots.values():
        assert cfg.data.machines == tuple(dev["dev"])
        assert not set(cfg.data.machines) & set(dev["eval"])


def test_untrained_aes_are_absent_and_rejected(grid, slots):
    for cfg in slots.values():
        assert not set(cfg.run_models) & set(UNTRAINED)
    assert not set(grid.families) & set(UNTRAINED) and not set(grid.untuned) & set(UNTRAINED)


# ------------------------------------------------------------------ file generati
def test_generated_files_equal_generator_output():
    files = tg.generate(GRID, BASE, DEV_FILE)
    assert sorted(files) == sorted(tg.SLOT_FILE.format(s) for s in range(1, K + 1))
    for name, text in files.items():
        assert (SLOTS_DIR / name).read_text(encoding="utf-8") == text, name
    assert sorted(p.name for p in SLOTS_DIR.glob("slot_*.yaml")) == sorted(files)


def test_script_check_mode(tmp_path):
    script = ROOT / "scripts" / "make_tuning_experiments.py"
    run = lambda *a: subprocess.run([sys.executable, str(script), *a], capture_output=True, text=True)  # noqa: E731
    assert run("--check").returncode == 0
    out = tmp_path / "tuning"
    assert run("--check", "--out", str(out)).returncode == 1 and not out.exists()     # --check non scrive
    assert run("--out", str(out)).returncode == 0
    assert run("--check", "--out", str(out)).returncode == 0
    (out / "slot_3.yaml").write_text("# modificato a mano\n", encoding="utf-8")
    assert run("--check", "--out", str(out)).returncode == 1
    (out / "slot_9.yaml").write_text("x: 1\n", encoding="utf-8")
    r = run("--out", str(out))
    assert r.returncode == 1 and "slot_9.yaml" in r.stderr
    assert (out / "slot_3.yaml").read_text(encoding="utf-8") == "# modificato a mano\n"   # nulla scritto


# ------------------------------------------------------------------ piano di analisi del tuning
def test_tuning_plan_matches_d41(grid):
    plan = load_analysis_plan(PLAN)
    assert plan.scenarios == ("primary",) and plan.ts_aggs == ("last",)
    assert set(plan.window_time_aggs) == set(grid.tie_break["window_time_aggs"]) == {"last", "mean"}
    assert set(plan.scorings) == set(grid.tie_break["scorings"]) == set(D41_SCORINGS)
    assert plan.roll_quantile == () and plan.roll_robust_z == () and plan.oracle is False and plan.pa == ()
    assert plan.static_pct == (98.0,)                   # requisito tecnico (thr non vuota), non usato dalla selezione
    assert plan.legacy_export is None


def test_scoring_levels_per_family_from_the_plan():
    plan = load_analysis_plan(PLAN)
    got = {m: len(tg.scoring_combos(plan, m)) for m in TUNED + UNTUNED}
    assert got == EXPECTED_SCORING_LEVELS
    assert {w for w, _, _ in tg.scoring_combos(plan, "lstm_forecaster")} == {"last"}      # D-74 b


def test_tuning_plan_runs_on_synthetic_fits_with_the_expected_levels(tmp_path):
    """Il piano di tuning gira davvero in ``analyze`` (macchina dev sintetica, split3) e i livelli nella tabella free
    coincidono con ``scoring_combos``."""
    import pandas as pd
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from tests.fixtures.datasets import SYNTH_DEV, write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo

    paths = write_synth_dataset(tmp_path / "synth")
    repo = make_repo(tmp_path / "repo")
    models = ("pca", "input_norm", "rolling_zscore", "lstm_forecaster")
    spec = synth_spec(str(paths["data"]), str(tmp_path / "store"), mode="split3", seeds=(42,), run_models=models)
    spec["data"]["machines"] = [SYNTH_DEV[0]]
    spec["models"]["lstm_forecaster"] = {"hidden_dim": 4}
    spec["training"]["max_epochs"], spec["training"]["patience"] = 1, 1
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo)
    an.run_analysis(spec, PLAN, datasets_dir=paths["spec"].parent, repo_root=repo)
    free = pd.concat(pd.read_csv(p) for p in sorted((tmp_path / "store").rglob("free.csv.gz")))
    plan = load_analysis_plan(PLAN)
    for m in models:
        rows = free[free["model"] == m]
        combos = set(zip(rows["window_time_agg"], rows["sensor_agg"], rows["normalizer"]))
        assert combos == set(tg.scoring_combos(plan, m)), m
        assert set(rows["scenario"]) == {"primary"} and set(rows["ts_agg"]) == {"last"}
        assert rows["metric"].eq("ap_norm").sum() == len(tg.scoring_combos(plan, m))


# ------------------------------------------------------------------ prove di difetto
def test_defect_budget_check_catches_uneven_experiments(slots):
    cfgs = list(slots.values())
    def with_models(cfg, run_models, models):
        return dataclasses.replace(cfg, run_models=tuple(run_models), models=models)
    # famiglia assente da uno slot
    c8 = cfgs[7]
    drop = with_models(c8, [m for m in c8.run_models if m != "conv_autoencoder"],
                       {m: s for m, s in c8.models.items() if m != "conv_autoencoder"})
    assert any("conv_autoencoder" in e for e in check_even_budget(cfgs[:7] + [drop], K))
    # punto ripetuto in due slot (7 configurazioni invece di 8)
    rep = with_models(c8, c8.run_models, {**c8.models, "lstm_forecaster": cfgs[6].models["lstm_forecaster"]})
    assert any("lstm_forecaster: 7" in e for e in check_even_budget(cfgs[:7] + [rep], K))
    # input_norm in un secondo slot con un'altra configurazione non è possibile (nessun iperparametro): basta la sua
    # assenza dallo slot 1
    c1 = cfgs[0]
    no_in = with_models(c1, [m for m in c1.run_models if m != "input_norm"],
                        {m: s for m, s in c1.models.items() if m != "input_norm"})
    assert any("input_norm" in e for e in check_even_budget([no_in] + cfgs[1:], K))
    # AE non addestrato in uno slot
    v1 = load_experiment(V1)
    ua = with_models(c1, c1.run_models + ("conv_autoencoder_untrained",),
                     {**c1.models, "conv_autoencoder_untrained": v1.models["conv_autoencoder_untrained"]})
    assert any("non addestrato" in e for e in check_even_budget([ua] + cfgs[1:], K))
    # regola K = 6 applicata a un insieme sbagliato di slot
    assert check_even_budget(cfgs[:5], K_REDUCED) != []


def _raw_grid():
    return load_yaml_strict(GRID)


def _mut(path, fn):
    raw = _raw_grid()
    node = raw
    for p in path[:-1]:
        node = node[p]
    fn(node, path[-1])
    return raw


@pytest.mark.parametrize("raw,match", [
    (_mut(["families", "conv_autoencoder", "points"], lambda n, k: n[k].pop()), "esattamente k = 8"),
    (_mut(["families", "pca", "points"], lambda n, k: n[k].append({"n_components": 0.55})), "esattamente k = 8"),
    (_mut(["families", "lstm_forecaster", "points"], lambda n, k: n[k].__setitem__(7, {"hidden_dim": 64})),
     "uguale a quello dello slot 1"),
    (_mut(["families", "pca", "points"], lambda n, k: n[k].__setitem__(1, {"n_components": 5})), "D-57"),
    (_mut(["families", "pca_pointwise", "points"], lambda n, k: n[k].__setitem__(1, {"n_components": 5})),
     "atteso un float"),
    (_mut(["families", "pca"], lambda n, k: (n[k]["fixed"].__setitem__("random_state", "from_seed"),
                                             n[k]["points"].__setitem__(1, {"n_components": 5}))),
     "atteso un float"),                                # intero ammesso dallo schema con from_seed, non dalla griglia
    (_mut(["families"], lambda n, k: n[k].__setitem__(
        "conv_autoencoder_untrained", {"points": [{"channels": [32, 64], "latent_dim": 16,
                                                   "score_batch_size": 64}] * 8})), "non addestrato"),
    (_mut(["untuned"], lambda n, k: n[k].__setitem__("lstm_autoencoder_untrained", {})), "non addestrato"),
    (_mut(["untuned"], lambda n, k: n[k].__setitem__("rolling_zscore", {"top_k": 5, "eps": 0.01})),
     "già tra le famiglie"),
    (_mut(["k_reduced"], lambda n, k: n.__setitem__(k, 8)), "k_reduced"),
    (_mut(["machines"], lambda n, k: n.__setitem__(k, "eval")), "D-08"),
    (_mut(["tie_break", "order"], lambda n, k: n.__setitem__(k, ["slot", "complexity", "scoring"])), "D-41"),
    (_mut(["tie_break", "complexity", "by_family"], lambda n, k: n[k].__setitem__("rolling_zscore",
                                                                                  "trainable_params")),
     "solo per i modelli addestrati"),
    (_mut(["families", "pca", "points"], lambda n, k: n[k].__setitem__(0, {"n_components": 0.9,
                                                                            "random_state": 42})), "anche in fixed"),
    (_mut(["families", "conv_autoencoder", "points"], lambda n, k: n[k][2].__setitem__("dropout", 0.1)),
     "chiavi sconosciute"),
], ids=["k7", "k9", "duplicato", "pca_int", "pcap_int", "pca_int_from_seed", "untrained_fam", "untrained_untuned",
        "untuned_doppio", "k_reduced", "eval", "ordine", "complessita", "fixed", "chiave"])
def test_defect_grid_validator(raw, match):
    with pytest.raises(ConfigError, match=match):
        tg.validate_grid(raw)


def test_defect_family_removed_from_the_grid_is_caught_by_the_budget():
    """Una famiglia tolta lascia una griglia valida: la ferma il controllo del budget sugli esperimenti generati."""
    raw = _raw_grid()
    raw["families"].pop("lstm_autoencoder")
    raw["tie_break"]["complexity"]["by_family"].pop("lstm_autoencoder")
    raw["pilot"]["families"].remove("lstm_autoencoder")
    g = tg.validate_grid(raw)
    base, dev = load_yaml_strict(BASE), tg.load_dev_machines(DEV_FILE)
    cfgs = [schema.validate_experiment(tg.slot_experiment(g, s, base, dev)) for s in g.slots()]
    assert any("lstm_autoencoder" in e for e in check_even_budget(cfgs, K))


def test_defect_untuned_with_hyperparameters_is_rejected():
    raw = _raw_grid()
    fam = raw["families"].pop("rolling_zscore")
    raw["untuned"]["rolling_zscore"] = fam["points"][0]
    with pytest.raises(ConfigError, match="va tarato con k = 8"):
        tg.validate_grid(raw)


def test_defect_duplicate_yaml_key_and_window_size_mismatch(tmp_path, grid):
    bad = tmp_path / "grid.yaml"
    bad.write_text(GRID.read_text(encoding="utf-8").replace("k: 8 ", "k: 8\nk: 8 ", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicata"):
        tg.load_grid(bad)
    base = copy.deepcopy(load_yaml_strict(BASE))
    base["windows"]["window_size"] = 50
    with pytest.raises(ConfigError, match="window_size"):
        tg.slot_experiment(grid, 1, base, tg.load_dev_machines(DEV_FILE))
    base = copy.deepcopy(load_yaml_strict(BASE))
    base["scoring"] = {"x": 1}                          # blocco nuovo in base.yaml: errore, non perso in silenzio
    with pytest.raises(ConfigError, match="in più"):
        tg.slot_experiment(grid, 1, base, tg.load_dev_machines(DEV_FILE))


def test_defect_slot_1_different_from_v1_changes_the_fit_payload(grid, slots):
    """Se lo slot 1 non fosse V1 (es. pca.random_state diverso), il confronto dei payload fallirebbe."""
    raw = _raw_grid()
    raw["families"]["pca"]["fixed"]["random_state"] = 0
    g = tg.validate_grid(raw)
    exp = tg.slot_experiment(g, 1, load_yaml_strict(BASE), tg.load_dev_machines(DEV_FILE))
    cfg = schema.validate_experiment(exp)
    dvs = {m: "dv" for m in cfg.data.machines}
    a = {(t.machine, t.model): t.fit_key for t in plan_tasks(cfg, cfg.data.machines, dvs) if t.model == "pca"}
    b = {(t.machine, t.model): t.fit_key for t in plan_tasks(slots[1], cfg.data.machines, dvs) if t.model == "pca"}
    assert a.keys() == b.keys() and all(a[k] != b[k] for k in a)


# ------------------------------------------------------------------ terminazioni di riga (core.autocrlf)
def _crlf_copy(src, dst):
    dst.write_bytes(src.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    assert b"\r\n" in dst.read_bytes()
    return dst


def test_crlf_checkout_gives_the_same_files_hashes_and_fit_keys(tmp_path):
    """Un clone con ``core.autocrlf`` diverso (CRLF al posto di LF in griglia, base, macchine e slot) dà gli stessi
    testi generati, ``--check`` a 0 e gli stessi ``config_sha256`` e ``fit_key``: nessun hash è calcolato sui byte."""
    from src.utils.config import config_sha256
    g, b, d = (_crlf_copy(p, tmp_path / p.name) for p in (GRID, BASE, DEV_FILE))
    assert tg.generate(g, b, d) == tg.generate(GRID, BASE, DEV_FILE)
    assert tg.generate_pilot(g, b, d) == tg.generate_pilot(GRID, BASE, DEV_FILE)
    out, pout = tmp_path / "tuning", tmp_path / "tuning_pilot"
    out.mkdir()
    pout.mkdir()
    for s in range(1, K + 1):
        _crlf_copy(_slot_path(s), out / tg.SLOT_FILE.format(s))
    for s in range(2, K + 1):
        _crlf_copy(tg.DEFAULT_PILOT_OUT / tg.SLOT_FILE.format(s), pout / tg.SLOT_FILE.format(s))
    script = ROOT / "scripts" / "make_tuning_experiments.py"
    r = subprocess.run([sys.executable, str(script), "--check", "--grid", str(g), "--base", str(b), "--dev", str(d),
                        "--out", str(out), "--pilot-out", str(pout)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    dvs = {m: f"dv-{m}" for m in load_experiment(_slot_path(1)).data.machines}
    for s in range(1, K + 1):
        lf = load_experiment(_slot_path(s))
        cr = load_experiment(out / tg.SLOT_FILE.format(s), base_path=b)
        assert config_sha256(cr) == config_sha256(lf)
        assert [t.fit_key for t in plan_tasks(cr, cr.data.machines, dvs)] == \
            [t.fit_key for t in plan_tasks(lf, lf.data.machines, dvs)]
