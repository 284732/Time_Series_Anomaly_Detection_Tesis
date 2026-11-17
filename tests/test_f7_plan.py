"""
F7 passo 1: piano statistico (D-73 P5, P8, P14; D-25, D-30, D-36, D-44).

Il piano d'esempio ``_plan()`` usa tutte le sezioni. I test sistematici percorrono ogni chiave (rimozione, chiave
sconosciuta, tipo errato) e ogni foglia (mutazione valida → hash diverso); i test specifici coprono i rifiuti elencati
nel piano F7. La validazione è una funzione pura sul dizionario: nessuna lettura di store, dati o risultati.
"""

import copy
import random

import pytest
import yaml

from src.pipeline import table_schema as ts
from src.stats import plan as sp
from src.utils.schema import ConfigError

SHA_A, SHA_B, SHA_C, SHA_D = "a" * 64, "b" * 64, "c" * 64, "d" * 64
CID_PCA, CID_CONV, CID_LSTM, CID_RZ = "0123456789ab", "1123456789ab", "2123456789ab", "3123456789ab"
CID_UNTR = "4123456789ab"          # AE non addestrato (D-69): stesso nome di modello, config_id diverso
EXP_P, EXP_S = "synth_conv_pca", "synth_scaler"   # esperimento primario e variante di scaler (livelli di preproc)


def _pca():
    return {"model": "pca", "config_id": CID_PCA, "window_time_agg": "last", "sensor_agg": "mean",
            "normalizer": "std"}


def _conv():
    return {"model": "conv_autoencoder", "config_id": CID_CONV, "window_time_agg": "last", "sensor_agg": "top5",
            "normalizer": "std"}


def _lstm():
    return {"model": "lstm_autoencoder", "config_id": CID_LSTM, "window_time_agg": "mean", "sensor_agg": "mean",
            "normalizer": "robust"}


def _rz():
    return {"model": "rolling_zscore", "config_id": CID_RZ, "window_time_agg": "none", "sensor_agg": "native",
            "normalizer": "none"}


def _untr(**scoring):
    """Baseline descrittiva (D-69) nella forma con scoring, per i tentativi di usarla come modello confrontato."""
    d = {"model": "conv_autoencoder", "config_id": CID_UNTR, "window_time_agg": "last", "sensor_agg": "top5",
         "normalizer": "std"}
    d.update(scoring)
    return d


def _rq3():
    """Sezione rq3 (passo 9): ρ entro gruppo con IC di Fisher (Bonett–Wright), nessun bootstrap."""
    return {"descriptors": ["prevalence", "n_events"], "alpha": 0.05, "statistic": "within_group",
            "ci": {"method": "fisher_bonett_wright"}, "min_machines_per_group": 5}


def _plan():
    """Piano valido con tutte le sezioni (esplorativo)."""
    return {
        "stats": {"name": "synth_example", "kind": "exploratory", "machine_role": "eval"},   # RQ2 non su dev (D-76)
        "inputs": [{"experiment": EXP_P, "config_sha256": SHA_A, "plan_sha256": SHA_B},
                   {"experiment": EXP_S, "config_sha256": SHA_C, "plan_sha256": SHA_D}],
        "primary_experiment": EXP_P,
        "exclusions": {
            "failed_fit": "exclude_machine_pair",
            "nonfinite_loss": "error",
            "nonfinite_cal": "exclude_fit",
            "nonfinite_test": "keep",
            "nan_rows": {r: "exclude_machine_pair" for r in ts.NAN_REASONS},
            "min_events": {"n": 2, "action": "exclude_machine_pair"},
            "exclude_fit_sensitivity": True,
        },
        "descriptive_baselines": [{"model": "conv_autoencoder", "config_id": CID_UNTR}],
        "descriptive": [
            {"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last", "threshold": None},
            {"table": "thr", "metric": "event_recall", "scenario": "primary", "ts_agg": "last",
             "threshold": {"policy": "static_pct", "level": 99.0}},
            {"table": "thr", "metric": "detection_delay", "scenario": "primary", "ts_agg": "last",
             "threshold": {"policy": "static_pct", "level": 99.0}},
        ],
        "rq2": {
            "metric": "ap_norm",
            "reference": _pca(),
            "family": [_conv(), _lstm()],
            "delta": 0.05,
            "alpha": 0.05,
            "ci": {"method": "stratified_bootstrap_t", "n_boot": 2000, "seed": 20261116},
            "sign_flip": True,
            "baselines": [_untr()],                       # righe descrittive di RQ2 (passo 7)
            "random_baseline": True,
            "min_machines_per_group": 5,
        },
        "rq1a": {
            "eta2_principal": "without_machine",
            "blocks": [
                {"name": "f1_protocol", "table": "thr", "metric": "f1",
                 "models": [_pca(), _conv(), _rz()],
                 "factors": {"ts_agg": ["last", "mean", "max"], "scenario": ["primary", "test_context"],
                             "threshold": [{"policy": "static_pct", "level": 99.0}, {"policy": "oracle"}],
                             "pa": [False, True]},
                 "tau_reference": {"ts_agg": "last", "scenario": "primary",
                                   "threshold": {"policy": "static_pct", "level": 99.0}, "pa": False},
                 "delta_reference": _pca()},
                {"name": "apnorm_protocol", "table": "free", "metric": "ap_norm",
                 "models": [_pca(), _conv()],
                 "factors": {"ts_agg": ["last", "mean"], "scenario": ["primary"]},
                 "tau_reference": {"ts_agg": "last", "scenario": "primary"},
                 "delta_reference": _pca()},
            ],
        },
        "rq1b": {
            "eta2_principal": "total",
            "blocks": [
                {"name": "design_scoring", "table": "thr", "metric": "f1",
                 "models": [{"model": "pca", "config_id": CID_PCA},
                            {"model": "conv_autoencoder", "config_id": CID_CONV}],
                 "factors": {
                     "scoring": [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"},
                                 {"window_time_agg": "mean", "sensor_agg": "top5", "normalizer": "robust"}],
                     "threshold": [{"policy": "static_pct", "level": 99.0},
                                   {"policy": "roll_quantile", "level": 99.0, "n_past": 1440}],
                     "preproc": [EXP_P, EXP_S]},
                 "tau_reference": {"scoring": {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"},
                                   "threshold": {"policy": "static_pct", "level": 99.0}, "preproc": EXP_P},
                 "delta_reference": {"model": "pca", "config_id": CID_PCA}},
                {"name": "design_native", "table": "thr", "metric": "fpr",
                 "models": [{"model": "rolling_zscore", "config_id": CID_RZ}],
                 "factors": {"threshold": [{"policy": "static_pct", "level": 99.0},
                                           {"policy": "roll_robust_z", "level": 99.5, "n_past": 720}],
                             "preproc": [EXP_P]},
                 "tau_reference": None,
                 "delta_reference": None},
            ],
        },
        "rq3": _rq3(),
        "rq4": {
            "eta2_principal": "without_machine",
            "metrics": ["f1", "fpr", "false_alarms_per_1440"],
            "descriptive_metrics": ["event_recall", "detection_delay"],
            "models": [_pca(), _conv()],
            "thresholds": [{"policy": "static_pct", "level": 99.0},
                           {"policy": "roll_robust_z", "level": 99.0, "n_past": 720}],
            "offset_indicator": True,
        },
    }


def _official():
    p = _plan()
    p["stats"]["kind"] = "official"
    p["rq2"]["ci"]["n_boot"] = 10_000
    return p


def _err(p, match=None):
    with pytest.raises(ConfigError, match=match):
        sp.validate_stats_plan(p)


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def _set(d, path, value):
    _get(d, path[:-1])[path[-1]] = value


def _mapping_paths(obj, path=()):
    """Percorsi di tutte le mappe (per i test sulle chiavi); nelle liste si percorrono tutti gli elementi."""
    if isinstance(obj, dict):
        yield path
        for k, v in obj.items():
            yield from _mapping_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _mapping_paths(v, path + (i,))


def _leaf_paths(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaf_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaf_paths(v, path + (i,))
    else:
        yield path


# ---------------------------------------------------------------- piano valido, versione
def test_example_plan_is_valid_and_normalized_form_is_stable():
    plan = sp.validate_stats_plan(_plan())
    again = sp.validate_stats_plan(plan.to_dict())
    assert again.to_dict() == plan.to_dict()
    assert sp.stats_plan_sha256(again) == sp.stats_plan_sha256(plan)
    assert plan.kind == "exploratory" and plan.machine_role == "eval"


def test_m_is_the_number_of_declared_comparisons():
    p = _plan()
    assert sp.validate_stats_plan(p).m == 2
    p["rq2"]["family"] = [_conv()]
    assert sp.validate_stats_plan(p).m == 1


def test_stats_code_version_exists_and_is_outside_the_plan_hash(monkeypatch):
    assert isinstance(sp.STATS_CODE_VERSION, int) and not isinstance(sp.STATS_CODE_VERSION, bool)
    assert sp.STATS_CODE_VERSION >= 1
    h = sp.stats_plan_sha256(sp.validate_stats_plan(_plan()))
    monkeypatch.setattr(sp, "STATS_CODE_VERSION", sp.STATS_CODE_VERSION + 1)
    assert sp.stats_plan_sha256(sp.validate_stats_plan(_plan())) == h


def test_validation_is_pure(monkeypatch):
    import builtins

    def no_io(*a, **k):
        raise AssertionError("la validazione del piano non deve aprire file")
    sp.validate_stats_plan(_plan())          # import delle classi dei modelli (native_only) prima del divieto
    monkeypatch.setattr(builtins, "open", no_io)
    sp.validate_stats_plan(_plan())


# ---------------------------------------------------------------- hash e canonicalizzazione
def _shuffled(obj, rng):
    """Stesso piano con ordine delle chiavi e delle liste non significative rimescolato."""
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: _shuffled(v, rng) for k, v in items}
    if isinstance(obj, list):
        items = [_shuffled(v, rng) for v in obj]
        rng.shuffle(items)
        return items
    return obj


@pytest.mark.parametrize("seed", range(5))
def test_hash_ignores_key_and_list_order(seed):
    h = sp.stats_plan_sha256(sp.validate_stats_plan(_plan()))
    assert sp.stats_plan_sha256(sp.validate_stats_plan(_shuffled(_plan(), random.Random(seed)))) == h


def test_hash_ignores_comments_and_yaml_format(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("# commento\n" + yaml.safe_dump(_plan(), sort_keys=True, default_flow_style=False), encoding="utf-8")
    q = tmp_path / "b.yaml"
    q.write_text(yaml.safe_dump(_plan(), sort_keys=False, default_flow_style=True), encoding="utf-8")
    assert sp.stats_plan_sha256(sp.load_stats_plan(p)) == sp.stats_plan_sha256(sp.load_stats_plan(q))


def test_integer_and_float_levels_are_the_same_level_for_type_and_hash():
    a, b = _plan(), _plan()
    b["rq1a"]["blocks"][0]["factors"]["threshold"][0]["level"] = 99
    b["rq1a"]["blocks"][0]["tau_reference"]["threshold"]["level"] = 99
    b["rq2"]["delta"] = 1 / 20          # stesso float
    pa, pb = sp.validate_stats_plan(a), sp.validate_stats_plan(b)
    assert sp.stats_plan_sha256(pa) == sp.stats_plan_sha256(pb)
    block = next(b for b in pb.to_dict()["rq1a"]["blocks"] if b["name"] == "f1_protocol")
    levels = [t["level"] for t in block["factors"]["threshold"] if "level" in t]
    assert levels == [99.0] and all(isinstance(x, float) for x in levels)


def test_duplicates_after_normalization_are_rejected():
    p = _plan()
    p["rq1a"]["blocks"][0]["factors"]["threshold"] = [{"policy": "static_pct", "level": 99},
                                                      {"policy": "oracle"},
                                                      {"policy": "static_pct", "level": 99.0}]
    _err(p, "duplicat")


def test_non_adjacent_duplicates_are_rejected_before_sorting():
    p = _plan()
    p["rq1a"]["blocks"][0]["factors"]["ts_agg"] = ["mean", "last", "mean"]
    _err(p, "duplicat")
    p = _plan()
    p["rq2"]["family"] = [_conv(), _lstm(), _conv()]
    _err(p, "duplicat")


# Mutazioni valide (con le foglie accoppiate mutate insieme). Il test di copertura verifica che ogni foglia del piano
# d'esempio sia cambiata da almeno una mutazione (salvo i valori a scelta unica e i null obbligatori).
def _mutations():
    muts = [
        {("stats", "name"): "other_name"},
        {("stats", "kind"): "official", ("rq2", "ci", "n_boot"): 10_000},
        {("stats", "machine_role"): "dev", ("rq2",): None, ("rq3",): None},   # su dev niente RQ2 e RQ3 (D-76)
        {("exclusions", "failed_fit"): "error"},
        {("exclusions", "nonfinite_loss"): "exclude_machine_pair"},
        {("exclusions", "nonfinite_cal"): "error", ("exclusions", "exclude_fit_sensitivity"): False},
        {("exclusions", "nonfinite_test"): "error"},
        {("exclusions", "min_events", "n"): 3},
        {("exclusions", "min_events", "action"): "error"},
        {("rq2", "metric"): "roc_auc"},
        {("rq2", "delta"): 0.07},
        {("rq2", "alpha"): 0.1},
        {("rq2", "ci", "n_boot"): 3000},
        {("rq2", "ci", "seed"): 1},
        {("rq2", "sign_flip"): False},
        {("rq2", "min_machines_per_group"): 6},
        {("rq1a", "eta2_principal"): "total"},
        {("rq1a", "blocks", 0, "name"): "renamed"},
        {("rq1a", "blocks", 1, "name"): "renamed_1"},
        {("rq1b", "eta2_principal"): "without_machine"},
        {("rq1b", "blocks", 0, "name"): "renamed_b"},
        {("rq1b", "blocks", 1, "name"): "renamed_n"},
        {("rq1b", "blocks", 1, "metric"): "f1"},
        {("rq1b", "blocks", 0, "metric"): "recall"},
        {("rq4", "eta2_principal"): "total"},
        {("rq4", "offset_indicator"): False},
        {("rq3", "alpha"): 0.1},
        {("rq3", "min_machines_per_group"): 6},
        {("descriptive_baselines", 0, "model"): "lstm_autoencoder", ("rq2", "baselines", 0, "model"): "lstm_autoencoder"},
        {("descriptive_baselines", 0, "config_id"): "5123456789ab", ("rq2", "baselines", 0, "config_id"): "5123456789ab"},
        {("rq2", "random_baseline"): False},
        {("rq2", "baselines", 0, "window_time_agg"): "mean"},
        {("rq2", "baselines", 0, "sensor_agg"): "max"},
        {("rq2", "baselines", 0, "normalizer"): "robust"},
    ]
    for r in ts.NAN_REASONS:
        muts.append({("exclusions", "nan_rows", r): "error"})
    for i in (0, 1):
        muts.append({("inputs", i, "config_sha256"): "e" * 63 + str(i)})
        muts.append({("inputs", i, "plan_sha256"): "f" * 63 + str(i)})
    return muts


def _apply(p, mut):
    for path, value in mut.items():
        _set(p, path, value)
    return p


@pytest.mark.parametrize("mut", _mutations(), ids=lambda m: "/".join(map(str, next(iter(m)))))
def test_each_mutated_value_changes_the_hash(mut):
    h = sp.stats_plan_sha256(sp.validate_stats_plan(_plan()))
    assert sp.stats_plan_sha256(sp.validate_stats_plan(_apply(_plan(), mut))) != h


def _m(model, cid, w, s, n):
    return {"model": model, "config_id": cid, "window_time_agg": w, "sensor_agg": s, "normalizer": n}


def _structural_mutations():
    """Mutazioni delle foglie dentro liste (livelli, modelli, soglie), con riferimenti e livelli accoppiati."""
    out = []

    def m(fn):
        out.append(fn)
        return fn

    @m
    def desc_free(p):
        p["descriptive"][0].update(metric="roc_auc", scenario="test_context", ts_agg="mean")

    @m
    def desc_thr(p):
        for e in p["descriptive"][1:]:
            e.update(scenario="test_context", ts_agg="mean")
            e["threshold"]["level"] = 98.0

    @m
    def desc_thr_policy(p):
        for e in p["descriptive"][1:]:
            e["threshold"] = {"policy": "roll_quantile", "level": 99.0, "n_past": 360}

    @m
    def desc_tables(p):
        base = {"scenario": "primary", "ts_agg": "last"}
        p["descriptive"] = [dict(base, table="thr", metric="event_recall",
                                 threshold={"policy": "static_pct", "level": 99.0}),
                            dict(base, table="free", metric="roc_auc", threshold=None),
                            dict(base, table="free", metric="ap", threshold=None)]

    @m
    def rq2_models(p):
        p["rq2"]["reference"] = _m("input_norm", "9" * 12, "mean", "max", "robust")
        p["rq2"]["family"] = [_m("pca", "8" * 12, "mean", "max", "none"),
                              _m("pca_pointwise", "7" * 12, "last", "top5", "std")]

    @m
    def rq1a_models(p):
        b = p["rq1a"]["blocks"][0]
        b["models"] = [_m("input_norm", "7" * 12, "mean", "max", "robust"),
                       _m("lstm_autoencoder", "8" * 12, "mean", "max", "none"),
                       _m("pca_pointwise", "6" * 12, "last", "max", "std")]
        b["delta_reference"] = copy.deepcopy(b["models"][1])
        b2 = p["rq1a"]["blocks"][1]
        b2["models"] = [_m("lstm_autoencoder", "5" * 12, "mean", "max", "robust"),
                        _m("input_norm", "4" * 12, "mean", "mean", "none")]
        b2["delta_reference"] = copy.deepcopy(b2["models"][0])
        b2["metric"] = "roc_auc"

    @m
    def rq1a_levels(p):
        b = p["rq1a"]["blocks"][0]
        b["factors"] = {"ts_agg": ["mean", "last"], "scenario": ["test_context"],
                        "threshold": [{"policy": "oracle"}, {"policy": "static_pct", "level": 98.0}],
                        "pa": [True, False]}
        b["tau_reference"] = {"ts_agg": "mean", "scenario": "test_context", "threshold": {"policy": "oracle"},
                              "pa": True}
        b2 = p["rq1a"]["blocks"][1]
        b2["factors"] = {"ts_agg": ["max", "last"], "scenario": ["test_context", "primary"]}
        b2["tau_reference"] = {"ts_agg": "max", "scenario": "test_context"}

    @m
    def rq1a_tables(p):
        b = p["rq1a"]["blocks"][0]
        b.update(table="free", metric="ap_norm")
        b["factors"] = {"ts_agg": ["last", "mean"], "scenario": ["primary"]}
        b["tau_reference"] = {"ts_agg": "last", "scenario": "primary"}
        b2 = p["rq1a"]["blocks"][1]
        b2.update(table="thr", metric="f1")
        b2["factors"] = {"ts_agg": ["last"], "scenario": ["primary"],
                         "threshold": [{"policy": "static_pct", "level": 99.0}], "pa": [False]}
        b2["tau_reference"] = {"ts_agg": "last", "scenario": "primary",
                               "threshold": {"policy": "static_pct", "level": 99.0}, "pa": False}

    @m
    def rq1b_levels(p):
        b = p["rq1b"]["blocks"][0]
        b["models"] = [{"model": "input_norm", "config_id": "5" * 12},
                       {"model": "lstm_autoencoder", "config_id": "4" * 12}]
        b["delta_reference"] = copy.deepcopy(b["models"][1])
        b["factors"] = {
            "scoring": [{"window_time_agg": "mean", "sensor_agg": "max", "normalizer": "none"},
                        {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"}],
            "threshold": [{"policy": "roll_quantile", "level": 97.0, "n_past": 360},
                          {"policy": "roll_robust_z", "level": 98.0, "n_past": 360}],
            "preproc": [EXP_S, EXP_P]}
        b["tau_reference"] = {"scoring": {"window_time_agg": "mean", "sensor_agg": "max", "normalizer": "none"},
                              "threshold": {"policy": "roll_quantile", "level": 97.0, "n_past": 360},
                              "preproc": EXP_S}
        n = p["rq1b"]["blocks"][1]
        n["models"][0]["config_id"] = "3" * 12
        n["factors"] = {"threshold": [{"policy": "roll_quantile", "level": 95.0, "n_past": 360},
                                      {"policy": "static_pct", "level": 98.0}],
                        "preproc": [EXP_S, EXP_P]}

    @m
    def rq1b_tables(p):
        b = p["rq1b"]["blocks"][0]
        b.update(table="free", metric="ap_norm")
        del b["factors"]["threshold"]
        del b["tau_reference"]["threshold"]
        n = p["rq1b"]["blocks"][1]
        n.update(table="free", metric="ap_norm")
        n["factors"] = {"preproc": [EXP_P, EXP_S]}

    @m
    def rename_experiments(p):
        ren = {EXP_P: "exp_primary_x", EXP_S: "exp_scaler_x"}
        for e in p["inputs"]:
            e["experiment"] = ren[e["experiment"]]
        p["primary_experiment"] = ren[p["primary_experiment"]]
        for b in p["rq1b"]["blocks"]:
            b["factors"]["preproc"] = [ren[x] for x in b["factors"]["preproc"]]
            if b["tau_reference"] is not None:
                b["tau_reference"]["preproc"] = ren[b["tau_reference"]["preproc"]]

    @m
    def primary_other(p):
        p["primary_experiment"] = EXP_S
        p["rq1b"]["blocks"][1]["factors"]["preproc"] = [EXP_S]

    @m
    def baselines_empty(p):
        p["descriptive_baselines"] = []
        p["rq2"]["baselines"] = []

    @m
    def rq3_rq4(p):
        p["rq3"]["descriptors"] = ["n_events"]
        p["rq4"]["metrics"] = ["precision", "recall", "fa_segments"]
        p["rq4"]["descriptive_metrics"] = ["f1", "event_recall", "fpr"]           # fpr per l'indicatore (P10)
        p["rq4"]["models"] = [_m("input_norm", "2" * 12, "mean", "top5", "none"),
                              _m("lstm_autoencoder", "1" * 12, "mean", "max", "robust")]
        p["rq4"]["thresholds"] = [{"policy": "roll_quantile", "level": 95.0, "n_past": 360},
                                  {"policy": "static_pct", "level": 98.0}]

    return out


@pytest.mark.parametrize("fn", _structural_mutations(), ids=lambda f: f.__name__)
def test_each_structural_mutation_changes_the_hash(fn):
    h = sp.stats_plan_sha256(sp.validate_stats_plan(_plan()))
    p = _plan()
    fn(p)
    assert sp.stats_plan_sha256(sp.validate_stats_plan(p)) != h


def test_mutations_cover_every_leaf():
    base = _plan()
    covered = set()
    for mut in _mutations():
        covered |= {tuple(k) for k in mut}
    for fn in _structural_mutations():
        p = _plan()
        fn(p)
        for path in _leaf_paths(base):
            try:
                if _get(p, path) != _get(base, path):
                    covered.add(path)
            except (KeyError, IndexError, TypeError):
                covered.add(path)
    single_choice = {("rq2", "ci", "method"), ("rq3", "ci", "method"),          # D-76, passo 9: metodi unici
                     ("rq3", "statistic"),
                     ("rq1b", "blocks", 1, "models", 0, "model")}               # unico modello native
    nulls = {("rq1b", "blocks", 1, "tau_reference"), ("rq1b", "blocks", 1, "delta_reference")}
    missing = sorted(set(_leaf_paths(base)) - covered - single_choice - nulls, key=str)
    assert not missing, f"foglie senza mutazione: {missing}"


# ---------------------------------------------------------------- rifiuti sistematici
@pytest.mark.parametrize("path", list(_mapping_paths(_plan())), ids=str)
def test_removing_any_key_is_rejected(path):
    keys = list(_get(_plan(), path))
    for k in keys:
        p = _plan()
        del _get(p, path)[k]
        with pytest.raises(ConfigError):
            sp.validate_stats_plan(p)


@pytest.mark.parametrize("path", list(_mapping_paths(_plan())), ids=str)
def test_unknown_key_is_rejected(path):
    p = _plan()
    _get(p, path)["zzz_unknown"] = 1
    _err(p)


@pytest.mark.parametrize("path", list(_leaf_paths(_plan())), ids=str)
def test_wrong_type_of_any_leaf_is_rejected(path):
    p = _plan()
    old = _get(p, path)
    _set(p, path, 123 if isinstance(old, str) else "x")
    _err(p)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    f = tmp_path / "dup.yaml"
    f.write_text("stats:\n  name: a\n  name: b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicata"):
        sp.load_stats_plan(f)


def test_empty_or_missing_file_is_rejected(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        sp.load_stats_plan(f)
    with pytest.raises(ConfigError):
        sp.load_stats_plan(tmp_path / "missing.yaml")


@pytest.mark.parametrize("path", [("rq2", "delta"), ("rq2", "alpha"), ("rq2", "ci", "n_boot"),
                                  ("rq2", "ci", "seed"), ("rq2", "min_machines_per_group"),
                                  ("exclusions", "min_events", "n")])
def test_bool_is_never_a_number(path):
    p = _plan()
    _set(p, path, True)
    _err(p)


@pytest.mark.parametrize("path", [("rq2", "ci", "n_boot"), ("rq2", "ci", "seed"), ("rq2", "min_machines_per_group"),
                                  ("exclusions", "min_events", "n")])
def test_float_is_not_an_integer(path):
    p = _plan()
    _set(p, path, float(_get(p, path)))
    _err(p)


# ---------------------------------------------------------------- δ, α, B
@pytest.mark.parametrize("delta", [0, 0.0, -0.05, float("nan"), float("inf"), None, "0.05"])
def test_delta_must_be_a_finite_positive_number(delta):
    p = _plan()
    p["rq2"]["delta"] = delta
    _err(p)


def test_delta_missing_is_rejected():
    p = _plan()
    del p["rq2"]["delta"]
    _err(p, "delta")


@pytest.mark.parametrize("alpha", [0, 1, 1.0, -0.1, 1.5, float("nan")])
def test_alpha_must_be_in_open_unit_interval(alpha):
    p = _plan()
    p["rq2"]["alpha"] = alpha
    _err(p)
    s = _rq3()
    s["alpha"] = alpha
    with pytest.raises(ConfigError, match="rq3.alpha"):
        sp._rq3(s, "exploratory")


@pytest.mark.parametrize("n_boot", [0, -1, 10.5, 2000.0])
def test_n_boot_must_be_a_positive_integer(n_boot):
    p = _plan()
    p["rq2"]["ci"]["n_boot"] = n_boot
    _err(p)


def test_official_requires_at_least_10000_bootstrap_replicates():
    p = _official()
    p["rq2"]["ci"]["n_boot"] = 9_999
    _err(p, "10")
    sp.validate_stats_plan(_official())                     # 10.000: accettato


def test_exploratory_accepts_fewer_than_10000_replicates():
    p = _plan()
    p["rq2"]["ci"]["n_boot"] = 9_999
    sp.validate_stats_plan(p)


# ---------------------------------------------------------------- RQ2: famiglia, metodo, metrica
def test_empty_family_is_rejected():
    p = _plan()
    p["rq2"]["family"] = []
    _err(p)


def test_family_with_duplicates_is_rejected():
    p = _plan()
    p["rq2"]["family"] = [_conv(), _conv()]
    _err(p, "duplicat")


def test_reference_compared_with_itself_is_rejected():
    p = _plan()
    p["rq2"]["family"] = [_conv(), _pca()]
    _err(p, "riferimento")


@pytest.mark.parametrize("where", ["reference", "family"])
def test_random_baseline_is_never_a_model_level(where):
    p = _plan()
    rnd = {"model": "random_baseline", "config_id": "random_basel", "window_time_agg": "none",
           "sensor_agg": "native", "normalizer": "none"}
    if where == "reference":
        p["rq2"]["reference"] = rnd
    else:
        p["rq2"]["family"] = [_conv(), rnd]
    _err(p, "D-73 P7")


@pytest.mark.parametrize("method", ["hierarchical", "percentile", "bca", "stratified_bca", "", None])
def test_only_the_stratified_bootstrap_t_method_is_admitted_for_rq2(method):
    p = _plan()
    p["rq2"]["ci"]["method"] = method
    _err(p, "rq2.ci.method")


def test_the_old_bca_method_is_rejected_with_a_pointer_to_d76():
    p = _plan()
    p["rq2"]["ci"]["method"] = "stratified_bca"
    _err(p, "D-76")


@pytest.mark.parametrize("metric", ["roc_auc", "ap", "ap_over_p"])
def test_rq2_metric_free_when_exploratory_and_ap_norm_when_official(metric):
    p = _plan()
    p["rq2"]["metric"] = metric
    sp.validate_stats_plan(p)
    q = _official()
    q["rq2"]["metric"] = metric
    _err(q, "ap_norm")
    sp.validate_stats_plan(_official())


@pytest.mark.parametrize("metric", ["best_f1_oracle", "f1", "vus_pr"])
def test_rq2_metric_must_be_a_non_diagnostic_free_metric(metric):
    p = _plan()
    p["rq2"]["metric"] = metric
    _err(p)


@pytest.mark.parametrize("bad", [
    {"window_time_agg": "none", "sensor_agg": "native", "normalizer": "none"},     # non native con terna native
    {"window_time_agg": "max", "sensor_agg": "mean", "normalizer": "std"},         # max non salvato (D-33)
    {"window_time_agg": "last", "sensor_agg": "native", "normalizer": "std"},
])
def test_scoring_of_a_non_native_model_must_be_a_residual_scoring(bad):
    p = _plan()
    p["rq2"]["family"][0].update(bad)
    _err(p)


def test_native_model_requires_the_native_scoring():
    p = _plan()
    p["rq2"]["family"][0] = dict(_rz(), window_time_agg="last", sensor_agg="mean", normalizer="std")
    _err(p, "native")
    p["rq2"]["family"][0] = _rz()
    sp.validate_stats_plan(p)


@pytest.mark.parametrize("cid", ["0123", "0123456789AB", "0123456789abc", 123])
def test_config_id_is_twelve_lowercase_hex(cid):
    p = _plan()
    p["rq2"]["reference"]["config_id"] = cid
    _err(p)


# ---------------------------------------------------------------- esclusioni (D-44, P8)
def test_missing_exclusions_block_is_rejected():
    p = _plan()
    del p["exclusions"]
    _err(p, "exclusions")


@pytest.mark.parametrize("key", ["failed_fit", "nonfinite_loss", "nonfinite_cal", "nonfinite_test"])
@pytest.mark.parametrize("action", ["drop", "ignore", "", None])
def test_unknown_action_is_rejected(key, action):
    p = _plan()
    p["exclusions"][key] = action
    _err(p)


@pytest.mark.parametrize("key", ["failed_fit", "nonfinite_loss", "nonfinite_cal"])
def test_keep_only_for_nonfinite_test(key):
    p = _plan()
    p["exclusions"][key] = "keep"
    _err(p, "solo per exclusions.nonfinite_test")


@pytest.mark.parametrize("reason", ts.NAN_REASONS)
def test_keep_is_not_an_action_for_nan_rows(reason):
    p = _plan()
    p["exclusions"]["nan_rows"][reason] = "keep"
    _err(p, "solo per exclusions.nonfinite_test")


@pytest.mark.parametrize("reason", ts.NAN_REASONS)
def test_every_nan_reason_must_have_a_rule(reason):
    p = _plan()
    del p["exclusions"]["nan_rows"][reason]
    _err(p, reason)


def test_unknown_nan_reason_is_rejected():
    p = _plan()
    p["exclusions"]["nan_rows"]["made_up_reason"] = "error"
    _err(p, "made_up_reason")


def test_exclude_fit_requires_the_exclude_machine_pair_sensitivity():
    p = _plan()
    p["exclusions"]["exclude_fit_sensitivity"] = False
    _err(p, "exclude_fit")
    q = _plan()                                     # exclude_fit in nan_rows: stessa regola
    q["exclusions"]["nonfinite_cal"] = "error"
    q["exclusions"]["nan_rows"]["nonfinite_score"] = "exclude_fit"
    q["exclusions"]["exclude_fit_sensitivity"] = False
    _err(q, "exclude_fit")


def test_sensitivity_without_exclude_fit_is_rejected():
    p = _plan()
    p["exclusions"]["nonfinite_cal"] = "error"
    _err(p, "exclude_fit")
    p["exclusions"]["exclude_fit_sensitivity"] = False
    sp.validate_stats_plan(p)


@pytest.mark.parametrize("n", [0, -1])
def test_min_events_n_at_least_one(n):
    p = _plan()
    p["exclusions"]["min_events"]["n"] = n
    _err(p)


@pytest.mark.parametrize("action", ["exclude_fit", "keep"])
def test_min_events_is_a_machine_level_rule(action):
    p = _plan()
    p["exclusions"]["min_events"]["action"] = action
    _err(p)


# ---------------------------------------------------------------- sezioni RQ
@pytest.mark.parametrize("sec", ["rq2", "rq1a", "rq1b", "rq3", "rq4"])
def test_missing_rq_key_is_rejected_but_explicit_null_is_admitted(sec):
    p = _plan()
    del p[sec]
    _err(p, sec)
    q = _plan()
    q[sec] = None
    if sec == "rq2":
        q["rq3"] = None                     # rq3 richiede rq2
    if sec == "rq1b":
        q["inputs"] = q["inputs"][:1]       # senza RQ1b l'esperimento dello scaler non ha ruolo
    sp.validate_stats_plan(q)


def test_plan_with_all_rq_null_is_rejected():
    p = _plan()
    for sec in ("rq2", "rq1a", "rq1b", "rq3", "rq4"):
        p[sec] = None
    _err(p, "RQ")


def test_rq3_requires_rq2():
    p = _plan()
    p["rq2"] = None
    _err(p, "rq2")


@pytest.mark.parametrize("method", ["stratified_bootstrap_t", "stratified_bca", "bootstrap", "fisher", None])
def test_rq3_ci_method_is_only_fisher_bonett_wright(method):
    p = _plan()
    p["rq3"]["ci"]["method"] = method
    _err(p, "rq3.ci.method")


@pytest.mark.parametrize("key,value", [("n_boot", 2000), ("seed", 1)])
def test_rq3_ci_rejects_bootstrap_keys(key, value):
    p = _plan()
    p["rq3"]["ci"][key] = value
    _err(p, "nessun bootstrap")


@pytest.mark.parametrize("value", ["overall", "by_group", None, 1])
def test_rq3_statistic_is_within_group(value):
    p = _plan()
    p["rq3"]["statistic"] = value
    _err(p, "rq3.statistic")
    q = _plan()
    del q["rq3"]["statistic"]
    _err(q, "statistic")


def test_rq3_normalized_section():
    d = sp.validate_stats_plan(_plan()).to_dict()["rq3"]
    assert d == {"descriptors": ["n_events", "prevalence"], "alpha": 0.05, "statistic": "within_group",
                 "ci": {"method": "fisher_bonett_wright"}, "min_machines_per_group": 5}


def test_offset_indicator_requires_fpr_among_the_rq4_metrics():
    p = _plan()
    p["rq4"]["metrics"] = ["f1", "false_alarms_per_1440"]
    _err(p, "fpr")
    q = _plan()
    q["rq4"]["metrics"] = ["f1"]
    q["rq4"]["descriptive_metrics"] = ["fpr"]
    sp.validate_stats_plan(q)
    q["rq4"]["offset_indicator"] = False
    q["rq4"]["descriptive_metrics"] = []
    sp.validate_stats_plan(q)


@pytest.mark.parametrize("sec", ["rq1a", "rq1b", "rq4"])
@pytest.mark.parametrize("value", [None, "both", "machine", ""])
def test_eta2_principal_must_be_declared(sec, value):
    p = _plan()
    p[sec]["eta2_principal"] = value
    _err(p)


@pytest.mark.parametrize("descriptor", ["drift", "out_of_range", "pca_rank", "prevalence_test"])
def test_rq3_only_table_descriptors(descriptor):
    s = _rq3()
    s["descriptors"] = ["prevalence", descriptor]
    with pytest.raises(ConfigError, match=descriptor):
        sp._rq3(s, "exploratory")


def test_rq4_detection_delay_is_never_decomposed():
    p = _plan()
    p["rq4"]["metrics"] = ["f1", "detection_delay"]
    _err(p, "detection_delay")


def test_rq4_descriptive_detection_delay_requires_event_recall():
    p = _plan()
    p["rq4"]["descriptive_metrics"] = ["detection_delay"]
    _err(p, "event_recall")


def test_descriptive_detection_delay_requires_event_recall_with_same_condition():
    p = _plan()
    p["descriptive"] = [p["descriptive"][0], p["descriptive"][2]]
    _err(p, "event_recall")
    q = _plan()
    q["descriptive"][1]["threshold"]["level"] = 98.0           # event_recall a un'altra soglia
    _err(q, "event_recall")


def test_rq4_offset_indicator_requires_a_static_threshold():
    p = _plan()
    p["rq4"]["thresholds"] = [{"policy": "roll_robust_z", "level": 99.0, "n_past": 720}]
    _err(p, "static_pct")
    p["rq4"]["offset_indicator"] = False
    sp.validate_stats_plan(p)


@pytest.mark.parametrize("policy", [{"policy": "oracle"}])
def test_rq4_rejects_the_oracle(policy):
    p = _plan()
    p["rq4"]["thresholds"].append(policy)
    _err(p, "oracle")


@pytest.mark.parametrize("table,metric", [("free", "best_f1_oracle"), ("thr", "detection_delay")])
def test_decomposed_metrics_exclude_diagnostic_and_nan_by_definition(table, metric):
    p = _plan()
    b = p["rq1a"]["blocks"][1] if table == "free" else p["rq1a"]["blocks"][0]
    b["metric"] = metric
    _err(p)
    q = _plan()
    q["rq1b"]["blocks"][1]["metric"] = "detection_delay"
    _err(q, "detection_delay")


@pytest.mark.parametrize("metric", ["best_f1_oracle"])
def test_descriptive_rejects_diagnostic_metrics(metric):
    p = _plan()
    p["descriptive"][0]["metric"] = metric
    _err(p)


@pytest.mark.parametrize("threshold", [{"policy": "oracle"}])
def test_descriptive_rejects_the_oracle(threshold):
    p = _plan()
    p["descriptive"][1]["threshold"] = threshold
    p["descriptive"][2]["threshold"] = threshold
    _err(p, "oracle")


def test_descriptive_free_has_null_threshold_and_thr_a_threshold():
    p = _plan()
    p["descriptive"][0]["threshold"] = {"policy": "static_pct", "level": 99.0}
    _err(p)
    q = _plan()
    q["descriptive"][1]["threshold"] = None
    _err(q)


def test_rolling_threshold_requires_last_projection_in_descriptive():
    p = _plan()
    for e in p["descriptive"][1:]:
        e["ts_agg"] = "mean"
        e["threshold"] = {"policy": "roll_quantile", "level": 99.0, "n_past": 360}
    _err(p, "D-24")


@pytest.mark.parametrize("thr,bad", [
    ({"policy": "static_pct"}, "level mancante"),
    ({"policy": "static_pct", "level": 99.0, "n_past": 10}, "n_past con statica"),
    ({"policy": "roll_quantile", "level": 99.0}, "n_past mancante"),
    ({"policy": "roll_quantile", "level": 99.0, "n_past": 0}, "n_past < 1"),
    ({"policy": "static_pct", "level": 0}, "livello 0"),
    ({"policy": "static_pct", "level": 100.5}, "livello > 100"),
    ({"policy": "oracle", "level": 99.0}, "livello con oracolo"),
    ({"policy": "made_up"}, "politica sconosciuta"),
])
def test_threshold_spec_rules(thr, bad):
    p = _plan()
    p["rq4"]["thresholds"] = [{"policy": "static_pct", "level": 95.0}, thr]
    _err(p)


# ---------------------------------------------------------------- RQ1a
def test_rq1a_rejects_rolling_thresholds():
    p = _plan()
    p["rq1a"]["blocks"][0]["factors"]["threshold"].append({"policy": "roll_quantile", "level": 99.0, "n_past": 360})
    _err(p, "D-25")


@pytest.mark.parametrize("factor", ["threshold", "pa"])
def test_rq1a_free_block_rejects_threshold_factors(factor):
    p = _plan()
    p["rq1a"]["blocks"][1]["factors"][factor] = [False] if factor == "pa" else [{"policy": "oracle"}]
    _err(p, factor)


@pytest.mark.parametrize("factor", ["scoring", "preproc"])
def test_rq1a_rejects_design_factors(factor):
    p = _plan()
    p["rq1a"]["blocks"][0]["factors"][factor] = [EXP_P]
    _err(p, factor)


@pytest.mark.parametrize("missing", ["ts_agg", "scenario", "threshold", "pa"])
def test_rq1a_all_table_factors_are_mandatory(missing):
    p = _plan()
    del p["rq1a"]["blocks"][0]["factors"][missing]
    _err(p, missing)


def test_rq1a_metric_must_belong_to_the_block_table():
    p = _plan()
    p["rq1a"]["blocks"][1]["metric"] = "f1"
    _err(p)
    q = _plan()
    q["rq1a"]["blocks"][0]["metric"] = "ap_norm"
    _err(q)


@pytest.mark.parametrize("metric", ["precision", "fpr", "event_recall"])
def test_rq1a_oracle_or_pa_levels_require_f1(metric):
    p = _plan()
    p["rq1a"]["blocks"][0]["metric"] = metric
    _err(p, "f1")
    q = _plan()                                           # senza oracolo e senza PA: ammessa
    b = q["rq1a"]["blocks"][0]
    b["metric"] = metric
    b["factors"]["threshold"] = [{"policy": "static_pct", "level": 99.0}]
    b["factors"]["pa"] = [False]
    sp.validate_stats_plan(q)


def test_rq1a_needs_at_least_two_models():
    p = _plan()
    b = p["rq1a"]["blocks"][1]
    b["models"] = [_pca()]
    _err(p)


def test_rq1a_rejects_random_and_duplicate_models():
    p = _plan()
    p["rq1a"]["blocks"][1]["models"].append({"model": "random_baseline", "config_id": "random_basel",
                                             "window_time_agg": "none", "sensor_agg": "native",
                                             "normalizer": "none"})
    _err(p, "D-73 P7")
    q = _plan()
    q["rq1a"]["blocks"][1]["models"].append(_pca())
    _err(q, "duplicat")


def test_rq1a_model_counts_as_a_factor_with_two_levels():
    p = _plan()
    b = p["rq1a"]["blocks"][1]
    b["factors"] = {"ts_agg": ["last"], "scenario": ["primary"]}
    b["tau_reference"] = {"ts_agg": "last", "scenario": "primary"}
    sp.validate_stats_plan(p)                     # fattori a un livello, ma 2 modelli: ammesso


@pytest.mark.parametrize("case", ["ts_agg", "scenario", "threshold", "pa"])
def test_rq1a_tau_reference_must_be_a_declared_combination(case):
    p = _plan()
    b = p["rq1a"]["blocks"][0]
    if case == "ts_agg":
        b["factors"]["ts_agg"] = ["last", "mean"]
        b["tau_reference"]["ts_agg"] = "max"
    elif case == "scenario":
        b["factors"]["scenario"] = ["primary"]
        b["tau_reference"]["scenario"] = "test_context"
    elif case == "threshold":
        b["tau_reference"]["threshold"] = {"policy": "static_pct", "level": 98.0}
    else:
        b["factors"]["pa"] = [False]
        b["tau_reference"]["pa"] = True
    _err(p, "tau_reference")


def test_rq1a_tau_reference_must_have_exactly_the_factor_keys():
    p = _plan()
    del p["rq1a"]["blocks"][0]["tau_reference"]["pa"]
    _err(p)


def test_rq1a_delta_reference_must_be_one_of_the_block_models():
    p = _plan()
    p["rq1a"]["blocks"][1]["delta_reference"] = _lstm()
    _err(p, "delta_reference")


def test_rq1a_native_models_are_admitted_with_their_native_scoring():
    """Fatto verificato (analysis_grid.fit_scores): i modelli native hanno righe per ogni scenario e ts_agg."""
    p = _plan()
    assert _rz() in p["rq1a"]["blocks"][0]["models"]
    sp.validate_stats_plan(p)


def test_block_names_are_distinct():
    p = _plan()
    p["rq1a"]["blocks"][1]["name"] = p["rq1a"]["blocks"][0]["name"]
    _err(p, "duplicat")


# ---------------------------------------------------------------- RQ1b
def test_rq1b_rejects_the_oracle():
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"]["threshold"].append({"policy": "oracle"})
    _err(p, "oracle")


@pytest.mark.parametrize("factor", ["ts_agg", "scenario", "pa"])
def test_rq1b_rejects_measurement_factors(factor):
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"][factor] = ["last"]
    _err(p, factor)


@pytest.mark.parametrize("missing", ["scoring", "threshold", "preproc"])
def test_rq1b_all_table_factors_are_mandatory(missing):
    p = _plan()
    del p["rq1b"]["blocks"][0]["factors"][missing]
    _err(p, missing)


def test_rq1b_free_block_has_no_threshold_factor():
    p = _plan()
    b = p["rq1b"]["blocks"][0]
    b["table"], b["metric"] = "free", "ap_norm"
    _err(p, "threshold")
    del b["factors"]["threshold"]
    del b["tau_reference"]["threshold"]
    sp.validate_stats_plan(p)


def test_rq1b_native_block_without_scoring_factor():
    p = _plan()
    n = p["rq1b"]["blocks"][1]
    n["factors"]["scoring"] = [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"}]
    _err(p, "native")


def test_rq1b_native_and_non_native_models_are_not_mixed():
    p = _plan()
    p["rq1b"]["blocks"][0]["models"].append({"model": "rolling_zscore", "config_id": CID_RZ})
    _err(p, "native")
    q = _plan()
    q["rq1b"]["blocks"][1]["models"].append({"model": "pca", "config_id": CID_PCA})
    _err(q, "native")


def test_rq1b_scoring_levels_are_residual_scorings():
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"]["scoring"].append(
        {"window_time_agg": "none", "sensor_agg": "native", "normalizer": "none"})
    _err(p)


def test_rq1b_single_model_block_has_null_references_and_multi_model_block_has_them():
    p = _plan()
    p["rq1b"]["blocks"][1]["tau_reference"] = {"threshold": {"policy": "static_pct", "level": 99.0},
                                               "preproc": EXP_P}
    _err(p, "null")
    q = _plan()
    q["rq1b"]["blocks"][0]["delta_reference"] = None
    _err(q, "delta_reference")


def test_rq1b_needs_a_factor_with_two_levels():
    p = _plan()
    n = p["rq1b"]["blocks"][1]
    n["factors"]["threshold"] = [{"policy": "static_pct", "level": 99.0}]
    _err(p, "livelli")


@pytest.mark.parametrize("pre", ["", "a/b", 12, "aaaaaaaaaaaa"])
def test_rq1b_preproc_levels_are_declared_experiment_names(pre):
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"]["preproc"] = [EXP_P, pre]
    _err(p)


# ---------------------------------------------------------------- stats e inputs
@pytest.mark.parametrize("field,value", [("kind", "official_x"), ("kind", None), ("machine_role", "both"),
                                         ("machine_role", "all")])
def test_stats_kind_and_machine_role(field, value):
    p = _plan()
    p["stats"][field] = value
    _err(p)


def test_inputs_must_be_a_non_empty_list_of_distinct_experiments():
    p = _plan()
    p["inputs"] = []
    _err(p)
    q = _plan()
    q["inputs"][1]["experiment"] = q["inputs"][0]["experiment"]
    _err(q, "duplicat")


@pytest.mark.parametrize("field", ["plan_sha256", "config_sha256"])
def test_inputs_require_both_hashes(field):
    p = _plan()
    del p["inputs"][0][field]
    _err(p, field)


@pytest.mark.parametrize("value", ["a" * 63, "A" * 64, "g" * 64, "a" * 65, 7])
@pytest.mark.parametrize("field", ["plan_sha256", "config_sha256"])
def test_input_hashes_are_64_lowercase_hex(field, value):
    p = _plan()
    p["inputs"][0][field] = value
    _err(p, field)


@pytest.mark.parametrize("name", ["", "a/b", "con", "x."])
def test_names_are_store_path_names(name):
    p = _plan()
    p["stats"]["name"] = name
    _err(p)
    q = _plan()
    q["inputs"][0]["experiment"] = name
    _err(q)


# ---------------------------------------------------------------- passo 1-bis: famiglia, baseline descrittive, ruoli
def test_family_of_three_gives_m_3_and_bonferroni_level():
    p = _plan()
    p["rq2"]["family"] = [_conv(), _lstm(), _m("input_norm", "6" * 12, "last", "mean", "std")]  # 3 confronti (D-71)
    plan = sp.validate_stats_plan(p)
    assert plan.m == 3
    assert plan.ci_level == 1 - 0.05 / 3          # Bonferroni al 1 − α/m (D-36)
    assert sp.validate_stats_plan(_plan()).ci_level == 1 - 0.05 / 2


def test_without_rq2_m_is_zero_and_no_ci_level():
    p = _plan()
    p["rq2"] = p["rq3"] = None
    plan = sp.validate_stats_plan(p)
    assert plan.m == 0 and plan.ci_level is None


def test_descriptive_baselines_are_mandatory_but_may_be_empty():
    p = _plan()
    p["descriptive_baselines"] = []
    p["rq2"]["baselines"] = []
    assert sp.validate_stats_plan(p).to_dict()["descriptive_baselines"] == []
    p = _plan()
    del p["descriptive_baselines"]
    _err(p, "descriptive_baselines")
    p = _plan()
    p["descriptive_baselines"] = None
    _err(p, "descriptive_baselines")


def test_descriptive_baselines_are_normalized_and_sorted():
    p = _plan()
    p["descriptive_baselines"] = [{"model": "lstm_autoencoder", "config_id": "5123456789ab"},
                                  {"model": "conv_autoencoder", "config_id": CID_UNTR}]
    out = sp.validate_stats_plan(p).to_dict()["descriptive_baselines"]
    assert out == [{"config_id": CID_UNTR, "model": "conv_autoencoder"},
                   {"config_id": "5123456789ab", "model": "lstm_autoencoder"}]


def test_duplicate_descriptive_baselines_are_rejected():
    p = _plan()
    p["descriptive_baselines"] = [{"model": "conv_autoencoder", "config_id": CID_UNTR},
                                  {"model": "lstm_autoencoder", "config_id": "5123456789ab"},
                                  {"model": "conv_autoencoder", "config_id": CID_UNTR}]
    _err(p, "duplicat")


def test_random_baseline_is_not_a_descriptive_baseline():
    p = _plan()
    p["descriptive_baselines"] = [{"model": "random_baseline", "config_id": CID_UNTR}]
    _err(p, "D-73 P7")


def test_descriptive_baseline_is_never_in_the_confirmatory_family():
    p = _plan()
    p["rq2"]["family"].append(_untr())
    _err(p, r"rq2\.family.*baseline descrittiva")


def test_descriptive_baseline_is_never_the_rq2_reference():
    p = _plan()
    p["rq2"]["reference"] = _untr()
    _err(p, r"rq2\.reference.*baseline descrittiva")


def test_baseline_identity_is_model_and_config_id_whatever_the_scoring():
    p = _plan()
    p["rq2"]["family"].append(_untr(window_time_agg="mean", sensor_agg="max", normalizer="none"))
    _err(p, "baseline descrittiva")


def test_same_model_name_with_another_config_id_is_not_the_baseline():
    plan = sp.validate_stats_plan(_plan())        # Conv-AE addestrato (CID_CONV) nella famiglia, non addestrato fuori
    pairs = [(e["model"], e["config_id"]) for e in plan.to_dict()["rq2"]["family"]]
    assert ("conv_autoencoder", CID_CONV) in pairs
    assert plan.to_dict()["descriptive_baselines"] == [{"config_id": CID_UNTR, "model": "conv_autoencoder"}]


@pytest.mark.parametrize("block", [0, 1])
def test_descriptive_baseline_is_never_a_model_of_an_rq1a_block(block):
    p = _plan()
    p["rq1a"]["blocks"][block]["models"].append(_untr())
    _err(p, r"rq1a\.blocks.*baseline descrittiva")


def test_descriptive_baseline_is_never_a_model_of_an_rq1b_block():
    p = _plan()
    p["rq1b"]["blocks"][0]["models"].append({"model": "conv_autoencoder", "config_id": CID_UNTR})
    _err(p, r"rq1b\.blocks.*baseline descrittiva")


def test_descriptive_baseline_is_never_a_model_of_rq4():
    p = _plan()
    p["rq4"]["models"].append(_untr())
    _err(p, r"rq4\.models.*baseline descrittiva")


def test_descriptive_baseline_is_never_a_delta_reference():
    p = _plan()                                   # nel blocco e come riferimento: rifiuto sul riferimento
    b = p["rq1a"]["blocks"][1]
    b["models"].append(_untr())
    b["delta_reference"] = _untr()
    _err(p, r"delta_reference.*baseline descrittiva")
    p = _plan()                                   # solo come riferimento: non è un modello del blocco
    p["rq1b"]["blocks"][0]["delta_reference"] = {"model": "conv_autoencoder", "config_id": CID_UNTR}
    _err(p, "delta_reference: deve essere uno dei modelli del blocco")


def test_baseline_rules_apply_only_to_declared_sections():
    p = _plan()                                   # RQ null: nessun controllo su sezioni assenti
    p["rq2"] = p["rq3"] = p["rq4"] = p["rq1b"] = None
    p["inputs"] = p["inputs"][:1]
    sp.validate_stats_plan(p)


def test_model_roles_of_the_output_schema():
    assert sp.MODEL_ROLES == ("compared", "reference", "descriptive_baseline", "random_baseline")


# ---------------------------------------------------------------- passo 1-ter: esperimento primario
def test_primary_experiment_is_mandatory():
    p = _plan()
    del p["primary_experiment"]
    _err(p, "primary_experiment")


@pytest.mark.parametrize("name", ["other_experiment", "", "a/b", None, 12])
def test_primary_experiment_must_be_a_declared_input(name):
    p = _plan()
    p["rq1b"] = None                              # solo il controllo sull'esperimento primario
    p["primary_experiment"] = name
    _err(p, "primary_experiment")
    if name == "other_experiment":
        _err(p, "non è tra gli esperimenti di inputs")


def test_primary_experiment_is_required_even_without_rq2():
    p = _plan()
    p["rq2"] = p["rq3"] = None
    assert sp.validate_stats_plan(p).primary_experiment == EXP_P
    del p["primary_experiment"]
    _err(p, "primary_experiment")


def test_rq1b_preproc_level_must_be_a_declared_input():
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"]["preproc"] = [EXP_P, "unknown_experiment"]
    _err(p, r"preproc.*unknown_experiment.*inputs")


@pytest.mark.parametrize("block", [0, 1])
def test_every_rq1b_block_includes_the_primary_experiment(block):
    p = _plan()
    b = p["rq1b"]["blocks"][block]
    b["factors"]["preproc"] = [EXP_S]
    if b["tau_reference"] is not None:
        b["tau_reference"]["preproc"] = EXP_S
    _err(p, r"primary_experiment.*D-40")


def test_rq1b_tau_reference_preproc_is_one_of_the_block_experiments():
    p = _plan()
    p["rq1b"]["blocks"][0]["factors"]["preproc"] = [EXP_P]
    p["rq1b"]["blocks"][0]["factors"]["threshold"].pop()     # resta lo scoring a due livelli
    p["rq1b"]["blocks"][0]["tau_reference"]["preproc"] = EXP_S
    _err(p, "tau_reference.preproc")


def test_single_input_plan_with_rq1b_on_the_primary_only():
    p = _plan()
    p["inputs"] = p["inputs"][:1]
    p["rq1b"]["blocks"][0]["factors"]["preproc"] = [EXP_P]
    p["rq1b"]["blocks"][0]["tau_reference"]["preproc"] = EXP_P
    plan = sp.validate_stats_plan(p)
    assert plan.primary_experiment == EXP_P


def test_every_input_has_a_role():
    p = _plan()                                   # synth_scaler compare solo come livello di preproc di RQ1b
    p["rq1b"] = None
    _err(p, r"inputs.*synth_scaler.*nessun ruolo")
    p = _plan()
    for blk in p["rq1b"]["blocks"]:
        blk["factors"]["preproc"] = [EXP_P]
        if blk["tau_reference"] is not None:
            blk["tau_reference"]["preproc"] = EXP_P
    _err(p, r"inputs.*synth_scaler.*nessun ruolo")


# ---------------------------------------------------------------- passo 6: gruppo minimo e sign-flip booleano
@pytest.mark.parametrize("value", [4, 2, 1, 0, -2, 5.5, True, None, "5"])
def test_min_machines_per_group_is_an_integer_of_at_least_five(value):
    p = _plan()
    p["rq2"]["min_machines_per_group"] = value
    _err(p, "rq2.min_machines_per_group")
    s = _rq3()
    s["min_machines_per_group"] = value
    with pytest.raises(ConfigError, match="rq3.min_machines_per_group"):
        sp._rq3(s, "exploratory")


def test_min_machines_per_group_five_is_admitted():
    p = _plan()
    p["rq2"]["min_machines_per_group"] = 5
    sp.validate_stats_plan(p)
    assert sp.MIN_MACHINES_PER_GROUP == 5


def test_min_machines_per_group_is_mandatory():
    p = _plan()
    del p["rq2"]["min_machines_per_group"]
    _err(p, "min_machines_per_group")
    s = _rq3()
    del s["min_machines_per_group"]
    with pytest.raises(ConfigError, match="min_machines_per_group"):
        sp._rq3(s, "exploratory")


def test_rq2_on_development_machines_is_rejected():
    p = _plan()
    p["stats"]["machine_role"] = "dev"
    _err(p, "2 macchine per gruppo")
    p["rq2"] = p["rq3"] = None
    sp.validate_stats_plan(p)                              # descrittive, RQ1 e RQ4 restano ammesse su dev


def test_rq3_on_development_machines_is_rejected_with_the_dev_message():
    p = _plan()
    p["stats"]["machine_role"] = "dev"
    p["rq2"] = None
    p["rq3"] = _rq3()
    _err(p, "2 macchine per gruppo")


@pytest.mark.parametrize("value", [{"n_perm": 2000, "seed": 1}, 1, "true", None])
def test_sign_flip_is_a_boolean(value):
    p = _plan()
    p["rq2"]["sign_flip"] = value
    _err(p, "sign_flip")


def test_min_machines_per_group_and_sign_flip_in_the_normalized_plan():
    d = sp.validate_stats_plan(_plan()).to_dict()
    assert d["rq2"]["min_machines_per_group"] == 5 and d["rq3"]["min_machines_per_group"] == 5
    assert d["rq2"]["sign_flip"] is True


# ---------------------------------------------------------------- passo 7: righe descrittive di RQ2
def test_rq2_baselines_and_random_baseline_are_mandatory():
    for key in ("baselines", "random_baseline"):
        p = _plan()
        del p["rq2"][key]
        _err(p, key)


def test_rq2_baselines_may_be_empty_and_random_baseline_is_a_boolean():
    p = _plan()
    p["rq2"]["baselines"] = []
    p["rq2"]["random_baseline"] = False
    d = sp.validate_stats_plan(p).to_dict()["rq2"]
    assert d["baselines"] == [] and d["random_baseline"] is False
    for v in (1, "true", None):
        q = _plan()
        q["rq2"]["random_baseline"] = v
        _err(q, "random_baseline")


def test_rq2_baselines_must_be_declared_descriptive_baselines():
    p = _plan()
    p["rq2"]["baselines"] = [_untr(), _m("lstm_autoencoder", "5123456789ab", "last", "mean", "none")]
    _err(p, r"rq2\.baselines.*descriptive_baselines")


def test_rq2_baselines_cannot_be_compared_models():
    p = _plan()
    p["rq2"]["baselines"] = [_conv()]                     # Conv-AE addestrato: confrontato, non baseline
    _err(p, "rq2.baselines")


def test_rq2_baselines_duplicates_and_random_are_rejected():
    p = _plan()
    p["rq2"]["baselines"] = [_untr(), _untr()]
    _err(p, "duplicat")
    q = _plan()
    q["rq2"]["baselines"] = [{"model": "random_baseline", "config_id": CID_UNTR, "window_time_agg": "none",
                              "sensor_agg": "native", "normalizer": "none"}]
    _err(q, "D-73 P7")


def test_rq2_baselines_in_the_normalized_plan():
    d = sp.validate_stats_plan(_plan()).to_dict()["rq2"]
    assert d["baselines"] == [dict(sorted(_untr().items()))] and d["random_baseline"] is True
