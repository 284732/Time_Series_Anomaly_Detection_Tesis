"""F6 passo 1: piano di analisi (D-72 C8), ``plan_sha256``, coerenza con l'esperimento, ``ANALYSIS_CODE_VERSION`` e
``analysis_id`` (D-72 B12, revisione del passo 0). Solo dati costruiti a mano: nessun file del repository, nessun fit."""

import copy
import re

import pytest

from src.utils import analysis_plan as ap
from src.utils.config import canonical_hash
from src.utils.schema import ConfigError, validate_experiment
from tests.fixtures.config_factory import raw_config

SHA_A = "a" * 64
SHA_B = "b" * 64


def raw_plan() -> dict:
    return copy.deepcopy({
        "scenarios": ["primary"],
        "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"},
                     {"sensor_agg": "mean", "normalizer": "std"},
                     {"sensor_agg": "max", "normalizer": "robust"}],
        "ts_aggs": ["last", "mean"],
        "thresholds": {
            "static_pct": [98.0, 99],
            "roll_quantile": [{"level": 99.0, "n_past": 1000}],
            "roll_robust_z": [{"level": 98, "n_past": 500}],
            "oracle": True,
        },
        "pa": ["static_pct", "oracle"],
    })


def _set(d, path, value):
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return d


def _del(d, path):
    node = d
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    return d


def _exp(**over):
    raw = raw_config()
    for path, v in over.items():
        _set(raw, path.split("."), v)
    return validate_experiment(raw)


# ------------------------------------------------------------------ schema (C8)
def test_valid_plan_is_parsed_and_levels_are_normalized():
    p = ap.validate_plan(raw_plan())
    assert p.scenarios == ("primary",)
    assert p.window_time_aggs == ("last", "mean")
    assert p.scorings == (("max", "robust"), ("mean", "none"), ("mean", "std"))
    assert p.ts_aggs == ("last", "mean")
    assert p.static_pct == (98.0, 99.0) and all(type(x) is float for x in p.static_pct)
    assert p.roll_quantile == ((99.0, 1000),)
    assert p.roll_robust_z == ((98.0, 500),) and type(p.roll_robust_z[0][0]) is float
    assert p.oracle is True
    assert p.pa == ("oracle", "static_pct")
    assert p.legacy_export is None


@pytest.mark.parametrize("key", ["scenarios", "window_time_aggs", "scorings", "ts_aggs", "thresholds", "pa"])
def test_every_top_key_is_required_no_defaults(key):
    with pytest.raises(ConfigError, match=key):
        ap.validate_plan(_del(raw_plan(), [key]))


@pytest.mark.parametrize("key", ["static_pct", "roll_quantile", "roll_robust_z", "oracle"])
def test_every_threshold_family_is_required_no_defaults(key):
    with pytest.raises(ConfigError, match=key):
        ap.validate_plan(_del(raw_plan(), ["thresholds", key]))


@pytest.mark.parametrize("path", [
    ["metrics"], ["thresholds", "burnin"], ["scorings", 0, "k"], ["thresholds", "roll_quantile", 0, "q"],
])
def test_unknown_keys_are_rejected(path):
    with pytest.raises(ConfigError, match="sconosciut"):
        ap.validate_plan(_set(raw_plan(), path, 1))


def test_unknown_key_in_legacy_export_is_rejected():
    raw = raw_plan()
    raw["legacy_export"] = {"threshold_percentile": 98.0, "timestamp_agg": "last", "window_time_agg": "last",
                            "plots": True}
    with pytest.raises(ConfigError, match="sconosciut"):
        ap.validate_plan(raw)


@pytest.mark.parametrize("path,value", [
    (["scenarios"], "primary"),                 # non lista
    (["scenarios"], []),                        # vuota
    (["scenarios"], ["primary", "primary"]),    # duplicati
    (["scenarios"], ["contesto"]),              # livello sconosciuto
    (["window_time_aggs"], ["max"]),            # solo le aggregazioni salvate (D-33)
    (["window_time_aggs"], ["none"]),           # none è solo la convenzione native, automatica
    (["ts_aggs"], ["median"]),
    (["ts_aggs"], []),
    (["scorings"], []),
    (["scorings"], [{"sensor_agg": "native", "normalizer": "none"}]),   # native: convenzione automatica
    (["scorings"], [{"sensor_agg": "mean", "normalizer": "minmax"}]),
    (["scorings"], [{"sensor_agg": "mean", "normalizer": "std"}, {"sensor_agg": "mean", "normalizer": "std"}]),
    (["scorings"], [{"sensor_agg": "mean"}]),
    (["thresholds"], [98.0]),
    (["thresholds", "static_pct"], 98.0),
    (["thresholds", "static_pct"], [0.0]),
    (["thresholds", "static_pct"], [100.5]),
    (["thresholds", "static_pct"], [98.0, 98]),          # duplicato dopo la normalizzazione
    (["thresholds", "static_pct"], [True]),
    (["thresholds", "static_pct"], [float("nan")]),
    (["thresholds", "roll_quantile"], [{"level": 99.0}]),
    (["thresholds", "roll_quantile"], [{"level": 99.0, "n_past": 0}]),
    (["thresholds", "roll_quantile"], [{"level": 99.0, "n_past": 10.0}]),
    (["thresholds", "roll_quantile"], [{"level": 99.0, "n_past": True}]),
    (["thresholds", "roll_quantile"], [{"level": 0, "n_past": 10}]),
    (["thresholds", "roll_robust_z"], [{"level": 98, "n_past": 5}, {"level": 98.0, "n_past": 5}]),
    (["thresholds", "oracle"], 1),
    (["thresholds", "oracle"], "true"),
    (["pa"], "oracle"),
    (["pa"], ["oracle", "oracle"]),
    (["pa"], ["burnin"]),
])
def test_invalid_values_are_rejected(path, value):
    with pytest.raises(ConfigError):
        ap.validate_plan(_set(raw_plan(), path, value))


def test_empty_threshold_families_are_explicit_and_allowed():
    raw = raw_plan()
    raw["thresholds"] = {"static_pct": [], "roll_quantile": [], "roll_robust_z": [], "oracle": False}
    raw["pa"] = []
    p = ap.validate_plan(raw)
    assert p.static_pct == () and p.roll_quantile == () and p.roll_robust_z == () and p.oracle is False


@pytest.mark.parametrize("policy,disable", [
    ("static_pct", ("static_pct", [])),
    ("oracle", ("oracle", False)),
    ("roll_quantile", ("roll_quantile", [])),
])
def test_pa_only_for_active_policies(policy, disable):
    raw = raw_plan()
    raw["thresholds"][disable[0]] = disable[1]
    raw["pa"] = [policy]
    with pytest.raises(ConfigError, match="pa"):
        ap.validate_plan(raw)


def test_rolling_thresholds_require_last_among_ts_aggs():
    """D-24: le soglie mobili si applicano solo a ``last``; un piano che le chiede senza ``last`` non darebbe righe."""
    raw = raw_plan()
    raw["ts_aggs"] = ["mean", "max"]
    with pytest.raises(ConfigError, match="D-24"):
        ap.validate_plan(raw)
    raw["thresholds"]["roll_quantile"] = []
    raw["thresholds"]["roll_robust_z"] = []
    ap.validate_plan(raw)          # senza soglie mobili gli offline da soli sono ammessi


def test_legacy_export_block():
    raw = raw_plan()
    raw["legacy_export"] = {"threshold_percentile": 98, "timestamp_agg": "max", "window_time_agg": "mean"}
    p = ap.validate_plan(raw)
    assert p.legacy_export == ap.LegacyExport(98.0, "max", "mean")
    for k in ("threshold_percentile", "timestamp_agg", "window_time_agg"):
        bad = copy.deepcopy(raw)
        del bad["legacy_export"][k]
        with pytest.raises(ConfigError, match=k):
            ap.validate_plan(bad)
    for k, v in (("threshold_percentile", 0), ("timestamp_agg", "median"), ("window_time_agg", "max")):
        bad = copy.deepcopy(raw)
        bad["legacy_export"][k] = v
        with pytest.raises(ConfigError):
            ap.validate_plan(bad)


def test_plan_must_be_a_mapping():
    for raw in (None, [], "piano", 3):
        with pytest.raises(ConfigError):
            ap.validate_plan(raw)


# ------------------------------------------------------------------ file YAML
def test_load_plan_from_yaml(tmp_path):
    f = tmp_path / "piano.yaml"
    f.write_text(
        "scenarios: [primary]\nwindow_time_aggs: [last]\n"
        "scorings:\n  - {sensor_agg: mean, normalizer: std}\nts_aggs: [last]\n"
        "thresholds:\n  static_pct: [98.0]\n  roll_quantile: []\n  roll_robust_z: []\n  oracle: true\n"
        "pa: []\n", encoding="utf-8")
    p = ap.load_analysis_plan(f)
    assert p.scorings == (("mean", "std"),) and p.static_pct == (98.0,)


def test_load_plan_rejects_duplicate_yaml_keys(tmp_path):
    f = tmp_path / "piano.yaml"
    f.write_text("scenarios: [primary]\nscenarios: [test_context]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicata"):
        ap.load_analysis_plan(f)


def test_load_plan_rejects_missing_or_empty_file(tmp_path):
    with pytest.raises(ConfigError):
        ap.load_analysis_plan(tmp_path / "manca.yaml")
    f = tmp_path / "vuoto.yaml"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        ap.load_analysis_plan(f)


# ------------------------------------------------------------------ plan_sha256
def test_plan_sha256_is_hex_and_stable():
    a = ap.plan_sha256(ap.validate_plan(raw_plan()))
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    assert a == ap.plan_sha256(ap.validate_plan(raw_plan()))


def test_plan_sha256_ignores_order_int_float_and_yaml_form(tmp_path):
    """Stesso contenuto semantico → stesso hash: ordine delle liste, 98 contro 98.0, commenti e formato del file."""
    raw = raw_plan()
    perm = raw_plan()
    for k in ("window_time_aggs", "scorings", "ts_aggs", "pa"):
        perm[k] = list(reversed(perm[k]))
    perm["thresholds"]["static_pct"] = [99.0, 98]
    perm["thresholds"]["roll_robust_z"] = [{"n_past": 500, "level": 98.0}]
    assert ap.plan_sha256(ap.validate_plan(raw)) == ap.plan_sha256(ap.validate_plan(perm))
    base = ("window_time_aggs: [last]\nscorings:\n  - {sensor_agg: mean, normalizer: std}\nts_aggs: [last]\n"
            "thresholds:\n  static_pct: [98]\n  roll_quantile: []\n  roll_robust_z: []\n  oracle: false\npa: []\n")
    f1, f2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    f1.write_text("scenarios: [primary]\n" + base, encoding="utf-8")
    f2.write_text("# commento\n" + base.replace("[98]", "[98.0]") + "scenarios:\n  - primary\n", encoding="utf-8")
    assert ap.plan_sha256(ap.load_analysis_plan(f1)) == ap.plan_sha256(ap.load_analysis_plan(f2))


@pytest.mark.parametrize("path,value", [
    (["scenarios"], ["primary", "test_context"]),
    (["window_time_aggs"], ["last"]),
    (["scorings", 0, "normalizer"], "robust"),
    (["ts_aggs"], ["last", "max"]),
    (["thresholds", "static_pct"], [98.0]),
    (["thresholds", "roll_quantile", 0, "level"], 99.5),
    (["thresholds", "roll_quantile", 0, "n_past"], 1001),
    (["thresholds", "roll_robust_z"], []),
    (["thresholds", "oracle"], False),
    (["pa"], ["static_pct"]),
    (["legacy_export"], {"threshold_percentile": 98.0, "timestamp_agg": "last", "window_time_agg": "last"}),
])
def test_plan_sha256_changes_with_every_level(path, value):
    raw = raw_plan()
    if path == ["thresholds", "oracle"]:
        raw["pa"] = ["static_pct"]
    ref = ap.plan_sha256(ap.validate_plan(raw))
    assert ap.plan_sha256(ap.validate_plan(_set(copy.deepcopy(raw), path, value))) != ref


# ------------------------------------------------------------------ coerenza con l'esperimento
def test_test_context_requires_split3_with_step_1():
    """D-60: lo scenario ``test_context`` esiste solo in ``split3`` con ``step`` = 1."""
    p = ap.validate_plan(_set(raw_plan(), ["scenarios"], ["primary", "test_context"]))
    with pytest.raises(ConfigError, match="test_context"):
        ap.check_plan_for_experiment(p, _exp())                                    # legacy2
    split3 = {"split": {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}}
    raw = raw_config()
    raw.update(split3)
    raw["windows"]["step"] = 2
    with pytest.raises(ConfigError, match="test_context"):
        ap.check_plan_for_experiment(p, validate_experiment(raw))
    raw["windows"]["step"] = 1
    ap.check_plan_for_experiment(p, validate_experiment(raw))


def test_legacy_export_requires_legacy2():
    raw = raw_plan()
    raw["legacy_export"] = {"threshold_percentile": 98.0, "timestamp_agg": "last", "window_time_agg": "last"}
    p = ap.validate_plan(raw)
    ap.check_plan_for_experiment(p, _exp())
    rc = raw_config()
    rc["split"] = {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}
    with pytest.raises(ConfigError, match="legacy_export"):
        ap.check_plan_for_experiment(p, validate_experiment(rc))


def test_primary_plan_is_coherent_with_both_modes():
    p = ap.validate_plan(raw_plan())
    ap.check_plan_for_experiment(p, _exp())
    rc = raw_config()
    rc["split"] = {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}
    ap.check_plan_for_experiment(p, validate_experiment(rc))


def test_vocabularies_come_from_table_schema_and_residual_cache():
    """Un solo punto per i livelli: quelli del piano sono un sottoinsieme di quelli della tabella lunga."""
    from src.pipeline import table_schema as ts
    from src.pipeline import residual_cache as rc
    from src.utils.schema import RESIDUAL_AGGS
    assert ap.SCENARIOS == ts.SCENARIOS and ap.TS_AGGS == ts.TS_AGGS and ap.NORMALIZERS == ts.NORMALIZERS
    assert ap.WINDOW_TIME_AGGS == RESIDUAL_AGGS
    assert set(ap.SENSOR_AGGS) == set(ts.SENSOR_AGGS) - {"native"}
    assert set(ap.PA_POLICIES) == set(ts.THRESHOLD_POLICIES)
    assert ap.ANALYSIS_SEGMENTS == tuple(s for s in rc.SEGMENT_ORDER if s not in rc.DIAGNOSTIC_SEGMENTS)
    assert ap.RESIDUAL_SEP == rc.SEP


# ------------------------------------------------------------------ ANALYSIS_CODE_VERSION e analysis_id
SHA_M = "c" * 64          # meta_sha256 fittizio


def F(key, status, residuals, meta=SHA_M):
    return ap.FitIdentity(key, status, meta, residuals)


def _fits():
    return [F("k2" * 10, "complete", {"cal__last": SHA_A, "test__last": SHA_B}), F("k1" * 10, "failed", {})]


ENV = {"python": "3.12.0", "packages": {"numpy": "2.3.0", "pandas": "2.3.0", "scipy": "1.16.0",
                                        "sklearn": "1.9.0"}, "lock_sha256": "4" * 64}


def _components(**over):
    kw = dict(plan_sha256="0" * 64, fits=_fits(),
              labels_versions={"machine-9-2": "lv1-b", "machine-9-1": "lv1-a"}, manifest_sha256="1" * 64,
              random_baseline_keys=["rb2" * 5, "rb1" * 5], environment=ENV)
    kw.update(over)
    return ap.analysis_id_components(**kw)


def test_analysis_code_version_is_a_positive_int():
    assert isinstance(ap.ANALYSIS_CODE_VERSION, int) and not isinstance(ap.ANALYSIS_CODE_VERSION, bool)
    assert ap.ANALYSIS_CODE_VERSION >= 1


def test_analysis_id_is_12_hex_of_the_components_in_clear():
    c = _components()
    assert set(c) == {"plan_sha256", "analysis_code_version", "fits", "labels_versions", "manifest_sha256",
                      "random_baseline_keys", "environment"}
    assert c["analysis_code_version"] == ap.ANALYSIS_CODE_VERSION
    assert c["plan_sha256"] == "0" * 64 and c["manifest_sha256"] == "1" * 64
    assert [f["fit_key"] for f in c["fits"]] == ["k1" * 10, "k2" * 10]          # ordinati per fit_key
    assert c["fits"][0] == {"fit_key": "k1" * 10, "status": "failed", "meta_sha256": SHA_M, "residual_sha256": {}}
    assert c["fits"][1]["residual_sha256"] == {"cal__last": SHA_A, "test__last": SHA_B}
    assert list(c["labels_versions"]) == ["machine-9-1", "machine-9-2"]           # ordinate per macchina
    assert c["random_baseline_keys"] == ["rb1" * 5, "rb2" * 5]                    # ordinate
    assert c["environment"] == ENV
    aid = ap.analysis_id(c)
    assert re.fullmatch(r"[0-9a-f]{12}", aid)
    assert aid == canonical_hash(c)[:12]


def test_analysis_id_ignores_input_order():
    a = ap.analysis_id(_components())
    b = ap.analysis_id(_components(fits=list(reversed(_fits())),
                                   labels_versions={"machine-9-1": "lv1-a", "machine-9-2": "lv1-b"},
                                   random_baseline_keys=["rb1" * 5, "rb2" * 5]))
    assert a == b


@pytest.mark.parametrize("change", [
    "plan", "code_version", "fit_key", "status", "meta", "residual_sha", "residual_set", "labels", "manifest",
    "random_baseline_seed", "random_baseline_machine", "environment_package", "environment_lock",
])
def test_analysis_id_changes_with_every_component(change, monkeypatch):
    ref = ap.analysis_id(_components())
    fits = _fits()
    over = {}
    if change == "plan":
        over["plan_sha256"] = "2" * 64
    elif change == "code_version":
        monkeypatch.setattr(ap, "ANALYSIS_CODE_VERSION", ap.ANALYSIS_CODE_VERSION + 1)
    elif change == "fit_key":
        fits[1] = F("k3" * 10, "failed", {})
    elif change == "status":
        fits[1] = F("k1" * 10, "complete", {"cal__last": SHA_A})
    elif change == "meta":                       # es. set_seed_value, epoche, fit_sec, errore diversi
        fits[1] = F("k1" * 10, "failed", {}, meta="d" * 64)
    elif change == "residual_sha":
        fits[0] = F("k2" * 10, "complete", {"cal__last": SHA_A, "test__last": SHA_A})
    elif change == "residual_set":
        fits[0] = F("k2" * 10, "complete", {"cal__last": SHA_A, "test__last": SHA_B, "test_ctx_head__last": SHA_A})
    elif change == "labels":
        over["labels_versions"] = {"machine-9-1": "lv1-a", "machine-9-2": "lv1-c"}
    elif change == "manifest":
        over["manifest_sha256"] = "3" * 64
    elif change == "random_baseline_seed":       # seed in più: stessi fit_key dei deterministici, righe diverse
        over["random_baseline_keys"] = ["rb2" * 5, "rb1" * 5, "rb3" * 5]
    elif change == "random_baseline_machine":
        over["random_baseline_keys"] = ["rb1" * 5]
    elif change == "environment_package":
        over["environment"] = {**ENV, "packages": {**ENV["packages"], "sklearn": "1.9.1"}}
    elif change == "environment_lock":
        over["environment"] = {**ENV, "lock_sha256": "5" * 64}
    if change in ("fit_key", "status", "meta", "residual_sha", "residual_set"):
        over["fits"] = fits
    assert ap.analysis_id(_components(**over)) != ref


@pytest.mark.parametrize("fits", [
    [],                                                                          # nessun fit
    [F("k1" * 10, "failed", {"cal__last": SHA_A})],                              # failed con residui
    [F("k1" * 10, "complete", {})],                                              # complete senza residui
    [F("k1" * 10, "complete", {"cal__last": SHA_A}), F("k1" * 10, "complete", {"cal__last": SHA_B})],  # duplicato
    [F("k1" * 10, "running", {})],                                               # stato non ammesso
    [F("k1" * 10, "complete", {"cal__last": "abc"})],                            # sha256 non valido
    [F("k1" * 10, "complete", {"cal__last": SHA_A}, meta="zz")],                 # meta_sha256 non valido
    [F("k1" * 10, "complete", {"es__last": SHA_A})],                             # es mai letto (D-04)
    [F("k1" * 10, "complete", {"cal__max": SHA_A})],                             # aggregazione non salvata
    [F("k1" * 10, "complete", {"cal": SHA_A})],                                  # nome senza segmento
])
def test_analysis_id_components_reject_invalid_fits(fits):
    with pytest.raises(ValueError):
        _components(fits=fits)


@pytest.mark.parametrize("over", [
    {"labels_versions": {}},
    {"labels_versions": {"machine-9-1": ""}},
    {"plan_sha256": "xyz"},
    {"manifest_sha256": ""},
    {"random_baseline_keys": []},
    {"random_baseline_keys": ["rb1" * 5, "rb1" * 5]},
    {"random_baseline_keys": [""]},
    {"environment": {}},
])
def test_analysis_id_components_reject_invalid_inputs(over):
    with pytest.raises(ValueError):
        _components(**over)


def _meta(**over):
    from src.utils import config as cfgmod
    payload = {"fit_code_version": 1, "machine": "machine-9-1", "model": {"name": "pca"}}
    m = {"fit_key": cfgmod.canonical_hash(payload)[:cfgmod.FIT_KEY_LEN], "fit_payload": payload,
         "status": "complete", "seed": {"key_seed": 42, "set_seed_value": 42},
         "fit_info": {"best_epoch": None, "n_components": 3}, "timings": {"fit_seconds": 1.5},
         "residuals": {"es__last": {"sha256": SHA_B}, "cal__last": {"sha256": SHA_A},
                       "test__last": {"sha256": SHA_B}},
         "provenance": {"timestamp_utc": "2026-11-13T00:00:00Z"}, "created_by": {"run_id": "r1"}}
    m.update(over)
    return m


def test_fit_identity_from_meta_uses_the_array_sha256_of_meta_json():
    """Identità dei residui = sha256 per array di ``meta.json`` (F5), mai ``npz_sha256``; mai i residui di es."""
    meta = _meta()
    fi = ap.fit_identity_from_meta(meta, ["test__last", "cal__last"])
    assert (fi.fit_key, fi.status) == (meta["fit_key"], "complete")
    assert dict(fi.residual_sha256) == {"cal__last": SHA_A, "test__last": SHA_B}
    with pytest.raises(ValueError, match="es"):
        ap.fit_identity_from_meta(meta, ["es__last"])
    with pytest.raises(ValueError):
        ap.fit_identity_from_meta(meta, ["test__mean"])                         # non registrato
    failed = _meta(status="failed", error={"type": "RuntimeError", "message": "x", "phase": "fit"})
    del failed["residuals"]
    fi = ap.fit_identity_from_meta(failed, ["cal__last"])
    assert fi.status == "failed" and dict(fi.residual_sha256) == {}
    with pytest.raises(ValueError):
        ap.fit_identity_from_meta(_meta(status="running"), [])


def _full_meta(status="complete"):
    """meta.json con tutti i campi di META_ID_FIELDS (complete) o quelli di un fit failed, più i campi misurati."""
    m = _meta(status=status, meta_format=1, fit_code_version=1, machine="machine-9-1", dataset="synth", group="9",
              data_version="dv1-x", model={"name": "pca", "native_only": False},
              timings={"fit_seconds": 1.5, "residual_seconds": 0.5, "total_seconds": 2.0})
    if status == "complete":
        m.update(segments={"mode": "legacy2", "cal": [1600, 2000], "n_test": 2000}, windows={"window_size": 20},
                 scoring={"score_batch_requested": None}, device={"requested": "cpu", "effective": "cpu"},
                 fit_info={"best_epoch": 2, "stopped_epoch": 2, "n_components": 3, "fit_seconds": 0.7},
                 nonfinite_loss=False, checkpoint={"hashable_sha256": SHA_A}, scaler={"arrays": {}},
                 loaded_with_version_mismatch=False)
    else:
        del m["residuals"], m["fit_info"]
        m["error"] = {"type": "RuntimeError", "message": "x", "phase": "fit", "traceback": "File C:\\a.py, line 1"}
    return m


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_every_meta_field_is_classified(status):
    """Ogni campo di meta.json è in META_ID_FIELDS o in META_MEASURED_FIELDS (elenchi espliciti, disgiunti)."""
    meta = _full_meta(status)
    assert not set(ap.META_ID_FIELDS) & set(ap.META_MEASURED_FIELDS)
    top_measured = {f for f in ap.META_MEASURED_FIELDS if "." not in f}
    assert set(meta) <= set(ap.META_ID_FIELDS) | top_measured
    ap.fit_identity_from_meta(meta, ["cal__last"])


def test_timing_subfields_of_fitinfo_are_measured():
    """I campi di tempo di FitInfo (D-67) sono misure: un nuovo campo di tempo va aggiunto all'elenco."""
    from src.models.base import FitInfo
    assert {f"fit_info.{f}" for f in FitInfo.TIMING_FIELDS} <= set(ap.META_MEASURED_FIELDS)


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_measured_fields_do_not_change_the_identity(status):
    names = ["cal__last"]
    ref = ap.fit_identity_from_meta(_full_meta(status), names).meta_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", ref)
    for field in ap.META_MEASURED_FIELDS:
        meta = _full_meta(status)
        top, _, sub = field.partition(".")
        if top not in meta or (sub and sub not in meta[top]):
            continue
        if sub:
            meta[top][sub] = "valore misurato diverso"
        else:
            meta[top] = {"diverso": 1}
        assert ap.fit_identity_from_meta(meta, names).meta_sha256 == ref, field


@pytest.mark.parametrize("status", ["complete", "failed"])
def test_every_identity_field_changes_the_identity(status):
    names = ["cal__last"]
    base = _full_meta(status)
    ref = ap.fit_identity_from_meta(base, names).meta_sha256
    changed = 0
    for field in ap.META_ID_FIELDS:
        if field not in base or field in ("fit_key", "fit_payload", "status"):
            continue                       # fit_key/fit_payload verificati a parte; status cambia la forma
        meta = _full_meta(status)
        meta[field] = dict(meta[field], test__mean={"sha256": SHA_A}) if field == "residuals" else {"diverso": field}
        assert ap.fit_identity_from_meta(meta, names).meta_sha256 != ref, field
        changed += 1
    assert changed >= (8 if status == "failed" else 18)
    for sub_change in ({"seed": {"key_seed": 42, "set_seed_value": 43}},
                       {"error": dict(base.get("error") or {}, message="y")} if status == "failed" else
                       {"fit_info": dict(base["fit_info"], stopped_epoch=3)}):
        meta = _full_meta(status)
        meta.update(sub_change)
        assert ap.fit_identity_from_meta(meta, names).meta_sha256 != ref, sub_change


def test_unknown_meta_field_is_an_explicit_error():
    meta = _full_meta()
    meta["campo_nuovo"] = 1
    with pytest.raises(ValueError, match="campo_nuovo"):
        ap.fit_identity_from_meta(meta, ["cal__last"])


def test_fit_identity_rejects_fit_key_not_matching_payload():
    """Il fit_key deve essere l'hash del ``fit_payload`` registrato: è ciò che garantisce che finestre, confini dei
    segmenti, preprocessing e data_version sono coperti dall'id."""
    with pytest.raises(ValueError, match="fit_payload"):
        ap.fit_identity_from_meta(_meta(fit_key="f" * 20), ["cal__last"])


def test_manifest_sha256_is_the_canonical_hash_of_to_dict():
    class FakeManifest:
        def to_dict(self):
            return {"dataset": "x", "machines": {"m": {"labels_version": "lv1-a"}}}
    assert ap.manifest_sha256(FakeManifest()) == canonical_hash(FakeManifest().to_dict())


def test_analysis_environment_lists_the_numeric_libraries_without_torch():
    """Ambiente dell'analisi: Python, librerie numeriche dell'analisi e sha256 del lock; niente torch né thread
    (l'analisi non usa torch se non nella rigenerazione, verificata contro gli sha256 dei residui)."""
    env = ap.analysis_environment()
    assert set(env) == {"python", "packages", "lock_sha256"}
    assert set(env["packages"]) == {"numpy", "scipy", "pandas", "sklearn"}
    assert all(isinstance(v, str) and v for v in env["packages"].values())


def test_random_baseline_keys_cover_seeds_machines_and_segments():
    """Le pseudo-chiavi della baseline random (``config.random_baseline_key``) cambiano con seed, macchina, split,
    finestre e data_version: è ciò che copre la baseline, che non ha fit nello store."""
    from src.utils.config import random_baseline_key
    cfg = _exp()
    ref = random_baseline_key(cfg, "machine-9-1", 42, "dv-a")
    assert random_baseline_key(cfg, "machine-9-1", 43, "dv-a") != ref
    assert random_baseline_key(cfg, "machine-9-2", 42, "dv-a") != ref
    assert random_baseline_key(cfg, "machine-9-1", 42, "dv-b") != ref
    assert random_baseline_key(_exp(**{"split.val_ratio": 0.3}), "machine-9-1", 42, "dv-a") != ref
    assert random_baseline_key(_exp(**{"windows.window_size": 50}), "machine-9-1", 42, "dv-a") != ref


# ------------------------------------------------------------------ livelli ripetuti
@pytest.mark.parametrize("path,value", [
    (["scenarios"], ["primary", "primary"]),
    (["window_time_aggs"], ["last", "last"]),
    (["ts_aggs"], ["last", "last"]),
    (["scorings"], [{"sensor_agg": "max", "normalizer": "std"}, {"sensor_agg": "max", "normalizer": "std"}]),
    (["thresholds", "static_pct"], [99.5, 99.5]),
    (["thresholds", "roll_quantile"], [{"level": 99, "n_past": 10}, {"level": 99.0, "n_past": 10}]),
    (["pa"], ["static_pct", "static_pct"]),
])
def test_repeated_levels_are_rejected(path, value):
    with pytest.raises(ConfigError, match="duplicat"):
        ap.validate_plan(_set(raw_plan(), path, value))
