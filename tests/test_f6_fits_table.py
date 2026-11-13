"""
F6 passo 6.4: tabella ``fits`` dai ``meta.json`` (D-72 C6, V12; D-44) in ``analysis_grid.py``.

Una riga per fit pianificato: stato, ``set_seed_value`` (colonna seed), ``config_id``, epoche, componenti, tempi,
finestre, colonne dei non finiti (somma degli elementi sui record ``cal__*`` e ``test__*`` + ``test_ctx_head__*``;
null per i fit ``failed``). Meta reali da ``run_fit`` su dati sintetici in ``tmp_path`` (D-16), più casi a mano.
"""

import copy
import json

import pytest

from src.pipeline import analysis_grid as ag
from src.pipeline import fit as fitmod
from src.pipeline import table_schema as ts
from src.pipeline.store import Store
from src.utils import config as cfgmod
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo


@pytest.fixture(scope="module")
def metas(tmp_path_factory):
    root = tmp_path_factory.mktemp("f6_fits")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    spec = synth_spec(str(paths["data"]), str(root / "store"), mode="split3", step=1,
                      run_models=("pca", "conv_autoencoder", "rolling_zscore"), seeds=[42])
    real = fitmod.fit_one

    def fail_rolling(name, *a, **k):
        if name == "rolling_zscore":
            raise RuntimeError("errore simulato")
        return real(name, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(fitmod, "fit_one", fail_rolling)
    try:
        fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo)
    finally:
        mp.undo()
    store = Store(root / "store")
    out = {}
    for d in sorted((store.root / "fits").iterdir()):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        out[(m["machine"], m["model"]["name"])] = m
    return out


def test_one_valid_row_per_fit(metas):
    rows = [ag.fits_row(m) for m in metas.values()]
    df = ag.to_frame(rows, "fits")
    assert len(df) == len(metas) == 6 and set(df.status) == {"complete", "failed"}
    assert set(df.fit_key) == {m["fit_key"] for m in metas.values()}


def test_complete_fit_fields(metas):
    m = metas[("machine-9-1", "conv_autoencoder")]
    r = ag.fits_row(m)
    fi = m["fit_info"]
    assert (r["status"], r["model"], r["machine"], r["dataset"]) == ("complete", "conv_autoencoder", "machine-9-1",
                                                                     "synth")
    assert r["seed"] == m["seed"]["set_seed_value"] and r["deterministic"] is False and r["split_mode"] == "split3"
    assert (r["best_epoch"], r["stopped_epoch"]) == (fi["best_epoch"], fi["stopped_epoch"])
    assert r["n_components"] is None and r["error"] is None
    assert r["fit_sec"] == m["timings"]["fit_seconds"]
    assert r["n_fit_windows"] == fi["n_fit_samples"] and r["n_es_windows"] == fi["n_val_samples"]
    assert r["n_cal_windows"] == m["residuals"]["cal__last"]["shape"][0]
    assert r["nonfinite_loss"] is False and (r["n_nonfinite_cal"], r["n_nonfinite_test"]) == (0, 0)
    p = ag.fits_row(metas[("machine-9-1", "pca")])
    assert p["n_components"] == metas[("machine-9-1", "pca")]["fit_info"]["n_components"] and p["best_epoch"] is None
    assert p["deterministic"] is True and p["n_es_windows"] is None


def test_config_id_comes_from_the_fit_payload_and_matches_meta(metas):
    """config_id per tutti i fit dal fit_payload (anche i failed, il cui meta non lo registra), uguale a quello
    registrato nei meta dei fit complete e a config.config_id."""
    for m in metas.values():
        r = ag.fits_row(m)
        if m["status"] == "complete":
            assert r["config_id"] == m["model"]["config_id"]
        assert len(r["config_id"]) == 12
        payload = m["fit_payload"]["model"]
        assert r["config_id"] == cfgmod.canonical_hash({"model": payload["name"], "params": payload["params"]})[:12]


def test_failed_fit_row(metas):
    """Fit failed: errore «tipo: messaggio», seed = set_seed_value, campi del fit e colonne dei non finiti null."""
    m = metas[("machine-9-2", "rolling_zscore")]
    r = ag.fits_row(m)
    assert r["status"] == "failed" and r["error"] == "RuntimeError: errore simulato"
    assert r["seed"] == m["seed"]["set_seed_value"] and r["model"] == "rolling_zscore"
    for c in ("best_epoch", "stopped_epoch", "n_components", "fit_sec", "n_fit_windows", "n_es_windows",
              "n_cal_windows", "nonfinite_loss", "n_nonfinite_cal", "n_nonfinite_test"):
        assert r[c] is None, c
    ag.to_frame([r], "fits")


def test_nonfinite_counts_sum_elements_over_records(metas):
    """Somma degli elementi non finiti: cal = cal__last + cal__mean; test = test__* + test_ctx_head__* (es escluso).
    Caso a mano: cal__last 2, cal__mean 3, es__last 7, test__last 1, test_ctx_head__mean 4 → (5, 5)."""
    m = copy.deepcopy(metas[("machine-9-1", "pca")])
    for name, n in {"cal__last": 2, "cal__mean": 3, "es__last": 7, "test__last": 1, "test_ctx_head__mean": 4}.items():
        m["residuals"][name]["n_nonfinite"] = n
    r = ag.fits_row(m)
    assert (r["n_nonfinite_cal"], r["n_nonfinite_test"]) == (5, 5)


def test_nonfinite_loss_is_copied(metas):
    m = copy.deepcopy(metas[("machine-9-1", "conv_autoencoder")])
    m["nonfinite_loss"] = True
    assert ag.fits_row(m)["nonfinite_loss"] is True


def test_failed_fits_appear_only_in_fits(metas):
    """D-44: un fit failed non ha righe di metriche (nessun residuo da cui calcolarle); solo la riga in fits."""
    m = metas[("machine-9-1", "rolling_zscore")]
    assert m["status"] == "failed" and "residuals" not in m
    assert ag.fits_row(m)["status"] == "failed"


def test_measured_times_stay_in_the_table(metas):
    """fit_sec è nella tabella come misura (non entra nell'id, D-72 B12)."""
    m = copy.deepcopy(metas[("machine-9-1", "pca")])
    m["timings"]["fit_seconds"] = 123.5
    assert ag.fits_row(m)["fit_sec"] == 123.5
    assert "fit_sec" in ts.columns("fits")
