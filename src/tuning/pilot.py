"""
Stima del costo dei fit di tuning dal pilota sui tempi (F9 passo 3, D-41) e regola K = 8 / 6.

Formula di D-41 (tempo a muro: secondi di ``timings`` di ``meta.json``, non tempo di CPU), sommata sui fit nuovi
degli slot 2..k (lo slot 1 riusa i fit di V1):
  * modelli addestrati: secondi per epoca del pilota (``timings.fit_seconds`` / ``fit_info.stopped_epoch`` del fit del
    pilota con stessa macchina e stesso punto, seed del pilota, usato per ogni seed) × mediana di ``stopped_epoch`` dei
    fit di V1 della stessa famiglia + mediana di ``residual_seconds`` dei fit di V1 della stessa famiglia;
  * PCA, PCA puntuale e baseline (deterministici): ``total_seconds`` del fit di V1 della stessa macchina.
Regola: K = ``k_reduced`` solo se la stima con K = k supera ``threshold_hours`` (confronto stretto: 40 h esatte danno
K = k). Si riporta anche la stima con gli slot 2..k_reduced, solo come informazione.

Ingressi: solo ``meta.json`` (tempi, epoche, ``fit_payload``, stato). Nessuna metrica, nessuna tabella, nessuna label.
Controlli (``EstimateError``): ogni fit di riferimento e del pilota presente e ``complete``; ``stopped_epoch`` ≥ 1;
il fit del pilota ha lo stesso ``fit_payload`` del fit di tuning che rappresenta, salvo seed e
``training.max_epochs`` / ``training.patience``, che devono essere quelli del pilota.
"""

import copy
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils import config as cfgmod
from src.utils import schema as _schema

HOUR = 3600.0


class EstimateError(ValueError):
    """Ingressi del pilota o del riferimento mancanti o incoerenti: nessuna stima."""


def _complete(meta: Optional[Mapping[str, Any]], what: str, task: Any) -> Mapping[str, Any]:
    if meta is None:
        raise EstimateError(f"{what}: fit mancante {task.machine} {task.model} seed={task.seed} {task.fit_key}")
    if meta.get("status") != "complete":
        raise EstimateError(f"{what}: fit {task.fit_key} ({task.machine} {task.model}) con stato "
                            f"{meta.get('status')!r}, atteso 'complete'")
    return meta


def _epochs(meta: Mapping[str, Any], what: str) -> int:
    e = (meta.get("fit_info") or {}).get("stopped_epoch")
    if not isinstance(e, int) or isinstance(e, bool) or e < 1:
        raise EstimateError(f"{what}: stopped_epoch {e!r} non valido nel fit {meta.get('fit_key')}")
    return e


def _timing(meta: Mapping[str, Any], key: str) -> float:
    v = (meta.get("timings") or {}).get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not v >= 0:
        raise EstimateError(f"timings.{key} {v!r} non valido nel fit {meta.get('fit_key')}")
    return float(v)


def _comparable(payload: Mapping[str, Any]) -> str:
    """Payload senza seed ed epoche, come JSON canonico del ``fit_key`` (tuple → liste; 1 ≠ 1.0 ≠ True, che il
    confronto tra oggetti Python confonderebbe): il ``meta.json`` riletto ha liste dove il payload pianificato ha
    tuple."""
    p = copy.deepcopy(dict(payload))
    p.pop("seed", None)
    if "training" in p:
        p["training"] = {k: v for k, v in p["training"].items() if k not in ("max_epochs", "patience")}
    return cfgmod.canonical_json(p)


def estimate(*, reference: Sequence[Tuple[Any, Optional[Mapping[str, Any]]]],
             pilot: Mapping[int, Sequence[Tuple[Any, Optional[Mapping[str, Any]]]]],
             targets: Mapping[int, Sequence[Any]], pilot_max_epochs: int, pilot_patience: int,
             threshold_hours: float, k: int, k_reduced: int) -> Dict[str, Any]:
    """Stima e decisione K. ``reference``: (FitTask, meta) dei fit di V1 (slot 1); ``pilot``: {slot: (FitTask, meta)};
    ``targets``: {slot: FitTask} dei fit di tuning degli slot 2..k."""
    slots = sorted(targets)
    if slots != list(range(2, k + 1)):
        raise EstimateError(f"slot di tuning {slots}, attesi 2..{k} (lo slot 1 riusa i fit di V1)")
    if not 1 <= k_reduced < k:
        raise EstimateError(f"k_reduced {k_reduced} non valido con k = {k}")

    # riferimento (V1): mediane per famiglia addestrata, total_seconds per (modello, macchina) dei deterministici
    stops: Dict[str, List[int]] = {}
    resid: Dict[str, List[float]] = {}
    det_total: Dict[Tuple[str, str], float] = {}
    for task, meta in reference:
        meta = _complete(meta, "riferimento (V1)", task)
        if _schema.MODEL_SCHEMAS[task.model].trained:
            stops.setdefault(task.model, []).append(_epochs(meta, "riferimento (V1)"))
            resid.setdefault(task.model, []).append(_timing(meta, "residual_seconds"))
        else:
            det_total[(task.model, task.machine)] = _timing(meta, "total_seconds")
    med_stop = {m: statistics.median(v) for m, v in stops.items()}
    med_res = {m: statistics.median(v) for m, v in resid.items()}

    # pilota: secondi per epoca per (slot, modello, macchina), con il controllo del fit_payload
    spe: Dict[Tuple[int, str, str], float] = {}
    pay: Dict[Tuple[int, str, str], str] = {}
    for slot, items in pilot.items():
        for task, meta in items:
            meta = _complete(meta, f"pilota slot {slot}", task)
            tr = (meta.get("fit_payload") or {}).get("training") or {}
            if (tr.get("max_epochs"), tr.get("patience")) != (pilot_max_epochs, pilot_patience):
                raise EstimateError(f"pilota slot {slot}: fit {task.fit_key} con max_epochs / patience "
                                    f"{tr.get('max_epochs')} / {tr.get('patience')}, attesi {pilot_max_epochs} / "
                                    f"{pilot_patience}")
            key = (slot, task.model, task.machine)
            if key in spe:
                raise EstimateError(f"pilota slot {slot}: più fit per {task.model} su {task.machine}")
            spe[key] = _timing(meta, "fit_seconds") / _epochs(meta, f"pilota slot {slot}")
            pay[key] = _comparable(meta["fit_payload"])

    rows: List[Dict[str, Any]] = []
    for slot in slots:
        for t in targets[slot]:
            if _schema.MODEL_SCHEMAS[t.model].trained:
                key = (slot, t.model, t.machine)
                if key not in spe:
                    raise EstimateError(f"pilota slot {slot}: manca il fit di {t.model} su {t.machine}")
                if pay[key] != _comparable(t.payload):
                    raise EstimateError(f"pilota slot {slot}: il fit di {t.model} su {t.machine} non ha lo stesso "
                                        "fit_payload del fit di tuning (salvo seed ed epoche): il pilota non misura "
                                        "quel punto")
                if t.model not in med_stop:
                    raise EstimateError(f"riferimento (V1): nessun fit di {t.model}")
                sec = spe[key] * med_stop[t.model] + med_res[t.model]
            else:
                if (t.model, t.machine) not in det_total:
                    raise EstimateError(f"riferimento (V1): manca il fit di {t.model} su {t.machine}")
                sec = det_total[(t.model, t.machine)]
            rows.append({"slot": slot, "model": t.model, "machine": t.machine, "seed": t.seed,
                         "fit_key": t.fit_key, "seconds": sec})

    def total(upto: int) -> float:
        return sum(r["seconds"] for r in rows if r["slot"] <= upto)

    sec_k, sec_red = total(k), total(k_reduced)
    hours_k = sec_k / HOUR
    chosen = k_reduced if hours_k > threshold_hours else k
    per_model: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = per_model.setdefault(r["model"], {"n_fits": 0, "seconds": 0.0})
        d["n_fits"] += 1
        d["seconds"] += r["seconds"]
    for m, d in per_model.items():
        if m in med_stop:
            d.update(v1_median_stopped_epoch=med_stop[m], v1_median_residual_seconds=med_res[m],
                     pilot_seconds_per_epoch_median=statistics.median(v for (s, mm, _), v in spe.items() if mm == m))
    return {
        "threshold_hours": threshold_hours,
        "estimate_hours_k": hours_k,
        "estimate_hours_k_reduced": sec_red / HOUR,
        "k": k, "k_reduced": k_reduced,
        "chosen_k": chosen,
        "rule": f"K = {k_reduced} solo se la stima con K = {k} supera {threshold_hours} h a muro, altrimenti K = {k}",
        "per_model": per_model,
        "per_slot_hours": {s: sum(r["seconds"] for r in rows if r["slot"] == s) / HOUR for s in slots},
        "n_fits": len(rows),
        "rows": rows,
    }


def tasks_with_meta(tasks: Iterable[Any], read_meta) -> List[Tuple[Any, Optional[Mapping[str, Any]]]]:
    """(FitTask, meta.json o None) leggendo solo ``meta.json`` dallo store."""
    return [(t, read_meta(t.fit_key)) for t in tasks]


def run_estimate(grid: Any, reference_cfg: Any, pilot_cfgs: Mapping[int, Any], target_cfgs: Mapping[int, Any],
                 data_versions: Mapping[str, str], reference_read_meta, pilot_read_meta) -> Dict[str, Any]:
    """Compiti dalle config (``plan_tasks``, nessuna lettura di dati) e ``meta.json`` dagli store, poi ``estimate``.
    Le config devono avere le stesse macchine; il riferimento è lo slot 1 (= V1 di F8, stessi ``fit_key``)."""
    from src.pipeline.fit_plan import plan_tasks
    machines = tuple(reference_cfg.data.machines)
    for name, cfg in [("pilota", c) for c in pilot_cfgs.values()] + [("tuning", c) for c in target_cfgs.values()]:
        if tuple(cfg.data.machines) != machines or cfg.data.dataset != reference_cfg.data.dataset:
            raise EstimateError(f"{name} {cfg.name}: macchine o dataset diversi dal riferimento {reference_cfg.name}")
    if sorted(pilot_cfgs) != sorted(target_cfgs):
        raise EstimateError(f"slot del pilota {sorted(pilot_cfgs)} diversi da quelli di tuning {sorted(target_cfgs)}")
    for slot, cfg in pilot_cfgs.items():
        if set(cfg.run_models) != set(grid.pilot.families):
            raise EstimateError(f"pilota {cfg.name}: modelli {list(cfg.run_models)}, attesi "
                                f"{list(grid.pilot.families)}")
    ref = tasks_with_meta(plan_tasks(reference_cfg, machines, data_versions), reference_read_meta)
    pil = {s: tasks_with_meta(plan_tasks(c, machines, data_versions), pilot_read_meta) for s, c in pilot_cfgs.items()}
    tgt = {s: plan_tasks(c, machines, data_versions) for s, c in target_cfgs.items()}
    return estimate(reference=ref, pilot=pil, targets=tgt, pilot_max_epochs=grid.pilot.max_epochs,
                    pilot_patience=grid.pilot.patience, threshold_hours=grid.pilot.threshold_hours,
                    k=grid.k, k_reduced=grid.k_reduced)
