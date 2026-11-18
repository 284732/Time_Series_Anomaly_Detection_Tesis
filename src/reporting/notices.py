"""
Avvertenze obbligatorie del report: testi fissi in italiano, senza numeri (i numeri vengono solo dalle statistiche
pubblicate). Ogni avvertenza ha un titolo (``marker``) identico in Markdown e LaTeX e una condizione di applicazione
letta da ``stats.json`` e dall'elenco delle tabelle pubblicate.
"""

from typing import Any, Dict, List

TITLES = {
    "exploratory": "Avvertenza: risultati esplorativi",
    "d45": "Avvertenza D-45: lettura di RQ2",
    "c4": "Avvertenza C4: ritardo di rilevamento",
    "d34": "Avvertenza D-34: righe diagnostiche",
    "rq3_power": "Avvertenza: potenza di RQ3",
    "p8": "Avvertenza P8: analisi di sensibilità",
    "not_computable": "Avvertenza: RQ non calcolabili",
}

TEXTS = {
    "exploratory": ("Tutti i risultati di questo report sono esplorativi: in F7 il piano statistico non è congelato "
                    "e i run ufficiali sono rifiutati fino a F10. Nessun esito va letto come confermativo."),
    "d45": ("RQ2 si descrive come «confermativa rispetto al protocollo congelato, su dati già osservati con altri "
            "protocolli» (D-45). Gli esiti si riportano solo come superiore, equivalente, inferiore o inconcludente "
            "rispetto al margine δ fissato nel piano, insieme all'intervallo di confidenza; le conclusioni valgono "
            "per la famiglia di modelli confrontata, non per una classe di metodi in generale."),
    "c4": ("Il ritardo di rilevamento (detection_delay) è definito solo sugli eventi rilevati: si legge sempre "
           "insieme alla quota di eventi rilevati (event_recall) e non si confronta tra scenari diversi, perché gli "
           "eventi valutati cambiano con lo scenario (C4)."),
    "d34": ("Le righe diagnostiche (point adjustment, soglia oracolo) sono limiti superiori, non metriche "
            "principali: restano separate e non entrano nelle conclusioni (D-34). Tabelle diagnostiche pubblicate:"),
    "rq3_power": ("RQ3 ha potenza bassa con poche macchine per gruppo: un intervallo che comprende lo zero non "
                  "indica assenza di associazione. Avvertenze registrate dallo stadio di statistica:"),
    "p8": ("Analisi di sensibilità P8: le stesse statistiche con i fit dai residui di calibrazione non finiti "
           "esclusi. È sempre esplorativa e non sostituisce l'analisi principale."),
    "not_computable": ("Una RQ non calcolabile non ha esito; il motivo è quello registrato in stats.json."),
}

FORBIDDEN_PHRASES = ("vince", "non vince", "il deep learning", "significativ", "dimostra", "prova che")

ORDER = ("exploratory", "d45", "c4", "d34", "rq3_power", "p8", "not_computable")
RQS = ("rq1a", "rq1b", "rq2", "rq3", "rq4")
DELAY = "detection_delay"


def marker(key: str) -> str:
    return TITLES[key]


def rq_status(block) -> str:
    """Stato di una RQ da ``stats.json`` (sempre esplicito: ``source.load_source`` rifiuta i blocchi senza)."""
    return block["status"]


def diagnostic_tables(source) -> List[str]:
    return sorted(n for n in source.tables if n.startswith("diagnostic_"))


def _rq1a_has_diagnostic_levels(block: Dict[str, Any]) -> bool:
    for b in (block or {}).get("blocks", {}).values():
        f = b.get("factors", {})
        if "true" in [str(x).lower() for x in f.get("pa", [])]:
            return True
        if any('"oracle"' in str(x) for x in f.get("threshold", [])):
            return True
    return False


def _uses_delay(source) -> bool:
    for tables in (source.tables, source.sensitivity):
        for name in ("descriptives", "rq4_descriptive"):
            df = tables.get(name)
            if df is not None and DELAY in set(df["metric"]):
                return True
    return False


def not_computable(doc_rq: Dict[str, Any]) -> List[str]:
    return [rq for rq in RQS if rq in doc_rq and rq_status(doc_rq[rq]) == "not_computable"]


def applicable(source) -> List[str]:
    doc = source.doc
    rq = doc.get("rq", {})
    sens = doc.get("sensitivity") or {}
    keys = []
    if doc.get("plan", {}).get("stats", {}).get("kind") != "official":
        keys.append("exploratory")
    if rq_status(rq.get("rq2")) != "skipped":
        keys.append("d45")
    if _uses_delay(source):
        keys.append("c4")
    if diagnostic_tables(source) or _rq1a_has_diagnostic_levels(rq.get("rq1a")):
        keys.append("d34")
    if rq_status(rq.get("rq3")) == "computable":
        keys.append("rq3_power")
    if doc.get("sensitivity") is not None:
        keys.append("p8")
    if not_computable(rq) or not_computable(sens.get("rq", {})):
        keys.append("not_computable")
    return [k for k in ORDER if k in keys]
