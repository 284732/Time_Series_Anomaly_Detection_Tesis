"""
Report dalle statistiche pubblicate (F7 passo 11, D-73 P13 e P15).

Il report legge solo ``store/stats/<stats_id>/`` (``source.py``), seleziona righe e colonne delle tabelle
pubblicate e le formatta: nessuna statistica si calcola qui. Uscite in una cartella nuova, fuori dallo store:

- ``report.md`` e ``report.tex`` (frammento da includere con ``\\input``; richiede ``booktabs`` e ``graphicx``);
- ``tables/<nome>.{md,tex,csv}`` (CSV: colonne pubblicate a precisione piena, sempre col punto);
- ``figures/<nome>.{pdf,png}``;
- ``report.json``: identità delle statistiche, opzioni di formattazione e sha256 di ogni file (nessuna data).

Le sezioni seguono il piano statistico: una RQ non richiesta ha solo la riga «Non richiesta dal piano statistico.».
"""

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.reporting import figures as fg
from src.reporting import notices as nt
from src.reporting import tables as tb
from src.reporting.source import ReportInputError, StatsSource, load_source

REPORT_FORMAT = 1
REPORT_CODE_VERSION = 1
REPORT_JSON = "report.json"
SHA_PREFIX = 12
NOT_REQUESTED = "Non richiesta dal piano statistico."
SENS_TITLE = "Analisi di sensibilità (P8) — esplorativa"
RQ4_TITLE = "RQ4: soglie e scostamento cal → test"     # non «deriva»: l'indicatore non la misura (D-73 P10)
NO_EXCLUDE_FIT = ("Nessun caso exclude_fit: la sensibilità coincide con l'analisi principale per costruzione.")
DESC_SECTION = "Righe descrittive (senza esito)"
VARIANT = ["model", "config_id", "window_time_agg", "sensor_agg", "normalizer"]

__all__ = ["build", "render_files", "render_markdown", "render_latex", "write_report", "ReportError",
           "ReportWriteError", "ReportInputError", "REPORT_JSON", "REPORT_CODE_VERSION"]


class ReportError(Exception):
    """Cartella di uscita non ammessa (esistente, senza cartella madre, dentro lo store)."""


class ReportWriteError(Exception):
    """Scrittura o rinomina della cartella di uscita non riuscita."""


# ---------------------------------------------------------------- blocchi del documento
@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class Bullets:
    items: List[str]


@dataclass
class Notice:
    key: str
    text: str
    items: List[str] = field(default_factory=list)


@dataclass
class TableBlock:
    name: str


@dataclass
class FigureBlock:
    name: str
    caption: str


@dataclass
class Built:
    blocks: List[Any]
    tables: Dict[str, tb.Table]
    csv: Dict[str, pd.DataFrame]
    csv_source: Dict[str, Tuple[str, List[str]]]
    figures: List[fg.FigureSpec]


class _Doc:
    def __init__(self):
        self.blocks: List[Any] = []
        self.tables: Dict[str, tb.Table] = {}
        self.csv: Dict[str, pd.DataFrame] = {}
        self.csv_source: Dict[str, Tuple[str, List[str]]] = {}

    def h(self, level, text):
        self.blocks.append(Heading(level, text))

    def p(self, text):
        self.blocks.append(Para(text))

    def notice(self, key, items=()):
        self.blocks.append(Notice(key, nt.TEXTS[key], list(items)))

    def table(self, t: tb.Table, pub_name: Optional[str] = None, pub: Optional[pd.DataFrame] = None,
              cols: Optional[List[str]] = None):
        self.tables[t.name] = t
        if pub is not None:
            cols = [c for c in (cols or list(pub.columns)) if c in pub.columns]
            self.csv[t.name] = pub[cols]
            self.csv_source[t.name] = (pub_name, cols)
        self.blocks.append(TableBlock(t.name))


# ---------------------------------------------------------------- celle
def _txt(v) -> tb.Cell:
    if v is None or (isinstance(v, float) and v != v):
        return tb.Cell(tb.MISSING)
    return tb.Cell(str(v))


def _num(v, kind="value") -> tb.Cell:
    return tb.Cell(tb.fmt_num(v, tb.DIGITS[kind]), num=True)


def _int(v) -> tb.Cell:
    return tb.Cell(tb.fmt_int(v), num=True)


def _pct(v) -> tb.Cell:
    return tb.Cell(tb.fmt_pct(v, tb.DIGITS["eta2_pct"]), num=True)


def _exact(v) -> tb.Cell:
    return tb.Cell(tb.fmt_exact(v), num=True)


def _variant(r) -> str:
    return f"{r['window_time_agg']}/{r['sensor_agg']}/{r['normalizer']}"


def _status_label(st: str) -> str:
    return {"computed": "calcolata", "computable": "calcolata", "not_computable": "non calcolabile",
            "skipped": "non richiesta dal piano"}.get(st, st)


def _short_sha(h: str) -> str:
    """sha256 abbreviato nella prosa; il valore completo è in report.json."""
    return str(h)[:SHA_PREFIX]


def _groups_text(d: Dict[str, Any]) -> str:
    return "; ".join(f"gruppo {g}: {n}" for g, n in sorted(d.items(), key=lambda kv: str(kv[0])))


# ---------------------------------------------------------------- sezioni
def _header(doc: _Doc, s: StatsSource, keys: List[str]) -> None:
    d = s.doc
    st = d["plan"]["stats"]
    doc.h(1, f"Report statistico: {st['name']}")
    doc.p(f"stats_id {d['stats_id']}; tipo {st['kind']}; ruolo delle macchine {st['machine_role']}; "
          f"stats_code_version {d['stats_code_version']}; stats_plan_sha256 {_short_sha(d['stats_plan_sha256'])} "
          f"(valori completi in {REPORT_JSON}).")
    if "exploratory" in keys:
        doc.notice("exploratory")
    t = tb.Table("inputs", "Ingressi (analisi pubblicate)", ["esperimento", "tipo", "analysis_id",
                                                              "analysis_code_version", "preproc"])
    for i in d["inputs"]:
        t.add_row([_txt(i["experiment"]), _txt(i["kind"]), _txt(i["analysis_id"]),
                   _int(i["analysis_code_version"]),
                   _txt(i.get("preproc") if isinstance(i.get("preproc"), str)
                        else json.dumps(i.get("preproc"), sort_keys=True))])
    doc.table(t)
    t = tb.Table("rq_status", "Stato delle RQ", ["RQ", "stato", "motivo"])
    for rq in nt.RQS:
        b = d["rq"].get(rq)
        t.add_row([_txt(rq.upper().replace("RQ1A", "RQ1a").replace("RQ1B", "RQ1b")),
                   _txt(_status_label(nt.rq_status(b))), _txt((b or {}).get("reason"))])
    doc.table(t)
    rq3w = set(((d["rq"].get("rq3") or {}).get("warnings")) or [])
    other = [w for w in d.get("warnings", []) if w not in rq3w]
    if other:
        doc.p("Avvertenze registrate dallo stadio di statistica:")
        doc.blocks.append(Bullets(other))
    if "d34" in keys:
        doc.notice("d34", nt.diagnostic_tables(s) or ["(nessuna tabella; livelli PA o oracolo tra i fattori di "
                                                      "RQ1a)"])
    if "not_computable" in keys:
        items = [f"{rq.upper()} non calcolabile: {d['rq'][rq].get('reason')}" for rq in nt.not_computable(d["rq"])]
        sens = (d.get("sensitivity") or {}).get("rq", {})
        items += [f"{rq.upper()} non calcolabile (sensibilità P8): {sens[rq].get('reason')}"
                  for rq in nt.not_computable(sens)]
        doc.notice("not_computable", items)


def _descriptives(doc: _Doc, s: StatsSource, tables, plan, lvl: int, prefix: str) -> None:
    doc.h(lvl, "Descrittive per modello")
    df = tables.get("descriptives")
    entries = plan.get("descriptive") or []
    if df is None or df.empty or not entries:
        doc.p("Nessuna voce descrittiva nel piano statistico.")
        return
    for i, e in enumerate(entries):
        sub = df[df["unit"] == f"descriptive/{i}"]
        delay = e["metric"] == nt.DELAY
        cond = (f"tabella {e['table']}, metrica {e['metric']}, scenario {e['scenario']}, ts_agg {e['ts_agg']}"
                + (f", soglia {json.dumps(e['threshold'], sort_keys=True)}" if e.get("threshold") else ""))
        cols = ["esperimento", "modello", "variante", "ruolo", "media", "sd", "mediana", "rango medio",
                "n macchine"] + (["n definite", "event_recall medio"] if delay else [])
        t = tb.Table(f"{prefix}descriptives_{i}", f"Descrittive, voce {i + 1}: {cond}", cols)
        for _, r in sub.iterrows():
            cells = [_txt(r["experiment"]), _txt(r["model"]), _txt(_variant(r)),
                     _txt(tb.ROLE_LABELS.get(r["model_role"], r["model_role"])), _num(r["mean"]), _num(r["sd"]),
                     _num(r["median"]), _num(r["mean_rank"]), _int(r["n_machines"])]
            if delay:
                cells += [_int(r["n_machines_defined"]), _num(r["event_recall_mean"])]
            t.add_row(cells)
        pub_cols = ["unit", "metric", "experiment"] + VARIANT + ["model_role", "mean", "sd", "median", "mean_rank",
                                                                 "n_machines", "n_machines_ranked"]
        if delay:
            pub_cols += ["n_machines_defined", "event_recall_mean"]
        doc.table(t, f"{prefix}descriptives", sub, pub_cols)


def _rq2(doc: _Doc, s: StatsSource, tables, block, lvl: int, prefix: str, figures: bool) -> None:
    doc.h(lvl, "RQ2: confronto con il riferimento")
    st = nt.rq_status(block)
    if st == "skipped":
        doc.p(NOT_REQUESTED)
        return
    doc.notice("d45")
    if st == "not_computable":
        doc.p(f"RQ2 non calcolabile: {block.get('reason')}")
    ref = block["reference"]
    doc.p(f"Metrica {block['metric']}; riferimento {ref['model']} ({_variant(ref)}); "
          f"δ = {tb.fmt_exact(block['delta'])}; α = {tb.fmt_exact(block['alpha'])}; m = {block['m']}; "
          f"livello dell'IC {tb.fmt_exact(block['ci_level'])}; metodo {block['method']} con B = {block['n_boot']} "
          f"e seed {block['seed']}; macchine per gruppo: {_groups_text(block.get('n_per_group') or {})}.")
    summ = tables["rq2_summary"]
    delta = block["delta"]
    cols = ["modello", "variante", "ruolo", "Δ̂ / media", "mediana", "L", "U", "δ", "esito", "stato IC", "n+",
            "n−", "n0", "n", "p sign-flip", "inferenza"]
    t = tb.Table(f"{prefix}rq2", f"RQ2: {block['metric']}, IC al livello {tb.fmt_exact(block['ci_level'])}, "
                 f"δ = {tb.fmt_exact(delta)}", cols)

    def row(r, compared: bool):
        outcome = tb.OUTCOME_LABELS.get(r["outcome"], tb.MISSING) if compared else tb.MISSING
        p = tb.fmt_p(r["sign_flip_p"])                      # il ruolo (di supporto) è nella nota sotto la tabella
        return [_txt(r["model"]), _txt(_variant(r)), _txt(tb.ROLE_LABELS.get(r["model_role"], r["model_role"])),
                _num(r["theta_hat"]), _num(r["median"]),
                tb.Cell(tb.fmt_bound(r["lower"], delta, tb.DIGITS["value"]), num=True),
                tb.Cell(tb.fmt_bound(r["upper"], delta, tb.DIGITS["value"]), num=True),
                _exact(r["delta"]) if compared else tb.Cell(tb.MISSING),
                tb.Cell(outcome), _txt(r["ci_status"] if compared else None),
                _int(r["n_pos"]), _int(r["n_neg"]), _int(r["n_zero"]), _int(r["n_machines"]),
                tb.Cell(p, num=p != tb.MISSING), _txt(r["inference"])]
    comp = summ[summ["model_role"] == "compared"]
    desc = summ[summ["model_role"] != "compared"]
    for _, r in comp.iterrows():
        t.add_row(row(r, True))
    if not desc.empty:
        t.add_section(DESC_SECTION)
        for _, r in desc.iterrows():
            t.add_row(row(r, False))
    doc.table(t, f"{prefix}rq2_summary", summ,
              VARIANT + ["model_role", "statistic", "metric", "theta_hat", "median", "lower", "upper", "ci_level",
                         "ci_status", "outcome", "delta", "m", "alpha", "n_pos", "n_neg", "n_zero", "n_machines",
                         "se", "sign_flip_p", "sign_flip_role", "status", "reason", "inference"])
    roles = sorted({str(x) for x in comp["sign_flip_role"].dropna()})
    if roles:
        doc.p(f"p del sign-flip esatto: ruolo {', '.join(roles)} (non entra nella regola dei quattro esiti).")
    doc.p("Le righe descrittive (riferimento, baseline random e baseline descrittive) non hanno esito: riportano "
          "la media della metrica o di Δ come contesto.")
    if figures:
        for sp in fg._rq2_specs(s, prefix, tables):
            doc.blocks.append(FigureBlock(sp.name, sp.caption))


def _terms_table(doc: _Doc, name: str, caption: str, pub_name: str, terms: pd.DataFrame) -> None:
    t = tb.Table(name, caption, ["termine", "tipo", "ordine", "gl", "SS", "η² totale (%)", "η² senza macchina (%)"])
    for _, r in terms.iterrows():
        bold = bool(r["model_interaction"])
        cells = [_txt(r["term"]), _txt(r["kind"]), _int(r["order"]), _int(r["df"]), _num(r["ss"]),
                 _pct(r["eta2_total"]), _pct(r["eta2_without_machine"])]
        for c in cells:
            c.bold = bold
        t.add_row(cells)
    doc.table(t, pub_name, terms, ["term", "factors", "order", "df", "ss", "kind", "model_interaction",
                                   "eta2_total", "eta2_without_machine", "principal"])


def _rq1(doc: _Doc, s: StatsSource, tables, sec: str, block, lvl: int, prefix: str, figures: bool) -> None:
    title = {"rq1a": "RQ1a: sensibilità alla misura", "rq1b": "RQ1b: sensibilità al design"}[sec]
    doc.h(lvl, title)
    if nt.rq_status(block) == "skipped":
        doc.p(NOT_REQUESTED)
        return
    for name, b in sorted(block["blocks"].items()):
        pre = f"{sec}_{name}"
        doc.h(lvl + 1, f"Blocco {name}")
        doc.p(f"Tabella {b['table']}, metrica {b['metric']}; η² principale: {b['eta2_principal']} "
              f"(stato {b['eta2_status']}); indice R = {tb.fmt_num(b['r'], tb.DIGITS['r'])} (stato {b['r_status']}); "
              f"macchine per gruppo: {_groups_text(b.get('n_per_group') or {})}; configurazione di riferimento: "
              f"{b['c0']}. Inferenza: {b['inference']}.")
        terms = tables.get(f"{pre}_terms")
        if terms is not None:
            _terms_table(doc, f"{prefix}{pre}_terms", f"{title}, blocco {name}: decomposizione della varianza "
                         "(in grassetto le interazioni con il modello)", f"{prefix}{pre}_terms", terms)
        summ = tables.get(f"{pre}_tau_summary")
        codes = {}
        if summ is not None:
            t = tb.Table(f"{prefix}{pre}_tau_summary", f"{title}, blocco {name}: τ-b per macchina rispetto a "
                         "c0 (distribuzione sulle macchine)", ["cod.", "configurazione", "media", "mediana", "min",
                                                               "q25", "q75", "n definite", "n non definite"])
            for i, (_, r) in enumerate(summ.iterrows()):
                codes[r["config"]] = f"c{i + 1}"
                t.add_row([_txt(f"c{i + 1}"), _txt(r["config"]), _num(r["mean"], "tau"), _num(r["median"], "tau"),
                           _num(r["min"], "tau"), _num(r["q25"], "tau"), _num(r["q75"], "tau"),
                           _int(r["n_defined"]), _int(r["n_undefined"])])
            doc.table(t, f"{prefix}{pre}_tau_summary", summ)
        means = tables.get(f"{pre}_tau_means")
        if means is not None:
            t = tb.Table(f"{prefix}{pre}_tau_means", f"{title}, blocco {name}: τ-b sui ranking delle medie",
                         ["cod.", "configurazione", "τ-b", "stato"])
            for _, r in means.iterrows():
                t.add_row([_txt(codes.get(r["config"])), _txt(r["config"]), _num(r["tau_b"], "tau"), _txt(r["status"])])
            doc.table(t, f"{prefix}{pre}_tau_means", means)
        inv = tables.get(f"{pre}_inversions")
        if inv is not None:
            t = tb.Table(f"{prefix}{pre}_inversions", f"{title}, blocco {name}: inversioni del segno di Δ per "
                         "macchina rispetto a c0", ["modello", "cod.", "configurazione", "invertite", "mantenute",
                                                    "pari", "n macchine", "quota invertite", "quota pari"])
            for _, r in inv.iterrows():
                t.add_row([_txt(r["model"]), _txt(codes.get(r["config"])), _txt(r["config"]), _int(r["n_inverted"]),
                           _int(r["n_kept"]), _int(r["n_tie"]), _int(r["n_machines"]),
                           _num(r["share_inverted"], "share"), _num(r["share_tie"], "share")])
            doc.table(t, f"{prefix}{pre}_inversions", inv)
        mi = tables.get(f"{pre}_mean_inversions")
        if mi is not None:
            t = tb.Table(f"{prefix}{pre}_mean_inversions", f"{title}, blocco {name}: inversioni del segno di Δ "
                         "medio tra configurazioni", ["modello", "invertite", "mantenute", "pari", "configurazioni",
                                                      "quota invertite"])
            for _, r in mi.iterrows():
                t.add_row([_txt(r["model"]), _int(r["n_configs_inverted"]), _int(r["n_configs_kept"]),
                           _int(r["n_configs_tie"]), _int(r["n_configs"]), _num(r["share_configs_inverted"], "share")])
            doc.table(t, f"{prefix}{pre}_mean_inversions", mi)
        if figures:
            for sp in fg._tau_specs(prefix, {f"{pre}_tau": tables[f"{pre}_tau"]} if f"{pre}_tau" in tables else {}):
                doc.blocks.append(FigureBlock(sp.name, sp.caption + " Codici delle configurazioni nella tabella "
                                              "della distribuzione di τ-b."))


def _rq3(doc: _Doc, s: StatsSource, tables, block, lvl: int, prefix: str, figures: bool, keys) -> None:
    doc.h(lvl, "RQ3: condizioni")
    st = nt.rq_status(block)
    if st == "skipped":
        doc.p(NOT_REQUESTED)
        return
    if "rq3_power" in keys and st == "computable":
        doc.notice("rq3_power", block.get("warnings") or [])
    if st == "not_computable":
        doc.p(f"RQ3 non calcolabile: {block.get('reason')}")
    doc.p(f"Statistica {block['statistic']}, IC {block['method']} al livello {tb.fmt_exact(block['ci_level'])}; "
          f"correlazioni: {block['n_correlations']}; macchine: {block['n_machines']} "
          f"({_groups_text(block.get('n_per_group') or {})}). Inferenza: {block['inference']}.")
    corr = tables.get("rq3_correlations")
    if corr is not None:
        t = tb.Table(f"{prefix}rq3_correlations", "RQ3: correlazioni entro gruppo (ρ_W) con IC",
                     ["tipo", "modello", "variante", "descrittore", "descrittore 2", "ρ_W", "L", "U", "stato IC", "n",
                      "n_eff", "ρ complessivo", "ρ per gruppo"])
        for _, r in corr.iterrows():
            var = _variant(r) if isinstance(r["window_time_agg"], str) else None
            t.add_row([_txt(r["kind"]), _txt(r["model"]), _txt(var), _txt(r["descriptor"]), _txt(r["descriptor_2"]),
                       _num(r["rho_w"], "rho"), _num(r["lower"], "rho"), _num(r["upper"], "rho"), _txt(r["ci_status"]),
                       _int(r["n"]), _int(r["n_eff"]), _num(r["rho_overall"], "rho"),
                       tb.Cell(tb.fmt_group_values(r["rho_by_group"], tb.DIGITS["rho"]), inline=True)])
        doc.table(t, f"{prefix}rq3_correlations", corr)
    if figures:
        for sp in fg._rq3_specs(s, prefix, tables):
            doc.blocks.append(FigureBlock(sp.name, sp.caption))


def _rq4(doc: _Doc, s: StatsSource, tables, block, lvl: int, prefix: str, figures: bool, keys) -> None:
    doc.h(lvl, RQ4_TITLE)
    if nt.rq_status(block) == "skipped":
        doc.p(NOT_REQUESTED)
        return
    doc.p(f"η² principale: {block['eta2_principal']}; macchine per gruppo: "
          f"{_groups_text(block.get('n_per_group') or {})}. Inferenza: {block['inference']}.")
    for metric, m in sorted(block["metrics"].items()):
        doc.p(f"Metrica {metric}: indice R = {tb.fmt_num(m['r'], tb.DIGITS['r'])} (stato {m['r_status']}); "
              f"stato di η² {m['eta2_status']}.")
        terms = tables.get(f"rq4_{metric}_terms")
        if terms is not None:
            _terms_table(doc, f"{prefix}rq4_{metric}_terms", f"RQ4, {metric}: decomposizione della varianza "
                         "(modello × soglia; in grassetto le interazioni con il modello)",
                         f"{prefix}rq4_{metric}_terms", terms)
    desc = tables.get("rq4_descriptive")
    if desc is not None and not desc.empty:
        delay = nt.DELAY in set(desc["metric"])
        if delay and "c4" in keys:
            doc.notice("c4")
        cols = ["metrica", "modello", "soglia", "media", "mediana", "n macchine", "n definite", "event_recall medio"]
        t = tb.Table(f"{prefix}rq4_descriptive", "RQ4: metriche descrittive per modello e soglia", cols)
        for _, r in desc.iterrows():
            t.add_row([_txt(r["metric"]), _txt(r["model"]), _txt(r["threshold"]), _num(r["mean"]), _num(r["median"]),
                       _int(r["n_machines"]), _int(r["n_machines_defined"]), _num(r["event_recall_mean"])])
        doc.table(t, f"{prefix}rq4_descriptive", desc)
    off = block.get("offset_indicator") or {}
    summ = tables.get("rq4_offset_summary")
    if off.get("computed") and summ is not None:
        doc.p(f"Indicatore di scostamento: {off.get('definition')}. Limite: {off.get('limit')}.")
        t = tb.Table(f"{prefix}rq4_offset_summary", "RQ4: scostamento FPR − quota nominale (soglie statiche)",
                     ["modello", "livello", "media", "mediana", "min", "q25", "q75", "n macchine"])
        for _, r in summ.iterrows():
            t.add_row([_txt(r["model"]), _exact(r["threshold_level"]), _num(r["mean"]), _num(r["median"]),
                       _num(r["min"]), _num(r["q25"]), _num(r["q75"]), _int(r["n_machines"])])
        doc.table(t, f"{prefix}rq4_offset_summary", summ)
        if figures:
            for sp in fg._rq4_specs(prefix, tables):
                doc.blocks.append(FigureBlock(sp.name, sp.caption))


def _excluded(doc: _Doc, s: StatsSource, filt: Dict[str, Any], tables, lvl: int, prefix: str) -> None:
    reg = tables.get("exclusions")
    doc.h(lvl, "Registro delle esclusioni (D-44)")
    if reg is None or reg.empty:
        doc.p("Nessuna esclusione dal registro.")
    else:
        t = tb.Table(f"{prefix}exclusions", "Registro delle esclusioni", ["passo", "regola", "motivo", "azione",
                     "esperimento", "modello", "config_id", "macchina", "seed", "unità", "n righe"])
        for _, r in reg.iterrows():
            t.add_row([_int(r["step"]), _txt(r["rule"]), _txt(r["reason"]), _txt(r["action"]), _txt(r["experiment"]),
                       _txt(r["model"]), _txt(r["config_id"]), _txt(r["machine"]),
                       _int(r["seed"]) if r["seed"] == r["seed"] and r["seed"] is not None else _txt(None),
                       _txt(r["units"]), _int(r["n_rows"])])
        doc.table(t, f"{prefix}exclusions", reg)
    counts = filt.get("counts") or []
    if counts:
        t = tb.Table(f"{prefix}exclusion_counts", "Esclusioni per regola, motivo e modello",
                     ["regola", "motivo", "modello", "azione", "n"])
        for c in counts:
            t.add_row([_txt(c["rule"]), _txt(c["reason"]), _txt(c["model"]), _txt(c["action"]), _int(c["n"])])
        doc.table(t)
    doc.h(lvl, "NaN in celle non usate")
    un = filt.get("unused_nan") or []
    if un:
        t = tb.Table(f"{prefix}unused_nan", "Righe NaN in celle non usate dal piano, per motivo e modello",
                     ["motivo", "modello", "n"])
        for u in un:
            t.add_row([_txt(u["reason"]), _txt(u["model"]), _int(u["n"])])
        doc.table(t)
    else:
        doc.p("Nessuna riga NaN in celle non usate.")
    doc.h(lvl, "Repliche della baseline random rimosse")
    dd = filt.get("dedup_removed") or {}
    if dd:
        t = tb.Table(f"{prefix}dedup_removed", "Repliche della baseline random rimosse tra ts_agg (si tiene la riga "
                     "last)", ["esperimento", "repliche rimosse"])
        for e, n in sorted(dd.items()):
            t.add_row([_txt(e), _int(n)])
        doc.table(t)
    else:
        doc.p("Nessuna replica rimossa.")
    doc.h(lvl, "Macchine per unità di inferenza")
    units = filt.get("units") or {}
    t = tb.Table(f"{prefix}units", "Macchine per gruppo all'inizio e dopo i filtri", ["unità", "all'inizio",
                                                                                      "usate"])
    for u, x in sorted(units.items()):
        t.add_row([_txt(u), _txt(_groups_text(x.get("groups_start") or {})), _txt(_groups_text(x.get("groups") or {}))])
    doc.table(t)
    dl = tables.get("rq2_deltas")
    if dl is not None:
        ex = dl[dl["status"] != "included"]
        doc.h(lvl, "Macchine escluse da RQ2")
        if ex.empty:
            doc.p("Nessuna macchina esclusa da RQ2.")
        else:
            t = tb.Table(f"{prefix}rq2_excluded", "Macchine escluse da RQ2", ["modello", "variante", "macchina",
                                                                              "gruppo", "stato", "motivo"])
            for _, r in ex.iterrows():
                t.add_row([_txt(r["model"]), _txt(_variant(r)), _txt(r["machine"]), _txt(r["group"]),
                           _txt(r["status"]), _txt(r["reason"])])
            doc.table(t, f"{prefix}rq2_deltas", ex, VARIANT + ["machine", "group", "status", "reason"])


# ---------------------------------------------------------------- assemblaggio
def _rq_sections(doc, s, tables, rq, plan, lvl, prefix, figures, keys):
    _descriptives(doc, s, tables, plan, lvl, prefix)
    _rq2(doc, s, tables, rq.get("rq2"), lvl, prefix, figures)
    _rq1(doc, s, tables, "rq1a", rq.get("rq1a"), lvl, prefix, figures)
    _rq1(doc, s, tables, "rq1b", rq.get("rq1b"), lvl, prefix, figures)
    _rq3(doc, s, tables, rq.get("rq3"), lvl, prefix, figures, keys)
    _rq4(doc, s, tables, rq.get("rq4"), lvl, prefix, figures, keys)


def build(source: StatsSource, decimal: str = tb.DEFAULT_DECIMAL) -> Built:
    tb.localize("0", decimal, "md")
    keys = nt.applicable(source)
    d = source.doc
    doc = _Doc()
    _header(doc, source, keys)
    _rq_sections(doc, source, source.tables, d["rq"], d["plan"], 2, "", True, keys)
    sens = d.get("sensitivity")
    if sens is not None:
        doc.h(2, SENS_TITLE)
        doc.notice("p8")
        n_cases = sens["n_exclude_fit_cases"]                 # conteggio pubblicato, non ricalcolato
        if n_cases == 0:
            doc.p(NO_EXCLUDE_FIT)
        else:
            doc.p(f"Casi exclude_fit nell'analisi principale: {n_cases}; analysis_variant "
                  f"{sens.get('analysis_variant')}; inferenza {sens.get('inference')}.")
            _rq_sections(doc, source, source.sensitivity, sens.get("rq", {}), d["plan"], 3, "sensitivity/", False,
                         keys)
    doc.h(2, "Dati esclusi o non usati")
    _excluded(doc, source, d.get("filter") or {}, source.tables, 3, "")
    if sens is not None:
        doc.h(3, "Sensibilità P8")
        if sens["n_exclude_fit_cases"] == 0:
            doc.p(NO_EXCLUDE_FIT)
        else:
            _excluded(doc, source, sens.get("filter") or {}, source.sensitivity, 4, "sensitivity/")
    figs = fg.figure_specs(source)
    return Built(doc.blocks, doc.tables, doc.csv, doc.csv_source, figs)


# ---------------------------------------------------------------- scrittura dei testi
def _inline(text: str, decimal: str, target: str) -> str:
    return tb.localize_inline(text, decimal, target)


def _md(text: str, decimal: str) -> str:
    return _inline(tb.md_escape(text), decimal, "md")


def _tex(text: str, decimal: str) -> str:
    return _inline(tb.tex_escape(text), decimal, "tex")


def render_markdown(b: Built, decimal: str) -> str:
    out: List[str] = []
    for blk in b.blocks:
        if isinstance(blk, Heading):
            out += ["#" * blk.level + " " + _md(blk.text, decimal), ""]
        elif isinstance(blk, Para):
            out += [_md(blk.text, decimal), ""]
        elif isinstance(blk, Bullets):
            out += [f"- {_md(i, decimal)}" for i in blk.items] + [""]
        elif isinstance(blk, Notice):
            out.append(f"> **{_md(nt.marker(blk.key), decimal)}.** {_md(blk.text, decimal)}")
            out += [f"> - {_md(i, decimal)}" for i in blk.items]
            out.append("")
        elif isinstance(blk, TableBlock):
            out += [tb.to_markdown(b.tables[blk.name], decimal).rstrip("\n"), ""]
        elif isinstance(blk, FigureBlock):
            out += [f"![{_md(blk.caption, decimal)}](figures/{blk.name}.png)", ""]
    return "\n".join(out).rstrip("\n") + "\n"


_TEX_HEAD = {1: "section*", 2: "subsection*", 3: "subsubsection*", 4: "paragraph"}


def render_latex(b: Built, decimal: str) -> str:
    out = ["% Report statistico generato da src/reporting: frammento da includere con \\input.",
           "% Pacchetti richiesti nel preambolo: booktabs, graphicx.", ""]
    for blk in b.blocks:
        if isinstance(blk, Heading):
            out += ["\\" + _TEX_HEAD[blk.level] + "{" + _tex(blk.text, decimal) + "}", ""]
        elif isinstance(blk, Para):
            out += [_tex(blk.text, decimal), ""]
        elif isinstance(blk, Bullets):
            out += ["\\begin{itemize}"] + [f"\\item {_tex(i, decimal)}" for i in blk.items] + ["\\end{itemize}", ""]
        elif isinstance(blk, Notice):
            out.append("\\par\\noindent\\textbf{" + _tex(nt.marker(blk.key), decimal) + ".} " + _tex(blk.text, decimal))
            if blk.items:
                out += ["\\begin{itemize}"] + [f"\\item {_tex(i, decimal)}" for i in blk.items] + ["\\end{itemize}"]
            out.append("")
        elif isinstance(blk, TableBlock):
            out += [tb.to_latex(b.tables[blk.name], decimal).rstrip("\n"), ""]
        elif isinstance(blk, FigureBlock):
            out += ["\\begin{figure}[htbp]", "\\centering", f"\\includegraphics[width=\\linewidth]{{figures/{blk.name}.pdf}}",
                    "\\caption{" + _tex(blk.caption, decimal) + "}", "\\end{figure}", ""]
    return "\n".join(out).rstrip("\n") + "\n"


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def render_files(source: StatsSource, decimal: str = tb.DEFAULT_DECIMAL) -> Dict[str, bytes]:
    b = build(source, decimal)
    files: Dict[str, bytes] = {"report.md": render_markdown(b, decimal).encode("utf-8"),
                               "report.tex": render_latex(b, decimal).encode("utf-8")}
    for name, t in b.tables.items():
        files[f"tables/{name}.md"] = tb.to_markdown(t, decimal).encode("utf-8")
        files[f"tables/{name}.tex"] = tb.to_latex(t, decimal).encode("utf-8")
    for name, df in b.csv.items():
        files[f"tables/{name}.csv"] = _csv_bytes(df)
    for sp in b.figures:
        for ext, data in fg.render_bytes(sp, decimal).items():
            files[f"figures/{sp.name}.{ext}"] = data
    files[REPORT_JSON] = _json_bytes({
        "report_format": REPORT_FORMAT, "report_code_version": REPORT_CODE_VERSION,
        "stats_id": source.doc["stats_id"], "stats_json_sha256": source.stats_json_sha256, "decimal": decimal,
        "stats_plan_sha256": source.doc["stats_plan_sha256"], "sha_prefix": SHA_PREFIX,
        "digits": dict(tb.DIGITS), "p_sci_below": tb.P_SCI_BELOW, "p_sig": tb.P_SIG,
        "figures": [fg.spec_manifest(sp) for sp in b.figures],
        "files": {p: hashlib.sha256(data).hexdigest() for p, data in sorted(files.items())},
    })
    return files


def write_report(stats_dir, out, decimal: str = tb.DEFAULT_DECIMAL) -> Path:
    """Legge e verifica le statistiche, poi scrive il report in ``out`` (nuova, fuori dallo store)."""
    source = load_source(stats_dir)
    out = Path(os.path.abspath(out))
    if out.exists():
        raise ReportError(f"la cartella di uscita esiste già: {out}")
    if not out.parent.is_dir():
        raise ReportError(f"la cartella madre di --out non esiste: {out.parent}")
    store = source.store_root.resolve()
    resolved = out.parent.resolve() / out.name
    if resolved == store or store in resolved.parents:
        raise ReportError(f"la cartella di uscita è dentro lo store ({store}): il report non scrive nello store")
    files = render_files(source, decimal)
    tmp = out.parent / f".{out.name}.tmp-{uuid.uuid4().hex[:12]}"
    try:
        tmp.mkdir()
        for rel, data in sorted(files.items()):
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        os.rename(tmp, out)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ReportWriteError(f"scrittura del report non riuscita: {exc}") from exc
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out
