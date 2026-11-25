"""Raccolta in sola lettura per docs/f8_protocol_switch.md (identificativi, provenance, riuso, epoche).

**Non è coperto da test**: script di supporto in sola lettura per ``docs/f8_protocol_switch.md`` (F8 passo 6),
copiato dalla cartella di lavoro di Claude con i soli percorsi d'ingresso resi argomenti. Si lancia dalla radice del repository.
Uso: python scripts/f8_extra/f8_collect.py [--store store]
"""
import json, glob, os, collections
import pandas as pd

import argparse

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--store", default="store", help="radice dello store (sola lettura)")
_ARGS = _ap.parse_args()
STORE = _ARGS.store.rstrip("/\\")

S = STORE + "/runs/exploratory/"
V = {"V0": ("protocol_switch_dev_v0_legacy", "07ff59c9b83f", "20261124T170125-7c9a833d"),
     "V3": ("protocol_switch_dev_v3_split3", "c53662adf11b", "20261124T180508-914894ed"),
     "V2": ("protocol_switch_dev_v2_epochs", "eadfaab3dca3", "20261124T190607-b3ad329a"),
     "V1": ("protocol_switch_dev_v1_canonical", "d538a64944dd", "20261124T212456-534d69d0")}
CTX = ("protocol_switch_dev_v1_canonical", "523248b05cea")
M1_KEYS = {"1b6bed56f28a9f9a35f8", "e36bdcca91691bdedf45", "7446399ad5ee2c3e11e8", "03016bc72ba2540faa53",
           "e7d9ec5a8fa63259edf3", "b039d8551e90542cc4f6", "a5dfd16af5e7a01b7edf", "5b5b6248fd02b13cb9e1"}


def meta(k):
    with open(f"{STORE}/fits/{k}/meta.json", encoding="utf-8") as f:
        return json.load(f)


for v, (exp, aid, rid) in list(V.items()) + [("V1ctx", CTX + (None,))]:
    a = json.load(open(f"{S}{exp}/analyses/{aid}/analysis.json", encoding="utf-8"))
    keys = [f["fit_key"] for f in a["components"]["fits"]]
    print(f"== {v} {exp} {aid}: fits in analysis {len(keys)}; status {collections.Counter(f['status'] for f in a['components']['fits'])}")
    top = {k: a[k] for k in a if k not in ("components", "absent_rows")}
    print("   top-level keys:", {k: (str(x)[:80]) for k, x in top.items()})
    comp = {k: str(x)[:90] for k, x in a["components"].items() if k not in ("fits",)}
    print("   components:", comp)
    if v == "V1ctx":
        continue
    ms = [meta(k) for k in keys]
    by_run = collections.Counter(m["created_by"]["run_id"] for m in ms)
    print("   created_by:", dict(by_run))
    print("   commits:", collections.Counter((m["provenance"]["git"]["commit"][:7], m["provenance"]["git"]["dirty"]) for m in ms))
    print("   lock:", {m["provenance"]["environment"]["lock_sha256"][:12] for m in ms},
          "threads:", {(m["provenance"]["environment"]["torch_num_threads"], m["provenance"]["environment"]["torch_num_interop_threads"]) for m in ms},
          "device:", {(m["device"]["requested"], m["device"]["effective"]) for m in ms})
    reused = [m for m in ms if m["created_by"]["run_id"] != rid]
    print("   reused:", len(reused), sorted((m["machine"], m["model"]["name"], m["seed"]["key_seed"] if m.get("seed") else None,
                                            m["created_by"]["experiment"], m["fit_key"] in M1_KEYS) for m in reused)[:12])
    nf = [(m["fit_key"], m.get("n_nonfinite_cal"), m.get("n_nonfinite_test"), m.get("nonfinite_loss")) for m in ms]
    print("   nonfinite keys present:", sorted({k for k in ms[0] if "nonfinite" in k}),
          "any>0:", [x for x in nf if (x[1] or 0) > 0 or (x[2] or 0) > 0 or x[3]])
    # epoche dal meta (indipendente dal report)
    rows = []
    for m in ms:
        fi = m.get("fit_info") or {}
        tr = (m["fit_payload"].get("training") or {})
        if fi.get("stopped_epoch") is None or not tr:
            continue
        if m["model"]["name"].endswith("_untrained"):
            continue
        rows.append(dict(model=m["model"]["name"], machine=m["machine"], seed=m["seed"]["key_seed"],
                         best=fi.get("best_epoch"), stop=fi["stopped_epoch"], cap=tr.get("max_epochs"), pat=tr.get("patience")))
    e = pd.DataFrame(rows)
    e["at_cap"] = (e.stop == e.cap) & ((e.stop - e.best) < e.pat)
    g = e.groupby("model").agg(n=("stop", "size"), at_cap=("at_cap", "sum"), best_null=("best", lambda s: s.isna().sum()),
                               best_min=("best", "min"), best_med=("best", "median"), best_max=("best", "max"),
                               stop_min=("stop", "min"), stop_med=("stop", "median"), stop_max=("stop", "max"),
                               cap=("cap", "first"), pat=("pat", "first"))
    print(g.to_string())
    cap_rows = e[e.at_cap]
    print("   at cap:", sorted((r.model[:9], r.machine[8:], r.seed, r.best, r.stop) for r in cap_rows.itertuples()))
    if v == "V0":
        print("   M1 epochs (1-1, 3-8, seed 42):", sorted((r.model, r.machine, r.best, r.stop) for r in e.itertuples()
                                                     if r.seed == 42 and r.machine in ("machine-1-1", "machine-3-8")))
    # tabella fits dell'analisi: nonfinite e coerenza
    ft = pd.read_csv(f"{S}{exp}/analyses/{aid}/tables/fits.csv.gz")
    nfc = [c for c in ft.columns if "nonfinite" in c]
    print("   fits table nonfinite cols:", {c: ft[c].sum() for c in nfc})
    thr = pd.read_csv(f"{S}{exp}/analyses/{aid}/tables/thr.csv.gz", usecols=["nan_reason"])
    fr = pd.read_csv(f"{S}{exp}/analyses/{aid}/tables/free.csv.gz", usecols=["nan_reason"])
    print("   nan_reason free:", fr.nan_reason.value_counts(dropna=False).to_dict(), "thr:", thr.nan_reason.value_counts(dropna=False).to_dict())

# trial: 9 fit riusati da V1
tr_keys = set()
for p in glob.glob(STORE + "/fits/*/meta.json"):
    m = json.load(open(p, encoding="utf-8"))
    if m["created_by"]["experiment"] == "protocol_switch_dev_v1_trial":
        tr_keys.add((m["fit_key"], m["provenance"]["git"]["commit"][:7], m["provenance"]["git"]["dirty"]))
print("trial fits:", len(tr_keys), {(c, d) for _, c, d in tr_keys})
print("store fits totali:", len(glob.glob(STORE + "/fits/*/meta.json")),
      collections.Counter(json.load(open(p, encoding="utf-8"))["status"] for p in glob.glob(STORE + "/fits/*/meta.json")))
tot = 0
per = collections.Counter()
for root, _, files in os.walk(STORE):
    for f in files:
        s = os.path.getsize(os.path.join(root, f))
        tot += s
        per[os.path.relpath(root, STORE).split(os.sep)[0]] += s
print("store bytes:", tot, {k: round(b / 2**30, 3) for k, b in per.items()}, "GiB", round(tot / 2**30, 3), "GB", round(tot / 1e9, 3))
print("tmp:", os.listdir(os.path.join(STORE, "tmp")))
