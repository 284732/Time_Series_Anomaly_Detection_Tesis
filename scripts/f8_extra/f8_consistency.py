"""Controlli di coerenza interna, sola lettura: (a) V2->V1 su machine-1-7; (b) V3->V2 a parità di best_epoch.

**Non è coperto da test**: script di supporto in sola lettura per ``docs/f8_protocol_switch.md`` (F8 passo 6),
copiato dalla cartella di lavoro di Claude con i soli percorsi d'ingresso resi argomenti. Si lancia dalla radice del repository.
Uso: python scripts/f8_extra/f8_consistency.py [--store store]
"""
import pandas as pd

import argparse

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--store", default="store", help="radice dello store (sola lettura)")
_ARGS = _ap.parse_args()
STORE = _ARGS.store.rstrip("/\\")

S = STORE + "/runs/exploratory/"
A = {"V3": "protocol_switch_dev_v3_split3/analyses/c53662adf11b",
     "V2": "protocol_switch_dev_v2_epochs/analyses/eadfaab3dca3",
     "V1": "protocol_switch_dev_v1_canonical/analyses/d538a64944dd"}
ID = ["machine", "model", "seed", "scenario", "window_time_agg", "sensor_agg", "normalizer", "ts_agg", "metric"]
TID = ["machine", "model", "seed", "scenario", "window_time_agg", "sensor_agg", "normalizer", "ts_agg",
       "threshold_policy", "threshold_level", "n_past", "pa", "metric"]


def load(v, t):
    d = pd.read_csv(f"{S}{A[v]}/tables/{t}.csv.gz")
    d["seed"] = d["seed"].fillna(-1)
    return d


def cmp(a, b, keys):
    m = a[keys + ["value"]].merge(b[keys + ["value"]], on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m._merge == "both"]
    eq = (both.value_a == both.value_b) | (both.value_a.isna() & both.value_b.isna())
    return len(m), int((m._merge != "both").sum()), int((~eq).sum()), both[~eq]


free = {v: load(v, "free") for v in A}
thr = {v: load(v, "thr") for v in A}
fits = {v: load(v, "fits") for v in A}
print("fits cols:", fits["V2"].columns.tolist())

# (a) machine-1-7, V2 -> V1, tutti i modelli e tutte le righe
for t, keys, d in (("free", ID, free), ("thr", TID, thr)):
    n, only, ndiff, ex = cmp(d["V2"][d["V2"].machine == "machine-1-7"], d["V1"][d["V1"].machine == "machine-1-7"], keys)
    print(f"(a) 1-7 V2->V1 {t}: righe {n}, solo in una {only}, diverse {ndiff}")
    if ndiff:
        print(ex.groupby("model").size().to_dict())
# altre macchine: quante righe uguali per modello (diagnostica)
for mach in sorted(free["V2"].machine.unique()):
    n, only, ndiff, ex = cmp(free["V2"][free["V2"].machine == mach], free["V1"][free["V1"].machine == mach], ID)
    print(f"    V2->V1 free {mach}: diverse {ndiff}/{n}")

# (b) V3 -> V2 per i modelli addestrati: fit con lo stesso best_epoch
f3, f2 = fits["V3"], fits["V2"]
k = ["machine", "model", "seed"]
ep = f3[k + ["best_epoch", "stopped_epoch"]].merge(f2[k + ["best_epoch", "stopped_epoch"]], on=k, suffixes=("_v3", "_v2"))
ep = ep[ep.model.isin(["conv_autoencoder", "lstm_autoencoder", "lstm_forecaster"])]
same = ep[ep.best_epoch_v3 == ep.best_epoch_v2]
print(f"(b) fit addestrati {len(ep)}; con best uguale in V3 e V2: {len(same)}")
print(same.to_string())
for t, keys, d in (("free", ID, free), ("thr", TID, thr)):
    tot_n = tot_diff = 0
    for r in same.itertuples():
        sel = lambda x: x[(x.machine == r.machine) & (x.model == r.model) & (x.seed == r.seed)]
        n, only, ndiff, _ = cmp(sel(d["V3"]), sel(d["V2"]), keys)
        tot_n += n
        tot_diff += ndiff + only
    print(f"    {t}: righe {tot_n}, diverse {tot_diff}")
# controllo positivo: best diverso -> righe diverse
diffb = ep[ep.best_epoch_v3 != ep.best_epoch_v2]
r = diffb.iloc[0]
sel = lambda x: x[(x.machine == r.machine) & (x.model == r.model) & (x.seed == r.seed)]
print("    controllo positivo", r.machine, r.model, r.seed, cmp(sel(free["V3"]), sel(free["V2"]), ID)[:3])
# V3 -> V2: fermati per ES in V3 (stop-best == 5) e best uguale in V2 => stop V2 = best + 10?
es3 = ep[(ep.stopped_epoch_v3 - ep.best_epoch_v3 == 5)]
print("    V3 fermati per ES:", len(es3), "di cui best invariato in V2:", int((es3.best_epoch_v3 == es3.best_epoch_v2).sum()),
      "; best V2 < best V3 (atteso 0):", int((ep.best_epoch_v2 < ep.best_epoch_v3).sum()))
