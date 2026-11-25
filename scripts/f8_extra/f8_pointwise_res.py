"""PCA puntuale V2 vs V1: dove cambiano i residui di test (agg last) dei sensori, sola lettura dello store.

**Non è coperto da test**: script di supporto in sola lettura per ``docs/f8_protocol_switch.md`` (F8 passo 6),
copiato dalla cartella di lavoro di Claude con i soli percorsi d'ingresso resi argomenti. Si lancia dalla radice del repository.
Uso: python scripts/f8_extra/f8_pointwise_res.py [--store store]
"""
import json
import numpy as np

import argparse

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--store", default="store", help="radice dello store (sola lettura)")
_ARGS = _ap.parse_args()
STORE = _ARGS.store.rstrip("/\\")

V2 = {'machine-1-1': '237deb87344a0aaa00d3', 'machine-3-11': '2b2747840f8f1b281763', 'machine-1-7': '7806519578870e817721',
      'machine-2-8': '9615ba294582fdca6610', 'machine-3-8': 'b4be79897e518b37bf91', 'machine-2-6': 'bd50409e93d6057c95ab'}
V1 = {'machine-1-7': '2187df0a9ccbe32e2738', 'machine-3-11': '229b97f763155db0e97a', 'machine-2-6': '27ac77e0119d057761fb',
      'machine-2-8': '4de64eada734f141e129', 'machine-3-8': '8a7bd97ff9c18bf7f7b5', 'machine-1-1': 'f23f528a2c9041c11586'}
ZV = {'machine-1-1': [26, 28], 'machine-2-8': [16, 17, 28], 'machine-3-11': [10, 31], 'machine-3-8': [28],
      'machine-1-7': [], 'machine-2-6': []}
W = 100
for m in sorted(V1):
    a = np.load(f"{STORE}/fits/{V2[m]}/residuals.npz")["test__last"]
    b = np.load(f"{STORE}/fits/{V1[m]}/residuals.npz")["test__last"]
    meta = json.load(open(f"{STORE}/fits/{V1[m]}/meta.json", encoding="utf-8"))["residuals"]["test__last"]
    n_win, T = a.shape[0], meta["n_timestamps"]
    # finestra j -> timestamp j + W - 1; regione primaria [W-1, T-W+1) -> finestre j < T - 2W + 2
    j_hi = T - 2 * W + 2
    diff = np.abs(b - a)
    rel = diff / np.maximum(np.abs(a), 1e-12)
    print(f"{m}: finestre {n_win}, T {T}, finestre in regione primaria [0, {j_hi})")
    others = [s for s in range(a.shape[1]) if s not in ZV[m]]
    print(f"   altri sensori: max |Δ| {diff[:, others].max():.3g}, max rel {rel[:, others].max():.3g}")
    for s in ZV[m]:
        nz = np.flatnonzero(diff[:, s] > 0)
        if len(nz) == 0:
            print(f"   sensore {s}: nessuna differenza")
            continue
        inside = nz[nz < j_hi]
        print(f"   sensore {s}: finestre diverse {len(nz)} (da {nz.min()} a {nz.max()}), in regione primaria {len(inside)}; "
              f"V2 max {a[:, s].max():.3g}, V1 max {b[:, s].max():.3g}; rapporto mediano {np.median(b[nz, s] / np.maximum(a[nz, s], 1e-300)):.4g}")
