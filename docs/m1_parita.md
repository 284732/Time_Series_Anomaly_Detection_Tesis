# M1 — Parità della nuova pipeline (`legacy2`) con i run storici

**Esito: parità esatta su entrambi i confronti (M1 superata, 2026-11-19).**

I report completi prodotti da `scripts/parity_report.py` sono in `reports_local/m1/parity_last/` e
`reports_local/m1/parity_mean/` (`report.json`, `report.md`), cartella esclusa da git: questo file ne riporta il
contenuto rilevante in forma versionata.

## 1. Scopo

Dimostrare che la nuova pipeline (`src/cli.py`: `fit` → `analyze`), in modalità `legacy2`, riproduce sui dati reali i
run storici di riferimento della pipeline congelata `main.py` (D-13), con tolleranza zero (D-17). È un controllo di
parità con il protocollo storico, non un risultato della tesi: nessun comando `stats` né `report` sui dati reali,
nessun esperimento `split3`, nessuna correzione degli errori storici (restano a F8 come cambiamenti dichiarati).
Piano, criteri di esito ed esclusioni sono stati fissati prima dei run, nel commit `4d8d248` (nota M1 in
`piano_operativo.md`, integrazione I2).

## 2. Riferimento (D-50)

| Variante | Run storico | Aggregazione nella finestra | Proiezione finestra → timestamp |
|---|---|---|---|
| `last` | `results/results_tests/results_repro_last_01` | `last` | `last` |
| `mean` | `results/results_tests/results_repro_mean_01` | `mean` | `last` |

Comuni alle due varianti: `machine-1-1` e `machine-3-8` (macchine di sviluppo, D-51), seed 42, soglia al 98°
percentile della validation, modelli `pca`, `pca_pointwise`, `conv_autoencoder`, `lstm_autoencoder` più la
`random_baseline` prodotta da `main.py`.

## 3. Commit e ambiente

- **Commit** di fit e analisi: `4d8d24897bce74ba0d7c22cccab87021db5974a7`, albero git pulito (`dirty: false`, nessun
  file modificato) in tutti gli 8 `meta.json` e nelle due righe `analyze` del registro.
- **Ambiente:** `C:\Users\predi\venvs\smd_m1` (D-55), CPython 3.13.9, `Windows-10-10.0.19045-SP0`.
- **`lock_sha256`:** `6e09688d74c4b22a23036c23b79bd826eeafceefcb93d56ea9f73d60d790ac59`, uguale nei fit, nell'analisi
  e nell'ambiente del report.
- **Versioni principali:** numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.9.0, torch 2.12.1+cpu,
  threadpoolctl 3.6.0, PyYAML 6.0.3.
- **Device:** richiesto ed effettivo `cpu` in tutti i fit (I4).
- **Thread registrati** (predefiniti, decisione D1; nessuna variabile `OMP/MKL/OPENBLAS_NUM_THREADS` impostata), uguali
  in tutti i fit:
  - torch: 4 intra-op, 4 inter-op;
  - OpenBLAS (`libscipy_openblas`, SkylakeX, pthreads): 8 (0.3.30, numpy) e 8 (0.3.29.dev, scipy);
  - OpenMP: `libiomp` 1 e 4 (due pool), `vcomp` 8.
- **Versione dei dati:** `data_version` = `dv1-c4ebeb5cf358f380ceb8`.

## 4. Comandi eseguiti (dall'utente, PowerShell)

`$py = "C:\Users\predi\venvs\smd_m1\Scripts\python.exe"`, dalla radice del repository.

```
B2  & $py -m pytest -q --real-data -m real_data
    & $py scripts/make_data_hashes.py --check
C1  & $py -B main.py configs/config_tests/m1_rerun_last.yaml
    & $py -B main.py configs/config_tests/m1_rerun_mean.yaml
D1  & $py -m src.cli fit configs/experiments/parity_repro.yaml
D2  & $py -m src.cli analyze configs/experiments/parity_repro.yaml configs/analysis/parity_last.yaml
    & $py -m src.cli analyze configs/experiments/parity_repro.yaml configs/analysis/parity_mean.yaml
D3  & $py -m src.cli status configs/experiments/parity_repro.yaml --verify
E1  & $py scripts/parity_report.py --reference results/results_tests/results_repro_last_01 --store store
        --experiment configs/experiments/parity_repro.yaml --analysis-id be0a03233647
        --rerun reports_local/m1/main_rerun_last --out reports_local/m1/parity_last
    & $py scripts/parity_report.py --reference results/results_tests/results_repro_mean_01 --store store
        --experiment configs/experiments/parity_repro.yaml --analysis-id a02e1de4a1e5
        --rerun reports_local/m1/main_rerun_mean --out reports_local/m1/parity_mean
```

Uscite e codici, riportati dall'utente. I codici di D1–E1 sono anche nella trascrizione
`reports_local/m1/console_D1_E1.txt`.

| Passo | Uscita | Codice |
|---|---|---|
| B2, test `real_data` | «3 passed, 3291 deselected in 44.91s» | 0 |
| B2, `make_data_hashes.py --check` | «smd_sha256.yaml: 84 file, tutti coincidenti» | 0 |
| C1, riesecuzione `last` | controllo successivo: `summary_timestamp_metrics.csv` presente, `failures.csv` assente, 34 righe | 0 |
| C1, riesecuzione `mean` | — | 0 |
| D1, `fit` | «fitted 8, skipped 0, failed 0», run `20261119T144234-c54fd9f8` | 0 |
| D2, `analyze` (ciascuna analisi) | «righe free 170, thr 340, fits 8 (failed 0); righe NaN 0; righe legacy 34» | 0 |
| D3, `status --verify` | «compiti: 8 (complete=8, failed=0, missing=0)», «orphan_tmp: 0», «orphan_analysis_tmp: 0», «verify: residui presenti integri» | 0 |
| E1, `parity_report.py` (ciascuna variante) | verdetto `parita_esatta` | 0 |

Epoche di arresto degli AE in D1, uguali allo storico: `machine-1-1` 40 (`conv_autoencoder`) e 13
(`lstm_autoencoder`); `machine-3-8` 22 e 26. Le epoche si confrontano anche nei CSV per macchina (§7). Alla chiusura
`store/tmp/` era vuota (verificato da Claude, in sola lettura).

## 5. Identificativi

- **Run di fit:** `20261119T144234-c54fd9f8`: 8 fit, 0 saltati, 0 falliti; 14:42:34–14:47:53 UTC (circa 5 min 19 s).
  `config_sha256` = `8723557658f3d2a768f7205562088b2984247bffc45f2d76e049b5d6df3d9816`.
- **Fit** (`fit_key`):

  | Macchina | `pca` | `pca_pointwise` | `conv_autoencoder` | `lstm_autoencoder` |
  |---|---|---|---|---|
  | `machine-1-1` | `1b6bed56f28a9f9a35f8` | `e36bdcca91691bdedf45` | `7446399ad5ee2c3e11e8` | `03016bc72ba2540faa53` |
  | `machine-3-8` | `e7d9ec5a8fa63259edf3` | `b039d8551e90542cc4f6` | `a5dfd16af5e7a01b7edf` | `5b5b6248fd02b13cb9e1` |

- **Analisi:**

  | Variante | Piano | `analysis_id` | Run di analisi |
  |---|---|---|---|
  | `last` | `configs/analysis/parity_last.yaml` | `be0a03233647` | `20261119T150922-6022c827` |
  | `mean` | `configs/analysis/parity_mean.yaml` | `a02e1de4a1e5` | `20261119T150936-80bbc35b` |

  `plan_sha256`: `last` = `6fccbf4f3b9c9828c44641f47364d8e875618385e09f24f1aca3865fc86a0285`, `mean` =
  `147e11353b5feb34cfbe6ae488ce9c04cb1f6c93c395dc9da71f47eaed9481fe`. I due piani usano gli stessi fit (D-33: il fit salva i residui sia `last` sia `mean`).
- **Riesecuzione di `main.py`:** `reports_local/m1/main_rerun_last/` e `reports_local/m1/main_rerun_mean/`.

## 6. Esito per controllo

Differenze trovate da `scripts/parity_report.py`, con tolleranza zero:

| Controllo | `last` / `last` | `mean` / `last` |
|---|---|---|
| storico contro nuovo (export legacy dell'analisi) | 0 | 0 |
| storico contro riesecuzione di `main.py` | 0 | 0 |
| riesecuzione contro nuovo | 0 | 0 |
| tabella lunga contro export legacy (I1) | 0 | 0 |
| fit e non finiti | 0 | 0 |
| provenance | 0 | 0 |
| **verdetto** | **`parita_esatta`** | **`parita_esatta`** |

(`last` / `last` = aggregazione nella finestra `last` con proiezione `last`; `mean` / `last` = aggregazione nella finestra
`mean` con proiezione `last`.)

Modelli per cui servirebbe l'approvazione esplicita (regola D5): nessuno. La regola D5 non si è applicata.

## 7. Conteggi

Uguali per le due varianti.

- **File confrontati:** 70 per ciascuno dei tre confronti tra cartelle: 34 JSON delle metriche e 34 `.npz` degli score
  (17 per macchina: 4 modelli × 4 varianti di scoring, cioè `raw_mean`, `z_mean`, `z_max`, `z_top5`, più
  `random_baseline`) e 2 CSV per macchina. `failures.csv` è assente in tutte le cartelle.
- **Tabella lunga contro export:** 408 confronti, cioè 34 righe del riepilogo × 12 colonne; righe escluse 0, di cui
  dichiarate 0; coppie saltate 0.
- **Fit:** 8, tutti con `n_nonfinite_cal` = 0, `n_nonfinite_test` = 0, `nonfinite_loss` falso; nessun caso dichiarato
  (D-72 C15).
- **Granularità del report:** `report.json` dà i conteggi solo a livello di file. Non contiene il numero di campi JSON,
  di celle dei CSV o di array `.npz` confrontati, ma ogni file si confronta per intero (tutti i campi, tutte le celle
  salvo le esclusioni sotto, tutti gli array con chiavi, dtype, forma e valori).

Esclusioni applicate, cioè l'elenco fissato prima dei run:
- colonne `FitSec` e `InferSec` dei CSV per macchina (durate);
- nessun campo dei JSON;
- byte dei file `.npz` (si confrontano gli array);
- `summary_timestamp_metrics.csv` e `plots/`.

La colonna `Epochs` è confrontata, quindi le epoche degli AE coincidono.

## 8. Limiti

- **Copertura:** solo 2 macchine di sviluppo (`machine-1-1`, `machine-3-8`) e il solo seed 42.
- **Modelli non verificati su dati reali:**
  - `input_norm` e `rolling_zscore` non sono nei run di D-50, quindi restano verificati solo dallo snapshot sintetico
    di F1;
  - gli AE non addestrati (D-69) e il modello di previsione (D-71, D-74) sono fuori da M1.
- **Non finiti:** il caso `n_nonfinite_cal` > 0 è assente su 1-1 e 3-8, quindi lo scostamento dichiarato su `raw_mean`
  (D-72 C15) non è stato esercitato sui dati reali. Va controllato in F8.
- **Run storici:** per i run `_01` non sono registrati la macchina, i thread e le versioni delle librerie (D-50,
  D-61; punto 0.6 del piano, limite dichiarato). La portata del limite è ridotta, ma non annullata: lo storico coincide
  bit a bit con la riesecuzione nell'ambiente registrato sopra. Quindi le eventuali differenze di macchina, thread o
  versioni non hanno avuto effetto su questi output, ma non è dimostrato che fossero le stesse.
- **Protocollo:** la parità vale per il protocollo storico `legacy2`. Il protocollo canonico (`split3`, D-05) e le sue
  differenze dichiarate sono oggetto di F8.

## 9. Conseguenze

- **Verifica di nessun cambiamento numerico:** da qui è il confronto tra due run dello `store/` con stessa config e
  stessi seed (D-14).
- **Pipeline storica:** non serve più come oracolo, ma `main.py` resta congelato e disponibile fino a F12.
- **Registro:** le note di rimando su D-14, D-17 e D-50 in `docs/decisioni.md` registrano l'esito.
