# F8 — Esperimento legacy contro canonico (catena V0 → V3 → V2 → V1)

**Stato: approvato dall'utente il 2026-11-25 (passo 6 di F8).** Tutti i controlli previsti sono superati (§3).

Il report completo prodotto da `scripts/protocol_switch_report.py` è in `reports_local/f8_protocol_switch/`
(`report.md`, `report.json` e 6 CSV), cartella esclusa da git. Questo file ne riporta in forma versionata il contenuto
rilevante, più alcuni controlli e calcoli fatti da Claude in sola lettura su store e report: sono indicati come tali
(«calcolo di Claude»), ciascuno con lo script che lo riproduce. Gli script sono in `scripts/f8_extra/`, sono in sola
lettura e **non sono coperti da test**:

| script | contenuto |
|---|---|
| `f8_collect.py` | identificativi, provenance, riuso dei fit, epoche, spazio |
| `f8_consistency.py` | controlli 6 e 7 |
| `f8_tables.py` | tabelle di §4.5 e §5, punti di valutazione, spazio per run |
| `f8_pointwise_res.py` | residui di test della PCA puntuale, V2 contro V1 |
| `f8_inline_checks.py` | `run.log`, baseline random per seed, checkpoint, ordine degli score, Δ per metrica e residuo massimo della PCA puntuale, zeri del contrasto di PA |
| `f8_abs_deltas.py` | tabelle di §4.1–4.4 con la mediana di \|Δ\| e confronto a parità di metrica (§4.5) |

**Convenzioni.**
- Δ = valore della variante di arrivo − valore della variante di partenza, sulla media dei 3 seed per macchina
  (unità di analisi = macchina). Le tabelle danno mediana [minimo; massimo] sulle 6 macchine e, tra graffe, la mediana
  di |Δ| sulle 6 macchine: per esempio «−0,0053 [−0,0792; +0,1082] {0,0097}».
- «0» = esattamente zero; «±0,0000…» = non nullo ma |Δ| < 0,00005.
- Nei paragrafi **Fatti** ci sono solo numeri letti; nei paragrafi **Ipotesi** ci sono interpretazioni non verificate.

## 1. Scopo e perimetro

Misurare, sulle 6 macchine di sviluppo (D-51), quanto cambiano i numeri passando dal protocollo storico
(`legacy2`, V0) a quello canonico (V1). Si cambia un blocco della config per passaggio, nell'ordine fissato prima dei
run (nota di F8 del 2026-11-24 in `piano_operativo.md`). È un esperimento descrittivo: niente IC, test o conclusioni
su RQ2.

| variante | esperimento | split | `min_range` (τ) | epoche (max / patience) |
|---|---|---|---|---|
| V0 | `protocol_switch_dev_v0_legacy` | `legacy2` (80/20, es = cal) | 0 | 40 / 5 |
| V3 | `protocol_switch_dev_v3_split3` | `split3` 60/20/20 | 0 | 40 / 5 |
| V2 | `protocol_switch_dev_v2_epochs` | `split3` 60/20/20 | 0 | 100 / 10 |
| V1 | `protocol_switch_dev_v1_canonical` | `split3` 60/20/20 | 0,02 | 100 / 10 |

Comuni a tutte le varianti:
- **Macchine:** `machine-1-1`, `machine-1-7`, `machine-2-6`, `machine-2-8`, `machine-3-11`, `machine-3-8`.
- **Seed:** 42, 43, 44.
- **Modelli (9):** `pca`, `pca_pointwise`, `conv_autoencoder`, `lstm_autoencoder`, `lstm_forecaster`,
  `conv_autoencoder_untrained`, `lstm_autoencoder_untrained`, `rolling_zscore`, `input_norm`. In più c'è la
  `random_baseline` dell'analisi.
- **Finestre e training:** W = 100, `step` 1, `train_step` 5 (D-38). Gli altri iperparametri sono quelli storici di M1;
  `lstm_forecaster.hidden_dim` = 64, provvisorio (D-41).
- **Piano primario** (`configs/analysis/protocol_switch_primary.yaml`):
  - scenario `primary`;
  - `window_time_agg` `last` e `mean`;
  - le 4 varianti di scoring storiche: mean/none, mean/std, max/std, top5/std;
  - `ts_agg` `last`, `mean`, `max`;
  - soglia `static_pct` al 98° percentile e oracolo, con e senza PA;
  - nessuna soglia mobile e nessun export legacy.
- **Piano con contesto** (`protocol_switch_context.yaml`, solo V1): lo stesso, più lo scenario `test_context`.

## 2. Identificativi

### 2.1 Commit e ambiente
- **Commit:** `91ceeb850d05e6fd093ddc2bfdea3223e7a7a623`, albero pulito (`dirty: false`) in tutti i `meta.json` dei
  fit nuovi e in tutti gli `analysis.json`. Fanno eccezione gli 8 fit di M1 riusati in V0: commit `4d8d248`, anche lì
  `dirty: false` (§3, controllo 3).
- **Ambiente:**
  - venv: `C:\Users\predi\venvs\smd_m1` (D-55);
  - Python: CPython 3.13.9, `Windows-10-10.0.19045-SP0`;
  - `lock_sha256`: `6e09688d74c4b22a23036c23b79bd826eeafceefcb93d56ea9f73d60d790ac59`, unico in tutti i fit usati
    (M1 compresi), nelle analisi e nel report.
- **Versioni:** numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.9.0, torch 2.12.1+cpu, threadpoolctl 3.6.0,
  PyYAML 6.0.3.
- **Thread** (predefiniti, nessuna variabile `*_NUM_THREADS` impostata):
  - torch: 4 intra-op e 4 inter-op in tutti i fit;
  - pool registrati: OpenBLAS 8 e 8, OpenMP `libiomp` 1 e 4, `vcomp` 8 (come in M1).
- **Device:** richiesto ed effettivo `cpu` in tutti i fit.
- **`manifest_sha256`:** `aa9a385c660431136e9b7adf74d4a4e8541503e20b81e8aec1b25357d057c963`, uguale nelle 5 analisi.

### 2.2 Config, run di fit e analisi

| variante | `config_sha256` | run di fit | `analysis_id` | run di analisi |
|---|---|---|---|---|
| V0 | `85448ae67aeb3cf7341ed461323d65ce2971309907d0843994de2e11e0dad340` | `20261124T170125-7c9a833d` | `07ff59c9b83f` | `20261125T091540-6d0f6aaa` |
| V3 | `eab6ecc40dbfa42fdf9cb036710f0997bacc28b8c8cba2a08bc7675b9755b339` | `20261124T180508-914894ed` | `c53662adf11b` | `20261125T091748-f8e30555` |
| V2 | `59f1514dde99d9eaaf3413b443b4d72b4a5bead38efcd0d35acb0b1390ae7566` | `20261124T190607-b3ad329a` | `eadfaab3dca3` | `20261125T091942-be6d6811` |
| V1 | `b30bc7ad0a043f4518cd76daa96d2cefc7b1c19c3fa3f60b899c2a32c2d65978` | `20261124T212456-534d69d0` | `d538a64944dd` | `20261125T092130-3fa2b154` |
| V1 con contesto | (stesso V1) | (stesso V1) | `523248b05cea` | `20261125T092316-15d4cf4c` |

**`plan_sha256`:**
- primario (V0–V3 e V1): `6495c5b7f738dcaf6a0ab07f5c19ebce174ad320bc55d482ac250e927e63a35e`;
- con contesto: `7756819c9513f2d708118dbf851c1bb55fd185dd27cdacd684c17c43d1fa26a5`.

**Uscite, riportate dall'utente:**
- tutte le analisi: codice 0, `failed` 0, righe NaN 0;
- `status --verify`: codice 0 su tutte e 4 le varianti.

**Righe delle analisi** (da `analysis.json`; nessun `nan_reason` in nessuna tabella):
- primarie: `free` 11.520, `thr` 25.344, `fits` 114;
- con contesto: `free` 23.040, `thr` 50.688.

Le righe mancanti del modello di previsione (proiezioni offline e `window_time_agg` degenere) sono dichiarate in
`absent_rows` (D-74).

### 2.3 Conteggi e durate dei fit

Le durate vanno dal `run_id` (UTC) all'ultima riga del `run.log` (ora locale, UTC+1, convertita).

| variante | fit | completati | saltati (origine) | falliti | inizio – fine (UTC) | durata |
|---|---|---|---|---|---|---|
| V0 | 114 | 106 | 8 (M1, `parity_repro`) | 0 | 17:01:25 – 18:04:54 | 1 h 03 min 29 s |
| V3 | 114 | 114 | 0 | 0 | 18:05:08 – 19:05:54 | 1 h 00 min 46 s |
| V2 | 114 | 54 | 60 (da V3) | 0 | 19:06:07 – 21:24:43 | 2 h 18 min 36 s |
| V1 | 114 | 105 | 9 (prova `protocol_switch_dev_v1_trial`) | 0 | 21:24:56 – 23:35:40 | 2 h 10 min 44 s |

- **Tempo totale dei fit:** 24/11 17:01–23:35 UTC, cioè 18:01–00:35 ora locale.
- **`run.log`:** nessuna riga `WARNING` o `ERROR` e nessun avviso «ambiente diverso» nei 4 file (calcolo di Claude,
  `f8_inline_checks.py`, controllo 1).
- **Analisi:** dal `run_id` al `created_utc` di `analysis.json`:
  - primarie: 1 min 38 s – 1 min 57 s ciascuna;
  - con contesto: 3 min 01 s.
- **Report:** `created_utc` 2026-11-25T09:40:00Z.

**Tempi per modello** (somma di `total_seconds` dei `meta.json`, dai 18 fit per modello di ogni variante; per V1
comprende i 3 fit della prova):

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `conv_autoencoder` | 530 s | 453 s | 712 s | 776 s |
| `lstm_autoencoder` | 1.460 s | 1.508 s | 3.907 s | 3.504 s |
| `lstm_forecaster` | 1.728 s | 1.252 s | 3.692 s | 3.556 s |
| `pca` (6 fit) | 101 s | 75 s | 75 s | 75 s |

**Finestre di training** (`n_fit_windows`):
- V0: 3.772–4.579;
- V3, V2, V1: 2.824–3.430, come nella misura di D-38;
- PCA puntuale (punti, non finestre): 18.958–22.994 in V0, 14.218–17.246 nelle altre.

**Finestre della loss di early stopping:** 928–1.130 in V0 (es = cal), 929–1.130 nelle altre.

### 2.4 Spazio su disco

Calcolo di Claude (`f8_collect.py`; spazio per run con `f8_tables.py`). `store/` occupa 4.535.035.304 byte (4,54 GB,
4,22 GiB): `fits/` 4,21 GiB, `runs/` 0,011 GiB, `tmp/` vuota. Ci sono 396 fit, tutti `complete`:

| creati da | fit | spazio |
|---|---|---|
| M1 (`parity_repro`) | 8 | 0,126 GB |
| prova (`protocol_switch_dev_v1_trial`) | 9 | 0,134 GB |
| V0 | 106 | 1,074 GB |
| V3 | 114 | 1,398 GB |
| V2 | 54 | 0,528 GB |
| V1 | 105 | 1,263 GB |

Il totale resta nel budget di D-56. Nello store restano anche le due analisi della prova (`50f0a6525c77`,
`88b7d9c3e1d5`): non sono usate qui.

## 3. Controlli

| # | controllo | atteso | esito | fonte |
|---|---|---|---|---|
| 1 | Fit condivisi V3 → V2 (stesso `fit_key`: PCA, PCA puntuale, baseline, AE non addestrati, `random_baseline`): valori per seed identici bit a bit | 0 differenze | **0 su 21.888 righe** | script |
| 2 | Stessi punti di valutazione ed eventi in tutte le varianti | 0 differenze | **0**; righe presenti in una sola variante: 0 per ogni passaggio | script |
| 3 | Riuso dei fit di M1 in V0 | 8 fit, quelli di `docs/m1_parita.md` §5 | **8 su 8**, vedi sotto | calcolo di Claude, `f8_collect.py` |
| 4 | C15: `n_nonfinite_cal` > 0, righe `nonfinite_residual_cal` | da contare | **0 fit, 0 righe** in tutte le varianti | script; tabella `fits` ricontata da Claude, `f8_collect.py` |
| 5 | Fit al tetto (criterio di D-39), `best_epoch` nulli | da contare | tabella sotto; `best_epoch` nulli 0 | script; ricontato da Claude dai `meta.json`, `f8_collect.py` |
| 6 | Coerenza interna (a): su machine-1-7 τ non cambia nessun valore scalato, quindi V2 → V1 è nullo | 0 differenze | **0 su 1.920 righe `free` e 4.224 `thr`** | calcolo di Claude, `f8_consistency.py` |
| 7 | Coerenza interna (b): V3 → V2, fit addestrati con lo stesso `best_epoch` | identici | **8 fit, 0 differenze su 860 + 1.892 righe**; controllo positivo: righe diverse quando il `best_epoch` cambia | calcolo di Claude, `f8_consistency.py` |

Note ai controlli:
- **Controllo 2:** `n_eval_points` ed `n_events` per macchina, uguali in tutte le varianti (tabella: calcolo di
  Claude, `f8_tables.py`):

  | macchina | punti valutati (`primary`) | eventi | prevalenza | punti con contesto (V1) |
  |---|---|---|---|---|
  | 1-1 | 28.281 | 8 | 0,0953 | 28.380 |
  | 1-7 | 23.499 | 13 | 0,1020 | 23.598 |
  | 2-6 | 28.545 | 8 | 0,0149 | 28.644 |
  | 2-8 | 23.505 | 1 | 0,0069 | 23.604 |
  | 3-11 | 28.498 | 3 | 0,0069 | 28.597 |
  | 3-8 | 28.506 | 6 | 0,0481 | 28.605 |

- **Controllo 3:**
  - i fit saltati da V0 sono esattamente gli 8 `fit_key` di M1: `pca`, `pca_pointwise`, `conv_autoencoder` e
    `lstm_autoencoder` (seed 42) su machine-1-1 e machine-3-8;
  - origine: run `20261119T144234-c54fd9f8`, commit `4d8d248`, stesso `lock_sha256`;
  - epoche (best / stop) uguali a M1: conv 1-1 39 / 40, lstm 1-1 8 / 13, conv 3-8 17 / 22, lstm 3-8 21 / 26.
- **Controllo 4:**
  - nella tabella `fits` di tutte le varianti `n_nonfinite_cal`, `n_nonfinite_test` e `nonfinite_loss` sono a zero;
  - lo scostamento dichiarato su `raw_mean` (D-72 C15) non si è esercitato nemmeno qui, come in M1.
- **Controlli 6 e 7:** sono controlli aggiunti da Claude, non richiesti dalla nota di F8.
  - (a) Su machine-1-7 nessun sensore ha range su fit in (0; τ) e i sensori a range nullo sono costanti anche nel test
    (nota del 2026-11-21 su D-40), quindi V1 = V2 bit a bit è atteso.
  - (b) Con lo stesso seed e gli stessi dati, V2 ripete le epoche di V3 e le prosegue. Il `best_epoch` di V2 non è mai
    minore di quello di V3: 0 casi su 54. Quando il `best_epoch` coincide, V2 ripristina lo stesso stato di V3, quindi
    i residui e le metriche devono essere identici.
  - **Gli 8 fit del controllo 7:**
    - 7 sono fit che in V3 si fermano per early stopping (`stopped_epoch` − `best_epoch` = 5) e in V2 non migliorano
      nelle 5 epoche in più, quindi si fermano a `best_epoch` + 10;
    - l'ottavo è `conv_autoencoder` su machine-1-7 con seed 44. In V3 si ferma **al tetto** (best 40, stop 40): non è
      tra i 38 fermati per early stopping. In V2 prosegue senza migliorare e si ferma per early stopping all'epoca 50,
      con best ancora 40.
- **Residui di 1 ulp:** il report mostra «−0,0000» sulla `random_baseline` in V0 → V3 (6 righe, |Δ| ≤ 2,7e−20) e
  V2 → V1 (9 righe, ≤ 1,1e−16). I valori per seed sono identici bit a bit (calcolo di Claude, `f8_inline_checks.py`,
  controllo 2). Il residuo nasce nella media sui seed dello script, perché le righe arrivano in un ordine diverso
  (42, 43, 44 contro 43, 44, 42). Non è una differenza della pipeline.

### 3.1 Fit al tetto ed epoche (criterio di D-39)

Al tetto = `stopped_epoch` = `max_epochs` e `stopped_epoch` − `best_epoch` < `patience`. Il conteggio dello script e il
ricalcolo di Claude dai `meta.json` (`f8_collect.py`) coincidono.

| variante | tetto / patience | `conv_autoencoder` | `lstm_autoencoder` | `lstm_forecaster` | totale al tetto |
|---|---|---|---|---|---|
| V0 | 40 / 5 | 5 / 18 | 2 / 18 | 9 / 18 | 16 / 54 |
| V3 | 40 / 5 | 3 / 18 | 6 / 18 | 7 / 18 | 16 / 54 |
| V2 | 100 / 10 | 0 / 18 | 2 / 18 | 2 / 18 | 4 / 54 |
| V1 | 100 / 10 | 0 / 18 | 0 / 18 | 4 / 18 | 4 / 54 |

`best_epoch` e `stopped_epoch`, minimo / mediana / massimo sui 18 fit:

| variante | conv best | conv stop | lstm best | lstm stop | forecaster best | forecaster stop |
|---|---|---|---|---|---|---|
| V0 | 11 / 25,5 / 39 | 16 / 30,5 / 40 | 6 / 15,5 / 40 | 11 / 20,5 / 40 | 16 / 34 / 40 | 21 / 38,5 / 40 |
| V3 | 12 / 29,5 / 40 | 17 / 34,5 / 40 | 5 / 28,5 / 40 | 10 / 33,5 / 40 | 24 / 31 / 40 | 29 / 36 / 40 |
| V2 | 19 / 39 / 71 | 29 / 49 / 81 | 13 / 48 / 96 | 23 / 58 / 100 | 27 / 64 / 94 | 37 / 74 / 100 |
| V1 | 19 / 47,5 / 71 | 29 / 57,5 / 81 | 22 / 47,5 / 71 | 32 / 57,5 / 81 | 12 / 65,5 / 98 | 22 / 75,5 / 100 |

I fit al tetto in V2 e V1, per macchina e seed (best / stop; calcolo di Claude, `f8_collect.py`):
- V2: lstm 2-8 s42 (92 / 100) e 3-11 s43 (96 / 100); forecaster 2-6 s42 (94 / 100) e 3-8 s42 (94 / 100);
- V1: forecaster 2-6 s42 (94 / 100), 2-6 s43 (98 / 100), 3-8 s42 (96 / 100), 3-8 s43 (94 / 100).

I 16 casi di V0 e i 16 di V3 sono in `report.json` e `fits_summary.csv`.

## 4. Δ per passaggio, per tipo di differenza

Tipi della nota di F8:
- correzioni dichiarate: P1, D-72 C15, D-72 C1;
- cambi di protocollo approvati: fit 80% → 60% (D-05, D-65);
- cambi di design dichiarati: D-39, D-40;
- fattori di protocollo della letteratura.

Le tabelle principali usano scenario `primary`, `window_time_agg` `last` e `ts_agg` `last`, cioè la proiezione
causale. Per `rolling_zscore` e `random_baseline`, che non hanno scoring per sensore, la cella è «native». Le
tabelle di robustezza coprono tutte le celle `free`:
- 2 `window_time_agg` × 3 `ts_agg` × 4 scoring × 6 macchine = 144 celle per modello;
- 24 celle per `lstm_forecaster`, che ha solo `last`;
- 18 celle per i modelli native.

I Δ con mediana [min; max] vengono da `delta_summary.csv` dello script. La mediana di |Δ| tra graffe e la colonna
«mediana |Δ|» sono un calcolo di Claude (`f8_abs_deltas.py`). Le tabelle complete, metriche con soglia comprese, sono
in `delta_summary.csv`.

**Metriche secondarie:** solo ROC-AUC. VUS-PR non è ancora implementata (D-34: passo separato prima di F10).

### 4.1 V0 → V3: correzione dichiarata P1 e cambio di protocollo approvato (fit 80% → 60%)

**Fatti: AP normalizzata.**

| modello | mean/none | mean/std | max/std | top5/std |
|---|---|---|---|---|
| `pca` | −0,0053 [−0,0792; +0,1082] {0,0097} | +0,0047 [−0,1203; +0,0191] {0,0056} | −0,0058 [−0,1675; +0,0061] {0,0065} | +0,0038 [−0,1434; +0,0098] {0,0058} |
| `pca_pointwise` | −0,0050 [−0,0836; +0,0833] {0,0193} | −0,0002 [−0,1065; +0,1636] {0,0165} | +0,0000… [−0,0427; +0,0986] {0,0134} | −0,0003 [−0,0961; +0,1559] {0,0202} |
| `conv_autoencoder` | +0,0023 [−0,0810; +0,0863] {0,0182} | −0,0009 [−0,0374; +0,0725] {0,0142} | −0,0001 [−0,0433; +0,0935] {0,0140} | −0,0004 [−0,0336; +0,0720] {0,0126} |
| `lstm_autoencoder` | +0,0014 [−0,1062; +0,0786] {0,0022} | +0,0010 [−0,0525; +0,1543] {0,0104} | +0,0047 [−0,0023; +0,1522] {0,0047} | −0,0002 [−0,0292; +0,1449] {0,0092} |
| `lstm_forecaster` | −0,0003 [−0,0728; +0,2074] {0,0166} | −0,0015 [−0,0326; +0,0592] {0,0037} | −0,0037 [−0,0505; +0,0250] {0,0067} | −0,0016 [−0,0403; +0,0411] {0,0044} |
| `conv_autoencoder_untrained` | +0,0003 [−0,0084; +0,0412] {0,0004} | −0,0001 [−0,0020; +0,0539] {0,0013} | −0,0002 [−0,0046; +0,0257] {0,0034} | +0,0006 [−0,0102; +0,0421] {0,0018} |
| `lstm_autoencoder_untrained` | +0,0003 [−0,0095; +0,0443] {0,0004} | −0,0000… [−0,0012; +0,0583] {0,0008} | −0,0000… [−0,0052; +0,0601] {0,0014} | +0,0011 [−0,0083; +0,0473] {0,0026} |
| `input_norm` | +0,0003 [−0,0096; +0,0457] {0,0004} | +0,0000… [−0,0030; +0,0541] {0,0021} | 0 [−0,0034; +0,0377] {0,0011} | −0,0000… [−0,0074; +0,0447] {0,0032} |
| `rolling_zscore` | −0,0010 [−0,0061; +0,0006] {0,0010} (native) | · | · | · |
| `random_baseline` | 0 [−0,0000…; +0,0000…] {0} (native) | · | · | · |

**Fatti: ROC-AUC.**

| modello | mean/none | mean/std | max/std | top5/std |
|---|---|---|---|---|
| `pca` | −0,0015 [−0,0537; +0,0548] {0,0157} | −0,0004 [−0,0838; +0,0173] {0,0171} | −0,0004 [−0,1059; +0,0162] {0,0149} | +0,0034 [−0,0962; +0,0182] {0,0164} |
| `pca_pointwise` | −0,0157 [−0,0837; +0,0294] {0,0279} | −0,0247 [−0,0483; +0,0526] {0,0474} | −0,0002 [−0,0686; +0,0437] {0,0239} | −0,0153 [−0,0427; +0,0515] {0,0348} |
| `conv_autoencoder` | −0,0029 [−0,0463; +0,0548] {0,0155} | +0,0008 [−0,0166; +0,0327] {0,0032} | −0,0004 [−0,0106; +0,0308] {0,0073} | −0,0002 [−0,0101; +0,0293] {0,0049} |
| `lstm_autoencoder` | −0,0009 [−0,0659; +0,0932] {0,0025} | −0,0025 [−0,0088; +0,0626] {0,0062} | +0,0003 [−0,0382; +0,0568] {0,0156} | +0,0005 [−0,0223; +0,0619] {0,0051} |
| `lstm_forecaster` | −0,0005 [−0,0348; +0,0986] {0,0080} | −0,0047 [−0,0331; +0,0163] {0,0115} | −0,0014 [−0,0512; +0,0080] {0,0049} | −0,0049 [−0,0419; +0,0073] {0,0069} |
| `conv_autoencoder_untrained` | −0,0000… [−0,0089; +0,0166] {0,0008} | −0,0001 [−0,0014; +0,0027] {0,0006} | −0,0003 [−0,0016; +0,0001] {0,0003} | +0,0001 [−0,0001; +0,0012] {0,0001} |
| `lstm_autoencoder_untrained` | −0,0000… [−0,0112; +0,0197] {0,0009} | −0,0001 [−0,0016; +0,0030] {0,0005} | −0,0004 [−0,0027; +0,0036] {0,0009} | −0,0001 [−0,0013; +0,0034] {0,0002} |
| `input_norm` | −0,0000… [−0,0112; +0,0198] {0,0008} | −0,0000… [−0,0012; +0,0025] {0,0005} | 0 [−0,0006; +0,0013] {0,0001} | −0,0001 [−0,0006; +0,0018] {0,0002} |
| `rolling_zscore` | −0,0005 [−0,0167; +0,0028] {0,0017} (native) | · | · | · |
| `random_baseline` | 0 [0; 0] {0} (native) | · | · | · |

**Fatti: robustezza, AP normalizzata su tutte le celle.**

| modello | celle | mediana | mediana \|Δ\| | min | max | Δ > 0 | Δ < 0 | \|Δ\| ≥ 0,01 |
|---|---|---|---|---|---|---|---|---|
| `pca` | 144 | −0,0003 | 0,0088 | −0,3397 | +0,1082 | 64 | 78 | 67 |
| `pca_pointwise` | 144 | −0,0003 | 0,0113 | −0,1065 | +0,1636 | 62 | 82 | 77 |
| `conv_autoencoder` | 144 | +0,0001 | 0,0098 | −0,3326 | +0,1690 | 77 | 67 | 71 |
| `lstm_autoencoder` | 144 | +0,0001 | 0,0077 | −0,3323 | +0,1733 | 74 | 70 | 68 |
| `lstm_forecaster` | 24 | −0,0015 | 0,0053 | −0,0728 | +0,2074 | 6 | 18 | 9 |
| `conv_autoencoder_untrained` | 144 | −0,0003 | 0,0030 | −0,3323 | +0,0576 | 54 | 88 | 51 |
| `lstm_autoencoder_untrained` | 144 | −0,0000… | 0,0022 | −0,3323 | +0,0601 | 58 | 81 | 43 |
| `input_norm` | 144 | −0,0000… | 0,0022 | −0,3323 | +0,0613 | 60 | 73 | 48 |
| `rolling_zscore` | 18 | −0,0002 | 0,0006 | −0,0197 | +0,0049 | 4 | 14 | 1 |
| `random_baseline` | 18 | 0 | 0 | −0,0000… | +0,0000… | 3 | 3 | 0 |

**Fatti:**
- **Mediane:** sulle 6 macchine sono tra −0,006 e +0,005 per ogni modello e scoring.
- **Mediane di |Δ|:** in `last`/`last` sono tra 0,0004 e 0,020:
  - 0,002–0,020 per le due PCA e i modelli addestrati;
  - 0,0004–0,0034 per gli AE non addestrati e `input_norm`.

  Su tutte le celle sono tra 0,002 e 0,011, esclusi `rolling_zscore` e `random_baseline`.
- **Estremi:** sulle singole macchine i Δ vanno fino a ±0,1–0,2 in `last`/`last`, sia per i modelli senza early
  stopping sia per gli AE addestrati.
- **Minimi vicini a −0,33:** in 6 modelli (PCA, AE addestrati e non, `input_norm`) cadono tutti sulla stessa cella,
  `machine-2-8`, mean/none, `ts_agg` `max`, che ha un solo evento. Lì quattro modelli diversi hanno lo stesso
  `ap_norm` alla sesta cifra: 0,835342 in V0 e 0,503085 in V3.
- **Baseline random:** Δ = 0 come atteso (stessa geometria di cal e test, D-23).

**Ipotesi:**
- Per i modelli senza early stopping l'effetto della sola riduzione dei dati di fit (80% → 60%) ha sulle singole
  macchine la stessa grandezza dei Δ degli AE addestrati, anche in mediana di |Δ| (PCA 0,006–0,010, PCA puntuale
  0,013–0,020, contro 0,002–0,018 degli AE addestrati). Quindi la parte dei Δ degli AE dovuta a P1 non è
  distinguibile, oltre a non essere separabile per costruzione (§6).
- Su machine-2-8 la coincidenza di modelli diversi alla sesta cifra fa pensare che, almeno con la proiezione offline
  `max`, un solo sensore molto amplificato (2-8 ha |max| fino a 497 senza pavimento, nota del 2026-11-21 su D-40)
  decida lo score di tutti i modelli che usano i dati scalati. Con un solo evento l'AP è poi molto sensibile.

### 4.2 Correzioni dichiarate senza un passaggio proprio: C15 e C1

- **D-72 C15. Fatti:** 0 fit con `n_nonfinite_cal` > 0 e 0 righe `nonfinite_residual_cal` in V0, V3, V2 e V1. La
  correzione non ha effetto osservabile su queste 6 macchine.
- **D-72 C1. Fatti:** non osservabile in questo esperimento:
  - `main.py` non ha mai usato `rules.py`;
  - entrambe le modalità usano `thresholds.py`;
  - il piano non ha soglie mobili, perché D-42 è APERTA.

  Resta tra le correzioni dichiarate e si misura quando le soglie mobili entreranno nelle analisi.
- **P1:** è nel passaggio V0 → V3 (§4.1), confusa con la riduzione dei dati.

### 4.3 V3 → V2: cambio di design dichiarato D-39 (epoche 40 / 5 → 100 / 10)

**Fatti:**
- **Modelli non addestrati:** PCA, PCA puntuale, AE non addestrati, `input_norm`, `rolling_zscore` e
  `random_baseline` hanno Δ = 0 esatto in tutte le 2.304 righe per modello (controllo 1). La mediana di |Δ| è quindi 0.
- **Modelli addestrati:**

  | metrica | modello | mean/none | mean/std | max/std | top5/std |
  |---|---|---|---|---|---|
  | `ap_norm` | `conv_autoencoder` | +0,0019 [−0,0010; +0,0172] {0,0027} | −0,0053 [−0,0787; +0,0426] {0,0059} | −0,0014 [−0,0260; +0,1205] {0,0091} | −0,0039 [−0,0272; +0,0579] {0,0063} |
  | `ap_norm` | `lstm_autoencoder` | +0,0004 [−0,0031; +0,0121] {0,0019} | +0,0001 [−0,0578; +0,0037] {0,0037} | +0,0001 [−0,1720; +0,0360] {0,0020} | −0,0041 [−0,0886; +0,0037] {0,0055} |
  | `ap_norm` | `lstm_forecaster` | +0,0005 [−0,0021; +0,0026] {0,0012} | −0,0003 [−0,0070; +0,0093] {0,0035} | +0,0039 [−0,0088; +0,0201] {0,0089} | −0,0001 [−0,0040; +0,0206] {0,0028} |
  | ROC-AUC | `conv_autoencoder` | +0,0007 [−0,0087; +0,0265] {0,0039} | +0,0018 [−0,0078; +0,0144] {0,0060} | −0,0014 [−0,0024; +0,0240] {0,0014} | +0,0004 [−0,0031; +0,0121] {0,0024} |
  | ROC-AUC | `lstm_autoencoder` | −0,0002 [−0,0041; +0,0087] {0,0027} | −0,0010 [−0,0292; +0,0035] {0,0014} | −0,0004 [−0,0325; +0,0115] {0,0051} | −0,0006 [−0,0295; +0,0058] {0,0010} |
  | ROC-AUC | `lstm_forecaster` | −0,0000… [−0,0049; +0,0073] {0,0006} | +0,0018 [−0,0055; +0,0242] {0,0037} | +0,0028 [−0,0000…; +0,0262] {0,0028} | +0,0037 [−0,0021; +0,0223] {0,0042} |

- **Robustezza (`ap_norm`, tutte le celle):**

  | modello | celle | mediana | mediana \|Δ\| | min | max | Δ > 0 | Δ < 0 | \|Δ\| ≥ 0,01 |
  |---|---|---|---|---|---|---|---|---|
  | `conv_autoencoder` | 144 | +0,0001 | 0,0050 | −0,0787 | +0,1205 | 76 | 68 | 53 |
  | `lstm_autoencoder` | 144 | −0,0002 | 0,0021 | −0,1720 | +0,0360 | 58 | 84 | 39 |
  | `lstm_forecaster` | 24 | −0,0002 | 0,0023 | −0,0088 | +0,0206 | 11 | 13 | 3 |

- **Fit al tetto:** da 16 su 54 (V3) a 4 su 54 (V2), §3.1.
- **Fit che restano identici:** 8 fit addestrati hanno lo stesso `best_epoch` in V3 e in V2 e danno valori identici
  (controllo 7):
  - 7 dei 38 fit che in V3 si fermano per early stopping;
  - più `conv_autoencoder` su machine-1-7 con seed 44, che in V3 si ferma al tetto con best = stop = 40 e in V2 non
    migliora più.

**Ipotesi:** con epoche e patience più ampie le mediane restano vicine a zero (mediana di |Δ| ≤ 0,0091 in
`last`/`last`), ma su singole macchine e seed l'epoca scelta cambia molto (mediana del best dell'LSTM-AE da 28,5 a
48). La direzione dei Δ non è costante tra modelli e scoring.

### 4.4 V2 → V1: cambio di design dichiarato D-40 (pavimento τ = 0,02)

**Fatti: AP normalizzata.**

| modello | mean/none | mean/std | max/std | top5/std |
|---|---|---|---|---|
| `pca` | −0,0056 [−0,0730; +0,0495] {0,0090} | +0,0008 [−0,0797; +0,0302] {0,0124} | −0,0001 [−0,0008; +0,0363] {0,0007} | −0,0004 [−0,0657; +0,0340] {0,0198} |
| `pca_pointwise` | +0,0000… [0; +0,0093] {0,0000…} | 0 [0; +0,0425] {0} | 0 [−0,0000…; +0,0151] {0,0000…} | +0,0000… [0; +0,0437] {0,0000…} |
| `conv_autoencoder` | +0,0006 [−0,0005; +0,0772] {0,0008} | +0,0012 [−0,0290; +0,1081] {0,0079} | +0,0001 [−0,0690; +0,1059] {0,0174} | +0,0021 [−0,0260; +0,1046] {0,0051} |
| `lstm_autoencoder` | −0,0000… [−0,0079; +0,0324] {0,0058} | +0,0024 [−0,0002; +0,0593] {0,0025} | +0,0011 [−0,0140; +0,1236] {0,0118} | +0,0013 [−0,0008; +0,0627] {0,0017} |
| `lstm_forecaster` | +0,0004 [−0,0115; +0,0330] {0,0021} | +0,0015 [−0,0077; +0,0516] {0,0049} | +0,0017 [−0,0106; +0,0271] {0,0079} | +0,0011 [−0,0091; +0,0538] {0,0051} |
| `conv_autoencoder_untrained` | −0,0000… [−0,0153; +0,0279] {0,0109} | +0,0006 [−0,0045; +0,0514] {0,0026} | +0,0007 [−0,0008; +0,0335] {0,0009} | +0,0004 [−0,0031; +0,0521] {0,0017} |
| `lstm_autoencoder_untrained` | −0,0000… [−0,0154; +0,0335] {0,0108} | +0,0000… [−0,0074; +0,0590] {0,0019} | +0,0002 [−0,0109; +0,0406] {0,0009} | +0,0007 [−0,0076; +0,0592] {0,0019} |
| `input_norm` | −0,0000… [−0,0153; +0,0337] {0,0109} | +0,0000… [−0,0056; +0,0552] {0,0025} | 0 [−0,0048; +0,0375] {0,0008} | +0,0000… [−0,0053; +0,0553] {0,0009} |
| `rolling_zscore` | +0,0011 [0; +0,0178] {0,0011} (native) | · | · | · |
| `random_baseline` | 0 [0; +0,0000…] {0} (native) | · | · | · |

**Fatti: ROC-AUC.**

| modello | mean/none | mean/std | max/std | top5/std |
|---|---|---|---|---|
| `pca` | −0,0028 [−0,0176; +0,0229] {0,0112} | −0,0013 [−0,0182; +0,0239] {0,0084} | −0,0011 [−0,0034; +0,0152] {0,0019} | −0,0020 [−0,0155; +0,0319] {0,0081} |
| `pca_pointwise` | +0,0000… [0; +0,0064] {0,0000…} | 0 [0; +0,0096] {0} | 0 [−0,0000…; +0,0038] {0,0000…} | +0,0000… [0; +0,0122] {0,0000…} |
| `conv_autoencoder` | +0,0026 [−0,0042; +0,0218] {0,0043} | −0,0003 [−0,0083; +0,0093] {0,0020} | +0,0002 [−0,0198; +0,0108] {0,0063} | +0,0010 [−0,0087; +0,0073] {0,0025} |
| `lstm_autoencoder` | −0,0005 [−0,0091; +0,0213] {0,0016} | +0,0009 [−0,0008; +0,0248] {0,0013} | +0,0013 [−0,0165; +0,0203] {0,0044} | −0,0001 [−0,0010; +0,0238] {0,0008} |
| `lstm_forecaster` | +0,0004 [−0,0055; +0,0046] {0,0012} | +0,0018 [0; +0,0208] {0,0018} | +0,0056 [−0,0025; +0,0182] {0,0056} | +0,0029 [0; +0,0232] {0,0029} |
| `conv_autoencoder_untrained` | −0,0005 [−0,0421; +0,0025] {0,0008} | +0,0002 [−0,0004; +0,0045] {0,0004} | +0,0004 [0; +0,0052] {0,0004} | +0,0002 [−0,0001; +0,0010] {0,0002} |
| `lstm_autoencoder_untrained` | −0,0005 [−0,0423; +0,0025] {0,0007} | +0,0003 [−0,0004; +0,0012] {0,0005} | +0,0004 [−0,0014; +0,0050] {0,0009} | +0,0002 [−0,0008; +0,0011] {0,0006} |
| `input_norm` | −0,0006 [−0,0414; +0,0024] {0,0008} | +0,0000… [−0,0003; +0,0048] {0,0002} | +0,0000… [−0,0002; +0,0051] {0,0001} | +0,0000… [−0,0002; +0,0007] {0,0002} |
| `rolling_zscore` | +0,0019 [−0,0035; +0,0100] {0,0036} (native) | · | · | · |
| `random_baseline` | 0 [0; 0] {0} (native) | · | · | · |

**Fatti: robustezza, AP normalizzata su tutte le celle.**

| modello | celle | mediana | mediana \|Δ\| | min | max | Δ > 0 | Δ < 0 | \|Δ\| ≥ 0,01 |
|---|---|---|---|---|---|---|---|---|
| `pca` | 144 | 0 | 0,0102 | −0,3677 | +0,3718 | 60 | 60 | 72 |
| `pca_pointwise` | 144 | 0 | 0 | −0,0055 | +0,0437 | 26 | 28 | 3 |
| `conv_autoencoder` | 144 | 0 | 0,0146 | −0,4161 | +0,3631 | 67 | 53 | 78 |
| `lstm_autoencoder` | 144 | 0 | 0,0123 | −0,3565 | +0,3775 | 70 | 50 | 75 |
| `lstm_forecaster` | 24 | +0,0010 | 0,0044 | −0,0115 | +0,0538 | 15 | 5 | 9 |
| `conv_autoencoder_untrained` | 144 | 0 | 0,0064 | −0,4109 | +0,3546 | 65 | 55 | 62 |
| `lstm_autoencoder_untrained` | 144 | 0 | 0,0089 | −0,4129 | +0,3451 | 58 | 62 | 68 |
| `input_norm` | 144 | 0 | 0,0063 | −0,4119 | +0,3532 | 64 | 54 | 63 |
| `rolling_zscore` | 18 | +0,0002 | 0,0009 | −0,0829 | +0,0752 | 11 | 3 | 6 |
| `random_baseline` | 18 | 0 | 0 | 0 | +0,0000… | 6 | 0 | 0 |

**Fatti:**
- **Mediane:** in `last`/`last` sono tra −0,006 e +0,003, e la mediana su tutte le celle è 0 per quasi tutti i
  modelli. Le celle uguali sono soprattutto quelle di machine-1-7, invariata per costruzione (controllo 6).
- **Mediane di |Δ|:** in `last`/`last` sono tra 0 (PCA puntuale) e 0,020 (PCA, top5/std). Su tutte le celle sono tra
  0,004 e 0,015 per i modelli diversi da PCA puntuale, `rolling_zscore` e `random_baseline`, cioè dello stesso ordine
  di V0 → V3.
- **Celle con |Δ| > 0,3** (`ap_norm`): sono tutte sulle proiezioni offline (`ts_agg` `max` o `mean`) e solo su due
  macchine:
  - machine-2-8, 1 evento: da circa 0,85 a circa 0,49 per PCA, AE addestrati e non, `input_norm`;
  - machine-3-11, 3 eventi: per AE addestrati e non e `input_norm`, da circa 0,57 a circa 0,92 con
    `window_time_agg` `last`, e da circa 0,46 a circa 0,05 con `window_time_agg` `mean` e max/std; per la PCA con
    `window_time_agg` `mean`, da circa 0,48 a 0,80–0,85.
- **Machine-2-6:** non ha sensori a range nullo variabili nel test, ma ha 3 sensori con range su fit in (0; τ)
  (6, 23, 25). Cambiano 1.633 righe `free` su 1.920.

**Ipotesi:** i Δ grandi su 2-8 e 3-11 vengono da due meccanismi del pavimento, presenti entrambi su queste due macchine
(nota del 2026-11-21 su D-40, §5):
- i **sensori quasi costanti su fit**, con range in (0; τ), la cui amplificazione scende da circa 500–900× a 50×
  (casi peggiori: 2-8 sensore 29, |max| 497; 3-11 sensore 9, |max| 922);
- i **sensori a range nullo su fit ma variabili nel test**, la cui scala sale da 1 a 1/τ = 50 (2-8 {16, 17, 28};
  3-11 {10, 31}).

Il primo meccanismo riduce il peso di un sensore, il secondo lo aumenta; questi dati non permettono di separarli. Con
uno o tre eventi l'AP si sposta molto quando cambia l'ordine di pochi punti. La direzione è opposta sulle due
macchine, quindi non c'è un segno comune del cambio.

### 4.5 Fattori di protocollo della letteratura

I contrasti sono calcolati dentro ogni variante, sulla media dei seed per macchina; poi si confronta V1 con V0. Celle:
- scoring mean/std con `window_time_agg` `last`, oppure native;
- per le soglie: `ts_agg` `last`, soglia `static_pct` 98.

Le altre combinazioni sono in `contrasts.csv` e `contrasts_v1_vs_v0.csv`. Le tabelle di questa sezione sono un calcolo
di Claude sui CSV dello script (`f8_tables.py`; confronto a parità di metrica con `f8_abs_deltas.py`).
`lstm_forecaster` non ha proiezioni offline (D-74). V3 e V2 coincidono per i modelli non addestrati, come atteso.

**Fatti: proiezione offline `ts_agg` mean − last (`ap_norm`).**

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `pca` | −0,0118 [−0,2272; +0,2649] | +0,0178 [−0,2229; +0,2524] | +0,0178 [−0,2229; +0,2524] | +0,0137 [−0,3360; +0,2946] |
| `pca_pointwise` | +0,0135 [−0,2565; +0,4706] | −0,0586 [−0,2407; +0,4667] | −0,0586 [−0,2407; +0,4667] | −0,0586 [−0,2417; +0,4242] |
| `conv_autoencoder` | −0,0442 [−0,1395; +0,4391] | −0,0243 [−0,1419; +0,4353] | −0,0218 [−0,1378; +0,4379] | −0,0076 [−0,2823; +0,5918] |
| `lstm_autoencoder` | −0,0144 [−0,1508; +0,2859] | −0,0079 [−0,1544; +0,2943] | −0,0149 [−0,1557; +0,2906] | −0,0112 [−0,3158; +0,6089] |
| `conv_autoencoder_untrained` | +0,0033 [−0,1683; +0,2904] | −0,0238 [−0,1682; +0,2878] | −0,0238 [−0,1682; +0,2878] | −0,0211 [−0,3611; +0,5900] |
| `lstm_autoencoder_untrained` | +0,0082 [−0,1667; +0,2926] | −0,0186 [−0,1666; +0,2898] | −0,0186 [−0,1666; +0,2898] | −0,0146 [−0,3666; +0,5746] |
| `input_norm` | +0,0050 [−0,1680; +0,2897] | −0,0210 [−0,1680; +0,2860] | −0,0210 [−0,1680; +0,2860] | −0,0177 [−0,3589; +0,5830] |
| `rolling_zscore` | −0,0062 [−0,0481; +0,1539] | −0,0141 [−0,0481; +0,1603] | −0,0141 [−0,0481; +0,1603] | +0,0233 [−0,0481; +0,1508] |

**Fatti: proiezione offline `ts_agg` max − last (`ap_norm`).**

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `pca` | −0,0202 [−0,2853; +0,2446] | +0,0447 [−0,2995; +0,2381] | +0,0447 [−0,2995; +0,2381] | +0,0159 [−0,3958; +0,2219] |
| `pca_pointwise` | +0,0004 [−0,3683; +0,4744] | −0,0422 [−0,3422; +0,4732] | −0,0422 [−0,3422; +0,4732] | −0,0422 [−0,3422; +0,4306] |
| `conv_autoencoder` | −0,0549 [−0,2892; +0,4471] | −0,0282 [−0,2691; +0,4514] | −0,0496 [−0,2085; +0,4569] | −0,0521 [−0,4538; +0,3572] |
| `lstm_autoencoder` | −0,0758 [−0,1796; +0,2840] | −0,0349 [−0,1943; +0,2920] | −0,0149 [−0,2022; +0,2880] | −0,0661 [−0,4678; +0,2456] |
| `conv_autoencoder_untrained` | +0,0650 [−0,1148; +0,2881] | +0,0649 [−0,1176; +0,2855] | +0,0649 [−0,1176; +0,2855] | +0,0643 [−0,4818; +0,2849] |
| `lstm_autoencoder_untrained` | +0,0777 [−0,1112; +0,2903] | +0,0735 [−0,1112; +0,2874] | +0,0735 [−0,1112; +0,2874] | +0,0687 [−0,4818; +0,2795] |
| `input_norm` | +0,0686 [−0,1135; +0,2874] | +0,0659 [−0,1135; +0,2837] | +0,0659 [−0,1135; +0,2837] | +0,0614 [−0,4819; +0,2794] |
| `rolling_zscore` | +0,0495 [−0,0244; +0,2500] | +0,0511 [−0,0234; +0,2561] | +0,0511 [−0,0234; +0,2561] | +0,0574 [−0,0252; +0,1554] |

**Fatti: PA con soglia statica (F1 con PA − F1 senza PA, `static_pct` 98).**

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `pca` | +0,0164 [0; +0,2635] | +0,0278 [0; +0,2619] | +0,0278 [0; +0,2619] | +0,0150 [0; +0,2619] |
| `pca_pointwise` | +0,0033 [0; +0,2641] | +0,0013 [0; +0,2625] | +0,0013 [0; +0,2625] | +0,0013 [0; +0,2625] |
| `conv_autoencoder` | +0,0049 [0; +0,2476] | +0,0098 [0; +0,2488] | +0,0048 [0; +0,2482] | +0,0072 [0; +0,2482] |
| `lstm_autoencoder` | +0,0186 [0; +0,7319] | +0,0092 [0; +0,7170] | +0,0096 [0; +0,5783] | +0,0086 [0; +0,5783] |
| `lstm_forecaster` | +0,0161 [0; +0,2748] | +0,0329 [0; +0,2743] | +0,0328 [0; +0,2750] | +0,0390 [0; +0,2750] |
| `conv_autoencoder_untrained` | +0,0926 [0; +0,8610] | +0,0923 [0; +0,8600] | +0,0923 [0; +0,8600] | +0,0926 [0; +0,8600] |
| `lstm_autoencoder_untrained` | +0,0872 [0; +0,8669] | +0,0878 [0; +0,8676] | +0,0878 [0; +0,8676] | +0,0880 [0; +0,8676] |
| `input_norm` | +0,0914 [0; +0,8563] | +0,0915 [0; +0,8570] | +0,0915 [0; +0,8570] | +0,0915 [0; +0,8570] |
| `rolling_zscore` | +0,6172 [+0,2234; +0,8474] | +0,6292 [+0,2235; +0,8463] | +0,6292 [+0,2235; +0,8463] | +0,6294 [+0,2258; +0,8463] |
| `random_baseline` | +0,6151 [+0,2191; +0,8941] | +0,6151 [+0,2191; +0,8941] | +0,6151 [+0,2191; +0,8941] | +0,6151 [+0,2191; +0,8941] |

**Fatti: PA con oracolo (best F1 con PA − best F1 senza PA).**

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `pca` | +0,4037 [+0,0288; +0,6228] | +0,4260 [+0,0253; +0,6178] | +0,4260 [+0,0253; +0,6178] | +0,4077 [+0,0698; +0,6232] |
| `pca_pointwise` | +0,5201 [+0,0282; +0,8321] | +0,5156 [+0,0282; +0,7871] | +0,5156 [+0,0282; +0,7871] | +0,5156 [+0,0282; +0,7216] |
| `conv_autoencoder` | +0,3600 [+0,0588; +0,7760] | +0,3811 [+0,0395; +0,7851] | +0,3782 [+0,0406; +0,7893] | +0,3857 [+0,0199; +0,6682] |
| `lstm_autoencoder` | +0,3795 [+0,0158; +0,6356] | +0,3881 [+0,0158; +0,5840] | +0,4311 [+0,0158; +0,5969] | +0,3891 [+0,0158; +0,5830] |
| `lstm_forecaster` | +0,4187 [+0,0179; +0,6404] | +0,4198 [+0,0190; +0,5819] | +0,4204 [+0,0256; +0,5810] | +0,4044 [+0,0244; +0,5769] |
| `conv_autoencoder_untrained` | +0,5606 [+0,0199; +0,7563] | +0,5541 [+0,0199; +0,7578] | +0,5541 [+0,0199; +0,7578] | +0,5416 [+0,0158; +0,7578] |
| `lstm_autoencoder_untrained` | +0,5598 [+0,0158; +0,7660] | +0,5587 [+0,0158; +0,7665] | +0,5587 [+0,0158; +0,7665] | +0,5437 [+0,0158; +0,7665] |
| `input_norm` | +0,5604 [+0,0189; +0,7490] | +0,5534 [+0,0189; +0,7507] | +0,5534 [+0,0189; +0,7507] | +0,5437 [+0,0158; +0,7507] |
| `rolling_zscore` | +0,7981 [+0,7638; +0,9072] | +0,7975 [+0,7525; +0,9127] | +0,7975 [+0,7525; +0,9127] | +0,7962 [+0,7484; +0,9060] |
| `random_baseline` | +0,7780 [+0,4117; +0,8171] | +0,7780 [+0,4117; +0,8171] | +0,7780 [+0,4117; +0,8171] | +0,7780 [+0,4117; +0,8171] |

**Fatti: oracolo − soglia statica (F1 senza PA; l'oracolo è un limite superiore diagnostico).**

| modello | V0 | V3 | V2 | V1 |
|---|---|---|---|---|
| `pca` | +0,2462 [+0,0047; +0,9489] | +0,2272 [+0,0023; +0,9522] | +0,2272 [+0,0023; +0,9522] | +0,2433 [+0,0023; +0,9078] |
| `pca_pointwise` | +0,1713 [+0,0122; +0,9510] | +0,2443 [+0,0256; +0,9510] | +0,2443 [+0,0256; +0,9510] | +0,2770 [+0,0256; +0,9510] |
| `conv_autoencoder` | +0,2368 [+0,0469; +0,9182] | +0,1954 [+0,0464; +0,9377] | +0,2259 [+0,0183; +0,9370] | +0,2684 [+0,0183; +0,9576] |
| `lstm_autoencoder` | +0,3579 [+0,1384; +0,9611] | +0,3526 [+0,1855; +0,9609] | +0,3016 [+0,1622; +0,9573] | +0,3003 [+0,1858; +0,9597] |
| `lstm_forecaster` | +0,2462 [+0,0104; +0,9584] | +0,2686 [+0,0104; +0,9581] | +0,2468 [+0,0121; +0,9504] | +0,2484 [+0,0121; +0,9529] |
| `conv_autoencoder_untrained` | +0,2167 [+0,0356; +0,9397] | +0,2224 [+0,0347; +0,9397] | +0,2224 [+0,0347; +0,9397] | +0,2216 [+0,0363; +0,9446] |
| `lstm_autoencoder_untrained` | +0,2180 [+0,0304; +0,9438] | +0,2204 [+0,0292; +0,9438] | +0,2204 [+0,0292; +0,9438] | +0,2196 [+0,0271; +0,9448] |
| `input_norm` | +0,2158 [+0,0367; +0,9405] | +0,2210 [+0,0344; +0,9405] | +0,2210 [+0,0344; +0,9405] | +0,2174 [+0,0341; +0,9437] |
| `rolling_zscore` | +0,0735 [+0,0115; +0,1061] | +0,0684 [+0,0115; +0,1079] | +0,0684 [+0,0115; +0,1079] | +0,0778 [+0,0113; +0,1147] |
| `random_baseline` | +0,0462 [+0,0077; +0,1561] | +0,0462 [+0,0077; +0,1561] | +0,0462 [+0,0077; +0,1561] | +0,0462 [+0,0077; +0,1561] |

**Fatti: contesto all'inizio del test (solo V1, `ap_norm`, `test_context` − `primary`).** Per tutti i modelli la
mediana è ±0,0000… e i valori stanno in [−0,0073; +0,0003]. Gli estremi negativi sono degli AE non addestrati e di
`input_norm` (fino a −0,0073); la `random_baseline` è ±0,0000…. Lo scenario con contesto aggiunge 99 punti per
macchina (§3).

**Fatti: V1 − V0 dei contrasti.**

| modello | mean − last | max − last | PA (statica) | PA (oracolo) | oracolo − statica |
|---|---|---|---|---|---|
| `pca` | −0,0175 [−0,1088; +0,0674] | −0,0103 [−0,2893; +0,0904] | −0,0008 [−0,0280; +0,0088] | +0,0163 [−0,0377; +0,0899] | +0,0115 [−0,0411; +0,0455] |
| `pca_pointwise` | −0,0071 [−0,1304; +0,0448] | +0,0006 [−0,0860; +0,0911] | −0,0008 [−0,0354; 0] | −0,0002 [−0,1386; +0,0931] | +0,0067 [−0,0486; +0,1293] |
| `conv_autoencoder` | +0,0137 [−0,1612; +0,1527] | −0,0158 [−0,3625; +0,0807] | +0,0003 [−0,0042; +0,0088] | −0,0283 [−0,1078; +0,0312] | +0,0255 [−0,0595; +0,1102] |
| `lstm_autoencoder` | −0,0099 [−0,1815; +0,3229] | −0,0305 [−0,3566; +0,0351] | 0 [−0,1536; +0,0296] | −0,0207 [−0,0661; +0,0279] | +0,0056 [−0,1279; +0,0474] |
| `lstm_forecaster` | — | — | +0,0001 [−0,0053; +0,0594] | −0,0006 [−0,0635; +0,0219] | −0,0019 [−0,0192; +0,0272] |
| `conv_autoencoder_untrained` | −0,0087 [−0,2430; +0,2996] | −0,0030 [−0,3670; +0,0014] | +0,0001 [−0,0010; +0,0018] | −0,0029 [−0,0284; +0,0015] | +0,0028 [−0,0021; +0,0284] |
| `lstm_autoencoder_untrained` | −0,0010 [−0,2507; +0,2820] | −0,0147 [−0,3706; +0,0005] | +0,0000… [−0,0013; +0,0014] | −0,0021 [−0,0299; +0,0006] | +0,0013 [−0,0032; +0,0298] |
| `input_norm` | −0,0053 [−0,2403; +0,2933] | −0,0093 [−0,3684; +0,0006] | +0,0001 [0; +0,0015] | −0,0015 [−0,0282; +0,0017] | +0,0016 [−0,0026; +0,0282] |
| `rolling_zscore` | +0,0005 [−0,0789; +0,0579] | +0,0010 [−0,0946; +0,0101] | +0,0077 [−0,0011; +0,0241] | −0,0025 [−0,0157; +0,0002] | −0,0000… [−0,0064; +0,0103] |
| `random_baseline` | 0 [0; 0] | 0 [0; 0] | 0 [0; 0] | 0 [0; +0,0000…] | 0 [0; 0] |

**Fatti: metrica.** Calcolo di Claude (`f8_tables.py`): lo script non ha un contrasto «metrica». Si confrontano i
ranking dei 9 modelli per `ap_norm` e per ROC-AUC (mean/std, `last`/`last`; `rolling_zscore` native), per macchina.

| variante | τ di Kendall, mediana [min; max] | macchine con lo stesso modello al primo posto |
|---|---|---|
| V0 | 0,78 [−0,33; 0,94] | 4 su 6 |
| V3 | 0,72 [−0,11; 0,94] | 5 su 6 |
| V2 | 0,83 [−0,17; 0,94] | 5 su 6 |
| V1 | 0,78 [0,06; 1,00] | 5 su 6 |

**Fatti: grandezze dei contrasti, per modello.** In mediana, sui modelli diversi da `rolling_zscore` e
`random_baseline`:
- +0,36…+0,56 per la PA con oracolo;
- +0,17…+0,36 per oracolo − statica;
- −0,08…+0,08 per le proiezioni offline (`ap_norm`).

Su `rolling_zscore` e `random_baseline` valgono +0,78…+0,80 per la PA con oracolo e +0,62…+0,63 per la PA con soglia
statica. Per gli altri modelli, la PA con soglia statica dà +0,001…+0,09. Tra V0 e V1 le mediane dei contrasti cambiano
di −0,031…+0,026; su singole macchine si arriva fino a −0,37 per `ts_agg` max.

#### 4.5.1 Confronto con la catena, a parità di metrica

Calcolo di Claude (`f8_abs_deltas.py`). Si confrontano grandezze assolute, nella stessa metrica e sullo stesso insieme
di celle:
- **AP normalizzata:** |Δ| dei passaggi della catena contro |contrasto| delle proiezioni offline;
- **F1 con soglia statica 98, senza PA:** |Δ| della catena contro il contrasto di PA, che non è mai negativo (PA non
  abbassa l'F1), quindi |contrasto| = contrasto.

Sono esclusi i contrasti con l'oracolo, che usa un'altra metrica (best F1 sul test). Modelli: i 9, senza
`random_baseline`, che ha Δ = 0 per costruzione nella catena. Celle:
- catena `last`/`last`: 4 scoring × 6 macchine per modello, `rolling_zscore` native, `lstm_forecaster` incluso, cioè
  198 celle;
- proiezioni offline con `window_time_agg` `last`: 174 celle, perché `lstm_forecaster` non ha proiezioni offline;
- «tutte le celle» aggiunge le altre `window_time_agg` e, per l'F1, le altre proiezioni.

**AP normalizzata.**

| grandezza | celle | mediana | 90° percentile | massimo |
|---|---|---|---|---|
| \|Δ\| V0 → V3, `last`/`last` | 198 | 0,0039 | 0,0788 | 0,2074 |
| \|Δ\| V3 → V2, `last`/`last` | 198 | 0 | 0,0089 | 0,1720 |
| \|Δ\| V2 → V1, `last`/`last` | 198 | 0,0025 | 0,0501 | 0,1236 |
| \|Δ\| V0 → V3, tutte le celle | 1.050 | 0,0053 | 0,0638 | 0,3397 |
| \|Δ\| V3 → V2, tutte le celle | 1.050 | 0 | 0,0085 | 0,1720 |
| \|Δ\| V2 → V1, tutte le celle | 1.050 | 0,0055 | 0,1101 | 0,4161 |
| \|mean − last\|, V0 / V3 / V2 / V1, `window_time_agg` `last` | 174 | 0,135 / 0,137 / 0,136 / 0,150 | 0,290 / 0,306 / 0,307 / 0,537 | 0,518 / 0,558 / 0,564 / 0,689 |
| \|max − last\|, V0 / V3 / V2 / V1, `window_time_agg` `last` | 174 | 0,136 / 0,147 / 0,147 / 0,176 | 0,293 / 0,325 / 0,333 / 0,453 | 0,498 / 0,508 / 0,508 / 0,495 |
| \|mean − last\|, V0 / V3 / V2 / V1, tutte le `window_time_agg` | 342 | 0,140 / 0,145 / 0,143 / 0,138 | 0,322 / 0,366 / 0,364 / 0,476 | 0,518 / 0,558 / 0,564 / 0,689 |
| \|max − last\|, V0 / V3 / V2 / V1, tutte le `window_time_agg` | 342 | 0,148 / 0,149 / 0,149 / 0,153 | 0,342 / 0,343 / 0,343 / 0,445 | 0,498 / 0,508 / 0,508 / 0,635 |

**F1 con soglia statica 98, senza PA.**

| grandezza | celle | mediana | 90° percentile | massimo |
|---|---|---|---|---|
| \|Δ\| V0 → V3, `last`/`last` | 198 | 0,0019 | 0,0343 | 0,1524 |
| \|Δ\| V3 → V2, `last`/`last` | 198 | 0 | 0,0086 | 0,1201 |
| \|Δ\| V2 → V1, `last`/`last` | 198 | 0,0004 | 0,0153 | 0,0844 |
| \|Δ\| V0 → V3, tutte le celle | 1.050 | 0,0018 | 0,0314 | 0,2069 |
| \|Δ\| V3 → V2, tutte le celle | 1.050 | 0 | 0,0045 | 0,1201 |
| \|Δ\| V2 → V1, tutte le celle | 1.050 | 0,0003 | 0,0161 | 0,1231 |
| PA statica, V0 / V3 / V2 / V1, `last`/`last` | 198 | 0,0145 / 0,0167 / 0,0152 / 0,0150 | 0,305 / 0,314 / 0,314 / 0,321 | 0,871 / 0,871 / 0,868 / 0,868 |
| PA statica, V0 / V3 / V2 / V1, tutte le celle | 1.050 | 0 / 0 / 0 / 0 | 0,248 / 0,247 / 0,247 / 0,250 | 0,871 / 0,871 / 0,868 / 0,868 |

**Fatti:**
- **Senza `rolling_zscore`** (8 modelli) il quadro è lo stesso:
  - |Δ| della catena in `ap_norm` `last`/`last`: mediane 0,0041 / 0 / 0,0026;
  - |contrasto| delle proiezioni offline: mediane 0,136–0,179;
  - PA statica in F1 `last`/`last`: mediane 0,011–0,015, 90° percentile 0,27.
- **Zeri della PA statica:** per i modelli con scoring per sensore, in V1, il contrasto è esattamente 0 nel 31–77% delle
  celle per combinazione di proiezioni (calcolo di Claude, `f8_inline_checks.py`, controllo 5). Le quote più alte sono
  con le proiezioni smussate (`window_time_agg` `mean`, `ts_agg` `mean` o `max`). Per definizione la PA lascia l'F1
  invariato quando in ogni evento nessun punto oppure tutti i punti superano la soglia. Per questo su tutte le celle la
  mediana del contrasto è 0.

**Ipotesi** (solo su questi confronti):
- **AP normalizzata:** sulle macchine dev la proiezione offline sposta l'AP normalizzata molto più di ogni passaggio
  della catena. Le mediane di |·| sono circa 0,14–0,18 contro ≤ 0,006, e anche i 90° percentili sono più alti:
  0,29–0,54 contro ≤ 0,11.
- **F1 statico:** la PA con soglia statica sposta l'F1 più dei passaggi della catena soprattutto nella coda. Il 90°
  percentile è circa 0,25–0,32 contro ≤ 0,034. In mediana la differenza dipende dall'insieme di celle: in `last`/`last`
  0,015 contro ≤ 0,002; su tutte le celle 0 per entrambi.
- **Portata:** è un'osservazione descrittiva su 6 macchine, da verificare in RQ1a sulle macchine di valutazione.
- **PA e allarmi sparsi:** l'aumento con PA della baseline random e di `rolling_zscore` è coerente con l'effetto noto
  della PA sugli allarmi sparsi.
- **Metrica:** la concordanza tra AP normalizzata e ROC-AUC è alta in mediana ma non uniforme: in ogni variante il τ di
  Kendall minimo è vicino a zero o negativo. Quindi anche la scelta della metrica può cambiare il ranking su singole
  macchine.

## 5. Verifica della motivazione di D-40 sulla PCA puntuale

**Affermazione da verificare** (D-40, «Motivazione»): la PCA puntuale ristandardizza ogni sensore con la deviazione
standard su fit. Quindi:
- per i sensori con range positivo è invariante rispetto allo scaler minmax, a meno degli arrotondamenti float32;
- τ agisce su di lei solo sui sensori a range nullo (std < 1e−8 → 1), che passano da 1 a 1/τ.

**Atteso, derivato dall'affermazione e dai gruppi calcolati dai dati** (lettura di `train/` e `test/` dello script,
senza label):
- **E1:** sulle macchine i cui sensori a range nullo su fit restano costanti anche nel test (1-7, 2-6), Δ V2 → V1 = 0
  a meno degli arrotondamenti;
- **E2:** sulle macchine con sensori a range nullo su fit variabili nel test, Δ può essere diverso da zero, con
  direzione non prevista:

  | macchina | sensori |
  |---|---|
  | 1-1 | {26, 28} |
  | 2-8 | {16, 17, 28} |
  | 3-11 | {10, 31} |
  | 3-8 | {28} |

**Fatti per gruppo** (script, `ap_norm`, `last`/`last`):

| gruppo | macchine | mean/none | mean/std | max/std | top5/std |
|---|---|---|---|---|---|
| costanti nel test | 2 | +0,0000… [0; +0,0000…] | 0 [0; 0] | −0,0000… [−0,0000…; 0] | +0,0000… [0; +0,0000…] |
| variabili nel test | 4 | +0,0000… [0; +0,0093] | +0,0000… [0; +0,0425] | +0,0000… [−0,0000…; +0,0151] | +0,0000… [0; +0,0437] |

**Fatti per macchina** (calcolo di Claude su `deltas.csv`: Δ e righe con Δ ≠ 0 da `f8_tables.py`, massimi per metrica
da `f8_inline_checks.py`, controllo 6):

| macchina | gruppo | Δ `ap_norm` mean/none | mean/std | max/std | top5/std | righe con Δ ≠ 0 (su 384) | massimo \|Δ\| di `ap_norm` |
|---|---|---|---|---|---|---|---|
| 1-1 | variabili {26, 28} | ±0,0000… | ±0,0000… | ±0,0000… | ±0,0000… | 55 | 3,6e−6 |
| 1-7 | costanti | 0 | 0 | 0 | 0 | 0 | 0 |
| 2-6 | costanti | ±0,0000… | 0 | ±0,0000… | ±0,0000… | 12 | 4,4e−8 |
| 2-8 | variabili {16, 17, 28} | 0 | 0 | 0 | 0 | 25 | 1,0e−3 |
| 3-11 | variabili {10, 31} | +0,0093 | +0,0425 | +0,0151 | +0,0437 | 102 | 4,4e−2 |
| 3-8 | variabili {28} | 0 | 0 | ±0,0000… | 0 | 52 | 3,6e−3 |

- **Metriche con soglia:** cambiano solo su 3-11 (F1, FPR, precisione, best F1 dell'oracolo). Sulle altre 5 macchine
  soglie e allarmi della PCA puntuale sono invariati (`f8_inline_checks.py`, controllo 6).
- **Residui di test** per sensore (`test__last`, store). Calcolo di Claude:
  - differenze per sensore: `f8_pointwise_res.py`;
  - residuo massimo: `f8_inline_checks.py`, controllo 7;
  - componenti e std interne: `f8_inline_checks.py`, controllo 3.

  Fatti:
  - sui sensori a range nullo variabili nel test il residuo massimo passa da 1 a 2.500 = (1/τ)². Questi sensori si
    scostano dal valore di fit di più di τ solo in poche finestre del test: da 1 (3-8, sensore 28) a 70 (2-8,
    sensore 16);
  - sugli altri sensori la variazione relativa è ≤ 1,8e−7. Fino a |Δ| 33,9 in assoluto su 1-1, ma su residui che
    arrivano a 2,07e8 (sensore 32, uguale in V2 e V1; 1,87e8 su 3-11, sensore 22): l'amplificazione interna 1/std
    della PCA puntuale resta;
  - le componenti differiscono di ≤ 3e−8 e `n_components_` è uguale su ogni macchina;
  - su 2-6 il rapporto delle std interne V1/V2 differisce da 1 solo per i 3 sensori con range su fit in (0; τ), e la
    standardizzazione lo annulla a meno degli arrotondamenti.
- **Ordine degli score** (mean/none, regione primaria; `f8_inline_checks.py`, controllo 4): è identico solo su 1-7.
  Sulle altre 5 macchine non lo è, ma la correlazione di Spearman tra V2 e V1 è ≥ 0,99993 (1 alla decima cifra su 1-1,
  2-6 e 3-8).

**Confronto con l'atteso:**
- **E1:** confermata. 1-7 è identica bit a bit; su 2-6 restano differenze ≤ 4,4e−8 su `ap_norm` e ≤ 2,9e−6 su
  `ap_over_p`, all'ordine degli arrotondamenti float32.
- **E2:** compatibile. Le differenze compaiono solo dove ci sono sensori a range nullo variabili nel test, e solo
  attraverso quei sensori (i residui degli altri sono invariati a meno degli arrotondamenti). La grandezza è
  rilevante solo su 3-11, fino a +0,044; su 2-8 e 3-8 Δ è esattamente 0 in `last`/`last` e piccolo sulle altre celle.
- **Esito:** l'affermazione di D-40 è coerente con i dati di queste 6 macchine.

**Ipotesi:** su 2-8 e 3-8 l'AP resta identica perché i pochi punti riordinati non attraversano il confine tra positivi
e negativi. Per verificarlo servono le label: non fatto.

## 6. Dichiarazioni

- **P1 e dati confusi:** per gli AE addestrati e il modello di previsione il passaggio V0 → V3 confonde due cose:
  - P1, la correzione dichiarata (validation usata tre volte);
  - la riduzione dei dati di fit dall'80% al 60%, cambio di protocollo approvato (D-05, D-65).

  I due effetti non si separano. Per i modelli senza early stopping (PCA, PCA puntuale, baseline, AE non addestrati)
  V0 → V3 dà solo l'effetto dei dati: scaler e parametri stimati sul 60% invece che sull'80%, con cal coincidente
  (D-23).
- **Dipendenza dall'ordine:** l'esito della catena dipende dall'ordine dei passaggi (V0 → V3 → V2 → V1, fissato prima
  dei run). Con interazioni tra i cambi, un altro ordine darebbe Δ diversi per ogni passaggio. Le interazioni non si
  misurano in questo esperimento.
- **V0 non è lo storico** per gli AE non addestrati e per il modello di previsione: `main.py` non li conteneva, quindi
  per loro V0 è solo il protocollo `legacy2` applicato. Per PCA, PCA puntuale, Conv-AE e LSTM-AE su 1-1 e 3-8 con
  seed 42 V0 contiene i fit di M1 (parità esatta con lo storico, `docs/m1_parita.md`).
- **D-72 C1:** non osservabile qui (§4.2); si misura quando le soglie mobili entreranno nelle analisi (D-42).
- **D-38** (`train_step` = 5): invariato, nessun passaggio della catena lo cambia.
- **Solo macchine dev:** solo le 6 macchine di sviluppo (D-51), mai quelle di valutazione. machine-1-1 e machine-3-8
  erano già state osservate prima dell'assegnazione (D-52).
- **Nessuna inferenza:** nessun IC, nessun test, nessuna conclusione su RQ2 o sul confronto DL − PCA. I numeri sono
  descrittivi, sulla media di 3 seed per macchina.
- **Vincoli delle voci:** i risultati non sono usati per rivedere D-38, D-39 o D-40 (vincolo di D-39: «il valore non si
  cambia dopo aver visto i risultati di F8»).
- **Metriche e soglie:**
  - VUS-PR non è disponibile (D-34);
  - l'oracolo è un limite superiore diagnostico;
  - con un solo evento (2-8) e tre eventi (3-11) l'AP di una macchina è molto sensibile all'ordine di pochi punti.
