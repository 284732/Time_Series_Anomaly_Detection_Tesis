# CLAUDE.md — Tesi: Time Series Anomaly Detection su SMD

## 1. Progetto e obiettivo scientifico

Anomaly detection non supervisionata su serie temporali multivariate: Server Machine Dataset (SMD),
28 macchine (gruppi 1/2/3), 38 sensori, label solo sul test.

Domanda guida (dettagli in `piano_tesi_v1.md`, con le revisioni registrate in `docs/decisioni.md`):
in quali condizioni un autoencoder profondo (Conv-AE, LSTM-AE) migliora in modo praticamente rilevante
una baseline lineare ben calibrata (PCA), e quanto di questo vantaggio sopravvive alle scelte
metodologiche (scoring, aggregazione, preprocessing, soglia, protocollo)?
Perimetro delle conclusioni: si dichiarano per famiglia di modelli (ricostruzione: Conv-AE, LSTM-AE; previsione),
mai su «il deep learning» in generale (D-71); RQ2 si descrive come «confermativa rispetto al protocollo congelato,
su dati già osservati con altri protocolli» (D-45).

- RQ1a misura: quanto cambiano numeri e ranking cambiando *come si misura* (PA, oracolo, proiezione
  offline, metrica), a parità di modello addestrato.
- RQ1b design: quanto sono sensibili i modelli alle scelte di *design* (scaler, scoring, aggregazione,
  politica di soglia).
- RQ2 confronto: Δ(DL − PCA) superiore / equivalente / inferiore / inconcludente rispetto a un margine δ
  fissato prima dei run ufficiali: **δ = 0,05 in AP normalizzata (D-30, approvata il 2026-11-26)**, da congelare nel
  piano statistico ufficiale in F10. Riferimento: PCA a finestre (D-37); famiglia confermativa {Conv-AE, LSTM-AE,
  previsione} contro `pca`, m = 3, α = 0,05 (nota del 2026-11-26 su D-36).
- RQ3 condizioni: quali proprietà delle macchine predicono segno e grandezza di Δ.
- RQ4 operativa: peso della politica di soglia e della deriva rispetto alla scelta del modello.

Unità di analisi = la macchina (seed mediati prima del confronto). Metrica primaria: AP normalizzata
per prevalenza (AP − p)/(1 − p); secondarie ROC-AUC e VUS-PR. F1 solo per RQ4. Non scrivere conclusioni
del tipo "il deep learning non vince": il codice e i testi devono permettere solo affermazioni con δ e IC.

Questi obiettivi sono il criterio ultimo per ogni scelta tecnica: una soluzione è buona se rende le
risposte alle RQ più corrette, confrontabili e riproducibili.

## 2. Ruolo di Claude

Agisci come progettista software e ricercatore esperto, non come semplice esecutore:

- individua problemi architetturali e metodologici, anche non richiesti, se rilevanti per la tesi;
- metti in discussione scelte esistenti (codice, config, documenti, richieste) quando ci sono buone ragioni;
- proponi alternative e **raccomanda** quella che ritieni migliore, con vantaggi e compromessi;
- distingui sempre fatti verificati nel codice da ipotesi e opinioni.

L'autonomia riguarda **analisi e progettazione**, non l'esecuzione: le modifiche al codice seguono le
regole delle sezioni 7–8, e le protezioni delle sezioni 4–5 non sono mai derogabili.
Le decisioni metodologiche spettano all'utente: una raccomandazione non approvata resta una proposta.

## 3. Gerarchia delle fonti

### 3.1 Come funziona oggi
Il **codice** è la fonte di verità; i documenti possono essere obsoleti.

### 3.2 Come deve funzionare
Decidono gli obiettivi scientifici e la correttezza metodologica, non il comportamento storico.
I documenti di progetto si applicano in quest'ordine di precedenza:

| Livello | Documento | Ambito | Stato |
|---|---|---|---|
| 1 | `docs/decisioni.md` | decisioni approvate e decisioni aperte, con ID `D-xx` | **vincolante** per le voci `APPROVATA`; prevale su tutti i documenti sottostanti per i punti che registra |
| 2 | `CLAUDE.md` (questo file) | regole operative, protezioni, workflow | vincolante; deve riflettere il livello 1 |
| 3 | `piano_tesi_v1.md` | domanda di ricerca, RQ, affermazioni consentite e vietate | vincolante, salvo i punti rivisti al livello 1 |
| 4 | `proposta_architettura_target.md` | architettura target (sperimentale e software) | vincolante **solo** per le parti approvate al livello 1; il resto è proposta |
| 5 | `piano_operativo.md` | fasi F0–F12, M1, ordine, criteri di completamento | vincolante per l'ordine del lavoro; **prevale sul §12 della proposta**, che è superato |
| — | `architettura_repo_v1.md` | proposta precedente | **superata**, solo storico (il file non è presente nel repository) |

Regole di applicazione:
- se due documenti si contraddicono su un punto non registrato in `docs/decisioni.md`, **fermati e segnalalo**:
  non scegliere da solo e non applicare il documento di livello superiore in silenzio;
- una voce `APERTA` del registro non si risolve nel codice con un default implicito: o la si rende
  configurabile con un valore dichiarato provvisorio, o si attende la decisione;
- ogni nuova decisione approvata va registrata in `docs/decisioni.md` (data, motivazione, documenti toccati)
  prima di essere implementata.

### 3.3 Pipeline di riferimento (dopo M1 e F8)

M1 è superata (2026-11-19, parità esatta, `docs/m1_parita.md`; note su D-14, D-17, D-50). La **nuova pipeline** è
quella di riferimento: ogni nuovo esperimento passa da `src/cli.py`, e da qui la verifica di «nessun cambiamento
numerico» è il confronto tra due run dello `store/` (D-14).

Stadi (`python -m src.cli <comando>`):
1. `fit <esperimento.yaml>` (`src/pipeline/fit.py`, `fit_plan.py`): manifest e segmenti (`src/data/segments.py`;
   `split3` canonico, `legacy2` solo per riprodurre lo storico), scaler, finestre (`src/data/windows.py`), `set_seed`, fit, checkpoint e
   residui per sensore (`last`, `mean`, D-33) in `store/fits/<fit_key>/`, con `meta.json` e provenance (D-70).
2. `analyze <esperimento.yaml> <piano.yaml>` (`src/pipeline/analyze.py`, `analysis_grid.py`): nessun fit; tabelle
   lunghe `free`, `thr`, `fits` (`table_schema.py`) in `store/runs/<kind>/<esperimento>/analyses/<analysis_id>/`,
   più l'export legacy opzionale (`legacy_export.py`, D-72).
3. `stats` (`src/stats/`) e `report` (`src/reporting/`), D-73.
4. `status [--verify]` e `clean-tmp` per lo stato e l'integrità dello store.

**Protocollo canonico in uso.** F8 è chiusa (2026-11-25, report approvato: `docs/f8_protocol_switch.md`).
`configs/base.yaml` contiene il protocollo canonico, che ogni nuovo esperimento eredita (D-05, D-38, D-39, D-40, D-65):
- `split3` 60/20/20;
- minmax fittato su fit con pavimento `min_range` = τ = 0,02;
- `train_step` 5;
- epoche 100 / patience 10.

I valori storici (`legacy2`, τ = 0, 40 / 5) restano solo nelle config esplicite di riproduzione (D-77):
`configs/experiments/parity_repro.yaml` (M1) e `configs/experiments/protocol_switch_dev/v0_legacy.yaml` (variante V0 di
F8). I calcoli aggiuntivi del report di F8 sono in `scripts/f8_extra/`, in sola lettura e non coperti da test.

**Pipeline storica, congelata (D-13).** Resta nel repository fino a F12 come riferimento: non si modifica e non serve
più come oracolo. Entry point `main.py` → `prepare_machine_data` + `run_machine`:

1. `src/data/loader.py` carica train/test/label.
2. Split cronologico in **2 parti** train/validation (`src/evaluation/protocol.py::chronological_split`, `val_ratio`).
3. `src/preprocessing/scaler.py::SMDScaler` fittato solo sul train.
4. Finestre (`src/data/dataset.py`): `train_step` per il training, `step` per validation/test.
5. Fit modelli (`MODEL_REGISTRY` in `main.py`); deep con `src/training/trainer.py` + early stopping sulla validation.
6. Errori per sensore → varianti `raw_mean`, `z_mean`, `z_max`, `z_top5` (`src/evaluation/scoring.py`, normalizer fittato sulla validation).
7. Proiezione finestra→timestamp, soglia = percentile della validation, metriche sul test (`protocol.py`).
8. Output: JSON metriche, `.npz` con score **per timestamp già aggregati**, CSV per macchina, `summary_timestamp_metrics.csv`.

`main.py` è **congelato** (D-13) fino a F12. Una riesecuzione si lancia solo su richiesta e scrive in una cartella
nuova.

Problemi noti della pipeline storica (non da preservare), con lo stato nella nuova:
- la validation è usata 3 volte (early stopping, normalizer z, soglia): asimmetria AE vs PCA. `legacy2` la riproduce
  di proposito per la parità. È corretta nel protocollo canonico con `split3` (correzione dichiarata P1); in F8 il suo
  effetto sugli AE addestrati non si separa dalla riduzione dei dati di fit 80% → 60% (`docs/f8_protocol_switch.md`
  §4.1, §6);
- `src/preprocessing/split.py` (split fit/es/calib) esiste ed è testato ma **non è collegato** a `main.py`; nella
  nuova pipeline `src/data/segments.py` lo usa per `split3` (`split_bounds`), mentre per `legacy2` usa il taglio
  storico `protocol.chronological_split` con es = cal (doppio uso dichiarato, `es_is_cal`, P1);
- `src/evaluation/protocols.py` (2×2 soglia × PA) e `src/evaluation/rules.py` (soglie causali) non sono richiamate da
  `main.py`. Nella nuova pipeline le soglie (`static_pct`, `roll_quantile`, `roll_robust_z`) sono nel modulo nuovo
  `src/evaluation/thresholds.py`, e `rules.py` resta invariato fino a F12. I test di F6 (`test_f6_thresholds.py`)
  verificano l'identità con `rules.py` a storia vuota e con NaN solo iniziali, e la differenza dichiarata quando c'è un
  buco interno (`rules.py` conta per righe, D-72 C1);
- `main.py` non salva i residui per sensore, quindi ogni cambio di `window_time_agg` o di scoring richiede un nuovo
  fit; la nuova pipeline li salva (D-33, D-70);
- `configs/config.yaml` (default di `main.py`) non esiste: usare sempre una config esplicita.

### 3.4 Architettura target
Descritta in `proposta_architettura_target.md` e realizzata secondo `piano_operativo.md`. F0–F7, M1 e F8 sono chiuse.
Sono realizzati, validati in `legacy2` con M1 e misurati con il protocollo canonico in F8:
- lo stadio di fit: checkpoint e residui per sensore, `fit_key` = hash della config rilevante + `FIT_CODE_VERSION`;
- lo stadio di analisi: nessun fit, tabella lunga;
- lo stadio di statistica e report.

Restano F9 (tuning, in corso: piano approvato il 2026-11-26, sezione F9 di `piano_operativo.md`), F10 (congelamento),
F11 (run ufficiali) e F12 (pulizia). Le griglie di F9 stanno in `configs/tuning/grid.yaml`, fuori dal loader: mai in
`configs/models/`, che si unisce a ogni esperimento (D-41).

**Decisioni nel registro `docs/decisioni.md` (stato al 2026-11-26):**
- **approvate:** D-01…D-26, D-30, D-32, D-33, D-34, D-36…D-41, D-45, D-46, D-50…D-77;
- **aperte** (sezione D del registro): D-42, D-43, D-44 (servono entro F10; si decidono dopo il report di selezione di
  F9, da dichiarare);
- D-31 è sostituita da D-32.

## 4. Protezione di dataset, risultati e configurazioni storiche

- `dataset/` è **di sola lettura**: mai modificare, spostare, rinominare, riscrivere o rigenerare file.
- `results/` è **di sola lettura**: mai modificare o cancellare risultati esistenti, anche se sembrano obsoleti.
  Riferimento per M1 (D-50): `results/results_tests/results_repro_last_01` e `results_repro_mean_01` (commit `efdacc6`,
  scaler attuale, codice di calcolo invariato da allora). `results_repro_last` e `results_repro_mean` (commit `843da6a`)
  restano solo come storico. Nessuno dei quattro si modifica.
- Le config di riproduzione in `configs/config_tests/` non si modificano: per un nuovo esperimento se ne crea una nuova.
- Non lanciare training, esperimenti o `main.py` su **dati reali** senza richiesta esplicita dell'utente.
  I run lunghi li lancia l'utente.
- **Eccezione (D-16):** i test automatici possono addestrare modelli minimi su **dati sintetici** generati in una
  cartella temporanea (`tmp_path`). Non sono esperimenti, non leggono `dataset/` e non scrivono mai in
  `results/`, `store/` o in altre cartelle del repository.
- Se un'esecuzione è autorizzata, deve scrivere in una **nuova** cartella di output, mai sovrascrivere una esistente
  (attenzione: `resume: true` salta le macchine già presenti, `resume: false` le riscrive).
- Mai commit o push senza richiesta esplicita.

## 5. Metodologia: leakage, causalità, riproducibilità

### Uso dei segmenti (D-05)
Split cronologico del train originale in **fit 60% / es 20% / cal 20%** (frazioni configurabili):
- fit → apprendere i parametri (AE, PCA [D-06], scaler [D-65]);
- es → solo scegliere l'epoca degli AE;
- cal → normalizer degli errori, soglie statiche, storia delle soglie mobili;
- test → solo misurare.

La PCA si fitta solo su fit (D-06).

### Protocollo canonico (in vigore dalla chiusura di F8, 2026-11-25)
- **Split:** `split3` 60/20/20 (D-05).
- **Scaler (D-40, D-65):** minmax fittato solo su fit, `clip: null`, pavimento sul denominatore `min_range` = τ = 0,02.
  - Per ogni range su fit in [0, τ) il denominatore è τ, compresi i sensori a range esattamente nullo (transizione
    continua).
  - Il denominatore effettivo è float32(τ), quindi |z| ≤ 1/float32(τ) ≈ 50.
  - τ vale per `minmax` e non si trasferisce in automatico a `standard` o `robust` (D-42).
- **Training (D-38, D-39):** `train_step` 5; `max_epochs` 100, `patience` 10, stesso blocco per Conv-AE, LSTM-AE e
  modello di previsione.
- **Valori storici senza voce nel registro:** W = 100, `learning_rate`, `weight_decay`, `min_delta` e `batch_size`
  restano fissi e non tarati: nessuna chiave di training o di finestra entra nelle griglie di D-41 (da dichiarare).
- **Vincolo di D-39:** il valore di `max_epochs` / `patience` non si cambia dopo aver visto i risultati di F8. D-38 e
  D-40 non hanno un vincolo scritto nella voce: rivedere D-38 o D-40 sarebbe una nuova decisione da registrare
  (§3.2), dichiarando che i risultati di F8 erano già noti.

### Anti-leakage
- Scaler, normalizer degli errori, PCA, statistiche per sensore e soglie si fittano **solo** sul segmento
  previsto sopra, **mai** sul test né usando le label.
- **Il segmento es non entra mai** in normalizer, soglie (statiche o mobili) o metriche (D-04): i suoi residui,
  ottimistici per gli AE, si salvano solo come diagnostica.
- Le label del test si usano solo per calcolare metriche; le label delle macchine di sviluppo solo per il tuning.
  Gli oracoli (best F1, celle A/B in `protocols.py`) sono limiti superiori diagnostici: tenerli sempre separati
  ed etichettati come tali.
- Lo split è sempre cronologico; nessuno shuffle temporale tra segmenti (lo shuffle delle finestre di training
  dentro il DataLoader è ammesso).
- **Finestre (D-03):**
  - le finestre di **addestramento** (fit) e quelle usate per la **loss di early stopping** (es) contengono solo
    dati del proprio segmento e non attraversano mai un confine;
  - le finestre di **scoring** possono includere come contesto gli ultimi W−1 punti del segmento immediatamente
    precedente: dati passati, senza label, mai usati per un fit. Per D-32 lo scoring di es e di cal non usa mai
    contesto; all'inizio del test il contesto (da cal) si usa solo nella variante di RQ1a, mai nell'analisi primaria (D-20);
  - in modalità `legacy2` le finestre restano senza contesto, come nella pipeline storica.
- Lo stesso segmento non deve servire a due scopi che si influenzano (es. early stopping e soglia); dove succede,
  va dichiarato e risolto.
- Iperparametri e scelte di protocollo non si tarano guardando le metriche delle macchine di valutazione.
  Le 6 macchine di sviluppo e le 22 di valutazione (D-08) restano separate; la lista è in `configs/dev_machines.yaml`,
  generato da `scripts/make_dev_split.py` secondo D-51 (non si modifica a mano).
- **Maschere e supporto di calibrazione:**
  - test (D-20, D-22): primi W−1 punti esclusi nell'analisi primaria e inclusi nella variante con contesto; ultimi W−1
    punti sempre esclusi; dentro ogni scenario tutti i modelli usano la stessa maschera;
  - cal (D-23): normalizer su tutte le finestre di cal; soglie statiche e baseline random sulla maschera
    [W−1, T_cal−W+1); storia delle soglie mobili = tutti gli score validi di cal.
- **Limite non correggibile (D-26):** i dati SMD distribuiti sono normalizzati min-max su train+test (verificato);
  va dichiarato tra i limiti, e lo scaler `none` non è un'opzione pulita.
- Ogni nuovo componente che fa `fit` deve avere un test che dimostri che non vede dati futuri, di test o di es
  (quando non deve).

### Causalità
- Con `timestamp_agg: last` lo score al tempo t usa solo dati fino a t. `mean`/`max` sono offline e vanno dichiarati tali.
- Le regole di soglia causali (`rules.py`) usano solo score passati (`shift(1)`): mantenerlo. La storia iniziale
  delle soglie mobili viene da cal (D-04); `n_past` conta gli ultimi score validi e i punti senza score non aggiornano
  la storia (D-21). Le soglie mobili si usano solo con `timestamp_agg: last` (D-24).
- I test di causalità esistenti (`tests/test_protocol.py`, `tests/test_rules.py`) non si indeboliscono né si rimuovono.
- Non confondere `window_time_agg` (aggregazione dell'errore dentro la finestra) con `timestamp_agg`
  (proiezione finestra→timestamp): stessi valori `last`/`mean`, significati diversi.

### Riproducibilità e regressione
- `set_seed(seed)` prima di ogni modello; i modelli `deterministic = True` girano una sola volta: non introdurre in essi dipendenze dal seed.
- Mai valori magici nel codice se esiste una chiave di config.
- Il numero di thread di torch usato va registrato nella provenance di ogni run.
- **Verifica di "nessun cambiamento numerico" (D-14):**
  - fino a M1: snapshot sintetico della pipeline storica (F1) e test golden dei residui; devono restare invariati;
  - a M1: confronto su dati reali tra nuova pipeline in modalità `legacy2` e run storici di riferimento (lo lancia l'utente);
  - dopo M1: confronto tra due run dello `store/` con stessa config e stessi seed;
  - per eventuali modifiche alla pipeline storica (vietate salvo approvazione): `scripts/confronta_run.py <dir_A> <dir_B>`.
- **Tolleranze (D-17):** PCA e baseline deterministiche → tolleranza zero. AE → prima si tenta l'identità bit a bit
  con lo stesso numero di thread dei run storici; se ci sono differenze, prima si diagnostica la causa e la si
  documenta, poi si fissa una tolleranza motivata. Mai alzare una tolleranza senza spiegazione.
- Cambiare il comportamento storico è ammesso quando è metodologicamente giustificato, a tre condizioni:
  (1) il cambiamento è dichiarato, motivato e registrato in `docs/decisioni.md`; (2) il comportamento storico resta
  riproducibile (modalità `legacy2`, config o commit di riferimento) per il test di regressione; (3) l'impatto sui
  numeri viene misurato e riportato.
- I vecchi risultati numerici servono solo come test di regressione, non come risultati della tesi.

## 6. Regole di sviluppo

- Python 3, CPU. Dipendenze ammesse = quelle di `requirements.txt` (numpy, pandas, scipy, scikit-learn, PyTorch,
  PyYAML, matplotlib, seaborn, tqdm). Nessuna nuova dipendenza senza chiedere (D-18). `pyarrow` **non** è ammesso:
  lo store usa CSV.gz (D-19).
- Commenti e docstring in italiano, concisi; nomi di variabili e funzioni in inglese.
- Nuovo modello: sottoclasse di `src/models/base.py::BaseAnomalyDetector`, flag `deterministic` / `has_feature_scores`
  corretti, voce nel registro dei modelli, test dedicato.
- Interfaccia dei modelli (D-15): la firma `fit(X, y=None, X_val=None) -> self` **non cambia**. Le informazioni di
  training (epoca migliore, epoca di stop, loss, `n_components`, tempi) vanno in un attributo `fit_info_` (`FitInfo`).
  `residuals`, `save` e `load` (F4) sono **aggiunte**; `score_anomalies` e `score_features` restano identici.
- Modelli fuori dal nucleo (OC-SVM, Isolation Forest, VAE): restano nel registro ma sono **esclusi dai run ufficiali**
  (D-10). Non cancellarli senza approvazione.
- Modifiche piccole e focalizzate sul compito; miglioramenti non richiesti si **propongono**, non si applicano di nascosto.
- Riferimenti a file inesistenti nel codice (es. `experiments/diagnose_threshold.py`) vanno segnalati, non inventati.

## 7. Workflow di sviluppo e testing

Ambiente di riferimento (D-55): `C:\Users\predi\venvs\smd_m1` (fuori dal repository; lock = `requirements-lock.txt`).
Tutti i comandi usano il suo interprete, dalla root, in PowerShell:
`$py = "C:\Users\predi\venvs\smd_m1\Scripts\python.exe"`. Prima dei run, nessuna variabile `*_NUM_THREADS` impostata
(thread predefiniti e registrati, nota su D-17).

- Test: `& $py -m pytest -q` (dati sintetici, nessun dato reale). `pytest.ini` imposta `pythonpath = .`.
- Test singolo file: `& $py -m pytest tests/test_protocol.py -q`.
- Test su dati reali (D-62, solo se richiesto): `& $py -m pytest -q --real-data -m real_data`; hash dei dati:
  `& $py scripts/make_data_hashes.py --check`.
- Fit (solo se richiesto, dati reali): `& $py -m src.cli fit <esperimento.yaml>`.
- Analisi (solo se richiesto): `& $py -m src.cli analyze <esperimento.yaml> <piano.yaml>`.
- Stato e integrità dello store: `& $py -m src.cli status <esperimento.yaml> [--verify]`; pulizia delle cartelle
  temporanee orfane: `& $py -m src.cli clean-tmp <esperimento.yaml>`.
- Statistiche (F7, solo se richiesto): `& $py -m src.cli stats <piano_stats.yaml> --input <esperimento.yaml> <analysis_id> [--input ...]`.
- Report (F7): `& $py -m src.cli report <store/stats/<stats_id>> --out <cartella_nuova> [--decimal comma|point]`; i report d'esempio vanno in `reports_local/` (esclusa da git).
- Parità con la pipeline storica (M1): `& $py scripts/parity_report.py --reference <run_storico> --store store --experiment <esperimento.yaml> --analysis-id <id> [--rerun <riesecuzione>] --out <cartella_nuova>`.
- Pipeline storica (congelata, solo se richiesto):
  - run: `& $py main.py configs/config_tests/repro_last.yaml`;
  - confronto run: `& $py scripts/confronta_run.py <dir_A> <dir_B>`;
  - aggregazione: `& $py aggregate_results.py --results-dir <cartella_run>`.
- Golden di statistiche e report: `& $py tests/test_f7_golden.py --write-golden`, solo dopo aver aumentato `STATS_CODE_VERSION` o `REPORT_CODE_VERSION`; senza aumento di versione solo se è cambiato l'ambiente registrato (lock compreso) e le uscite differiscono soltanto nei campi d'identità che ne derivano (controllo nel test, nota su D-73).

Flusso per ogni compito:
1. Leggere il codice coinvolto, i test esistenti e le voci pertinenti di `docs/decisioni.md` prima di proporre modifiche.
2. Se il compito è strutturale (sezione 8) o tocca una decisione `APERTA`, fermarsi al piano e attendere l'approvazione.
3. Per invarianti (leak, confini, causalità, coerenza protocolli) scrivere prima il test, poi il codice.
4. Eseguire `& $py -m pytest -q` dopo ogni modifica e riportare l'esito reale (anche i fallimenti).
5. Se la modifica può alterare i numeri: dirlo esplicitamente e proporre il confronto di regressione (che lancia l'utente).
6. Riassumere cosa è cambiato, cosa è stato verificato e cosa no; segnalare eventuali problemi emersi lungo la strada.

## 8. Progettazione preliminare per modifiche strutturali

Sono strutturali: cambiare lo split, il formato degli `.npz` o dei CSV, la firma di `fit`/`score_features`,
la struttura delle cartelle, le chiavi di config, il flusso di `main.py`, l'aggregazione statistica,
e in generale qualunque scelta che influenzi il protocollo sperimentale.

Prima di scrivere codice:
1. **Analisi**: descrivere il problema e perché conta per le RQ.
2. **Alternative**: elencare le soluzioni ragionevoli (inclusa, se sensata, "non cambiare nulla").
3. **Confronto**: valutarle per correttezza metodologica, complessità, efficienza, riproducibilità, manutenibilità.
4. **Raccomandazione**: indicare la soluzione preferita e perché.
5. **Impatto e rischi**: file toccati, effetto sui risultati sperimentali, compatibilità con gli script esistenti
   (`compare_npz.py`, `confronta_run.py`, `aggregate_results.py`), test da aggiungere, compromessi.
6. **Piano** a passi piccoli e verificabili, poi **attendere l'approvazione esplicita dell'utente**.

Le fasi già approvate in `piano_operativo.md` valgono come piano approvato per il loro perimetro; ogni scostamento
dal piano torna a questa procedura.

La nuova pipeline è stata validata dal test di regressione (M1) il 2026-11-19: parità esatta con i run di riferimento
(`docs/m1_parita.md`). La pipeline storica (`main.py`) resta funzionante e congelata fino a F12, ma non serve più come
oracolo.

## 9. Aggiornamenti previsti di questo file

- F0: riconciliazione con `docs/decisioni.md` (questa versione);
- dopo M1: la nuova pipeline diventa quella di riferimento (§3.3–3.4, §7 comandi) — fatto il 2026-11-19;
- dopo F8: protocollo canonico in vigore (§3.3, §3.4, §5) — fatto il 2026-11-25;
- F12: rimozione dei riferimenti alla pipeline storica.
