# CLAUDE.md — Tesi: Time Series Anomaly Detection su SMD

## 1. Progetto e obiettivo scientifico

Anomaly detection non supervisionata su serie temporali multivariate: Server Machine Dataset (SMD),
28 macchine (gruppi 1/2/3), 38 sensori, label solo sul test.

Domanda guida (dettagli in `piano_tesi_v1.md`, con le revisioni registrate in `docs/decisioni.md`):
in quali condizioni un autoencoder profondo (Conv-AE, LSTM-AE) migliora in modo praticamente rilevante
una baseline lineare ben calibrata (PCA), e quanto di questo vantaggio sopravvive alle scelte
metodologiche (scoring, aggregazione, preprocessing, soglia, protocollo)?

- RQ1a misura: quanto cambiano numeri e ranking cambiando *come si misura* (PA, oracolo, proiezione
  offline, metrica), a parità di modello addestrato.
- RQ1b design: quanto sono sensibili i modelli alle scelte di *design* (scaler, scoring, aggregazione,
  politica di soglia).
- RQ2 confronto: Δ(DL − PCA) superiore / equivalente / inferiore / inconcludente rispetto a un margine δ
  fissato prima dei run ufficiali. **Valore e criterio di δ: decisione aperta (D-30).**
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

### 3.3 Pipeline ATTUALE (implementata)

Entry point `main.py` → `prepare_machine_data` + `run_machine`:

1. `src/data/loader.py` carica train/test/label.
2. Split cronologico in **2 parti** train/validation (`src/evaluation/protocol.py::chronological_split`, `val_ratio`).
3. `src/preprocessing/scaler.py::SMDScaler` fittato solo sul train.
4. Finestre (`src/data/dataset.py`): `train_step` per il training, `step` per validation/test.
5. Fit modelli (`MODEL_REGISTRY` in `main.py`); deep con `src/training/trainer.py` + early stopping sulla validation.
6. Errori per sensore → varianti `raw_mean`, `z_mean`, `z_max`, `z_top5` (`src/evaluation/scoring.py`, normalizer fittato sulla validation).
7. Proiezione finestra→timestamp, soglia = percentile della validation, metriche sul test (`protocol.py`).
8. Output: JSON metriche, `.npz` con score **per timestamp già aggregati**, CSV per macchina, `summary_timestamp_metrics.csv`.

`main.py` è **congelato** (D-13): resta il riferimento storico finché la nuova pipeline non supera M1.

Problemi noti (da risolvere nella nuova pipeline, non da preservare):
- la validation è usata 3 volte (early stopping, normalizer z, soglia): asimmetria AE vs PCA;
- `src/preprocessing/split.py` (split fit/es/calib) esiste ed è testato ma **non è collegato** a `main.py`;
- `src/evaluation/protocols.py` (2×2 soglia × PA) e `src/evaluation/rules.py` (soglie causali) sono librerie non richiamate da nessuno script;
- non si salvano residui per sensore: ogni cambio di `window_time_agg` o di scoring richiede un nuovo fit;
- `configs/config.yaml` (default di `main.py`) non esiste: usare sempre una config esplicita.

### 3.4 Architettura target
Descritta in `proposta_architettura_target.md` e realizzata secondo `piano_operativo.md`: stadio di fit
(checkpoint + residui per sensore, `fit_key` = hash della config rilevante + `FIT_CODE_VERSION`), stadio di
analisi (nessun fit, tabella lunga), stadio di statistica e report. Le scelte già approvate sono in
`docs/decisioni.md` (D-03…D-24, D-26, D-32, D-33, D-34, D-50, D-51, D-52, D-55, D-56, D-58…D-65); quelle ancora aperte sono nelle
sezioni D ed E del registro.

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

Comandi (Windows, dalla root):
- Test: `python -m pytest -q` (dati sintetici, veloce, nessun dato reale). `pytest.ini` imposta `pythonpath = .`.
- Test singolo file: `python -m pytest tests/test_protocol.py -q`.
- Run (solo se richiesto): `python main.py configs/config_tests/repro_last.yaml`.
- Confronto run: `python scripts/confronta_run.py <dir_A> <dir_B>`.
- Aggregazione: `python aggregate_results.py --results-dir <cartella_run>`.

Flusso per ogni compito:
1. Leggere il codice coinvolto, i test esistenti e le voci pertinenti di `docs/decisioni.md` prima di proporre modifiche.
2. Se il compito è strutturale (sezione 8) o tocca una decisione `APERTA`, fermarsi al piano e attendere l'approvazione.
3. Per invarianti (leak, confini, causalità, coerenza protocolli) scrivere prima il test, poi il codice.
4. Eseguire `python -m pytest -q` dopo ogni modifica e riportare l'esito reale (anche i fallimenti).
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

Durante l'implementazione, la pipeline attuale resta funzionante finché la nuova non è validata dal test di
regressione (M1).

## 9. Aggiornamenti previsti di questo file

- F0: riconciliazione con `docs/decisioni.md` (questa versione);
- dopo M1: la nuova pipeline diventa quella di riferimento (§3.3–3.4, §7 comandi);
- dopo F8: protocollo canonico in vigore (§5);
- F12: rimozione dei riferimenti alla pipeline storica.
