# CLAUDE.md — Tesi: Time Series Anomaly Detection su SMD

## 1. Progetto e obiettivo scientifico

Anomaly detection non supervisionata su serie temporali multivariate: Server Machine Dataset (SMD),
28 macchine (gruppi 1/2/3), 38 sensori, label solo sul test.

Domanda guida (dettagli in `piano_tesi_v1.md`): in quali condizioni un autoencoder profondo
(Conv-AE, LSTM-AE) migliora in modo praticamente rilevante una baseline lineare ben calibrata (PCA),
e quanto di questo vantaggio sopravvive alle scelte metodologiche (scoring, aggregazione,
preprocessing, soglia, protocollo)?

- RQ1 misura: quanta variazione di performance/ranking è dovuta al protocollo di valutazione.
- RQ2 confronto: Δ(DL − PCA) superiore / equivalente / inferiore rispetto a un margine δ fissato prima.
- RQ3 condizioni: quali proprietà delle macchine predicono segno e grandezza di Δ.
- RQ4 operativa: peso della politica di soglia e della deriva rispetto alla scelta del modello.

Unità di analisi = la macchina (seed mediati prima del confronto). Metrica principale: AP normalizzata
per prevalenza, accanto a ROC-AUC. F1 solo per RQ4. Non scrivere conclusioni del tipo
"il deep learning non vince": il codice e i testi devono permettere solo affermazioni con δ e IC.

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

## 3. Codice attuale, documenti e soluzione desiderabile

Gerarchia delle fonti:
- **Come funziona oggi** → il codice è la fonte di verità; i documenti possono essere obsoleti.
- **Come dovrebbe funzionare** → decidono gli obiettivi scientifici e la correttezza metodologica,
  non il comportamento storico né i documenti.

Il codice attuale non è corretto per il solo fatto di esistere. Quando trovi una discrepanza tra codice,
obiettivi della tesi, architettura desiderabile e documentazione, **segnalala esplicitamente** e proponi
una soluzione.

### Pipeline ATTUALE (implementata)

Entry point `main.py` → `prepare_machine_data` + `run_machine`:

1. `src/data/loader.py` carica train/test/label.
2. Split cronologico in **2 parti** train/validation (`src/evaluation/protocol.py::chronological_split`, `val_ratio`).
3. `src/preprocessing/scaler.py::SMDScaler` fittato solo sul train.
4. Finestre (`src/data/dataset.py`): `train_step` per il training, `step` per validation/test.
5. Fit modelli (`MODEL_REGISTRY` in `main.py`); deep con `src/training/trainer.py` + early stopping sulla validation.
6. Errori per sensore → varianti `raw_mean`, `z_mean`, `z_max`, `z_top5` (`src/evaluation/scoring.py`, normalizer fittato sulla validation).
7. Proiezione finestra→timestamp, soglia = percentile della validation, metriche sul test (`protocol.py`).
8. Output: JSON metriche, `.npz` con score **per timestamp già aggregati**, CSV per macchina, `summary_timestamp_metrics.csv`.

Problemi noti (da risolvere, non da preservare):
- la validation è usata 3 volte (early stopping, normalizer z, soglia): asimmetria AE vs PCA;
- `src/preprocessing/split.py` (split fit/es/calib) esiste ed è testato ma **non è collegato** a `main.py`;
- `src/evaluation/protocols.py` (2×2 soglia × PA) e `src/evaluation/rules.py` (soglie causali) sono librerie non richiamate da nessuno script;
- non si salvano residui per sensore: ogni cambio di `window_time_agg` o di scoring richiede un nuovo fit;
- `configs/config.yaml` (default di `main.py`) non esiste: usare sempre una config esplicita.

### Architettura di riferimento: proposta non vincolante

`architettura_repo_v1.md` è **solo una proposta progettuale precedente**, non l'architettura target
(split 60/20/20, Stadio 1 che salva residui per sensore, Stadi 2–5 senza fit, provenance/hash,
6 macchine di sviluppo). Layout, cartelle e chiavi di config lì descritti non corrispondono al codice.

Puoi mantenerne le parti valide, modificare quelle discutibili, scartare quelle inadeguate o proporre
un'architettura alternativa. Ogni decisione architetturale va motivata rispetto a: obiettivi scientifici,
correttezza metodologica, modularità, riproducibilità, efficienza.

## 4. Protezione di dataset, risultati e configurazioni storiche

- `dataset/` è **di sola lettura**: mai modificare, spostare, rinominare, riscrivere o rigenerare file.
- `results/` è **di sola lettura**: mai modificare o cancellare risultati esistenti, anche se sembrano obsoleti
  (es. `results_repro_last` / `results_repro_mean` sono storico di riproduzione).
- Le config di riproduzione in `configs/config_tests/` non si modificano: per un nuovo esperimento se ne crea una nuova.
- Non lanciare training, esperimenti o `main.py` senza richiesta esplicita dell'utente. I run lunghi li lancia l'utente.
- Se un'esecuzione è autorizzata, deve scrivere in una **nuova** cartella di output, mai sovrascrivere una esistente
  (attenzione: `resume: true` salta le macchine già presenti, `resume: false` le riscrive).
- Mai commit o push senza richiesta esplicita.

## 5. Metodologia: leakage, causalità, riproducibilità

### Anti-leakage
- Scaler, normalizer degli errori, PCA, statistiche per sensore e soglie si fittano **solo** su segmenti di
  training (fit / early stopping / calibrazione), **mai** sul test né usando le label.
- Le label del test si usano solo per calcolare metriche. Gli oracoli (best F1, celle A/B in `protocols.py`)
  sono limiti superiori diagnostici: tenerli sempre separati ed etichettati come tali.
- Lo split è sempre cronologico; nessuno shuffle temporale tra segmenti (lo shuffle delle finestre di training dentro il DataLoader è ammesso).
- Nessuna finestra deve attraversare il confine tra segmenti: le finestre si costruiscono per segmento.
- Lo stesso segmento non deve servire a due scopi che si influenzano (es. early stopping e soglia); dove succede, va dichiarato e risolto.
- Iperparametri e scelte di protocollo non si tarano guardando le metriche delle macchine di valutazione.
- Ogni nuovo componente che fa `fit` deve avere un test che dimostri che non vede dati futuri o di test.

### Causalità
- Con `timestamp_agg: last` lo score al tempo t usa solo dati fino a t. `mean`/`max` sono offline e vanno dichiarati tali.
- Le regole di soglia causali (`rules.py`) usano solo score passati (`shift(1)`): mantenerlo.
- I test di causalità esistenti (`tests/test_protocol.py`, `tests/test_rules.py`) non si indeboliscono né si rimuovono.
- Non confondere `window_time_agg` (aggregazione dell'errore dentro la finestra) con `timestamp_agg`
  (proiezione finestra→timestamp): stessi valori `last`/`mean`, significati diversi.

### Riproducibilità e regressione
- `set_seed(seed)` prima di ogni modello; i modelli `deterministic = True` girano una sola volta: non introdurre in essi dipendenze dal seed.
- Mai valori magici nel codice se esiste una chiave di config.
- Una modifica dichiarata "senza cambiamento numerico" deve dare score identici (o entro una tolleranza float esplicitata),
  verificati confrontando un run di riferimento con `scripts/confronta_run.py <dir_A> <dir_B>` (stessa config, stessi seed).
- Cambiare il comportamento storico è ammesso quando è metodologicamente giustificato, a tre condizioni:
  (1) il cambiamento è dichiarato e motivato; (2) il comportamento storico resta riproducibile
  (tramite config o commit di riferimento) per il test di regressione; (3) l'impatto sui numeri viene misurato e riportato.
- I vecchi risultati numerici servono solo come test di regressione, non come risultati della tesi.

## 6. Regole di sviluppo

- Python 3, numpy / pandas / scikit-learn / PyTorch su CPU. Nessuna nuova dipendenza senza chiedere.
- Commenti e docstring in italiano, concisi; nomi di variabili e funzioni in inglese.
- Nuovo modello: sottoclasse di `src/models/base.py::BaseAnomalyDetector`, flag `deterministic` / `has_feature_scores`
  corretti, voce in `MODEL_REGISTRY`, test dedicato.
- Modifiche piccole e focalizzate sul compito; miglioramenti non richiesti si **propongono**, non si applicano di nascosto.
- Non cancellare moduli "fuori dal nucleo" (OC-SVM, Isolation Forest, VAE) senza approvazione: puoi proporne la rimozione motivandola.
- Riferimenti a file inesistenti nel codice (es. `experiments/diagnose_threshold.py`) vanno segnalati, non inventati.

## 7. Workflow di sviluppo e testing

Comandi (Windows, dalla root):
- Test: `python -m pytest -q` (dati sintetici, veloce, nessun dato reale). `pytest.ini` imposta `pythonpath = .`.
- Test singolo file: `python -m pytest tests/test_protocol.py -q`.
- Run (solo se richiesto): `python main.py configs/config_tests/repro_last.yaml`.
- Confronto run: `python scripts/confronta_run.py <dir_A> <dir_B>`.
- Aggregazione: `python aggregate_results.py --results-dir <cartella_run>`.

Flusso per ogni compito:
1. Leggere il codice coinvolto e i test esistenti prima di proporre modifiche.
2. Se il compito è strutturale (sezione 8), fermarsi al piano e attendere l'approvazione.
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

Durante l'implementazione, la pipeline attuale resta funzionante finché la nuova non è validata
dal test di regressione.
