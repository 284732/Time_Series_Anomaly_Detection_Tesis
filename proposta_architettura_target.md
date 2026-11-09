# Proposta di architettura target

Prima di progettare ho raccolto altri numeri, leggendo solo conteggi e i due CSV di riepilogo, senza toccare i dati grezzi:

- **Prevalenza delle anomalie:** dallo 0,4% al 15,7% a seconda della macchina.
- **Numero di eventi anomali:** da 1 a 30 per macchina; `machine-2-8` ne ha uno solo.
- **Tempi di fit (train_step=5, massimo 40 epoche):** 13–70 s per autoencoder, inferenza 1–11 s.
- **Effetto della deriva:** con la soglia al 98° percentile della validation la quota di allarmi sul test va da 0,10 a 0,61, contro lo 0,02 atteso.
- **Il ranking dipende dal protocollo:** su `machine-3-8` il Conv-AE passa da AP 0,39 con aggregazione `last` a 0,17 con `mean`, mentre la PCA fa il percorso opposto (da 0,30 a 0,41).

Questi numeri cambiano alcune premesse di `architettura_repo_v1.md`.

---

## 1. Sintesi dell'architettura raccomandata

**Due stadi separati, con una chiave di contenuto.**

- **Stadio *fit*:** per ogni combinazione (macchina, modello, iperparametri, seed) produce un modello addestrato e salvato (checkpoint) più i residui per sensore. Ogni artefatto è identificato da un hash della sola configurazione che ne influenza il contenuto.
- **Stadio *analisi*:** senza alcun fit, applica a quei residui tutte le varianti di scoring, proiezione, soglia e metrica, e scrive una sola tabella lunga.
- **Stadio *statistica*:** legge solo la tabella lunga.

**Perché separarli.** Il motivo principale non è il costo (i fit costano minuti, non ore) ma la **correttezza del confronto**. Tutte le varianti di protocollo vengono valutate sullo *stesso* modello addestrato, quindi le differenze dipendono solo dal protocollo. Per RQ1 è un requisito, non un'ottimizzazione.

**Scelte scientifiche chiave:**

1. **Split cronologico in 3 parti** fit / early stopping (es) / calibrazione (cal). La calibrazione serve solo per normalizzazione e soglie.
2. **Contesto causale per le finestre di scoring:** una finestra che *assegna uno score* può includere gli ultimi W−1 punti del segmento precedente (sono dati passati e non etichettati); una finestra che *addestra* contiene solo dati del segmento fit. Così ogni timestamp del test ha uno score e non si perdono i primi 99 punti.
3. **Due tipi di fattori metodologici, trattati in modo diverso:**
   - *fattori di design*, che fanno parte del detector (scaler, scoring, aggregazione, politica di soglia): si tarano sulle macchine di sviluppo con budget pari;
   - *fattori di misura*, cioè come si valuta (PA, oracolo, proiezione offline, metrica): li fissa il protocollo, e studiarne la variazione è il cuore di RQ1.
4. **Statistica sulla macchina come unità:**
   - bootstrap delle macchine **stratificato per gruppo**: con 3 gruppi un bootstrap gerarchico sui gruppi non funziona;
   - regola di equivalenza con un δ **fissato prima** dei run ufficiali;
   - una famiglia confermativa minima (Conv vs PCA, LSTM vs PCA), con correzione di Holm.
5. **Migrazione in 9 fasi.** Le fasi 1–5 dimostrano l'equivalenza numerica con la pipeline attuale (risultati identici bit a bit) prima di cambiare il protocollo; il cambio di protocollo arriva nella fase 6, dichiarato e misurato.

## 2. Problemi principali dell'architettura attuale

| # | Problema | Dove | Gravità |
|---|---|---|---|
| P1 | La validation serve a tre scopi (early stopping, normalizer, soglia): asimmetria AE/PCA | [main.py:95](main.py#L95), [main.py:124](main.py#L124), [main.py:231](main.py#L231) | alta (metodologica) |
| P2 | Gli output sono score per timestamp già aggregati: ogni variante richiede un nuovo fit e i confronti tra varianti non sono garantiti sullo stesso modello | [main.py:129-151](main.py#L129-L151) | alta (per RQ1) |
| P3 | Lo scaler fissa la soglia dove il test è lontano: quota di allarmi 5–30 volte quella nominale | risultati `*_01` | alta (per RQ4) |
| P4 | Le finestre per segmento più il taglio dei bordi escludono i primi e gli ultimi 99 punti del test, inclusi eventuali eventi | [protocol.py:98](src/evaluation/protocol.py#L98) | media |
| P5 | Le chiavi di config sono lette con `.get(default)`: un errore di battitura viene ignorato in silenzio | [main.py:199-220](main.py#L199-L220) | media (riproducibilità) |
| P6 | Nessuna provenance: niente hash della config, commit git o versioni delle librerie negli output | — | media |
| P7 | `MODEL_REGISTRY` sta dentro `main.py`, e i test importano `main` | [test_kim_baselines.py:7](tests/test_kim_baselines.py#L7) | bassa |
| P8 | `create_sliding_windows` copia tutte le finestre (≈430 MB per il solo test) | [dataset.py:61-62](src/data/dataset.py#L61-L62) | bassa (memoria) |
| P9 | Due aggregazioni diverse si chiamano entrambe `last`/`mean` | [main.py:263](main.py#L263), [main.py:345](main.py#L345) | bassa (chiarezza) |
| P10 | Aggregazione statistica: Wilcoxon senza correzione, nessun IC, nessuna equivalenza | [aggregate_results.py](aggregate_results.py) | alta (per RQ2) |

**Critica di `architettura_repo_v1.md`.**

*Cosa tengo:* split in 3 parti, residui salvati per sensore, analisi a valle senza fit, 6 macchine di sviluppo estratte con un seed, buona parte della lista di test.

*Cosa cambio:*

- **`run_id` con il commit git dentro l'hash.** Ogni commit, anche solo di documentazione, invaliderebbe la cache e costringerebbe a rifare i fit. Va sostituito da un numero di versione del codice di fit, aggiornato a mano.
- **Bootstrap gerarchico con cluster = gruppo.** Con 3 cluster i ricampionamenti possibili sono 10: non si ottiene un IC credibile. Lo stesso vale per il termine `(1|gruppo)` nel modello misto.
- **δ "≥ spread indotto dal protocollo".** È un margine ricavato dagli stessi dati che si vogliono testare (circolare): va fissato prima, oppure ricavato solo dalle macchine di sviluppo.
- **Finestre mai a cavallo dei segmenti.** È corretto per il fit, troppo restrittivo per lo scoring.
- **Residui dell'early stopping usati nell'analisi.** Per gli AE sono ottimistici, perché quell'epoca è stata scelta proprio per minimizzarli: vanno usati solo come storia per le soglie mobili.
- **Il layout con rinomina completa delle cartelle** è inutilmente distruttivo.

## 3. Principi progettuali adottati

1. **Le RQ comandano.** Ogni componente esiste perché serve a una RQ; tutto il resto è opzionale.
2. **Confronti appaiati.** Varianti di protocollo e modelli diversi si confrontano sugli stessi timestamp, sulle stesse macchine e, per il protocollo, sullo stesso modello addestrato.
3. **Il dato giusto per lo scopo giusto:**
   - fit → apprendere i parametri;
   - es → scegliere l'epoca;
   - cal → normalizzare e fissare le soglie;
   - test → solo misurare.
   Le label si usano solo sul test (per le metriche) e sulle macchine di sviluppo (per il tuning).
4. **Causalità di default;** l'offline è sempre dichiarato.
5. **Identità degli artefatti data dal contenuto,** non dal tempo o dal nome: stessa configurazione rilevante = stesso artefatto, quindi nessun fit ripetuto.
6. **Evoluzione e non riscrittura.** Ogni fase di migrazione è verificabile contro i risultati esistenti.
7. **Strumenti semplici:** numpy, pandas, scipy, YAML e dataclass. Niente MLflow, Hydra o DVC: con circa 1.000 fit e un solo autore il costo di adozione supera il beneficio.

## 4. Architettura sperimentale dettagliata

### 4.1 Split temporali

Per ogni macchina il train originale viene diviso in ordine cronologico: **fit 60% / es 20% / cal 20%** (≈14,2k / 4,7k / 4,7k punti su 23,7k). Il test resta intatto.

| Modello | Usa | Note |
|---|---|---|
| AE (Conv, LSTM) | fit (pesi), es (epoca) | massimo 100 epoche, patience 10: oggi il Conv-AE arriva al limite di 40 epoche su `machine-1-1` |
| PCA, PCA puntuale | fit | analisi di sensibilità: PCA su fit+es (stessi dati "di modello" degli AE) |
| Baseline (input norm, rolling z, AE non addestrati) | nulla oppure fit | — |
| Tutti | cal | normalizer z per sensore e soglie |

Perché cal è l'ultimo segmento: con la deriva osservata, è il più simile al test. Perché es è separato da cal: altrimenti la soglia degli AE viene tarata su dati già usati per selezionarli (P1).

*Proporzioni:* 60/20/20 è ragionevole ma non ottimizzata. Non la tarerei: renderei le frazioni configurabili e farei un'unica analisi di sensibilità sulle macchine di sviluppo.

### 4.2 Finestre

- W=100, step=1 per lo scoring. `train_step` è una decisione aperta (§11).
- **Regola dei confini (riformulata):**
  - le finestre di *addestramento* contengono solo dati del segmento fit (e quelle di early stopping solo dati di es);
  - le finestre di *scoring* di es, cal e test possono usare come contesto gli ultimi W−1 punti del segmento immediatamente precedente.
  
  Non è leakage: sono dati passati, non etichettati, e vengono solo trasformati, mai usati per un fit.
- **Assunzione da verificare (fase 0):** in SMD il test segue il train senza interruzioni. Se non fosse così, il contesto sul primo test resta causale ma diventa "innaturale"; in quel caso si torna al taglio dei bordi solo all'inizio del test.
- **Finestre senza copia:** si usa una vista `as_strided` e si convertono in float32 solo a lotti (P8).

### 4.3 Preprocessing

- **Canonico:** minmax fittato solo su fit, nessun clip, nessun pavimento. È il comportamento storico, è confrontabile con la letteratura e fa da ponte per la regressione.
- **Fattore di sensibilità (E4):** {storico, `min_range`, clip [−k, 1+k], standard, robust}. Applicato a tutte le macchine di valutazione per la PCA (costa poco) e con 1 seed per gli AE.
- **Attenzione:** SMD è distribuito già normalizzato in [0,1]. Non so (e non posso verificarlo dal repo) se gli autori originali abbiano usato anche il test per farlo. Per questo `none` non è un'opzione "pulita" e va interpretato con cautela.

### 4.4 Residui e score

Il residuo base è r[t,f] = (x − x̂)² per sensore, all'interno di una finestra. Ne derivano tre fattori in sequenza:

1. **`window_time_agg` ∈ {last, mean}** (`max` opzionale): da (W, F) a (F,) per finestra. Sono due detector diversi, entrambi causali: errore dell'ultimo istante oppure errore medio della finestra che termina in t.
2. **`sensor_agg` ∈ {raw_mean, z_mean, z_max, z_top5}:** da (F,) a uno scalare. Il normalizer z si fitta su **cal**; aggiungerei una variante robusta (mediana/IQR), visto che gli errori hanno code pesanti.
3. **`timestamp_agg` ∈ {last (causale, canonico), mean, max (offline)}:** da finestra a timestamp. Con step=1 e `last`, lo score in t è quello della finestra che termina in t.

I fattori 1 e 2 sono di **design**: si scelgono per ogni famiglia di modelli sulle macchine di sviluppo, come parte del budget di tuning. Il fattore 3 è di **misura**: canonico `last`, mentre `mean`/`max` servono solo a RQ1 per quantificare quanto un protocollo offline cambia i numeri.

### 4.5 Training e selezione dei modelli (RQ2: budget pari)

- **Macchine di sviluppo:** le 6 già estratte (seed 20261109). Il tuning usa le loro label (lecito, perché sono escluse dalla valutazione).
- **Budget:** lo stesso numero di configurazioni per famiglia, ad esempio K=8:
  - *PCA:* `n_components`, eventualmente combinato con `window_time_agg` e `sensor_agg`;
  - *AE:* latent, hidden/canali, learning rate, combinati con le stesse opzioni di scoring.
  
  Lo scoring rientra nel budget, così nessuna famiglia ottiene "gratis" la sua variante migliore.
- **Criterio di selezione:** AP normalizzata media sulle 6 macchine, con protocollo canonico e un'unica configurazione globale per famiglia (non una per macchina).
- **Seed:** 5 per gli AE nella valutazione (dai tempi misurati il costo è sostenibile), 3 nel tuning. PCA e baseline deterministiche girano una volta.
- **Alternativa scartata come principale:** selezione per macchina in base alla loss su es. Una loss bassa non implica una buona detection, e confonderebbe RQ2.

### 4.6 Soglie (RQ4)

Tutte le soglie si calibrano su cal e, per le regole mobili, usano la storia continua es → cal → test (la sequenza degli score è causale):

- **statiche:** percentile di cal ∈ {95, 98, 99, 99,5};
- **mobili:** `roll_quantile` e `roll_robust_z` con `n_past` ∈ {360, 720, 1440, 2880}, inizializzate con la storia invece che escludendo un tratto iniziale del test. Così tutte le regole vengono valutate sull'intero test;
- **oracolo:** solo come limite superiore diagnostico, sempre etichettato.

### 4.7 Metriche

| Livello | Metrica | Ruolo |
|---|---|---|
| Senza soglia | **AP normalizzata** (AP−p)/(1−p) | primaria (RQ2) |
| Senza soglia | ROC-AUC, AP grezza, AP/p | secondarie |
| Senza soglia | VUS-PR | secondaria, solo se implementata e validata contro un riferimento esterno |
| Con soglia | F1, precision, recall, FPR, **falsi allarmi al giorno** | RQ4 |
| Per evento | event recall, **ritardo di rilevamento**, segmenti di falso allarme | RQ4 (significato operativo) |
| Diagnostiche | best-F1 (oracolo), celle A–D del 2×2 PA/oracolo | RQ1: misurare il gonfiamento (oggi il random ottiene PA_F1 = 0,92) |

Due avvertenze da riportare nella tesi:

- **L'AP per punti pesa gli eventi lunghi.** Su `machine-1-6` un solo evento copre 3.161 dei 3.708 positivi. Le metriche per evento servono a compensare.
- **Le metriche per evento sono molto rumorose** sulle macchine con 1–4 eventi (2-8, 3-11, 3-1, 3-9).

### 4.8 Aggregazione statistica

1. **Per (macchina, modello, configurazione):** media sui seed, conservando la deviazione standard tra seed (che misura la componente di varianza dovuta al seed).
2. **RQ2:** Δ_m = APnorm(DL) − APnorm(PCA di riferimento) per ciascuna delle 22 macchine di valutazione.
   - IC al 95%: bootstrap delle macchine **stratificato per gruppo** (B = 10.000, metodo BCa). Sensibilità: bootstrap a due livelli (macchine, poi seed).
   - Regola: superiore / equivalente / inferiore / inconcludente rispetto a δ, con IC corretti con Holm sulla famiglia {Conv, LSTM}.
   - Supporto: test di permutazione a segni casuali (sign-flip) sulla media di Δ; Wilcoxon solo come statistica descrittiva.
3. **RQ1 (fattori di misura, disegno fattoriale completo e bilanciato):**
   - decomposizione della varianza con somme dei quadrati classiche. Su un disegno bilanciato è esatta, non richiede statsmodels ed è più trasparente di un modello misto;
   - stabilità del ranking con il τ di Kendall tra configurazioni;
   - quota di configurazioni in cui il segno di Δ si inverte.
4. **RQ3:** al massimo 4 descrittori dichiarati prima (deriva train→test, quota fuori range, rango PCA effettivo, prevalenza/numero di eventi), correlazione di Spearman con IC bootstrap. Con 22 macchine la potenza è bassa, quindi questa parte è esplicitamente esplorativa.
5. **Esplorativo e confermativo:** solo la famiglia RQ2 canonica è confermativa; tutto il resto viene etichettato come esplorativo.

### 4.9 Problemi nelle Research Questions (da discutere)

- **RQ1 mescola fattori di design e fattori di misura.** Propongo di dividerla: RQ1a "quanto cambiano numeri e ranking cambiando *come si misura*" e RQ1b "quanto sono sensibili i modelli alle scelte di *design*".
- **RQ2: quale "PCA tarata"?** Va deciso prima se il riferimento è la PCA a finestre (stesso input degli AE) o la migliore delle due PCA scelta sulle macchine di sviluppo.
- **δ** va fissato prima dei run ufficiali (vedi §2).
- **Il "secondo dataset" non è opzionale** se si vuole un'affermazione confermativa, perché SMD è già stato osservato in tutte le analisi precedenti. Deve avere più entità, altrimenti l'unità "macchina" non esiste: PSM ha una sola serie, quindi non va bene; ASD (12 server) sarebbe un candidato, ma i dettagli vanno verificati. Va scelto presto.
- **Il train di SMD non è etichettato** e può contenere anomalie: soglie e normalizer su cal possono esserne contaminati. Conviene usare statistiche robuste e dichiararlo.
- **La deriva domina le metriche con soglia.** L'F1 a soglia misura soprattutto la deriva, non il detector: è un argomento per RQ4, non per RQ2.

## 5. Architettura software dettagliata

### 5.1 Componenti e responsabilità

| Componente | Responsabilità | Interfaccia principale |
|---|---|---|
| `utils/config.py` | Carica i YAML a livelli (base + modello + esperimento), unisce, **valida contro uno schema di dataclass** (chiave sconosciuta = errore), calcola l'hash canonico | `load_experiment(path) -> ExperimentConfig`; `cfg.fit_key(machine, model, seed) -> str` |
| `utils/provenance.py` | Commit git, stato *dirty*, versioni di python/torch/sklearn/numpy, hash del dataset | `collect() -> dict` |
| `data/manifest.py` | Elenco delle macchine e dei gruppi, hash sha256 dei file (calcolato una volta), lista dev/eval | `machines(split="eval")`, `data_version()` |
| `data/segments.py` | Usa `preprocessing/split.py`; modalità `legacy2` (train/val) e `split3` | `make_segments(raw, cfg) -> Segments(fit, es, cal, test, bounds, labels)` |
| `data/windows.py` | Finestre senza copia, con contesto opzionale, mappa finestra ↔ timestamp | `windows(seg, W, step, context=None)`, `iter_batches(...)` |
| `preprocessing/scaler.py` | Esiste già | `fit(seg.fit)`, `transform(...)` |
| `models/base.py` | Interfaccia unica | `fit(fit_X, es_X=None) -> FitInfo`; `residuals(windows, aggs) -> {agg: (N,F)}`; `save(dir)` / `load(dir)` |
| `models/registry.py` | Nome → classe (spostato da `main.py`) | `get(name)` |
| `training/trainer.py` | Esiste già; aggiunte `max_epochs`, `best_epoch` e le loss per epoca in `FitInfo` | — |
| `pipeline/store.py` | Percorsi a indirizzamento per contenuto, scrittura atomica (tmp + rename), lettura di meta, registro | `path(fit_key)`, `exists_complete(key)`, `write(...)` |
| `pipeline/fit.py` | Per ogni fit_key mancante: segmenti → scaler → finestre → fit → checkpoint → residui → meta. Isola i fallimenti | `run_fit(experiment)` |
| `pipeline/residual_cache.py` | Calcola o rigenera i residui da un checkpoint (cache cancellabile) | `get_residuals(fit_key, segment, agg)` |
| `evaluation/scoring.py` | Esiste già, più il normalizer robusto | `normalize`, `aggregate` |
| `evaluation/protocol.py` | Proiezione e maschere (esiste) | — |
| `evaluation/thresholds.py` | Da `rules.py` + statiche + inizializzazione con la storia | `apply(policy, score_stream, cal_range) -> pred` |
| `evaluation/metrics.py` | AP, APnorm, AUC, metriche per evento, ritardo, FA/giorno; PA e oracolo da `protocols.py` | `free_metrics(y, s)`, `thr_metrics(y, s, pred)` |
| `pipeline/analyze.py` | Griglia dei fattori di analisi per ogni fit → righe della tabella lunga | `run_analysis(experiment, plan)` |
| `stats/` | `aggregate.py` (seed → macchina), `bootstrap.py`, `decisions.py` (δ, Holm), `ranking.py`, `variance.py`, `descriptors.py` | funzioni pure da DataFrame a DataFrame |
| `reporting/` | Tabelle e figure per la tesi a partire dalle uscite di `stats` | — |
| `cli.py` | `python -m src.cli fit | analyze | stats | report | status | compare` | — |

### 5.2 Interazioni

```
experiment.yaml ─► config (validazione + fit_key) ─► pipeline.fit ─► store/fits/<key>/
                                                   (data, preprocessing, models, training)
analysis_plan.yaml ─► pipeline.analyze ─► legge store/fits/* (residui, rigenerati se mancano)
                                         ─► evaluation (scoring → proiezione → soglie → metriche)
                                         ─► store/runs/<esperimento>/tables/{free,thr}.parquet
stats ─► legge tables/* + piano ─► stats/*.parquet ─► reporting ─► figure e tabelle LaTeX
```

Ogni stadio legge **solo** gli artefatti dello stadio precedente, mai gli oggetti in memoria dell'altro. Così si può rilanciare uno stadio senza ripetere quelli prima.

### 5.3 Configurazioni

- `configs/base.yaml`: dati, split, finestre, preprocessing, default del training.
- `configs/models/<famiglia>.yaml`: iperparametri e griglie di tuning.
- `configs/experiments/<nome>.yaml`: `kind: exploratory|official`, macchine (`dev|eval|lista`), modelli, seed, override.
- `configs/analysis/<nome>.yaml`: fattori e livelli di analisi, metriche, famiglia confermativa, δ.
- `configs/config_tests/`: invariata (legacy).

La **fit_key** include solo ciò che cambia il modello addestrato: versione dei dati, split, preprocessing, W/train_step, modello e iperparametri, training, seed, `FIT_CODE_VERSION`. Restano fuori l'analisi, i percorsi e il commit (che viene registrato in meta).

### 5.4 Formati

- `store/fits/<key>/model.pt` (AE) oppure `pca.npz` (componenti, media); `residuals.npz` con `res_{es,cal,test}_{last,mean}` in float32, forma (N_seg, F), e gli indici dei timestamp; `meta.json` con la config risolta, provenance, `FitInfo` (best/stopped epoch, loss, `n_components`), tempi e stato.
- Tabella lunga: `machine, group, model, config_id, seed, fit_key, window_time_agg, sensor_agg, ts_agg, threshold_policy, threshold_param, pa, metric, value`. Due tabelle: `free` (senza soglia) e `thr` (con soglia).
- `store/registry.csv`, a sola aggiunta: esperimento, tipo, data, commit, hash della config, hash del piano, stato, note.

### 5.5 Errori e logging

- Modulo `logging` al posto di `print`: livello INFO su console, DEBUG su file per ogni run.
- Un fit che fallisce scrive `meta.json` con `status: failed` e il traceback; il runner continua e alla fine stampa un riepilogo.
- Scrittura atomica: un run interrotto non lascia artefatti parziali, e la ripresa salta solo le chiavi complete.
- I run `official` rifiutano di partire con l'albero git *dirty*, senza un piano di analisi congelato, o se usano macchine di sviluppo.

## 6. Struttura delle cartelle proposta

```
configs/
  base.yaml
  models/{pca,pca_pointwise,conv_ae,lstm_ae,baselines}.yaml
  experiments/            # esperimenti di fit (esplorativi e ufficiali)
  analysis/               # piani di analisi (quello ufficiale viene congelato e registrato con hash)
  config_tests/           # invariata (legacy)
src/
  cli.py
  data/        loader.py, manifest.py, segments.py, windows.py, dataset.py (legacy)
  preprocessing/ scaler.py, split.py
  models/      base.py, registry.py, traditional/*, deep/*
  training/    trainer.py
  pipeline/    store.py, fit.py, residual_cache.py, analyze.py
  evaluation/  protocol.py, protocols.py, rules.py→thresholds.py, scoring.py, metrics.py
  stats/       aggregate.py, bootstrap.py, decisions.py, ranking.py, variance.py, descriptors.py
  reporting/   tables.py, figures.py
  utils/       config.py, provenance.py, seed.py, logging.py
scripts/       compare_npz.py, confronta_run.py (legacy), make_dev_split.py, check_contiguity.py
tests/         unit/, invariants/, integration/, regression/, stats/
store/         # fuori da git: fits/, runs/{exploratory,official}/, registry.csv
dataset/, results/   # invariati, sola lettura
main.py, aggregate_results.py   # pipeline legacy, mantenuta fino a parità dimostrata
```

Tengo `src/evaluation/` invece di rinominarlo: non cambia la sostanza e evita di rompere import e test.

## 7. Flusso completo dei dati e degli esperimenti

1. **Fase di sviluppo (esplorativa):**
   - `fit configs/experiments/tuning.yaml`: 6 macchine di sviluppo × griglie × 3 seed;
   - `analyze` con il piano di tuning;
   - selezione delle configurazioni globali → `configs/models/*` con `selected: ...`.
2. **Congelamento:**
   - si scrive `configs/analysis/official.yaml` (δ, metrica primaria, famiglia confermativa, fattori di RQ1, descrittori di RQ3);
   - il suo hash va nel registro e viene salvato in un commit.
3. **Run ufficiale (lo lanci tu):**
   - `fit configs/experiments/official.yaml`: 22 macchine × modelli selezionati × 5 seed;
   - `analyze` → tabelle lunghe;
   - `stats` → IC, decisioni, decomposizione;
   - `report`.
4. **Replica:** stesso piano congelato, applicato al secondo dataset.

Per ogni (macchina, fit) il percorso dei dati è:

```
raw → segmenti → scaler(fit) → finestre (+ contesto) → modello.fit(fit, es) → checkpoint
    → residui per sensore (es/cal/test × last/mean) → [analisi] normalizer(cal) → sensor_agg
    → proiezione ts → soglia(cal / storia) → metriche sul test (label) → righe della tabella
```

## 8. Caching e gestione degli artefatti

Cosa salvare: le opzioni valutate.

| Opzione | Dimensione per run | Pro | Contro |
|---|---|---|---|
| a. Score per timestamp per variante (attuale) | ~0,1 MB × varianti | piccolo | ogni nuova variante richiede di rifare il fit; combinatorio |
| b. Residui per sensore (N, F) × {last, mean} | ~10 MB | tutta l'analisi senza fit | ~5 GB per l'esperimento ufficiale |
| c. Ricostruzioni complete (N, W, F) | ~1 GB | qualunque aggregazione temporale | improponibile |
| d. Solo checkpoint | 0,1–5 MB | minimo, è la fonte dei residui | ogni analisi richiede inferenza (≈1–11 s per run) |
| **b + d (raccomandata)** | checkpoint durevoli, residui come cache rigenerabile | durevolezza e velocità | due livelli da gestire |

- **Stime:** sono basate su 33k righe × 38 sensori × 4 B × 2 aggregazioni. Per l'esperimento ufficiale: 22 macchine × circa 24 combinazioni modello×seed ≈ 530 run → ~5 GB di residui. Per il tuning conservo solo checkpoint e metriche; i residui sono cancellabili.
- **Rigenerazione:** se i residui mancano, `residual_cache` li ricalcola dal checkpoint (inferenza su CPU con numero di thread fisso). Un test verifica che i residui rigenerati siano identici.
- **Invalidazione:** un cambio di `FIT_CODE_VERSION` (aggiornato a mano) o di config produce chiavi nuove; quelle vecchie restano finché non le cancelli tu. Un test "golden" su dati sintetici (hash dei residui) fallisce se cambia la numerica del fit, e così ricorda di aggiornare la versione.
- **Baseline casuale:** viene generata nell'analisi a partire dal seed, senza artefatti.

## 9. Piano di testing e validazione

| Livello | Contenuto | Quando |
|---|---|---|
| Unit | Test esistenti + config (chiave sconosciuta → errore, stabilità dell'hash), finestre con contesto, metriche nuove (APnorm, ritardo, FA/giorno), normalizer robusto | sempre (< 1 min) |
| Invarianti | (1) lo scaler e il normalizer non cambiano se si perturbano es/test; (2) le finestre di fit contengono solo indici del fit; (3) i segmenti coprono tutto, in ordine, senza sovrapposizioni; (4) **causalità end-to-end**: perturbare il test dopo t lascia invariati gli score ≤ t con modelli piccoli reali; (5) oracolo esatto ≥ oracolo a griglia ≥ F1 calibrato; PA ≥ no-PA; (6) residui `last`/`mean` = calcolo manuale su un modello giocattolo; (7) checkpoint salvato e ricaricato → residui identici | sempre |
| Integrazione | Mini-dataset sintetico (2 macchine, 3 sensori, 2k punti) in una cartella temporanea: fit → analyze → stats; ripresa e isolamento dei fallimenti; rifiuto di un run ufficiale con albero *dirty* | sempre (< 30 s) |
| Regressione R0 (ponte) | La nuova pipeline in modalità `legacy2` riproduce `results_repro_*_01` (stessi score, tolleranza 0 o esplicitata) | fasi 1–5; lo lancia l'utente |
| Regressione R1 | Hash golden dei residui sintetici, legato a `FIT_CODE_VERSION` | sempre |
| Statistica | Copertura del bootstrap stratificato ≈ 95% su dati simulati; regola δ su casi noti; Holm; la decomposizione SS somma al totale | sempre |

Marcatori pytest: `slow` e `regression`, esclusi dalla suite veloce. I test di regressione su dati reali li lanci tu.

## 10. Alternative progettuali considerate

| Decisione | Alternative | Raccomandazione e motivo |
|---|---|---|
| Split | 2 parti (attuale) · **3 parti con ES** · 2 parti a epoche fisse · rolling-origin CV | 3 parti: elimina P1 mantenendo un ES per macchina. Epoche fisse come sensibilità; la CV è troppo costosa e poco naturale per serie con deriva |
| Confini delle finestre | rigidi per segmento · **contesto causale per lo scoring** | contesto: nessun punto del test perso, comparabilità tra W, nessun leakage |
| Artefatti | a · b · c · d · **b+d** | vedi §8 |
| Identità | `run_id` con commit · **hash del contenuto + `FIT_CODE_VERSION`** | il commit nella chiave distrugge la cache; va registrato in meta |
| Tuning | per macchina senza label · **globale su 6 macchine dev con label** · leave-one-group-out · nessuno | dev globale: semplice e pulito. Il leave-one-group-out permetterebbe di valutare su 28 macchine, ma triplica il tuning e produce configurazioni diverse per gruppo |
| IC | bootstrap gerarchico (3 cluster) · **bootstrap stratificato per gruppo** · modello misto | stratificato: valido con 3 gruppi; gruppo come effetto fisso |
| Decomposizione RQ1 | modello misto (statsmodels) · **SS su disegno bilanciato** | esatta, trasparente, nessuna dipendenza nuova |
| Tracciamento | MLflow / DVC · **registro CSV + hash** | proporzionato alla scala del progetto |
| Config | Hydra · **YAML + dataclass** | nessuna magia, errori espliciti |
| Tabella lunga | CSV.gz · **parquet** (richiede pyarrow) | da decidere (§11) |
| Layout | rinomina completa (v1) · **incrementale** | nessuna migrazione distruttiva |

## 11. Rischi e decisioni ancora aperte

**Decisioni da prendere insieme:**

1. **δ e metrica primaria.** Propongo δ = 0,05 in AP normalizzata, oppure un valore ricavato solo dalle macchine di sviluppo.
2. **PCA di riferimento per RQ2:** a finestre (la mia raccomandazione) oppure la migliore tra le due scelta sulle macchine dev.
3. **Riformulazione di RQ1** in RQ1a e RQ1b.
4. **`train_step`:** 5 (attuale, costo ≈ 1 min per fit) oppure 1 per tutti (≈ 5 volte più dati e più tempo). In ogni caso deve essere lo stesso per PCA e AE.
5. **Numero di seed:** 5 nella valutazione è la mia raccomandazione, 3 il minimo.
6. **Macchine di valutazione:** 6 dev + 22 eval (raccomandato) oppure leave-one-group-out.
7. **Preprocessing canonico:** storico (raccomandato, studiato poi come fattore) oppure con pavimento/clip.
8. **Secondo dataset:** quale e quando. Condiziona la forza delle conclusioni.
9. **Dipendenza `pyarrow`** e budget di disco (~5–10 GB in `store/`).
10. **Dataset e `__pycache__` sotto git:** toglierli dal tracciamento è una tua scelta, io non tocco nulla.
11. **OC-SVM, Isolation Forest e VAE:** restano nel registro, esclusi dagli esperimenti ufficiali.
12. **Scadenze e tempo disponibile:** non le conosco, e determinano quante fasi sono realistiche.

**Rischi:**

- L'assunzione di contiguità tra train e test non è verificata.
- Il train può contenere anomalie non etichettate.
- La deriva rende poco informative le metriche a soglia.
- Le macchine con pochi eventi pesano poco in informazione ma quanto le altre nella media.
- La riproducibilità bit a bit dipende dalle versioni di torch: vanno bloccate le versioni delle dipendenze.
- I costi computazionali sono stimati su 2 sole macchine.
- La potenza statistica con 22 unità è limitata.

## 12. Piano di migrazione incrementale

Ogni fase si chiude con `pytest` verde. Le fasi 1–5 non devono cambiare alcun numero.

**Fase 0 — Igiene e verifiche preliminari**
- *Obiettivo:* base pulita e decisioni prese.
- *File:* `.gitignore`, `requirements.txt` (versioni bloccate o file di lock), `README.md`, `scripts/check_contiguity.py` (sola lettura del dataset).
- *Modifiche:* nessuna al codice di calcolo.
- *Test:* suite esistente.
- *Impatto sui risultati:* nessuno.
- *Completata quando:* decisioni del §11 prese; contiguità verificata; run di riferimento `results_repro_*_01` confermati come baseline R0.

**Fase 1 — Config, provenance, registro dei modelli, logging**
- *File:* `utils/config.py`, `utils/provenance.py`, `utils/logging.py`, `models/registry.py`, `main.py` (usa i nuovi moduli e ri-esporta `MODEL_REGISTRY`).
- *Test:* schema, hash, chiave sconosciuta; i test esistenti invariati.
- *Impatto sui risultati:* nessuno (R0 identico).
- *Completata quando:* rilanciando le config di riproduzione in una nuova cartella, `confronta_run.py` dà file identici bit a bit.

**Fase 2 — Segmenti e finestre**
- *File:* `data/segments.py`, `data/windows.py`, `main.py` (`prepare_machine_data` usa `Segments` in modalità `legacy2`, senza contesto).
- *Test:* invarianti 2, 3 e contesto causale; uguaglianza degli array di finestre con la versione storica.
- *Impatto sui risultati:* nessuno.
- *Completata quando:* R0 è identico e il consumo di memoria è più basso.

**Fase 3 — Interfaccia dei residui e stadio di fit**
- *File:* `models/base.py` e tutti i modelli (`residuals`, `save`/`load`), `pipeline/store.py`, `pipeline/fit.py`, `pipeline/residual_cache.py`, `cli.py fit`.
- *Test:* invarianti 6 e 7, scrittura atomica, ripresa, isolamento dei fallimenti, R1.
- *Impatto sui risultati:* nessuno (residui coerenti con la vecchia pipeline).
- *Completata quando:* un fit in modalità `legacy2` produce checkpoint e residui da cui si ricostruiscono esattamente gli score degli `.npz` storici.

**Fase 4 — Stadio di analisi**
- *File:* `pipeline/analyze.py`, `evaluation/metrics.py`, `evaluation/thresholds.py` (da `rules.py`), `cli.py analyze`.
- *Modifiche:* prima la parità con il `summary` di `main.py`; poi APnorm, metriche per evento con ritardo, FA/giorno, celle 2×2, soglie con storia, tabella lunga.
- *Test:* ponte analisi ≡ `main.py` su dati sintetici; invarianti 4 e 5; R0.
- *Impatto sui risultati:* metriche storiche invariate; metriche nuove aggiunte.
- *Completata quando:* la tabella lunga contiene le stesse metriche del summary storico.

**Fase 5 — Statistica e report**
- *File:* `stats/*`, `reporting/*`, `cli.py stats|report`. `aggregate_results.py` resta finché non si dimostra la parità.
- *Test:* copertura, regola δ, Holm, somma delle SS.
- *Impatto sui risultati:* nessuno sui valori per macchina.
- *Completata quando:* le tabelle di `aggregate_results.py` sono riprodotte dalla nuova catena.

**Fase 6 — Nuovo protocollo canonico (cambiamento numerico dichiarato)**
- *File:* config di base (`split3`, contesto, cal per normalizer e soglie, PCA solo su fit, 100 epoche con patience 10), `scripts/make_dev_split.py`.
- *Test:* invarianti 1–4 end-to-end; un test che verifica che la modalità `legacy2` resti disponibile.
- *Impatto sui risultati:* **cambiano.** Il confronto legacy vs nuovo si fa solo sulle 6 macchine di sviluppo e viene documentato.
- *Completata quando:* il report delle differenze è approvato.

**Fase 7 — Tuning a budget pari (macchine di sviluppo)**
- *File:* `configs/models/*` (griglie), `configs/experiments/tuning.yaml`, `configs/analysis/tuning.yaml`.
- *Test:* il budget è pari per costruzione (stesso numero di configurazioni, verificato da un test sulla config).
- *Impatto sui risultati:* sceglie gli iperparametri finali.
- *Completata quando:* le configurazioni selezionate sono salvate e motivate.

**Fase 8 — Piano congelato e run ufficiali (li lanci tu)**
- *File:* `configs/analysis/official.yaml`, `configs/experiments/official.yaml`.
- *Test:* rifiuto con albero *dirty* o con un piano non registrato.
- *Impatto sui risultati:* produce i risultati della tesi.
- *Completata quando:* IC, decisioni RQ2 e analisi RQ1, RQ3 e RQ4 sono generati in modo riproducibile da piano e commit.

**Fase 9 — Pulizia (solo con la tua approvazione)**
- *File:* `main.py`, `aggregate_results.py`, `src/data/dataset.py` legacy, `architettura_repo_v1.md`.
- *Modifiche:* deprecazione o rimozione dopo la parità; aggiornamento dei documenti.
- *Test:* la suite completa.
- *Impatto sui risultati:* nessuno.
- *Completata quando:* esiste una sola pipeline e la documentazione è allineata.

---

Non ho modificato file né eseguito esperimenti; ho letto solo conteggi del dataset e due CSV piccoli. Prima di qualsiasi implementazione mi servono soprattutto le decisioni 1–4 e 8 del §11, che cambiano il disegno. Se vuoi, posso pubblicare questa proposta come pagina da condividere con il relatore.