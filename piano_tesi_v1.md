> Rivisto da docs/decisioni.md, che prevale sui punti che registra (D-01).

# Piano tesi v1 — SMD anomaly detection (bozza di progetto, 2026-11-06)

Stato: bozza da discutere. Nessun numero dei vecchi notebook è assunto come risultato; i vecchi risultati servono solo come test di regressione della nuova pipeline.

## 1. Domanda di ricerca (versione operativa)

**Domanda guida.** In quali condizioni un autoencoder profondo (Conv-AE, LSTM-AE) migliora in modo praticamente rilevante una baseline lineare ben calibrata (PCA) nell'anomaly detection multivariata, e quanto di questo vantaggio sopravvive alle scelte metodologiche (scoring, aggregazione, preprocessing, soglia, protocollo)?

Sotto-domande, ciascuna con una regola di decisione fissata prima di rieseguire:

- **RQ1 – Misura.** Quanta variazione di performance e di ranking tra modelli è dovuta alle scelte di valutazione, a parità di modello addestrato?
- **RQ2 – Confronto.** A protocollo causale, senza soglia, con budget di tuning pari: Δ(DL − PCA) è superiore, equivalente o inferiore rispetto a un margine δ fissato in anticipo?
- **RQ3 – Condizioni.** Quali proprietà delle macchine (misurate senza usare i modelli) predicono il segno e la grandezza di Δ?
- **RQ4 – Operativa.** A soglia, quanto pesano la politica di soglia e la deriva rispetto alla scelta del modello?

Affermazione da non fare: "il deep learning non vince". Si può dire: "gli autoencoder a ricostruzione testati, su SMD (+ dataset di conferma), non superano la PCA di oltre δ".

## 2. Disegno statistico

- Unità di analisi: la macchina. Seed mediati prima del confronto.
- Stimando: Δ_m = metrica(DL, m) − metrica(PCA tarata, m). Sintesi: media di Δ con IC bootstrap **gerarchico** (cluster = gruppo server 1/2/3 di SMD, poi macchina).
- Regola: DL superiore se limite inf. IC > +δ; equivalente se IC ⊂ [−δ, +δ]; inferiore se limite sup. < −δ; altrimenti inconcludente. δ fissato prima (proposta: δ ≥ spread indotto dalle scelte di protocollo, così "vantaggio rilevante" = "più grande del rumore metodologico").
- Metrica principale: AP normalizzata per prevalenza, (AP − p)/(1 − p), accanto a ROC-AUC e VUS-PR. F1 operativa solo in RQ4.
- Decomposizione della varianza: modello misto  metrica ~ modello * fattori_protocollo + (1|macchina) + (1|gruppo). Quota di varianza spiegata da modello vs protocollo vs interazioni.
- Stabilità del ranking: Kendall τ tra configurazioni di protocollo.
- Correzione per confronti multipli (Holm) solo per test confermativi dichiarati; il resto è esplorativo e va etichettato tale.

## 3. Rischi di validità da chiudere subito

1. Le 28 macchine sono già state viste da tutte le analisi esplorative. SMD non può essere un held-out pulito: la conferma richiede un secondo dataset con analisi congelata prima di eseguirlo.
2. Macchine di tuning scelte guardando l'AP degli autoencoder: sceglierle con descrittori indipendenti dai modelli (o leave-one-group-out).
3. Validation usato sia per early stopping sia per calibrare la soglia: asimmetria tra AE e PCA (la PCA non usa il validation per fittare). Soluzione: split cronologico in 3 parti (fit / early-stop / calibrazione) uguale per tutti, oppure epoche fisse senza early stopping.
4. "Non significativo" ≠ "equivalente": serve test di equivalenza/IC con δ.
5. Copertura famiglie DL: solo ricostruzione. O si restringe l'affermazione, o si aggiunge un modello predittivo/di altra famiglia.
6. AP media tra macchine pesata da prevalenza e struttura degli eventi: normalizzare, riportare anche mediana e ranghi.
7. Dipendenza tra macchine dello stesso gruppo server.

## 4. Architettura della pipeline (fit una volta, tutto il resto a posteriori)

| Stadio | Contenuto | Costo |
|---|---|---|
| S0 | Manifest dati, split cronologici, descrittori macchina (deriva train→test, frazione fuori range, rango effettivo PCA, sensori quasi costanti, prevalenza, n. e durata eventi) | trascurabile |
| S1 | Fit + inferenza per (macchina, modello, seed, config di training). Salva residui **per sensore** per timestamp, già aggregati su finestra in 3 modi (last, mean, max) per val e test + metadati (tempo, epoca migliore) | alto (CPU) |
| S2 | Score da residui: raw mean, z per sensore (stat dal validation), top-k, ... | basso |
| S3 | Politiche di soglia: percentile val (95–99.5), rolling robust-z, oracle | basso |
| S4 | Metriche: AP, ROC-AUC, VUS-PR, F1 pointwise/PA, metriche per evento | basso |
| S5 | Statistica e figure da un'unica tabella lunga (parquet) | basso |

Richiedono rifit: modello, scaler/clip, finestra, frazione di training, seed. NON richiedono rifit: aggregazione, scoring, soglia, PA/oracle, metriche.

## 5. Esperimenti, in ordine di priorità

**Nucleo (senza questi non c'è tesi)**
- E0 Sanity e test: causalità (modificare il futuro non cambia gli score passati), nessun leak, baseline random / input_norm / AE non addestrato / rolling z-score.
- E1 Confronto principale: PCA puntuale, PCA a finestre, Conv-AE, LSTM-AE; 28 macchine; 3 seed per i DL.
- E2 Tuning a budget pari con macchine scelte senza guardare i modelli; PCA e AE con lo stesso numero di config.
- E3 Decomposizione del protocollo (aggregazione × scoring × soglia × PA/oracle): tutto post-hoc su E1.
- E4 Preprocessing/scala: minmax, standard, robust; clip esplicito (none, [0,1], ±k). Richiede rifit; per i DL su sottoinsieme o 1 seed.
- E5 Deriva e soglie: controllo mancante = escludere i punti vicini agli eventi etichettati e verificare che la crescita dell'FPR resti. Poi politiche di soglia.
- E6 Condizioni: Δ vs descrittori macchina, poche ipotesi (≤4) fissate prima.
- E7 Replica su secondo dataset (analisi congelata).

**Secondari**
- Frazione di training, con direzione invertita (dati più vecchi vs più recenti) per testare l'ipotesi di deriva.
- Costo: tempo di fit e di inferenza su CPU.
- Un modello di altra famiglia (predittivo) o USAD/TranAD, se il tempo lo permette.

## 6. Cosa recuperare / scartare / rifare

| Elemento | Decisione | Note |
|---|---|---|
| `protocol.py`, `protocols.py`, `scoring.py`, `dataset.py`, `loader.py`, `base.py`, `trainer.py`, modelli PCA/rolling/input_norm/untrained | Recuperare | Con split a 3 parti nel trainer e salvataggio dei residui per sensore |
| `rules.py` | Recuperare e parametrizzare | `n_past` da config |
| `aggregate_results.py` | Recuperare e estendere | IC gerarchico, equivalenza, mixed model |
| `scaler.py` | Rifare | Clip esplicito e configurabile |
| `oc_svm.py`, `isolation_forest.py`, VAE | Scartare dal nucleo | Eventuale appendice |
| Esecuzioni separate per mean/last | Scartare | Ora post-hoc |
| Risultati numerici dei vecchi run | Non importare | Usare come test di regressione (seed 42, stessa config) |
| `experiments/tuning.py` (GRID, TUNING_MACHINES) | Recuperare solo la griglia | Rifare la scelta delle macchine |
| Grafici per singola macchina/evento, dashboard | Scartare | Un solo caso illustrativo in appendice |
| Caso `machine-3-2` | Opzionale | Usa PCA puntuale, non il modello del confronto |
| Test A≥B ecc. | Recuperare | Servono come test di coerenza dei protocolli |

File dal vecchio progetto ancora da vedere: `configs/config.yaml` reale, `experiments/tuning.py`, `experiments/diagnose_threshold.py`, lo script che produce `protocolli_2x2.csv`, la cartella `tests/`.

## 7. Struttura di capitoli (provvisoria)

1. Introduzione e domanda; 2. Stato dell'arte (benchmark e critiche alla valutazione); 3. Dati e descrittori delle macchine; 4. Protocollo e pipeline; 5. Modelli e tuning; 6. RQ1 effetto del protocollo; 7. RQ2 confronto; 8. RQ3 condizioni; 9. RQ4 soglia e deriva; 10. Replica; 11. Limiti e conclusioni.
