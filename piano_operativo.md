# Piano operativo di sviluppo

Questo piano trasforma in passi concreti l'architettura approvata. Non ho modificato file né eseguito codice.

## Cosa cambio rispetto all'ordine della proposta precedente, e perché

1. **`main.py` resta congelato.** La nuova pipeline cresce accanto a quella storica invece di modificarla dall'interno. Prima prevedevo di modificare `prepare_machine_data` già nelle fasi 1–2. Così il riferimento storico non si può rompere per errore, e la parità si dimostra confrontando le due pipeline. Le uniche modifiche ai moduli condivisi (modelli, trainer, scoring) sono **aggiunte**: nessun metodo esistente cambia comportamento.
2. **Un banco di prova sintetico come prima fase software.** Un piccolo dataset in formato SMD, generato nei test, più una "fotografia" (snapshot) degli output di `main.py` su quel dataset. La parità si verifica in automatico a ogni modifica, senza training su dati reali. Nella proposta precedente il controllo di regressione lo dovevi lanciare tu a ogni fase.
3. **La parità su dati reali si concentra in un solo punto di controllo (M1),** invece di ripeterla a ogni fase. Lo lanci tu una volta, quando la nuova pipeline è completa in modalità legacy.
4. **Lo schema della tabella lunga si fissa presto (F2).** Così il modulo di statistica (F7) dipende solo da quello schema e si può sviluppare anche in parallelo, su dati simulati.
5. **La verifica che test e train siano contigui nel tempo e il blocco delle versioni delle dipendenze vanno in F0,** perché condizionano due scelte di progetto: il contesto delle finestre e la riproducibilità bit a bit.

## Quadro generale

| Fase | Tipo | Training su dati reali? | Cambia i numeri? |
|---|---|---|---|
| F0 Decisioni, ambiente, contiguità | software + validazione metodologica | no | no |
| F1 Banco di prova sintetico e snapshot legacy | software | no | no |
| F2 Config, provenance, archivio artefatti, schema tabella | software | no | no |
| F3 Segmenti e finestre | software | no | no |
| F4 Interfaccia residui e checkpoint dei modelli | software | no | no |
| F5 Stadio di fit | software | no | no |
| F6 Stadio di analisi (prima parità, poi estensioni) | software | no | no (le metriche nuove si aggiungono) |
| F7 Statistica e report | software | no | no |
| **M1 Parità su dati reali** | validazione metodologica | **sì (lo lanci tu)**, 2 macchine | deve risultare nessun cambiamento |
| F8 Nuovo protocollo canonico | software + validazione metodologica | sì (lo lanci tu), 6 macchine dev | **sì, dichiarato** |
| F9 Tuning e sensibilità | esperimenti esplorativi | sì (lo lanci tu), 6 macchine dev | sceglie le configurazioni |
| F10 Congelamento del piano di analisi | validazione metodologica | no | no |
| F11 Run ufficiali | esperimenti ufficiali | sì (lo lanci tu), 22 macchine | risultati della tesi |
| F12 Pulizia | software | no | no |

**F0–F7 si implementano e si testano interamente senza training su dati reali.** Anche la parte software di F8 e le protezioni di F10 sono testabili su dati sintetici.

---

## Decisioni scientifiche da congelare

Non tutte servono subito. Le ho raggruppate in base al momento in cui bloccano il lavoro.

**Gruppo A — prima di iniziare l'implementazione (servono in F0):**
1. Politica del contesto per le finestre di scoring. Dipende dalla verifica di contiguità; senza contiguità si usa il contesto solo tra segmenti del train.
2. Quali aggregazioni dei residui salvare: `last` e `mean` (la mia raccomandazione), con `max` opzionale.
3. Metriche da implementare: AP normalizzata, ROC-AUC, metriche per evento con ritardo di rilevamento, falsi allarmi al giorno, celle 2×2. VUS-PR sì o no: la mia raccomandazione è di rimandarla.
4. Normalizer robusto (mediana/IQR) come variante aggiuntiva: sì o no.
5. Dipendenza `pyarrow` per il parquet oppure CSV.gz; dove si trova `store/` e quanto disco gli si può dedicare.

**Gruppo B — prima degli esperimenti esplorativi (servono in F8–F9):**
6. `train_step` (5 oppure 1), identico per tutti i modelli.
7. `max_epochs` e patience (proposta: 100 e 10).
8. Preprocessing canonico: storico (raccomandato) oppure con pavimento o clip.
9. Budget di tuning K per famiglia, griglie, criterio di selezione (AP normalizzata media sulle macchine dev) e seed per il tuning (3).
10. Macchine di sviluppo: confermare le 6 estratte.

**Gruppo C — prima dei run ufficiali (servono in F10):**
11. δ e metrica primaria.
12. PCA di riferimento (a finestre oppure la migliore delle due).
13. Famiglia confermativa {Conv vs PCA, LSTM vs PCA}, α e correzione di Holm.
14. Livelli dei fattori per RQ1a e RQ1b.
15. Descrittori per RQ3 (al massimo 4) e relative ipotesi.
16. Numero di seed per la valutazione (5).
17. Regole di esclusione: cosa fare con un fit fallito o con una macchina con un solo evento.
18. Secondo dataset: quale e quando. Il piano congelato per SMD va riapplicato invariato.

---

## Fasi

### F0 — Decisioni, ambiente, contiguità
- **Obiettivo:** prendere le decisioni del gruppo A, rendere l'ambiente riproducibile e verificare l'assunzione di contiguità.
- **File:**
  - nuovi: `docs/decisioni.md` (registro delle decisioni, una voce per decisione con data e motivazione), `requirements-lock.txt` (versioni esatte da `pip freeze`), `.gitignore` (`__pycache__/`, `.pytest_cache/`, `store/`), `scripts/check_contiguity.py`;
  - da modificare: `README.md` (istruzioni minime).
- **Modifiche:** `check_contiguity.py` legge il dataset in sola lettura e confronta la distanza tra l'ultima riga del train e la prima del test con la distribuzione delle distanze tra righe consecutive.
- **Dipendenze:** nessuna.
- **Test:** suite esistente verde.
- **Completata quando:** decisioni A registrate, esito della contiguità documentato, file di lock presente.
- **Rischi:** il `.gitignore` non rimuove i `.pyc` già tracciati. Toglierli dal tracciamento (`git rm --cached`) è una decisione tua, separata.
- **Impatto sui risultati:** nessuno.

### F1 — Banco di prova sintetico e snapshot legacy
- **Obiettivo:** poter dimostrare la parità con `main.py` in automatico.
- **File:**
  - nuovi: `tests/fixtures/synth_smd.py`, `tests/conftest.py`, `tests/golden/legacy_synth.json`, `tests/test_legacy_characterization.py`;
  - da modificare: `pytest.ini` (marcatori `slow`, `regression`, `real_data`).
- **Modifiche:**
  - il generatore scrive in una cartella temporanea 2 macchine finte (3–5 sensori, ~2.000 punti, eventi iniettati, una leggera deriva) con la stessa struttura `train/`, `test/`, `test_label/`;
  - una config minima esegue `main.run_experiment` con modelli piccolissimi (PCA, PCA puntuale, Conv e LSTM con 2 epoche, baseline);
  - lo snapshot salva gli hash degli array `.npz` e le metriche del summary.
- **Dipendenze:** F0.
- **Test:** il test di caratterizzazione stesso. Deve durare meno di 30 s e restituire lo stesso risultato ripetendolo.
- **Completata quando:** lo snapshot è stabile su due esecuzioni consecutive.
- **Rischi:**
  - non determinismo di torch tra esecuzioni (thread): si fissa `torch.set_num_threads(1)` *solo nei test*;
  - lo snapshot dipende dalle versioni delle librerie, ed è proprio il motivo del lock in F0.
- **Impatto sui risultati:** nessuno.

### F2 — Config, provenance, archivio, schema della tabella
- **Obiettivo:** infrastruttura che non tocca alcun calcolo.
- **File:**
  - nuovi: `src/utils/schema.py` (dataclass della configurazione), `src/utils/provenance.py`, `src/utils/logging.py`, `src/pipeline/store.py`, `src/pipeline/table_schema.py`, `configs/base.yaml`, `configs/models/*.yaml` (valori storici);
  - da modificare: `src/utils/config.py` (aggiunte: merge, validazione, hash; `load_config` resta invariato per `main.py`).
- **Modifiche:**
  - `fit_key` = hash canonico del sottoinsieme della config rilevante per il fit, più `FIT_CODE_VERSION`;
  - `store` gestisce scrittura atomica, `exists_complete`, il registro `store/registry.csv` e la protezione contro i run ufficiali con albero git modificato;
  - `table_schema` definisce colonne e tipi della tabella lunga, così F7 può partire.
- **Dipendenze:** F0 (decisione sul parquet).
- **Test:**
  - una chiave sconosciuta genera un errore;
  - l'hash non dipende dall'ordine delle chiavi né dai percorsi;
  - cambiare un parametro di analisi non cambia `fit_key`;
  - la scrittura atomica sopravvive a un'eccezione a metà;
  - il registro è a sola aggiunta;
  - provenance in un repository modificato.
- **Completata quando:** test verdi e snapshot legacy (F1) invariato.
- **Rischi:** uno schema troppo rigido può rifiutare chiavi legittime. Le config legacy non passano da questo schema.
- **Impatto sui risultati:** nessuno.

### F3 — Segmenti e finestre
- **Obiettivo:** implementare split e finestre nella nuova pipeline.
- **File:**
  - nuovi: `src/data/manifest.py`, `src/data/segments.py` (modalità `legacy2` e `split3`, basate su `preprocessing/split.py`), `src/data/windows.py` (contesto opzionale, viste senza copia, iterazione a lotti, mappa finestra ↔ timestamp);
  - `src/data/dataset.py` resta invariato.
- **Dipendenze:** F2.
- **Test:**
  - i segmenti coprono tutto il train, contigui e senza sovrapposizioni;
  - le finestre di fit contengono solo indici del segmento fit;
  - il contesto usa solo indici precedenti;
  - **parità:** in modalità `legacy2` senza contesto, le finestre coincidono con quelle di `main.prepare_machine_data` sul dataset sintetico;
  - il manifest elenca 28 macchine e i 3 gruppi;
  - l'hash dei dati è stabile.
- **Completata quando:** test verdi e parità esatta.
- **Rischi:**
  - errori di uno nella gestione degli indici del contesto (coperti dai test brute force);
  - le viste `as_strided` in sola scrittura vanno protette con `writeable=False`.
- **Impatto sui risultati:** nessuno.

### F4 — Interfaccia dei residui e checkpoint dei modelli
- **Obiettivo:** dare a ogni modello lo stesso output (residui per sensore) e la possibilità di essere salvato e ricaricato.
- **File:**
  - nuovi: `src/models/registry.py`;
  - da modificare, solo con aggiunte: `src/models/base.py`, ogni modello in `traditional/` e `deep/`, `src/training/trainer.py`.
- **Modifiche:**
  - `residuals(windows, aggs) -> {agg: (N,F)}`, calcolato a lotti;
  - `save(dir)` / `load(dir)`;
  - `FitInfo` (epoca migliore, epoca di stop, loss, `n_components`, tempi);
  - nel trainer, `max_epochs` come alias opzionale di `epochs` (se assente non cambia nulla);
  - rolling z-score: residuo = |z| per sensore.
  
  `score_anomalies` e `score_features` restano identici.
- **Dipendenze:** F2 (per `FIT_CODE_VERSION`).
- **Test:**
  - `residuals[last|mean]` coincide **esattamente** con `score_features(time_agg)` per ogni modello;
  - residui `last`/`mean` uguali al calcolo manuale su un modello giocattolo;
  - un checkpoint salvato e ricaricato produce residui identici;
  - test golden R1 (hash dei residui sintetici);
  - snapshot legacy invariato.
- **Completata quando:** tutto verde.
- **Rischi:**
  - lo stato di BatchNorm nel Conv-AE va salvato insieme ai pesi (con `state_dict` è incluso);
  - la PCA puntuale ha uno standardizzatore interno, che va salvato.
- **Impatto sui risultati:** nessuno.

### F5 — Stadio di fit
- **Obiettivo:** il fit eseguito da riga di comando, con artefatti indirizzati per contenuto.
- **File (tutti nuovi):** `src/pipeline/fit.py`, `src/pipeline/residual_cache.py`, `src/cli.py` (comandi `fit` e `status`).
- **Modifiche:** per ogni `fit_key` mancante: segmenti → scaler → finestre → `set_seed` → fit → checkpoint → residui → `meta.json`. I fallimenti restano isolati e si possono riprendere i run interrotti.
- **Dipendenze:** F3, F4.
- **Test (integrazione su dati sintetici):**
  - **parità:** in modalità `legacy2` i residui dello stadio di fit coincidono con quelli calcolati dalla pipeline storica;
  - rilanciare lo stesso esperimento non ripete alcun fit;
  - un errore iniettato in un modello non ferma gli altri;
  - un'interruzione simulata non lascia artefatti parziali;
  - residui cancellati e poi rigenerati sono identici.
- **Completata quando:** test verdi; il fit sintetico completo dura meno di 1 minuto.
- **Rischi:**
  - l'ordine di chiamata di `set_seed` deve replicare esattamente quello di `main.py` ([main.py:313](main.py#L313), [main.py:330](main.py#L330)), altrimenti la parità degli AE salta;
  - il generatore casuale della baseline random va replicato nell'analisi.
- **Impatto sui risultati:** nessuno.

### F6 — Stadio di analisi
- **Obiettivo:** prima riprodurre esattamente le metriche storiche dai residui, poi estendere.
- **File:**
  - nuovi: `src/pipeline/analyze.py`, `src/evaluation/metrics.py`, `src/evaluation/thresholds.py`;
  - da modificare: `src/evaluation/rules.py` (diventa un modulo di compatibilità che ri-esporta le funzioni), `src/evaluation/scoring.py` (aggiunta del normalizer robusto), `src/cli.py` (comando `analyze`).
- **Modifiche:**
  - **6a, parità:** analisi in modalità legacy → stessi `.npz` e stesso summary della pipeline storica;
  - **6b, estensioni:** AP normalizzata, ritardo di rilevamento, falsi allarmi al giorno, celle 2×2 (da `protocols.py`), soglie con storia continua es → cal → test, griglia dei fattori, tabella lunga.
- **Dipendenze:** F5.
- **Test:**
  - parità esatta con lo snapshot F1;
  - causalità end-to-end: perturbare il test dopo t lascia invariati gli score e le decisioni di soglia fino a t;
  - proprietà: l'oracolo è ≥ dell'F1 calibrato, il PA è ≥ del no-PA;
  - le soglie con storia coincidono con quelle senza storia quando la storia è vuota;
  - le metriche nuove sono verificate su casi calcolati a mano.
- **Completata quando:** 6a esatta; 6b verde; la tabella lunga rispetta lo schema di F2.
- **Rischi:** numero di combinazioni della griglia (si stimano circa 15 minuti per l'esperimento ufficiale). Va misurato su dati sintetici e, se serve, parallelizzato per macchina.
- **Impatto sui risultati:** metriche storiche invariate; metriche nuove aggiunte.

### F7 — Statistica e report
- **Obiettivo:** RQ1–RQ4 calcolate dalla tabella lunga.
- **File (tutti nuovi):** `src/stats/{aggregate,bootstrap,decisions,ranking,variance,descriptors}.py`, `src/reporting/{tables,figures}.py`, comandi `stats` e `report` in `src/cli.py`. `aggregate_results.py` resta.
- **Dipendenze:** solo lo schema di F2 (per i test su dati simulati); F6 per l'uso reale. Si può sviluppare in parallelo a F3–F6.
- **Test:**
  - copertura del bootstrap stratificato ≈ 95% su simulazioni (marcato `slow`);
  - regola di decisione su δ con casi noti;
  - correzione di Holm;
  - la decomposizione SS somma al totale;
  - τ di Kendall su casi noti;
  - **parità:** media ± std e confronti appaiati coincidono con `aggregate_results.py` sulla stessa tabella.
- **Completata quando:** test verdi e report generato dalla tabella sintetica.
- **Rischi:** troppi output. Il report produce solo ciò che il piano di analisi richiede.
- **Impatto sui risultati:** nessuno.

### M1 — Parità su dati reali (validazione metodologica, la lanci tu)
- **Obiettivo:** dimostrare che la nuova pipeline in modalità legacy riproduce `results_repro_*_01`.
- **File (nuovi):** `configs/experiments/parity_repro_{last,mean}.yaml`, `scripts/parity_report.py` (confronta `store/` con `results/` in sola lettura).
- **Esecuzione:** `cli fit` + `cli analyze` su `machine-1-1` e `machine-3-8`. Stima: pochi minuti.
- **Completata quando:** score identici (oppure differenze entro una tolleranza dichiarata e motivata), metriche identiche, report salvato in `docs/`.
- **Rischi:** differenze dovute a thread o versioni. In quel caso si diagnostica prima di procedere: non si alzano le tolleranze senza spiegazione.
- **Impatto sui risultati:** nessuno. **Da qui la pipeline storica non serve più come oracolo**, ma resta disponibile fino a F12.

### F8 — Nuovo protocollo canonico (cambiamento numerico dichiarato)
- **Obiettivo:** passare a `split3`, contesto causale, normalizer e soglie su cal, PCA solo su fit, nuovi `max_epochs`/patience.
- **File:**
  - da modificare: `configs/base.yaml` (valori canonici);
  - nuovi: `configs/experiments/protocol_switch_dev.yaml`, `scripts/make_dev_split.py` (riproduce l'estrazione delle 6 macchine), test di leakage end-to-end.
- **Dipendenze:** M1; decisioni del gruppo B (punti 6–8, 10).
- **Test (su dati sintetici):**
  - perturbare es, cal o test lascia invariati scaler e PCA;
  - perturbare il test lascia invariati normalizer e soglie;
  - la modalità `legacy2` resta riproducibile, verificata con lo snapshot.
- **Esperimento (lo lanci tu):** confronto legacy vs canonico sulle 6 macchine dev, con report delle differenze.
- **Completata quando:** report approvato da te.
- **Rischi:** con il contesto, la maschera di valutazione cambia: va confrontata anche sulla maschera comune, per separare l'effetto del protocollo da quello della maschera.
- **Impatto sui risultati:** **sì, dichiarato e documentato.**

### F9 — Tuning e sensibilità (esperimenti esplorativi)
- **Obiettivo:** scegliere le configurazioni globali per famiglia, con budget pari, sulle 6 macchine dev.
- **File (nuovi):** griglie in `configs/models/*.yaml`, `configs/experiments/tuning.yaml`, `configs/analysis/tuning.yaml`; ulteriori esperimenti esplorativi (preprocessing, epoche fisse, PCA su fit+es) sempre sulle sole macchine dev.
- **Dipendenze:** F8; decisione 9.
- **Test:** un test sulle config verifica che il budget sia pari per costruzione; lo stadio di fit rifiuta le macchine di valutazione negli esperimenti esplorativi, salvo un flag esplicito che viene registrato.
- **Completata quando:** configurazioni selezionate salvate, con motivazione in `docs/decisioni.md`.
- **Rischi:** tentazione di guardare le macchine di valutazione, impedita dal controllo sopra. Stima del costo: circa 7 ore di CPU con `train_step=5`, da verificare.
- **Impatto sui risultati:** sceglie gli iperparametri finali. Questi risultati sono **esplorativi**, non della tesi.

### F10 — Congelamento del piano di analisi
- **Obiettivo:** fissare RQ, δ e analisi prima di vedere i risultati ufficiali.
- **File (nuovi):** `configs/analysis/official.yaml` (decisioni 11–17), con il suo hash nel registro, più la sezione corrispondente in `docs/decisioni.md`.
- **Dipendenze:** F9; decisioni del gruppo C.
- **Test:** il comando `cli` con `kind: official` rifiuta di partire se il piano non è registrato, se l'albero git è modificato o se tra le macchine compare una di sviluppo.
- **Completata quando:** piano congelato in un commit (lo fai tu o lo chiedi tu).
- **Impatto sui risultati:** nessuno.

### F11 — Run ufficiali (esperimenti ufficiali, li lanci tu)
- **Obiettivo:** risultati della tesi sulle 22 macchine di valutazione.
- **File (nuovi):** `configs/experiments/official.yaml`.
- **Esecuzione:** `fit` → `analyze` → `stats` → `report`. Stima: circa 6 ore di CPU per i fit.
- **Completata quando:** IC, decisioni RQ2 e analisi RQ1, RQ3 e RQ4 sono generati dal piano congelato e da un commit noto; il registro è aggiornato.
- **Rischi:** un fit fallito si gestisce secondo la regola di esclusione congelata, non decidendo dopo.
- **Impatto sui risultati:** sono i risultati.

### F12 — Pulizia (solo con la tua approvazione)
- **Obiettivo:** un'unica pipeline e documentazione allineata.
- **File:** `main.py`, `aggregate_results.py`, `src/data/dataset.py`, `scripts/compare_npz.py`, `scripts/confronta_run.py` (deprecazione o rimozione); `architettura_repo_v1.md` (archiviato); `CLAUDE.md` (aggiornato).
- **Dipendenze:** M1 superata; F11 completata o almeno F8.
- **Test:** suite completa; lo snapshot legacy si archivia o si rimuove insieme a `main.py`.
- **Impatto sui risultati:** nessuno.

---

## Ordine consigliato e parallelismi

```
F0 → F1 → F2 ─┬→ F3 → F4 → F5 → F6 ─┐
              └→ F7 (in parallelo) ──┴→ M1 → F8 → F9 → F10 → F11 → F12
```

- **Si può interrompere in qualunque punto:** fino a M1 nessun risultato cambia e la pipeline storica resta intatta. Ogni fase è un branch e un commit separato, su tua richiesta.
- **`CLAUDE.md` va aggiornato due volte:** dopo M1, quando la nuova pipeline diventa quella di riferimento, e dopo F8, quando il protocollo canonico cambia.
- **Dimensioni relative:** F1, F4 e F6 sono le fasi più delicate; F2 e F7 sono le più lunghe da scrivere ma le meno rischiose; F3 e F5 stanno nel mezzo.

## Cosa mi serve da te per partire

Le decisioni del **gruppo A** (1–5). Con quelle posso iniziare F0 e F1. Le decisioni del gruppo B servono solo da F8 in poi, quelle del gruppo C solo da F10.

Mi fermo qui e attendo la tua approvazione.