# Piano operativo di sviluppo

Questo piano trasforma in passi concreti le parti approvate di `proposta_architettura_target.md`, registrate in `docs/decisioni.md`.

> **Stato del documento (2026-11-09).** Livello 5 della gerarchia (`CLAUDE.md` §3.2): vincolante per l'ordine del lavoro e i criteri di completamento; prevale sul §12 della proposta [D-13]. Per le scelte metodologiche prevale il registro `docs/decisioni.md`. I riferimenti `[D-xx]` rimandano alle sue voci.

## Cosa cambio rispetto all'ordine della proposta precedente, e perché

1. **`main.py` resta congelato.** La nuova pipeline cresce accanto a quella storica invece di modificarla dall'interno. Prima prevedevo di modificare `prepare_machine_data` già nelle fasi 1–2. Così il riferimento storico non si può rompere per errore, e la parità si dimostra confrontando le due pipeline. Le uniche modifiche ai moduli condivisi (modelli, trainer, scoring) sono **aggiunte**: nessun metodo esistente cambia comportamento.
2. **Un banco di prova sintetico come prima fase software.** Un piccolo dataset in formato SMD, generato nei test, più una "fotografia" (snapshot) degli output di `main.py` su quel dataset. La parità si verifica in automatico a ogni modifica, senza training su dati reali. Nella proposta precedente il controllo di regressione lo dovevi lanciare tu a ogni fase.
3. **La parità su dati reali si concentra in un solo punto di controllo (M1),** invece di ripeterla a ogni fase. Lo lanci tu una volta, quando la nuova pipeline è completa in modalità legacy.
4. **Lo schema della tabella lunga si fissa presto (F2).** Così il modulo di statistica (F7) dipende solo da quello schema e si può sviluppare anche in parallelo, su dati simulati.
5. **La verifica che test e train siano contigui nel tempo e il blocco delle versioni delle dipendenze vanno in F0,** perché condizionano due scelte di progetto: il contesto delle finestre e la riproducibilità bit a bit.

## Quadro generale

| Fase | Tipo | Training su dati reali? | Cambia i numeri? |
|---|---|---|---|
| F0 Riconciliazione, decisioni, ambiente, contiguità | software + validazione metodologica | no | no |
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

Il registro autorevole è `docs/decisioni.md`. Qui le decisioni sono raggruppate in base alla fase che bloccano.

**Già approvate (2026-11-09):** finestre e contesto [D-03], es escluso dall'analisi [D-04], split 60/20/20 [D-05],
PCA su fit [D-06], metriche primaria e secondarie [D-07], 6 di sviluppo + 22 di valutazione [D-08], 5 seed [D-09],
modelli fuori dal nucleo esclusi [D-10], normalizer robusto come variante [D-11], RQ1a/RQ1b [D-12], piano di migrazione
[D-13], verifica di non regressione [D-14], interfaccia di `fit` [D-15], training sintetico nei test [D-16],
procedura di M1 [D-17], dipendenze [D-18], CSV.gz [D-19].

**Approvate dopo (2026-11-09/10):** confine train → test [D-32], contesto iniziale del test e maschera [D-20],
soglie mobili e punti senza score [D-21], coda del test esclusa [D-22], supporto di calibrazione in cal [D-23],
soglie mobili solo con `last` [D-24], normalizzazione di SMD su train+test, fatto verificato [D-26]. D-31 è sostituita da D-32.

**Gruppo A — servono per chiudere F0:**
1. ~~Criterio della verifica di contiguità e comportamento in caso negativo~~ — **risolta da D-32**: la contiguità non è verificabile e nessuna conclusione primaria ne dipende.
2. ~~Aggregazioni dei residui da salvare~~ — **risolta da D-33** (approvata 2026-11-10): `last` e `mean`; `max` non si salva.
3. ~~Elenco delle metriche da implementare in F6~~ — **risolta da D-34** (approvata 2026-11-10); VUS-PR in un passo separato prima di F10.
4. ~~Posizione di `store/` e budget di disco~~ — **risolta da D-56** (approvata 2026-11-10): `store/` nella radice, esclusa da git, 5–10 GB.
5. ~~Ambiente di riferimento per il file di lock~~ — **risolta da D-55** (approvata 2026-11-10): Python 3.13.9, `requirements-lock.txt`.
6. ~~Run storici di riferimento per M1~~ — **risolta da D-50** (approvata 2026-11-10): `results_repro_last_01` / `results_repro_mean_01`.
7. ~~Lista delle 6 macchine di sviluppo~~ — **risolta da D-51** (approvata 2026-11-10): `configs/dev_machines.yaml`, generato da `scripts/make_dev_split.py`.

D-53 approvata (2026-11-10): `.pyc` tolti dal tracciamento, `dataset/` resta tracciato.

**Gruppo A' — servono prima di F3 e F7:**
8. ~~Quali segmenti ricevono il contesto nello scoring~~ — **risolta**: D-31 è sostituita da D-32; regole operative in D-20 e D-22.
9. ~~Metodo degli IC e famiglia confermativa~~ — **risolta da D-36** (approvata 2026-11-16): bootstrap stratificato per gruppo con BCa, IC di Bonferroni; famiglia, α e B parametri del piano statistico.
9-bis. ~~Metodo di decomposizione della varianza per RQ1a~~ — **risolta da D-25** (approvata 2026-11-16): somme dei quadrati classiche a blocchi bilanciati, una risposta per metrica.

**Gruppo B — prima degli esperimenti esplorativi (F8–F9):**
10. ~~`train_step` [D-38]~~ — **risolta da D-38** (approvata 2026-11-24): 5 per tutti i modelli a finestre.
11. ~~`max_epochs` e patience [D-39] (proposta: 100 e 10)~~ — **risolta da D-39** (approvata 2026-11-24): 100 e 10.
12. ~~Preprocessing canonico [D-40]~~ — **risolta da D-40** (approvata 2026-11-24): minmax su fit con pavimento τ = 0,02.
13. Budget di tuning, griglie, criterio di selezione, seed per il tuning, ruolo di scoring e normalizer robusto nel budget [D-41].
14. Sensibilità delle proporzioni dello split [D-46].

**Gruppo C — prima del congelamento (F10):**
15. δ [D-30] (la metrica primaria è già approvata, D-07).
16. PCA di riferimento per RQ2 [D-37].
17. Livelli dei fattori per RQ1a e RQ1b [D-42].
18. Descrittori per RQ3 [D-43].
19. Regole di esclusione [D-44].
20. ~~Secondo dataset~~ — **risolta da D-45** (approvata 2026-11-16): nessun secondo dataset in questa tesi, per ora; limite dichiarato.

**Dipendono dall'utente, senza una fase che le blocchi:** scadenze [D-54]. Macchine dei numeri preliminari [D-52] e file tracciati da git [D-53] sono approvate (2026-11-10).

---

## Fasi

### F0 — Riconciliazione, decisioni, ambiente, contiguità
- **Obiettivo:** documenti coerenti, decisioni del gruppo A registrate, ambiente riproducibile, confine train → test regolato da D-32 (la contiguità non è verificabile e nessuna conclusione primaria ne dipende).
- **File:**
  - nuovi: `docs/decisioni.md` (registro, livello 1), `requirements-lock.txt` (versioni esatte da `pip freeze`, generato dall'utente nel suo ambiente [D-55]), `.gitignore` (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `store/` [D-56]), `scripts/check_contiguity.py` (facoltativo, solo diagnostica descrittiva [D-32]), `scripts/make_dev_split.py` e `configs/dev_machines.yaml` (lista delle 6 macchine di sviluppo, seed e procedura [D-51]), `tests/test_dev_split.py`;
  - da modificare: `CLAUDE.md` (riconciliazione, D-02), `proposta_architettura_target.md` (note di stato e rimandi), `piano_operativo.md` (questo file), `README.md` (istruzioni minime).
- **Modifiche:**
  - `check_contiguity.py`, se si scrive, legge il dataset in sola lettura e confronta il salto tra l'ultima riga del train e la prima del test con la distribuzione dei salti tra righe consecutive. È solo una diagnostica descrittiva, senza ruolo decisionale [D-32]; i controlli strutturali bloccanti di D-32 si implementano nel manifest (F3);
  - la lista delle macchine di sviluppo si estrae qui con `scripts/make_dev_split.py`, secondo la procedura di D-51 (2 per gruppo, `machine-1-1` e `machine-3-8` fisse, seed 20261110), senza leggere label, metriche o risultati.
- **Dipendenze:** nessuna.
- **Test:** suite esistente verde, più `tests/test_dev_split.py` (6 + 22 disgiunte, 2 per gruppo, 1-1 e 3-8 presenti, stessa lista a ogni esecuzione, file coerente con la procedura).
- **Completata quando:** documenti riconciliati e approvati; decisioni del gruppo A registrate (D-32 registrata il 2026-11-09); file di lock presente; lista delle macchine di sviluppo registrata; run di riferimento per M1 identificati [D-50].
- **Rischi:** il lock è provvisorio (installazione globale di Python): prima di M1 va rigenerato da un ambiente virtuale dedicato [D-55].
- **Impatto sui risultati:** nessuno.

### F1 — Banco di prova sintetico e snapshot legacy
- **Obiettivo:** poter dimostrare la parità con `main.py` in automatico. È lo strumento di verifica di non regressione fino a M1 [D-14].
- **File:**
  - nuovi: `tests/fixtures/synth_smd.py`, `tests/conftest.py`, `tests/golden/legacy_synth.json`, `tests/test_legacy_characterization.py`;
  - da modificare: `pytest.ini` (marcatori `slow`, `regression`, `real_data`).
- **Modifiche:**
  - il generatore scrive in una cartella temporanea 2 macchine finte (3–5 sensori, ~2.000 punti, eventi iniettati, una leggera deriva) con la stessa struttura `train/`, `test/`, `test_label/`;
  - una config minima, scritta in `tmp_path` con `results_dir` in `tmp_path`, esegue `main.run_experiment` con modelli piccolissimi (PCA, PCA puntuale, Conv e LSTM con 2 epoche, baseline). È training su dati sintetici, ammesso dentro i test [D-16]; non legge `dataset/` e non scrive nel repository;
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
  - nuovi: `src/utils/schema.py` (dataclass della configurazione), `src/utils/provenance.py` (include il numero di thread di torch e `threadpoolctl.threadpool_info()` senza i percorsi dei file: backend BLAS/OpenMP, versione, numero di thread [D-17]), `src/utils/logging.py`, `src/pipeline/store.py`, `src/pipeline/table_schema.py`, `configs/base.yaml`, `configs/models/*.yaml` (valori storici);
  - da modificare: `src/utils/config.py` (aggiunte: merge, validazione, hash; `load_config` resta invariato per `main.py`).
- **Modifiche:**
  - `fit_key` = hash canonico del sottoinsieme della config rilevante per il fit, più `FIT_CODE_VERSION`; le aggregazioni dei residui salvate [D-33] fanno parte della config rilevante;
  - `store` gestisce scrittura atomica, `exists_complete`, il registro `store/registry.csv` e la protezione contro i run ufficiali con albero git modificato;
  - `table_schema` definisce colonne e tipi della tabella lunga, così F7 può partire.
- **Dipendenze:** F0. Formato: CSV.gz [D-19]; lettura e scrittura delle tabelle isolate in `store.py` / `table_schema.py`.
- **Test:**
  - una chiave sconosciuta genera un errore;
  - l'hash non dipende dall'ordine delle chiavi né dai percorsi;
  - cambiare un parametro di analisi non cambia `fit_key`;
  - la scrittura atomica sopravvive a un'eccezione a metà;
  - il registro è a sola aggiunta;
  - provenance in un repository modificato.
- **Completata quando:** test verdi e snapshot legacy (F1) invariato.
- **Rischi:**
  - uno schema troppo rigido può rifiutare chiavi legittime. Le config legacy non passano da questo schema.
  - fuori dai test l'ambiente usa 8 thread per BLAS e 4 per torch (misurato in F1); i run storici di M1 sono stati fatti così, quindi la provenance deve registrare anche i thread di BLAS, altrimenti una divergenza di qualche ULP sulla PCA in M1 non è diagnosticabile [D-17].
- **Impatto sui risultati:** nessuno.

### F3 — Segmenti e finestre
- **Obiettivo:** implementare split e finestre nella nuova pipeline.
- **File:**
  - nuovi: `src/data/manifest.py`, `src/data/segments.py` (modalità `legacy2` e `split3`, basate su `preprocessing/split.py`), `src/data/windows.py` (contesto opzionale, viste senza copia, iterazione a lotti, mappa finestra ↔ timestamp); Regola dei confini [D-03]: finestre di fit ed es solo dentro il proprio segmento; contesto solo per lo scoring, mai per es e cal, e all'inizio del test solo nella variante di RQ1a [D-32, D-20]; nessun contesto in `legacy2`. Maschere del test: [W−1, T−W+1) nell'analisi primaria, [0, T−W+1) nella variante con contesto, uguali per tutti i modelli dello scenario [D-20, D-22].
  - `src/data/dataset.py` resta invariato.
- **Dipendenze:** F2; decisioni D-32, D-20, D-22 (approvate) e D-51 (lista delle macchine di sviluppo, usata dal manifest).
- **Test:**
  - i segmenti coprono tutto il train, contigui e senza sovrapposizioni;
  - le finestre di fit contengono solo indici del segmento fit;
  - il contesto usa solo indici precedenti;
  - le finestre della loss di early stopping contengono solo indici di es;
  - es e cal non ricevono mai contesto; il test lo riceve solo nello scenario della variante di RQ1a [D-32, D-20];
  - maschere del test: primi W−1 punti esclusi nell'analisi primaria e inclusi nella variante, ultimi W−1 sempre esclusi, identiche per tutti i modelli dello scenario [D-20, D-22];
  - il manifest blocca la pipeline se un controllo strutturale fallisce: differenza tra lunghezze di train e test ≤ 1, 38 sensori, lunghezza delle label = test, valori in [0, 1], hash dei file [D-32];
  - **parità:** in modalità `legacy2` senza contesto, le finestre coincidono con quelle di `main.prepare_machine_data` sul dataset sintetico;
  - il manifest elenca 28 macchine e i 3 gruppi, con 6 di sviluppo e 22 di valutazione disgiunte [D-08, D-51];
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
  - `FitInfo` (epoca migliore, epoca di stop, loss, `n_components`, tempi), salvato nell'attributo `fit_info_`. La firma `fit(X, y=None, X_val=None) -> self` non cambia [D-15];
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
- **Nota (2026-11-11, piano F4 approvato):** decisioni D-66 (tipo nativo dei residui), D-67 (checkpoint), D-68
  (interfaccia, registro, `batching.py`, golden T0) e note su D-33 (rolling z-score solo `last`, `native_only`),
  D-39 (alias), D-57 (F4 non corregge il solver), D-64 (regola dei lotti). File in più rispetto all'elenco sopra:
  `src/models/batching.py`. Perimetro: i 6 modelli del nucleo; AE non addestrati rimandati (D-69). Ordine dei passi:
  0 golden T0 (commit separato) → 0-bis registro e piano → 1 `base.py` e trainer → 2 `batching.py` → 3 modelli
  tradizionali → 4 AE → 5 `registry.py` → 6 golden R1.
- **Stato: completata (2026-11-11).** Commit: passo 0 `79ea2cf` (golden T0), 0-bis `6a0a731` (registro e piano),
  1 `40754c7` (`FitInfo`, `base.py`, alias), 2 `1800b55` (`batching.py`), 3 `7aff6ae` (PCA, PCA puntuale, `input_norm`,
  rolling z-score, layout dei lotti), 4 `6dbf9e5` (Conv-AE, LSTM-AE), 5 `556ead1` (`registry.py`), 6 golden R1 e
  chiusura (commit che contiene questa riga). Suite di chiusura: vedi il messaggio del commit del passo 6.
  - **Verificato:** residui = `score_features` storico bit a bit per i 6 modelli (T1, anche contro gli hash di T0);
    checkpoint salva → ricarica → residui identici, anche in un processo nuovo per gli AE e passando da
    `load_checkpoint`; snapshot F1 e golden T0 invariati a ogni passo; golden R1 coerente con T0.
  - **Non verificato in F4:** parità su dati reali (M1); layout delle finestre di fit nella nuova pipeline (F5).
  - **Decisioni APERTE toccate e rimaste aperte:** D-57 (nessuna correzione del solver), D-69 (AE non addestrati),
    D-39 (solo l'alias, nessun valore).
  - **Passati a F5 (vedi Rischi di F5):** `load` mai tra `set_seed` e `fit`; tipo dei residui e
    `loaded_with_version_mismatch` in `meta.json`; tempi fuori dagli hash (`checkpoint_hashable_meta`); regola dei
    lotti in entrambe le modalità; lotti di scoring con `legacy_layout_batch`; layout delle finestre di fit;
    `native_only` per la convenzione none/native in F5–F6 (nota su D-33); golden R1 da rigenerare solo con un cambio
    motivato di `FIT_CODE_VERSION` o del lock.

### F5 — Stadio di fit
- **Obiettivo:** il fit eseguito da riga di comando, con artefatti indirizzati per contenuto.
- **File (tutti nuovi):** `src/pipeline/fit.py`, `src/pipeline/residual_cache.py`, `src/cli.py` (comandi `fit` e `status`).
- **Modifiche:** per ogni `fit_key` mancante: segmenti → scaler → finestre → `set_seed` → fit → checkpoint → residui → `meta.json`. I fallimenti restano isolati e si possono riprendere i run interrotti. La pipeline passa il segmento fit come `X` ed es come `X_val` [D-15]. I residui di es si salvano solo come diagnostica [D-04].
- **Dipendenze:** F3, F4.
- **Test (integrazione su dati sintetici):**
  - **parità:** in modalità `legacy2` i residui dello stadio di fit coincidono con quelli calcolati dalla pipeline storica;
  - rilanciare lo stesso esperimento non ripete alcun fit;
  - un errore iniettato in un modello non ferma gli altri;
  - un'interruzione simulata non lascia artefatti parziali;
  - residui cancellati e poi rigenerati sono identici.
- **Completata quando:** test verdi; il fit sintetico completo dura meno di 1 minuto.
- **Rischi:**
  - l'ordine di chiamata di `set_seed` deve replicare esattamente quello di `main.py` ([main.py:310](main.py#L310), [main.py:330](main.py#L330)), altrimenti la parità degli AE salta;
  - il generatore casuale della baseline random va replicato nell'analisi;
  - `load` di un checkpoint non va mai chiamato tra `set_seed` e `fit`: ricostruire la rete di un AE consuma il
    generatore casuale di torch [D-67];
  - `meta.json` registra il tipo dei residui di ogni modello [D-66] e `loaded_with_version_mismatch` quando un
    checkpoint è stato caricato forzando le versioni [D-67]; i tempi non entrano in alcun hash [D-67];
  - dimensione del lotto di scoring: regola di D-64 applicata in entrambe le modalità da F4; F5 può allentarla solo
    con una decisione esplicita (nota su D-64);
  - layout di memoria: anche le finestre di fit (`fit_X` con `train_step`, `fit_X_val` con la regola di D-63) devono
    avere lo stesso layout di `main.py` (`create_sliding_windows`, ordine dei passi conservato), non le copie
    C-contigue di `WindowSet.materialize`, finché un test non dimostra che il layout non conta; test sui passi da
    aggiungere in F5 (nota su D-64, passo 3 di F4). Sonda di F4 (passo 4, non è un test): per Conv-AE e LSTM-AE fit e
    scoring non dipendono dal layout (`torch.from_numpy` conserva i passi, ma il `DataLoader` impila lotti
    C-contigui: pesi, history e score identici); la PCA a finestre copia in ordine C con `reshape`; la PCA puntuale
    fitta su `train_scaled` (2D);
  - lo scoring dei residui passa da `batching.compute_residuals` (lotti con `legacy_layout_batch`, regola di D-64),
    mai da `iter_batches`;
  - i confronti di `checkpoint.json` e di `meta.json` usano la forma senza tempi (`checkpoint_hashable_meta`) [D-67];
  - `native_only` (rolling z-score) indica la convenzione none/native di `table_schema` per F5–F6: la chiave dei
    residui in cache resta `last` (nota su D-33);
  - `residuals` degli AE isola il generatore di torch (`fork_rng`); il `DataLoader` di `score_features` lo fa
    avanzare, senza effetti sulla baseline random storica, che usa un generatore numpy proprio (nota su D-67);
  - golden R1 (`tests/golden/residuals_r1.json`): si rigenera solo con un cambio motivato di `FIT_CODE_VERSION` o del
    lock (D-55), mai per far passare un test.
- **Impatto sui risultati:** nessuno.
- **Nota (2026-11-12, piano F5 approvato):** decisione D-70 (layout dello store, nomi dei residui, `meta.json` /
  `residuals.json` e regola degli hash, eccezione limitata per i file di cache, controlli preliminari, `official`
  rifiutato, valori non finiti, `interrupted`, `status` in sola lettura, lock) e note su D-04, D-44, D-58, D-64 (nessun
  allentamento dei lotti), D-67 (regola degli hash, ordine di `set_seed`).
  - **Perimetro:** i 6 modelli del nucleo; `legacy2` (parità) e `split3` (invarianti, `test_ctx_head` solo con
    `step` = 1); la baseline random resta a F6 (`default_rng(seed)`, poi `random(T_cal)` e `random(T_test)`, come
    `main.py:313-314`, sulla maschera M_cal di D-23); le protezioni sulle macchine e sui run ufficiali restano a F9–F10.
  - **File in più rispetto all'elenco sopra:** `src/pipeline/store.py`, elenco chiuso di aggiunte di D-70
    (`replace_fit_cache_file`, `orphan_tmp`, acquisizione e rilascio del lock, valore `interrupted` in `RUN_STATUSES`
    accettato da `finish_run` e `append_registry`; nessun metodo esistente cambia per i valori già ammessi); docstring di
    `WindowSet.materialize` (`src/data/windows.py`) corretta in un **commit separato** (nessun numero cambia): in F5 le
    finestre di fit non usano `materialize`.
  - **Finestre di fit:** `fit_X` = `legacy_layout_batch(train_windows, 0, N)`; `fit_X_val` =
    `legacy_layout_batch(score_windows(es), 0, N)[::max(1, train_step // step)]` (regola letterale di D-63, stessi passi
    di `main.py:229-231`); la PCA puntuale riceve il segmento fit scalato.
  - **Test aggiunti rispetto all'elenco sopra:** stato dei generatori all'ingresso di `fit` uguale a quello dopo un
    `set_seed(seed)` nuovo (nota su D-67); residui dei modelli deterministici identici con `set_seed(42)` e
    `set_seed(7)`; nessun modello diverso dalla PCA legge `random_state` dalla config (grep + test); passi di X e X_val
    passati a `fit` uguali a quelli di `main.prepare_machine_data`, PCA comprese; loss NaN iniettata → fit `complete`,
    `nonfinite_loss: true`, JSON valido; `KeyboardInterrupt` → riga `interrupted`, codice 130, `tmp/` vuota, nessun
    `fits/<key>`; secondo run concorrente rifiutato dal lock; lock attivo → `clean-tmp` rifiuta; lock di un pid morto →
    `clean-tmp` senza opzione rifiuta, con `--break-stale-lock` pulisce e annota la rottura nel registro; lock di un
    host diverso → `--break-stale-lock` rifiuta; `orphan_tmp` e `status` elencano i cestini orfani; rigenerazione con versioni diverse e hash uguali → ok con
    warning e `loaded_with_version_mismatch: true` in `residuals.json`, con hash diversi → errore anche forzando;
    anti-leakage in `split3` (perturbare cal e test lascia invariati scaler, PCA, PCA puntuale e AE) e in `legacy2`
    (perturbare il test lascia invariati scaler, checkpoint e residui di cal per tutti i modelli; perturbare cal = es
    cambia gli AE: doppio uso P1 dichiarato, verificato come comportamento atteso); `status` non scrive nulla; la suite
    non crea `<repo>/store`.
  - **Passi:** 0 registro e piano → 0-bis docstring di `materialize` (commit separato) → 1 layout delle finestre di fit,
    sequenza di `set_seed`, seed dei deterministici e `random_state` (prima i test), preparazione per macchina →
    2 `residual_cache.compute_all`, formato dei residui, aggiunte a `store.py` (con `interrupted`: se un test o lo
    schema di F2 fissa i valori ammessi per lo stato nel registro, si aggiorna insieme, dichiarandolo) → 3 `fit.py` (parità, rilancio, dtype,
    `native_only`, es, tempi fuori dagli hash, loss NaN; misura della durata) → 4 fallimenti isolati, interruzione,
    controlli preliminari, lock, anti-leakage `split3` e `legacy2`, `test_ctx_head` → 5 rigenerazione e regola degli
    hash → 6 `cli.py` (`fit`, `status`, `clean-tmp`, rifiuto di `official`) → 7 chiusura. Dopo ogni passo: snapshot F1,
    golden T0 e R1, poi `python -m pytest -q`.
  - **Decisioni APERTE toccate, stato invariato:** D-38, D-39, D-40, D-41, D-44, D-57, D-69.
  - **Passo 1 (2026-11-12):** `fit.py` contiene `prepare_machine` e `fit_one`; `fit_one` è anticipato rispetto al
    passo 3 perché i test sull'ordine di `set_seed` ne hanno bisogno (pianificazione, `run_fit`, meta e store restano
    al passo 3). Test: `tests/test_f5_layout.py`, `tests/test_f5_seed.py` (fixture `tests/fixtures/f5.py`).
  - **Passo 2 (2026-11-12):** `residual_cache.py` (`compute_all`, `array_record`) e aggiunte a `store.py` (D-70).
    Test: `tests/test_f5_residual_cache.py`, `tests/test_f5_store.py`.
    - *Scostamento:* una chiamata di `compute_residuals` per segmento con tutte le aggregazioni (come il golden R1),
      non una per aggregazione; il test sull'unica via (test 3) conta le chiamate di conseguenza.
    - *Fatto:* calcolando test e `test_ctx_head` nella stessa chiamata, i residui `test__*` restano identici bit a bit a
      un thread (condizioni della suite); non è garantito con BLAS a più thread o con kernel che dipendono dal numero
      di righe. La chiamata separata resta la garanzia per costruzione e la protegge il test 3; il test 6 verifica la
      proprietà.
    - *Per il passo 6:* (a) `orphan_tmp` distingue per `status` il cestino di un fit `failed` da una scrittura
      `complete` non installata; la pulizia elimina entrambi e non promuove mai un `tmp/` a `fits/<key>`; (b) un lock
      illeggibile (per esempio un processo ucciso tra la creazione con `O_EXCL` e la scrittura) non è mai considerato
      rimasto: `clean-tmp` deve dare un messaggio che spieghi come rimuoverlo a mano dopo aver verificato che nessun
      processo sia attivo.
    - *Funzioni di supporto aggiunte:* `residual_name`, `residual_aggs_for`, `npz_bytes`, `read_npz`, `ResidualSet`
      (`residual_cache.py`); `StoreLocked`, `read_lock`, `lock_path` (`store.py`).
  - **Passo 3 (2026-11-12):** `plan_fits`, `run_fit`, `meta.json`, `meta_hashable` in `fit.py`; `begin_run` con
    `run_id` facoltativo e `new_run_id` in `store.py` (nota su D-70); controllo del device effettivo prima di
    `write_fit` (nota su D-70). Test: `tests/test_f5_fit.py` (più tre test di `begin_run` in `tests/test_f5_store.py`).
    **Misura:** fit sintetico completo (16 fit, solo `run_fit`, esclusi import e riferimento storico) = 4,6 s, sotto il
    criterio di 1 minuto. `run_fit` rifiuta già `kind: official` (V6), oltre alla CLI del passo 6.
    **Regola:** identità dei residui = sha256 per array in `meta.json`; `npz_sha256` in `residuals.json` serve solo
    all'integrità del file sul disco e non si confronta tra scritture diverse. Verificato: con numpy 2.3.4 `npz_bytes`
    è deterministico anche con orologio diverso (voci zip a data fissa), fissato da
    `test_npz_bytes_deterministic_across_writes`; nessun punto del codice confronta `npz_sha256` tra due scritture.
  - **Passo 4 (2026-11-12):** in `fit.py` errori di modello isolati per `fit_key` (fasi `fit`, `device_check`,
    `residuals`, `save`, `serialize`; `meta.json` failed con tipo, messaggio, traceback e fase; `FitReport.counts`;
    colonna `phase` in `fit_summary.csv`); `KeyboardInterrupt` si propaga con registro `interrupted`; loss non
    finita → `complete` con `nonfinite_loss`. **Fuori da `fit.py`** (nota «F5 passo 4» su D-70, rimando su D-67):
    `FitInfo.to_dict`/`from_dict` in `src/models/base.py` serializzano i non finiti come `null`, con
    `nonfinite_fields` per i campi scalari (prima `save` falliva con `allow_nan=False`); invariato per valori finiti.
    **`run_fit` cambia firma:** riceve la specifica grezza o un percorso YAML e chiama `validate_experiment` prima di
    ogni scrittura (corpo privato `_run_fit`); i test dei passi 3 e 4 costruiscono le varianti sulla specifica grezza
    (`synth_spec` in `tests/fixtures/f5.py`). Lacuna di F2 da decidere a parte: `experiment.name` con `/` passa lo
    schema (test `xfail`). Test: `tests/test_f5_failures.py`, `tests/test_f5_leakage.py` (split3, legacy2 A4,
    `test_ctx_head`); il test del passo 3 su `DeviceMismatch` ora verifica l'isolamento.
  - **Passo 4-bis (2026-11-12):** `experiment.name` validato con la regola dei nomi di percorso dello store, in un
    solo punto (`schema.is_path_name`, riusata da `store._check_name`), rafforzata: `fullmatch`, niente punto finale,
    niente nomi riservati di Windows; vale anche per `run_id` e `stage`. Il test `xfail` su `name="a/b"` diventa
    normale; caso limite valido `f5-run-2026-11-12`; test della regola e di `new_run_id` in
    `tests/test_config_schema.py` (nota su D-70).
  - **Passo 5 (2026-11-12):** `get_residuals` e `regenerate` in `residual_cache.py`: lettura verificata (file e
    array contro `meta.json`, mai rigenerazione silenziosa); rigenerazione dal `fit_payload` registrato con `load`,
    senza `set_seed` né `fit`, hash sempre uguali a `meta.json`; lock con gettone esplicito («letture senza lock,
    scritture solo con il gettone del run»; campo `nonce` nel lock); firma con `lock`, `data_dir`, `datasets_dir`
    (scostamento registrato nella nota su D-70). `provenance.file_sha256` (CRLF → LF) è usato solo su file di testo:
    docstring e test che lo documentano. Test: `tests/test_f5_regenerate.py`.
  - **Passo 6 (2026-11-12):** `src/cli.py` con `fit`, `status` (sola lettura, `--verify`) e `clean-tmp`
    (`--dry-run`, `--break-stale-lock` annotato nel registro con un run di stage `clean-tmp`); codici 0/1/2/3/130
    (nota su D-70). Test: `tests/test_f5_cli.py`, con un test di fumo via `subprocess`.
- **Stato: completata (2026-11-12).** Commit: passo 0 `844f590` (registro D-70 e piano), 0-bis `3380c3d` (docstring
  di `materialize`), 1 `76c95d5` (`prepare_machine`, `fit_one`), 2 `b26cfda` (`compute_all`, aggiunte allo store),
  3 `37658d9` (`plan_fits`, `run_fit`, `meta.json`), 4 `38eca68` (fallimenti, interruzione, controlli bloccanti,
  anti-leakage, `test_ctx_head`), 4-bis `2d9138b` (regola dei nomi di percorso), 5 `a0c8a07` (`get_residuals`,
  `regenerate`, nonce del lock), 6 `aed44f2` (CLI), 7 chiusura (commit che contiene questa riga). Suite di chiusura:
  vedi il messaggio del commit del passo 7.
  - **Verificato (dati sintetici):**
    - parità `legacy2`: residui bit a bit uguali a quelli della pipeline storica calcolata dal vivo su tutte le 16
      chiavi (2 macchine × 6 modelli, seed 42 e 43) e al golden R1; stati fittati bit a bit uguali a quelli ottenuti
      con gli ingressi di `main.prepare_machine_data` per i 6 modelli con `train_step` 1 e 5 (machine-9-1, seed 42);
      stati identici tra run della nuova pipeline con ordine dei modelli invertito e tra rilanci;
    - ingressi di fit con lo stesso layout (passi, contiguità) di `main.py`, PCA comprese; stato dei generatori
      all'ingresso di `fit` uguale a un `set_seed` nuovo;
    - rilancio senza alcun fit, `set_seed` o `load`; fallimenti isolati per `fit_key` in ogni fase, rifatti al
      rilancio senza cestini; interruzione (durante il fit e dentro il writer) senza `fits/<key>` parziali, `tmp/` vuota,
      registro `interrupted`, lock rilasciato, e rilancio con gli stessi sha256 di un run mai interrotto;
    - rigenerazione dei residui identica a `meta.json` senza `set_seed` né `fit`, con un solo `load` per chiave e
      generatori invariati; integrità verificata in lettura, mai rigenerazione silenziosa;
    - anti-leakage `split3` (cal e test non raggiungono scaler, PCA, PCA puntuale, AE e residui di es; il test non
      raggiunge i residui di cal; es cambia solo la loss di early stopping) e `legacy2` (A4);
    - `test_ctx_head` solo in `split3` con `step` = 1, contesto = ultime W−1 righe di cal scalate, residui `test__*`
      identici a un run senza testa;
    - CLI: codici 0/1/2/3/130, `status` in sola lettura, `clean-tmp` sotto lock senza promozioni;
    - parametri di finestra dei record di `meta.json` (`n_timestamps`, W, `step`, `context`, `n_windows`): ricostruiscono
      `starts` ed `ends` identici ai `WindowSet` di scoring, `test_ctx_head` compreso, e `n_windows` coincide con la
      prima dimensione del residuo (C9, aggiunto alla chiusura);
    - durata del fit sintetico completo (16 fit, solo `run_fit`): 4,6 s, sotto il criterio di 1 minuto.
  - **Non verificato in F5:**
    - parità su dati reali (M1, la lancia l'utente);
    - `test_ctx_head` calcolato nella stessa chiamata del test con BLAS a più thread (la chiamata separata resta la
      garanzia per costruzione);
    - lock su POSIX: provato solo su Windows 10 (`O_EXCL`, run concorrente rifiutato, rilascio anche su eccezione,
      `is_stale` con un pid morto tramite `OpenProcess`/`GetExitCodeProcess`); il ramo POSIX di `_pid_alive`
      (`os.kill(pid, 0)`) non è mai stato eseguito;
    - `cuda`: non ammesso da `schema.DEVICES`; vincolo prima di ammetterlo nella nota «F5 passo 3» su D-70;
    - `clean_summary.csv` scritto parzialmente se `clean-tmp` si interrompe a metà: comportamento implementato, senza
      test.
  - **Suite:** durata osservata 72–128 s nella stessa giornata, per variabilità della macchina (nessuna modifica ai
    test tra le misure). `test_smoke_subprocess_fit_and_status` è marcato `slow`: la suite predefinita lo include;
    `python -m pytest -q -m "not slow"` fa risparmiare circa 11 s.
  - **Decisioni APERTE toccate e rimaste aperte:** D-57 (nessuna correzione del solver), D-69 (AE non addestrati),
    D-38, D-39, D-40, D-41 (valori storici o provvisori, nessuna scelta), D-44 (registrazione senza esclusioni).
  - **Passati a F6:**
    - `get_residuals` con il gettone del lock del run di analisi (letture senza lock, scritture solo con il gettone);
    - es solo con `allow_diagnostic=True`, mai in normalizer, soglie o metriche (D-04);
    - nomi `<segmento>__<agg>` e `table_convention` (none/native) del rolling z-score (nota su D-33);
    - `test_ctx_head` per la variante con contesto di RQ1a (D-20, D-32);
    - baseline random: `np.random.default_rng(seed)`, poi `random(T_cal)` e `random(T_test)` come `main.py:313-314`,
      soglia sulla maschera di cal (D-23);
    - `nonfinite_loss` e `n_nonfinite` da trattare senza esclusioni finché D-44 è aperta;
    - identità dei residui = sha256 per array in `meta.json` (`npz_sha256` solo per l'integrità del file).

### F6 — Stadio di analisi
- **Obiettivo:** prima riprodurre esattamente le metriche storiche dai residui, poi estendere.
- **File:**
  - nuovi: `src/pipeline/analyze.py`, `src/evaluation/metrics.py`, `src/evaluation/thresholds.py`, `src/utils/analysis_plan.py`, `src/pipeline/legacy_export.py` [D-72];
  - da modificare, solo con aggiunte: `src/evaluation/scoring.py` (normalizer robusto), `src/pipeline/table_schema.py` (tre colonne di `fits`), `src/pipeline/store.py` (cartella temporanea dell'analisi), `src/cli.py` (comando `analyze`, estensione di `clean-tmp`) [D-72];
  - invariati: `src/evaluation/rules.py` (resta fino a F12, la regola `burnin` non passa in `thresholds.py`), `src/evaluation/protocols.py`, `src/evaluation/protocol.py`, `main.py` [D-72 C1, C2]. *(rev. 2026-11-12: prima `rules.py` diventava un modulo di compatibilità che ri-esportava le funzioni.)*
- **Modifiche:**
  - **6a, parità:** analisi in modalità legacy → stessi `.npz` e stesso summary della pipeline storica;
  - **6b, estensioni:** AP normalizzata, normalizer robusto [D-11], metriche per evento, ritardo di rilevamento, falsi allarmi ogni 1.440 punti, celle 2×2 calcolate con la formula unica dell'F1 di `protocol._prf` (non da `protocols.py`, D-72 C2), secondo l'elenco fissato in D-34 e le righe per politica di soglia di D-72 (V10); soglie mobili con storia continua **cal → test** [D-04], fatta degli ultimi `n_past` score validi, senza aggiornamenti nei punti senza score [D-21], e applicate solo con `timestamp_agg: last` [D-24]; supporto di calibrazione in cal secondo D-23 (normalizer su tutte le finestre di cal, soglie statiche e baseline random su [W−1, T_cal−W+1), numero di punti di calibrazione registrato nella tabella lunga); griglia dei fattori; tabella lunga in CSV.gz [D-19]. VUS-PR secondo i tempi fissati in D-34.
- **Dipendenze:** F5.
- **Test:**
  - parità esatta con lo snapshot F1;
  - causalità end-to-end: perturbare il test dopo t lascia invariati gli score e le decisioni di soglia fino a t;
  - proprietà: l'oracolo è ≥ dell'F1 calibrato, il PA è ≥ del no-PA;
  - le soglie con storia coincidono con quelle senza storia quando la storia è vuota;
  - es [D-04, D-72 C9]: per i modelli che non usano `X_val`, perturbare es lascia invariati end-to-end normalizer, soglie e metriche; per gli AE in `split3` es sceglie l'epoca (atteso), quindi una spia verifica che l'analisi non legga mai i residui `es__*` *(rev. 2026-11-12: prima «perturbare il segmento es lascia invariati normalizer, soglie e metriche», falso per gli AE)*;
  - leakage [D-72 B13]: perturbare il test non cambia normalizer e soglie statiche; le soglie mobili restano invariate fino a t (test di causalità);
  - righe per politica di soglia secondo D-72 V10; proprietà A ≥ D, PA ≥ no-PA e B ≥ C a tolleranza zero [D-72 C2];
  - la storia delle soglie mobili conta solo score validi e salta i punti senza score [D-21]. L'attuale `rules.py` conta per righe: la nuova `thresholds.py` non deve farlo;
  - le politiche di soglia mobile rifiutano `timestamp_agg` offline [D-24];
  - il normalizer usa tutte le finestre di cal e le soglie statiche la maschera di cal; in `legacy2` coincidono con la pipeline storica [D-23];
  - le metriche nuove sono verificate su casi calcolati a mano.
- **Completata quando:** 6a esatta; 6b verde; la tabella lunga rispetta lo schema di F2.
- **Rischi:** numero di combinazioni della griglia (si stimano circa 15 minuti per l'esperimento ufficiale). Va misurato su dati sintetici e, se serve, parallelizzato per macchina.
- **Impatto sui risultati:** metriche storiche invariate; metriche nuove aggiunte.
- **Stato: completata (2026-11-16).** Commit: passo 0 `adfbcbf` (registro D-72 e piano), 1 `387b99e` (piano di
  analisi, `analysis_id`), 2 `1a8191f` (`analyze` con esportazione legacy, pubblicazione atomica), 3 `ed0d5f7`
  (metriche), 4 `94b1f40` (soglie), 5 `ef85d88` (normalizer robusto), 6.0 `3d19ce7` (non finiti), 6.1 `f27e92e`
  (colonne di `fits`), 6.2 `e2459d7` (nucleo di scoring), 6.3 `8a0225a` (righe `free`/`thr`, `nan_reason`), 6.4
  `33cec21` (tabella `fits`), 6.5 `b3c8a50` (integrazione in `analyze`), 6.6 `48c5499` (controlli incrociati), 7
  `435efb2` (CLI), 8 chiusura (commit che contiene questa riga). Suite di chiusura: vedi il messaggio del commit del
  passo 8.
  - **Verificato in F6 (dati sintetici):**
    - 6a: esportazione legacy dai residui dello store identica a tolleranza zero allo snapshot F1 (`last`/`mean`) e,
      dal vivo, a `main.run_experiment` con `timestamp_agg` `mean` e `max` (`test_f6_analyze_legacy.py`);
    - identità dell'analisi: componenti in chiaro, id provvisorio e definitivo sotto lock, rifiuto senza scritture,
      `meta_sha256` per elenco esplicito, tempi fuori dall'id (`test_f6_analysis_plan.py`, `test_f6_analyze_legacy.py`);
    - pubblicazione atomica, tentativi su `PermissionError`, `run_id` riservati, completezza V13
      (`test_f6_store_analysis.py`, `test_f6_analyze_legacy.py`);
    - metriche: casi a mano (ritardo, falsi allarmi, AP normalizzata, celle), uguaglianza bit a bit con
      `evaluate_timestamps`, A ≥ D, B ≥ C, PA ≥ no-PA a tolleranza zero (20.000 serie una tantum), scarto documentato
      con `protocols.four_cells` (`test_f6_metrics.py`);
    - soglie: storia cal → test dei soli validi, identità con `rules.py` a storia vuota (anche 99,5), differenza sul
      buco iniziale, C15, causalità (`test_f6_thresholds.py`);
    - normalizer robusto, conteggi di C5, invarianza empirica rispetto a 1,349 (`test_f6_robust_normalizer.py`);
    - non finiti: rifiutati in `metrics`/`thresholds` (6.0), controllati prima della proiezione con motivo uniforme
      (6.2), righe NaN con `nan_reason` contate, `inf` end-to-end (`test_f6_analysis_grid.py`, `test_f6_rows.py`,
      `test_f6_nan_reason.py`, `test_f6_analyze_tables.py`);
    - nucleo: varianti storiche bit a bit, contesto contro forza bruta, scenari, griglia (`test_f6_analysis_grid.py`);
      righe V10–V12 contate a mano (`test_f6_rows.py`); tabella `fits` (`test_f6_fits_table.py`);
    - integrazione: tabelle, `analysis.json`, letture (mai es), C13, C15, rigenerazione con contesto, B10 in processo
      separato (`test_f6_analyze_tables.py`);
    - tabella lunga = CSV per macchina dell'esportazione legacy per chiave, causalità con `last` sui residui reali,
      dipendenza offline di `mean`/`max` (D-24), leakage, C9 (`test_f6_crosscheck.py`);
    - CLI: codici 0/1/2/3/4/130, riepilogo delle righe NaN, `status` (analisi, orfane, V13), `clean-tmp`
      (`test_f6_cli_analyze.py`);
    - prova golden delle uscite legata ad `ANALYSIS_CODE_VERSION` = 1 (`test_f6_analyze_golden.py`, golden
      `tests/golden/analyze_synth.json`).
  - **Non verificato in F6:**
    - dati reali (M1, la lancia l'utente);
    - lock e rinomina su POSIX (provati solo su Windows 10);
    - test scritti dopo il codice e convalidati solo con prove di difetto: punti 1 (`meta_sha256` per elenco
      esplicito) e 2 (V13) della revisione del passo 2; «NaN nel supporto statico» del passo 6.0;
    - test del passo 2 modificati in 6.5, con il motivo (nessuno reso più permissivo): elenco nelle annotazioni sopra;
    - durata reale (solo misura sintetica e stima, 6.7).
  - **Suite:** 4–5,5 minuti a fine F6 (era circa 2 a inizio fase), cresciuta con i test di integrazione; i test di fumo
    e delle proprietà sono marcati `slow` e inclusi nella suite predefinita.
  - **Passati a M1:**
    - confronto con i run storici sui CSV per macchina o sui JSON delle metriche, non su
      `summary_timestamp_metrics.csv` (perde l'ultima cifra);
    - verifica con `n_nonfinite_cal` dello scostamento dichiarato su `raw_mean` (NaN nel supporto di cal);
    - rigenerare il lock da un ambiente dedicato (D-55) prima dei run di confronto.
  - **Passati a F7:**
    - D-44 da decidere prima di leggere RQ1b (nota del passo 5 su D-44);
    - filtrare prima per stato del fit, poi sui non finiti (colonne nullabili di `fits`);
    - `n_nonfinite_*` come indicatore (> 0), non come misura di gravità;
    - usare `nan_reason` per applicare D-44 riga per riga;
    - deduplicare le repliche della baseline random tra `ts_agg` (V12);
    - ritardo sempre insieme a `events_detected`/`event_recall`, con l'avvertenza sul confronto tra scenari (C4);
    - righe `diagnostic` (oracolo, PA) separate dalle metriche principali (D-34);
    - analisi complete = cartelle pubblicate `analyses/<analysis_id>/` con `analysis.json` (V13);
    - D-25, D-30, D-36 aperte.
  - **Passati a F8:** D-69, D-71; correzioni dichiarate (conteggio per righe di `rules.py`, esclusione silenziosa dei
    NaN su `raw_mean`, validation usata tre volte P1), da tenere separate dai fattori di protocollo da decomporre.
  - **Passati a F9:** nota sulla durata dell'analisi e sulle ottimizzazioni (sezione F9).
  - **Passati a F10 (non nell'elenco di chiusura richiesto, segnalati):** VUS-PR (D-34, esclusa da F6);
    `kind: official` di `analyze` rifiutato fino a F10, come per `fit`.
  - **Decisioni APERTE toccate e rimaste aperte:** D-42 (livelli = dati del piano, nessun default), D-44 (nessuna
    esclusione; colonne e `nan_reason` per applicarla), D-69, D-71; non toccate D-25, D-30, D-36, D-37, D-38, D-39,
    D-40, D-41, D-57.
- **Nota (2026-11-12, piano F6 approvato):** decisione D-72 (perimetro, C1–C15, V10–V13, B10–B13) e note di rimando su
  D-11, D-34, D-42, D-44, D-59, D-69; nuova voce APERTA D-71.
  - **Perimetro:** 6a parità esatta con `main.py` dai residui dello store (esportazione legacy nello stesso formato,
    confrontata con lo snapshot F1 a tolleranza zero e, per `timestamp_agg` `mean` e `max`, con `main.py` dal vivo); 6b
    metriche di D-34 senza VUS-PR, soglie, normalizer robusto, scenari, baseline random, tabelle `free` / `thr` / `fits`,
    `analyze`; `kind: official` rifiutato fino a F10. Fuori: VUS-PR, F7, file di M1, parallelizzazione.
  - **Flusso di `analyze`:** validazione di esperimento e piano → controlli preliminari bloccanti (manifest, segmenti,
    ruolo delle macchine, coerenza piano–modalità, fit presenti, `n_past`, `labels_version`) → lock e `begin_run`
    (stage `analyze`, `plan_sha256`) → per macchina label e maschere; per fit residui `cal`, `test` e, se serve,
    `test_ctx_head` (mai `es`), normalizer su tutte le finestre di cal, score, proiezione (con contesto: finestre di
    `test_ctx_head` e del test proiettate sulla serie estesa, poi tolti i W−1 punti di contesto), metriche e soglie;
    baseline random per seed → tabelle, esportazione legacy facoltativa e `analysis.json` (per ultimo) nella cartella
    temporanea del run, poi rinomina atomica in `runs/<kind>/<esperimento>/analyses/<analysis_id>/` → `finish_run`.
  - **Identità dell'analisi [D-72 B12, rev. passo 0 e passo 1]:** `analysis_id` = sha256 canonico di `plan_sha256`,
    `ANALYSIS_CODE_VERSION`, per ogni fit usato (ordinati per `fit_key`) `fit_key`, `status`, `meta_sha256` e sha256
    per array dei residui letti (fit non `failed`), `labels_version` per macchina, hash del manifest, pseudo-chiavi
    della baseline random e ambiente dell'analisi, troncato a 12; componenti in chiaro in `analysis.json`. Nessun residuo si legge per calcolare l'id (gli sha256 sono in `meta.json`): id
    provvisorio senza lock con rifiuto anticipato, id definitivo ricalcolato sotto lock, ogni array letto verificato da
    `get_residuals` contro lo stesso sha256. Id già presente e completo: codice 2, messaggio con la cartella esistente.
  - **Rinomina su Windows [D-72 B11, rev. passo 0]:** `os.rename` solo se la destinazione non esiste; su
    `PermissionError` al massimo `PUBLISH_RENAME_ATTEMPTS` = 5 tentativi con attese crescenti `PUBLISH_RENAME_WAITS_S`
    = (0,2, 0,4, 0,8, 1,6) s (fissate al passo 2), poi errore esplicito con codice di uscita 4 «pubblicazione non
    riuscita, nessun risultato» (run chiuso `failed`, cartella temporanea lasciata per `clean-tmp`; la CLI e i suoi
    test lo coprono al passo 7); mai copia di ripiego, mai risultati parziali nella cartella definitiva; test con rinomina che fallisce in
    modo simulato sempre e solo al primo tentativo.
  - **Copertura dei rischi passati da F5:** gettone del lock (rigenerazione sotto lock, gettone estraneo rifiutato);
    es mai letto (spia); convenzione none/native del rolling z-score (score nativo = `score_anomalies` bit a bit);
    `test_ctx_head` (proiezione con contesto contro forza bruta); baseline random (parità con lo snapshot, M_cal in
    `split3`); non finiti (righe NaN e colonne di `fits`, nessuna esclusione); identità dei residui = sha256 per array
    di `meta.json`, registrati in `analysis.json` (mai `npz_sha256`).
  - **Passi:** 0 registro e documentazione → 1 `analysis_plan.py` (schema, `plan_sha256`, `ANALYSIS_CODE_VERSION`,
    `analysis_id`, coerenza) → 2 6a: `analyze.py` (controlli, lock, residui, scoring storico), `legacy_export.py`,
    cartella temporanea del run e rinomina in `analyses/<analysis_id>/` (con i test di B11); parità con lo snapshot F1
    e dal vivo (`mean`, `max`); spia su es; test sull'import di `main.py` in un processo separato (B10) → 3 `metrics.py` (C2–C4, V10; test di B ≥ C su casi casuali con pari
    merito) → 4 `thresholds.py` (C1, C15; equivalenza con `rules.py` a storia vuota) → 5 normalizer robusto (C5) →
    6 6b: griglia completa, tre tabelle (C6, V11, V12), `split3`, `test_context`, NaN, fit falliti, `labels_version`,
    rigenerazione, causalità e leakage end-to-end, tabella lunga = summary legacy sulle metriche in comune, misura
    della durata → 7 CLI `analyze` (codici di uscita 0/1/2/3/4/130 di D-72) ed estensione di `clean-tmp` → 8 chiusura. Dopo ogni passo: snapshot F1, golden T0 e
    R1, poi `python -m pytest -q`.
  - **Completezza dell'analisi [D-72 V13, rev. passo 2]:** un'analisi è completa se e solo se esiste la cartella
    pubblicata `analyses/<analysis_id>/` con `analysis.json`; F7 legge solo quelle, qualunque sia lo stato del run. Se
    dopo la pubblicazione `run.json` o il registro non si aggiornano: avviso nel log, `status` segnala l'incongruenza
    (passo 7), codice di uscita 0. `analyses` è un `run_id` riservato.
  - **Identità per elenco esplicito [D-72 B12, rev. passo 2]:** `meta_sha256` usa solo `META_ID_FIELDS`; i tempi
    (`META_MEASURED_FIELDS`, colonne dei tempi di `fits`) sono misure non riproducibili, fuori dal significato dell'id; un
    campo di `meta.json` non classificato è un errore (codice 2).
  - **Annotazioni per i passi successivi (approvate al passo 2):**
    - passo 6: il test B10 «`split3` senza esportazione legacy» deve dimostrare di aver letto i residui (numero di
      chiamate a `get_residuals` > 0), non solo che `main.py` non è importato (al passo 2 il piano senza
      `legacy_export` è rifiutato e il test è banale);
    - passo 6: il controllo incrociato tabella lunga = summary legacy gestisce in modo esplicito un deterministico
      riusato con `seeds[0]` diverso (seed legacy = seed del ciclo di `main.py`, colonna `seed` = `set_seed_value`):
      confronto per chiave, nessuna tolleranza;
    - passo 6 (annotazione del passo 4): un test dimostra che `analyze` non passa mai `require_history=False` alle
      funzioni di `thresholds.py` (ammesso solo nei test di identità con `rules.py` e per lo z di cal);
    - passo 6 (annotazione del passo 5): test end-to-end con un `inf` nei residui di cal e uno nei residui del test,
      per soglia statica, soglie mobili e metriche senza soglia di scikit-learn, con il comportamento atteso esplicito
      (riga NaN contata, oppure errore): mai numeri finiti in silenzio (D-44);
    - **sotto-piano del passo 6 (approvato il 2026-11-13):** 6.0 non finiti rifiutati in `metrics.py` e
      `thresholds.py` (commit a sé) → 6.1 colonne di `fits` (C6) → 6.2 nucleo di scoring per fit in
      `src/pipeline/analysis_grid.py` (normalizer, aggregazione, proiezione, `test_context`, baseline random) → 6.3
      righe `free` / `thr` (V10–V12, C12) → 6.4 tabella `fits` → 6.5 integrazione in `analyze` (C13, C15,
      `analysis.json`, B10) → 6.6 controlli incrociati, causalità, leakage, C9 → 6.7 durata. Riepilogo e stop prima
      di ogni commit;
    - passo 6.6 (aggiunta approvata): con `ts_agg` `last`, perturbare dopo t lascia invariati score, soglie e
      decisioni fino a t, per ogni politica e scenario; con `mean` e `max` un test dimostra la dipendenza offline
      (perturbare dopo t cambia lo score solo da t − W + 1 in poi, mai prima), così l'etichetta «offline» di D-24 è
      verificata e non data per scontata;
    - passo 6.5 (revisione di 6.2): `analysis.json` registra `TOP5_K` con le costanti di C14; i motivi delle righe
      NaN si contano per causa (`ResidualNonFiniteError.reason`, uguale per ogni `ts_agg`: gli score di finestra si
      controllano prima della proiezione);
    - passo 8: prova golden (sha256 delle uscite di un'analisi sintetica) legata ad `ANALYSIS_CODE_VERSION`, che
      fallisce se le uscite cambiano senza aumentare la versione;
    - passo 8, riepilogo di chiusura di F6: annotare che i test di integrazione dei punti 1 (`meta_sha256` per
      elenco esplicito) e 2 (V13, errore del registro dopo la pubblicazione) della revisione del passo 2 sono stati
      scritti dopo il codice e convalidati solo con le prove di difetto (mai visti rossi prima dell'implementazione);
      lo stesso vale per il test «NaN nel supporto statico» del passo 6.0 (convalidato solo con la mutazione);
    - chiusura di F6: test del passo 2 modificati in 6.5, nessuno reso più permissivo:
      - `test_mean_export_reads_the_mean_residuals_except_for_native_models`: i residui letti (e nell'id) comprendono
        ora anche `last`, usato dalle tabelle; l'asserzione resta un'uguaglianza esatta di insiemi;
      - `test_refusals_happen_before_any_write`: tolto il caso «piano senza `legacy_export` rifiutato», perché il rifiuto
        era provvisorio (passo 2); gli altri quattro rifiuti restano; nuovo
        `test_legacy2_plan_without_legacy_export_produces_only_tables` per il comportamento definitivo;
      - `test_main_is_never_imported_without_legacy_export`: sostituito da
        `test_split3_without_legacy_export_reads_residuals_and_never_imports_main` (`test_f6_analyze_tables.py`), più
        forte: oltre a `main.py` mai importato verifica che l'analisi legga davvero i residui (chiamate > 0);
    - passo 6.2 (requisito approvato al passo 6.0): `analysis_grid` verifica che gli score NaN di cal e del test
      stiano solo dove la geometria delle finestre non dà score (primi W − 1 punti senza contesto, coda); un NaN
      altrove (da residuo non finito) è un errore → riga NaN contata, mai un punto saltato nella storia delle soglie
      mobili. Test: residuo NaN in cal con `roll_quantile` e `roll_robust_z`, e nel test, con l'esito atteso
      esplicito; prova di difetto «NaN da residuo trattato come punto senza score».
  - **Precisazioni approvate al passo 1 [D-72 C8]:** al passo 6 `analysis.json` registra quali modelli producono
    righe native (convenzione none/native applicata dall'analisi, non dichiarata nel piano); al passo 4 il test
    «storia vuota = `rules.py`» copre la conversione percentile → quantile, anche con livelli non interi (es. 99,5).
  - **Decisioni APERTE toccate, stato invariato:** D-42 (livelli = dati del piano), D-44 (nessuna esclusione), D-69,
    D-71; non toccate D-25, D-30, D-36, D-37, D-38, D-39, D-40, D-41, D-57.

### F7 — Statistica e report
- **Obiettivo:** RQ1–RQ4 calcolate dalla tabella lunga.
- **Nota (2026-11-13, da F6 passo 6.1):** nella tabella `fits` le colonne dei non finiti (`nonfinite_loss`,
  `n_nonfinite_cal`, `n_nonfinite_test`) sono nullabili (null per i fit `failed`, D-72 C6): filtrare prima sullo stato
  del fit, poi sui non finiti (pandas non ammette maschere booleane con valori nulli).
- **File:**
  - nuovi: `src/stats/{plan,inputs,aggregate,bootstrap,decisions,ranking,variance,descriptors,run}.py`,
    `src/reporting/{tables,figures,report}.py`, piani statistici `configs/stats/<nome>.yaml` [D-73];
  - da modificare, solo con aggiunte: `src/pipeline/store.py` (cartella delle statistiche, pubblicazione), `src/cli.py`
    (comandi `stats` e `report`, `status`) [D-73];
  - invariati: `aggregate_results.py`, `main.py`, `src/pipeline/table_schema.py`, `analyze`.
  *(rev. 2026-11-16: prima solo `aggregate`, `bootstrap`, `decisions`, `ranking`, `variance`, `descriptors` e
  `reporting/{tables,figures}`.)*
- **Dipendenze:** solo lo schema di F2 (per i test su dati simulati); F6 per l'uso reale. Metodo degli IC: D-36
  (approvata); decomposizione: D-25 (approvata). δ è un parametro senza valore predefinito [D-30]; D-44 è un blocco
  obbligatorio del piano statistico, con valori aperti. *(rev. 2026-11-16: prima «D-36 va deciso prima di F7, oppure F7
  implementa entrambi i metodi».)*
- **Test:**
  - copertura del bootstrap stratificato BCa [D-36] su simulazioni (marcato `slow`), con le bande fissate in D-36
    (normale [0,93; 0,97]; code pesanti e asimmetrico ≥ 0,90, valore documentato); il gerarchico solo per documentarne
    la copertura;
  - regola di decisione su δ con casi noti, confini compresi;
  - livello di Bonferroni 1 − α/m su casi noti (m da 1 a 3) *(rev. 2026-11-17: prima «Bonferroni e Holm su casi noti
    (Holm implementato come funzione pura, non usato dalla regola di RQ2)», e prima ancora solo «correzione di Holm».
    Holm non si implementa: D-36 l'ha scartato (M2 non adottato) e la nota su D-71 lo conferma per m = 3)*;
  - la decomposizione SS somma al totale; effetti noti recuperati; disegno sbilanciato rifiutato;
  - τ di Kendall su casi noti (τ-b, pari merito, uguaglianza con scipy);
  - deduplicazione della baseline random, filtro di D-44 nell'ordine fissato, righe `diagnostic` separate, ingressi
    secondo D-73 P6;
  - **parità:** media ± std e confronti appaiati coincidono con `aggregate_results.py` secondo i tre livelli di D-73 P12.
- **Completata quando:** test verdi e report generato dalla tabella sintetica.
- **Rischi:** troppi output. Il report produce solo ciò che il piano statistico richiede.
- **Impatto sui risultati:** nessuno.
- **Nota (2026-11-16, piano F7 approvato):** decisioni D-25, D-36 e D-73 (P1–P15); note di rimando su D-24, D-30, D-42,
  D-43, D-44, D-71.
  - **Flusso di `stats`:** validazione del piano statistico → ingressi da CLI controllati contro gli esperimenti e i
    `plan_sha256` dichiarati nel piano (codice 2 se non dichiarato, diverso o mancante) → solo analisi complete (V13)
    → lettura con `read_csv_gz` → coerenza tra ingressi (dataset, manifest, `labels_version`, ruolo) → `stats_id`
    provvisorio, rifiuto se presente → lock e `begin_run`, `stats_id` definitivo → D-44 nell'ordine stato del fit →
    non finiti del fit → `nan_reason` → eventi (`primary`) → righe `diagnostic` separate → deduplicazione del random →
    media sui seed (`skipna=False`, `n_seeds_used`) → solo le RQ richieste dal piano → tabelle e `stats.json`
    (esclusioni per regola, motivo e modello; macchine per blocco; B, seed, metodi) nella cartella temporanea →
    rinomina atomica in `store/stats/<stats_id>/` → registro.
  - **Passi:** 0 registro e documentazione → 1 `plan.py` → 2 `inputs.py` (ingressi, coerenza, `stats_id`) → 3 D-44,
    deduplicazione del random, `diagnostic` (test di invariante prima) → 4 `aggregate.py` e parità con
    `aggregate_results.py` → 5 `decisions.py` → 6 `bootstrap.py` e simulazioni di copertura → 7 RQ2 → 8 `ranking.py`
    e `variance.py`, RQ1a/RQ1b → 9 `descriptors.py`, RQ3 e RQ4 → 10 `run.py`, `store.py`, comando `stats`, `status` →
    11 comando `report` → 12 integrazione sintetica, prova golden legata a `STATS_CODE_VERSION`, chiusura. Dopo ogni
    passo: snapshot F1, golden T0, R1 e `analyze`, poi `python -m pytest -q`; riepilogo e stop prima di ogni commit.
  - **Bande di copertura (D-36):** fissate ora; se il caso normale esce dalla banda ci si ferma e lo si riporta,
    senza correggere test o metodo senza una voce di registro.
  - **Da fare fuori da F7 (D-73 P10):** controllo E5 sulla deriva, prima della lettura di RQ4.
  - **Decisioni APERTE toccate, stato invariato:** D-30, D-37, D-42, D-43, D-44, D-71 (parametri dichiarati, nessun
    default). *(rev. 2026-11-18, chiusura: D-71 è stata approvata al passo 0-bis; elenchi aggiornati sotto.)*
- **Stato: completata (2026-11-18).** Commit:
  - passi 0 `a637ced` (registro e documentazione), 0-bis `48a691e` (D-45, D-69, D-71), 1 `30e815d` (`plan.py`),
    1-bis `94e56b8` (baseline descrittive, livello degli IC), nota D-69 `71a4115`, 1-ter `2959201` (esperimento
    primario, livelli di preproc);
  - passi 2 `5887ce3` (`inputs.py`), 3 `2b74709` (D-44, deduplicazione, `diagnostic`), 4 `69831a0` (media sui seed,
    parità), 5 `11aa88b` (quattro esiti), 6a `8227813`, 6 `bfdce3b` (bootstrap-t, D-76), 7 `0153222` (RQ2),
    8 `f69fd35` (varianza, τ-b, RQ1), 9 `865120f` (RQ3, RQ4);
  - passi 10 `6bd21ad` (`run.py`, `stats`), 11 `ca8d8c0` (`report`), 11-bis `c3d75b1` (correzioni dal report
    d'esempio), 12 A+B `1334aef` (golden, prova completa), 12 C chiusura (commit che contiene questa riga).

  Suite di chiusura: vedi il messaggio del commit di chiusura.
  - **Corrispondenza tra test previsti e file:**
    - copertura del bootstrap: `test_f7_bootstrap.py` (bande del bootstrap-t stratificato; «BCa» superato da D-76,
      che ha rivisto D-36 A dopo il fallimento delle bande);
    - regola su δ con i confini e livello 1 − α/m: `test_f7_decisions.py`, `test_f7_plan.py`;
    - decomposizione SS (somma al totale, effetti noti, disegno sbilanciato rifiutato): `test_f7_variance.py`;
    - τ-b, pari merito, uguaglianza con scipy, inversioni: `test_f7_ranking.py`;
    - D-44 nell'ordine fissato, deduplicazione del random, righe `diagnostic`: `test_f7_filters.py`;
    - ingressi secondo D-73 P6, `stats_id`: `test_f7_inputs.py`;
    - parità con `aggregate_results.py` a tre livelli (P12), media sui seed, descrittive: `test_f7_aggregate.py`;
    - piano statistico, chiavi, hash, nomi brevi: `test_f7_plan.py`;
    - assemblaggio di RQ1a/RQ1b, RQ2, RQ3, RQ4: `test_f7_rq1.py`, `test_f7_rq2.py`, `test_f7_rq3.py`,
      `test_f7_rq4.py`;
    - descrittori e ρ_W con IC di Fisher–Bonett–Wright: `test_f7_descriptors.py`; coerenza con lo script di evidenza:
      `test_f7_spearman_sim.py`;
    - stadio completo, determinismo, sensibilità P8, `n_exclude_fit_cases`: `test_f7_run.py`;
    - cartella delle statistiche nello store, pubblicazione, percorsi lunghi: `test_f7_store_stats.py`;
    - comando `stats`, codici di uscita, `status`, `clean-tmp`, flussi della CLI: `test_f7_cli_stats.py`;
    - nessuna operazione BLAS in `src/stats`: `test_f7_no_blas.py`;
    - vocabolari degli stati (`DELAY_STATUSES`, `FIT_RULES`, `TAU_STATUSES`, `R_STATUSES`, `ETA2_STATUSES`,
      `FORMATS`): valori nelle uscite dentro l'elenco e stati scritti tramite le costanti (`test_f7_status_vocab.py`);
    - integrazione fit → analyze → stats: `test_f7_integration.py`;
    - report: `test_f7_report_inputs.py`, `test_f7_report_tables.py`, `test_f7_report_notices.py`,
      `test_f7_report_no_stats.py`, `test_f7_report_figures.py`, `test_f7_cli_report.py`;
    - «report generato dalla tabella sintetica» e golden: `test_f7_cli_e2e.py`, `test_f7_golden.py`.
  - **Scostamenti dal piano:**
    - nessun piano in `configs/stats/` in F7: un piano reale richiede δ (D-30) e i valori di D-44, senza default;
    - file in più rispetto a D-73 P15: `src/stats/{rq1,rq2,rq3,rq4,run}.py`, `src/pipeline/fit_plan.py` (parti di
      `fit.py` senza torch), `src/reporting/{source,notices}.py`.
  - **Non verificato in F7:**
    - dati reali;
    - lock e rinomina su POSIX (provati solo su Windows 10);
    - durata su 22 macchine (solo misura sintetica);
    - impaginazione LaTeX (rimandata al formato della tesi);
    - prova completa da riga di comando solo in `legacy2` (vedi «Passati a F8»);
    - test scritti dopo il codice: nei passi 1–4 i test sono stati scritti insieme al codice (la regola «test
      prima» vale dal passo 5, su indicazione dell'utente). Dal passo 5 in poi, sono stati scritti o rinforzati dopo
      il codice i test per le prove di difetto non rilevate al primo tentativo, con il motivo nelle note dei passi:
      - passi 3–9: condizione `pa` ridondante (rimossa, poi ripristinata come controllo), test del filtro dei
        descrittori diviso in due, test di m dalle righe reso esplicito, messaggi di RQ3;
      - passo 11: P8 in `applicable`, `matplotlibrc` dell'utente;
      - passo 11, aggiunte: percorso pubblicato, nomi degli esperimenti;
      - passo 12: `--write-golden`.

      L'elenco è ricostruito dai riepiloghi dei passi; dove la ricostruzione è incerta lo dicono le note;
    - un fallimento non identificato: una volta, 1 test su 147 nell'insieme snapshot F1 + golden, durante il
      passo 12, con il nome del test non registrato. Non è ricomparso in 3 esecuzioni manuali, in 10 esecuzioni
      con log completo (153 su 153) e in 2 suite complete. Ipotesi, non verificata: un test con sottoprocesso
      sensibile al carico.
  - **Suite:** da circa 264 s a inizio F7 a circa 480 s a fine F7 (misure tra 427 e 588 s secondo il carico). Una
    misura isolata di 929 s, durante la verifica di chiusura, non si è ripetuta: la suite successiva, con
    `--durations=20`, ha impiegato 480 s, e i test più lenti erano quelli attesi (prova completa, ambiente del
    report, `test_f6_metrics`, integrazione). Cause principali:
    - simulazioni di copertura del bootstrap (`slow`);
    - ambienti sintetici costruiti con fit e analisi veri (integrazione, report, golden);
    - prova completa da riga di comando, con sottoprocessi.

    In F8–F9 valutare se separare i test lenti dalla suite predefinita.
  - **Passati ad altre fasi:**
    - **F8:**
      - un esperimento `split3` scritto su file non si carica: `base.yaml` è `legacy2` con `split.val_ratio`,
        l'unione profonda dei livelli non toglie chiavi e lo schema di `split3` la rifiuta. È un prerequisito per
        `test_context` e per le analisi esplorative; la prova completa da riga di comando copre solo `legacy2`, e
        un test documenta il limite;
      - modello di previsione (D-71, D-74);
      - *(aggiunto il 2026-11-19, da M1, decisione dell'utente)* correzione dichiarata del docstring di
        `src/pipeline/legacy_export.py`: cita l'handler di `charset_normalizer`, pacchetto che non è più nell'ambiente
        dedicato (D-55). Nessun effetto sui numeri; in M1 il file non si tocca;
      - *(aggiunto il 2026-11-19, da M1, decisione dell'utente)* allineare la golden di F6
        (`tests/test_f6_analyze_golden.py`) alla regola di F7: rigenerazione per cambio di ambiente solo se cambiano
        soltanto gli identificativi che ne derivano (oggi la ammette senza questo controllo). Vedi la nota su D-73.
    - **M1** (da F6, ancora validi):
      - rigenerare il lock da un ambiente dedicato (D-55);
      - confronto con i run storici sui CSV per macchina o sui JSON delle metriche, non sul summary;
      - verifica con `n_nonfinite_cal` dello scostamento dichiarato su `raw_mean`.
    - **M1 e run ufficiali (F11):** le tabelle diagnostiche (PCA) dipendono nelle ultime cifre dal numero di thread
      BLAS. La golden usa `single_thread`; per M1 e F11 va fissato un thread, o registrato il numero, perché
      l'identità byte per byte valga.
    - **F10:**
      - `kind: official` e `configs/stats/official.yaml`;
      - VUS-PR (D-34);
      - proposta di Claude Project, approvata come voce da trattare: rifiutare i percorsi assoluti negli esperimenti
        `official` (oggi lo schema accetta `paths.store_root` assoluto; con percorsi relativi `config_sha256` non
        cambia spostando il repository, verificato).
    - **Prima dei run ufficiali:**
      - D-30 (δ) e i valori di D-44. Vincolo da ricordare: con il minimo di 5 macchine per gruppo (D-76), il gruppo
        da 6 macchine può perderne una sola;
      - controllo E5 sulla deriva prima di leggere RQ4 (D-73 P10).
    - **Formato della tesi:** impaginazione LaTeX delle tabelle larghe e delle etichette lunghe.
    - **Proposte di Claude Project non ancora decise dall'utente:**
      - PCA a finestre come riferimento di RQ2 (D-37);
      - budget di tuning pari anche per PCA puntuale, `input_norm` e `rolling_zscore` (D-41);
      - confronti esplorativi DL contro `input_norm` e DL contro `rolling_zscore` (RQ2-bis);
      - ruoli dei modelli in RQ1 e RQ4;
      - proposta su D-69 (AE non addestrati con zero epoche).
  - **Decisioni approvate durante F7:** D-25, D-36, D-45, D-69 (dettagli aperti, proposta registrata in nota), D-71,
    D-73, D-76.
  - **Decisioni toccate e rimaste aperte:** D-30, D-37, D-41, D-42, D-43, D-44, D-74, D-75.

### M1 — Parità su dati reali (validazione metodologica, la lanci tu)
- **Obiettivo:** dimostrare che la nuova pipeline in modalità `legacy2` riproduce i run storici di riferimento identificati in D-50.
- **File (nuovi):** `configs/experiments/parity_repro_{last,mean}.yaml`, `scripts/parity_report.py` (confronta `store/` con `results/` in sola lettura).
- **Vincolo sui seed (nota 2026-11-10, da F3):** le config `parity_repro_*.yaml` hanno i seed nell'ordine storico, con 42 per primo, e un'asserzione (test o controllo all'avvio di M1) lo verifica. Motivo: in `main.py` la PCA gira solo al primo seed con `random_state` = `seeds[0]`, mentre il nuovo schema fissa `random_state: 42` con un fit condiviso tra i seed. L'equivalenza vale solo se il primo seed è 42; altrimenti, con D-57, una PCA diversa sembrerebbe una divergenza di thread. Test di riferimento: `test_pca_random_state_and_sharing_across_seeds`. Provenienza dei dati: vedi la nota di rimando su D-61.
- **Confronto con i run storici (nota 2026-11-16, da F6 passo 6.6):** il confronto sui dati reali si fa sui CSV per
  macchina (`per_machine/<macchina>.csv`) o sui JSON delle metriche, non su `summary_timestamp_metrics.csv`:
  `main.run_experiment` rilegge i CSV per macchina con il parser predefinito di `pd.read_csv` e può perdere l'ultima
  cifra (fatto verificato: AP 0,06751194113624537 → 0,0675119411362453). I CSV letti con
  `float_precision="round_trip"` e i JSON sono esatti.
- **Punti passati a M1 da F6 (nota 2026-11-13):** verificare con `n_nonfinite_cal` (tabella `fits`) se sui dati reali
  si presenta un residuo non finito in cal; se sì, la variante `raw_mean` dello storico aveva una soglia finita
  (NaN scartato in silenzio) e la tabella lunga una riga NaN (D-72 C15, passo 6.0): differenza attesa e dichiarata.
- **Esecuzione:** `cli fit` + `cli analyze` su `machine-1-1` e `machine-3-8`. Stima: pochi minuti.
- **Completata quando:** PCA e baseline con score identici; AE identici bit a bit oppure con differenze diagnosticate, documentate e coperte da una tolleranza motivata [D-17]; metriche identiche; report salvato in `docs/`.
- **Procedura in caso di differenze [D-17]:** prima si verifica il numero di thread (registrato nella provenance) e le versioni delle librerie rispetto ai run storici; poi si isola il primo artefatto che diverge (finestre, scaler, residui, score). Non si alzano le tolleranze senza spiegazione.
- **Impatto sui risultati:** nessuno. **Da qui la pipeline storica non serve più come oracolo**, ma resta disponibile fino a F12.
- **Nota (2026-11-19, piano di M1 approvato dall'utente; criteri fissati prima dei run, I2).** Decisioni D1–D6 e
  integrazioni I1–I4 dell'utente, su raccomandazione di Claude Project; note di rimando su D-17, D-18, D-55.
  - **Run di riferimento, descritti con precisione:** `results_repro_last_01` = aggregazione nella finestra `last`,
    proiezione `last`; `results_repro_mean_01` = aggregazione nella finestra `mean`, proiezione `last` (in entrambe le
    `repro_*.yaml` `timestamp_agg` è `last`); soglia al 98° percentile della validation, seed 42, `machine-1-1` e
    `machine-3-8` (macchine di sviluppo, D-51).
  - **Perimetro:** controllo di parità con il protocollo storico (`legacy2`). Nessun comando `stats` né `report`,
    nessun esperimento `split3`, nessuna statistica del nuovo protocollo sui dati reali prima di F10. Gli errori storici
    non si correggono in M1 (restano a F8 come cambiamenti dichiarati).
  - **Scostamenti dal testo sopra:**
    - un solo esperimento, `configs/experiments/parity_repro.yaml`, al posto di `parity_repro_{last,mean}.yaml` (D3):
      le due config storiche differiscono solo per l'aggregazione nella finestra, che è un livello dell'analisi
      (D-33), quindi i fit sono gli stessi;
    - due piani di analisi, `configs/analysis/parity_last.yaml` e `parity_mean.yaml`, con la struttura di
      `legacy_plan` di `tests/test_f6_crosscheck.py` (i piani di analisi sono arrivati con F6, dopo il testo sopra);
    - riesecuzione di `main.py` (D2) con `configs/config_tests/m1_rerun_{last,mean}.yaml` (copie delle `repro_*`
      salvo le cartelle di uscita, sotto `reports_local/m1/`, esclusa da git); le config esistenti non cambiano;
    - ambiente dedicato fuori dal repository e lock rigenerato (D-55), con il criterio di confronto fissato prima
      (nota su D-55);
    - report finale in `docs/m1_parita.md`.
  - **Confronti (tolleranza zero, D-17), con `scripts/parity_report.py`:** storico contro nuovo, storico contro
    riesecuzione, riesecuzione contro nuovo, per ciascuno dei due piani:
    - insieme dei file: `metrics/<m>/*_metrics.json`, `scores/<m>/*_scores.npz`, `per_machine/<m>.csv`,
      presenza di `failures.csv`;
    - JSON delle metriche: tutti i campi, stesso tipo e stesso valore (NaN = NaN);
    - CSV per macchina letti con `float_precision="round_trip"`, righe abbinate per (Machine, Model, Variant, Seed),
      stesse colonne nello stesso ordine, stesso dtype, valori identici; `Epochs` si confronta;
    - `.npz` per array: stesse chiavi, dtype, forma e valori.
  - **Esclusioni dai confronti (elenco esatto, passo 0.5):**
    - colonne `FitSec` e `InferSec` dei CSV per macchina (durate);
    - nessun campo dei JSON delle metriche (nessuno dipende dal run);
    - byte dei file `.npz` (lo zip può contenere date): si confrontano gli array; esclusione di formato, non di
      contenuto;
    - `summary_timestamp_metrics.csv` (riletto da `main.run_experiment` con il parser predefinito, nota sopra) e le
      cartelle vuote dei grafici (`save_plots` falso).
  - **Tabella lunga contro export legacy (I1):** criterio di
    `tests/test_f6_crosscheck.py::test_long_table_equals_legacy_summary`, ripreso senza cambiarlo: 12 colonne (AP,
    ROC_AUC, BestF1_oracle, Prevalence, F1, Precision, Recall, FPR_at_thr, PA_F1, EventRecall, FalseAlarmSeg,
    N_events), abbinamento per (macchina, modello, `sensor_agg`, `normalizer`, metrica; per `thr` anche
    `threshold_policy = static_pct` e `pa`) con una sola riga, tolleranza zero, righe con `nan_reason` escluse e contate
    contro quelle dichiarate. Un NaN in entrambe le tabelle senza `nan_reason` → ci si ferma e lo si riporta; il
    criterio non cambia. Motivo: l'export riusa `main.evaluate_variant`, quindi storico contro export verifica fit,
    residui e scoring ma non il codice delle metriche di F6, che userà F11.
  - **Non finiti:** caso dichiarato (D-72 C15) = righe `raw_mean` con `nan_reason = nonfinite_residual_cal` di un fit
    con `n_nonfinite_cal` > 0, contate e riportate; qualunque altro `nan_reason`, residui non finiti nel test, loss non
    finita o fit `failed` sono differenze. Se `n_nonfinite_cal` = 0 ovunque, il punto si chiude come «assente su 1-1
    e 3-8» e resta da controllare in F8.
  - **Provenance:** `seeds[0]` = 42; nessuna variabile `*_NUM_THREADS`; thread di torch e pool BLAS/OpenMP uguali tra i
    fit e registrati (D1); `lock_sha256` = lock del commit; albero git pulito e stesso commit per fit e analisi; device
    richiesto ed effettivo `cpu` in `meta.json` (I4; per lo storico e la riesecuzione il device si verifica solo sulle
    config, perché `main.py` non lo registra).
  - **Criteri di esito:**
    - **parità esatta (M1 superata):** storico = nuovo e riesecuzione = nuovo su tutti i confronti; I1 senza
      differenze; non finiti assenti o solo nel caso dichiarato; provenance senza differenze; `status --verify`
      pulito; epoche uguali;
    - **parità nell'ambiente di riferimento (D5):** storico ≠ nuovo ma riesecuzione = nuovo bit a bit, con gli altri
      controlli puliti: M1 superata con una voce di registro. Se lo storico diverge su PCA, PCA puntuale o baseline
      random, prima si diagnostica (dati con `scripts/make_data_hashes.py --check`, poi versioni e thread) e
      l'accettazione richiede l'approvazione esplicita dell'utente;
    - **differenze ammesse per gli AE (D-17):** solo con causa diagnosticata e documentata, nell'ordine: confronto
      storico contro riesecuzione (deriva dell'ambiente), riesecuzione contro nuovo (deve essere esatto), thread e
      versioni, primo artefatto che diverge (dati scalati con i test `real_data`, finestre, epoche, score); una
      tolleranza solo con una voce di registro approvata;
    - **esito negativo:** riesecuzione ≠ nuovo, oppure differenze non spiegate su PCA, baseline o metriche, oppure I1,
      non finiti o provenance con differenze: ci si ferma, niente F8, diagnosi secondo D-17; non si alzano
      tolleranze, non si modificano `main.py` né la pipeline per far tornare i numeri.
  - **Esito del cancello B1 (suite con l'ambiente dedicato, 2026-11-19):** 3 golden fallite (analisi F6, statistiche
    e report F7) per `lock_sha256`, nessun altro test. Diagnosi scambiando una causa alla volta: ambiente globale con
    il lock nuovo → stesse 3 fallite; ambiente dedicato con il lock vecchio → tutte passate; quindi la causa è il lock,
    non l'ambiente. Valori identici: le uscite differiscono solo negli identificativi che derivano dal lock
    (`analysis_id` e sha256 che ne dipendono), verificato byte per byte dopo la sostituzione (nota su D-73). Golden
    rigenerate dopo la verifica; la regola delle golden F7 è stata resa più permissiva nel solo caso del cambio
    d'ambiente con differenze limitate all'identità (nota su D-73, decisione dell'utente).
  - **Limiti già noti:**
    - `input_norm` e `rolling_zscore` non sono nei run di D-50: su dati reali restano non verificati (solo snapshot
      sintetico F1);
    - AE non addestrati (D-69) e modello di previsione (D-71, D-74) sono fuori da M1;
    - versioni delle librerie, thread e macchina dei run storici non sono registrati (D-50, D-61): limite dichiarato;
      la riesecuzione lo mitiga ma non lo elimina.
- **Stato: completata (2026-11-19).** Commit: preparazione `4d8d248` (config di parità, `parity_report`, lock
  dall'ambiente dedicato, golden), chiusura (commit che contiene questa riga). Run lanciati dall'utente il 2026-11-19
  sul commit `4d8d248`. Report in `docs/m1_parita.md`; report dello script in `reports_local/m1/parity_{last,mean}/`
  (esclusa da git). Suite di chiusura: vedi il messaggio del commit di chiusura.
  - **Esito:** **parità esatta** (`parita_esatta`) su entrambe le varianti: aggregazione nella finestra `last` con
    proiezione `last` (analisi `be0a03233647`) e aggregazione nella finestra `mean` con proiezione `last` (analisi
    `a02e1de4a1e5`). Il run di fit è `20261119T144234-c54fd9f8` (8 fit, nessun fallito). Codici di uscita 0 per B2, C1,
    D1, D2, D3, E1.
  - **Verificato in M1 (dati reali, `machine-1-1` e `machine-3-8`, seed 42, tolleranza zero):**
    - storico = nuovo, storico = riesecuzione di `main.py`, riesecuzione = nuovo: 70 file per confronto (34 JSON delle
      metriche, 34 `.npz`, 2 CSV per macchina), nessuna differenza, epoche comprese. I tre modelli soggetti alla
      regola D5 sono coperti a tolleranza zero (`pca`, `pca_pointwise`, `random_baseline`);
    - tabella lunga contro export legacy (I1): 408 confronti (34 righe × 12 colonne), righe escluse 0;
    - non finiti: `n_nonfinite_cal` = `n_nonfinite_test` = 0 e nessuna loss non finita negli 8 fit;
    - provenance: commit unico, albero pulito, `lock_sha256` del commit, device `cpu`, `seeds[0]` = 42, thread e pool
      uguali tra i fit, nessuna variabile `*_NUM_THREADS`;
    - `status --verify` con codice 0.

    I conteggi del report sono a livello di file, non di campi, celle o array (`docs/m1_parita.md` §7).
  - **Corrispondenza tra test previsti e file:**
    - config e vincolo sui seed (`seeds[0]` = 42, equivalenza con `repro_*.yaml` e con `legacy_plan`):
      `test_m1_configs.py` (9 test);
    - `parity_report.py`: confronti, esclusioni, I1, non finiti, provenance, esiti e codici di uscita, con prove di
      difetto: `test_m1_parity_report.py` (38 test);
    - regola delle golden F7 al cambio d'ambiente: `test_f7_golden.py` (8 test nuovi).
  - **Scostamenti dal piano:**
    - quelli elencati nella nota del piano di M1 sopra: un esperimento e due piani di analisi, riesecuzione di
      `main.py`, ambiente dedicato e lock rigenerato, report in `docs/m1_parita.md`;
    - in più, dal cancello B1: regola delle golden F7 resa più permissiva solo per il cambio d'ambiente con
      differenze limitate all'identità (nota su D-73).
  - **Non verificato in M1:**
    - altre macchine e altri seed;
    - `input_norm` e `rolling_zscore` su dati reali;
    - AE non addestrati e modello di previsione;
    - caso `n_nonfinite_cal` > 0, assente su 1-1 e 3-8;
    - macchina, thread e versioni dei run storici (limite dichiarato, portata ridotta dall'identità storico =
      riesecuzione);
    - le uscite di B2–D3 sono quelle riportate dall'utente (`docs/m1_parita.md` §4); la trascrizione
      `reports_local/m1/console_D1_E1.txt` contiene solo i codici di D1–E1.
  - **Test modificati alla chiusura** (approvazione dell'utente, 2026-11-19):
    - i 6 controlli «`store/` assente» diventano «`store/` non modificato dai test» (nota su D-16; guardia in
      `tests/fixtures/store_guard.py`, prove in `tests/test_store_guard.py`, 9 test);
    - `test_f7_cli_e2e.py` fissa la codifica delle pipe del sottoprocesso (`PYTHONIOENCODING=cp1252`, decodifica
      cp1252) invece di ereditarla dalla shell. Con `PYTHONIOENCODING=utf-8` nella shell il test falliva in
      decodifica. Prova di difetto: senza `_safe_streams()` in `main` il test ora fallisce. Il caso «ρ» su cp1252
      resta coperto anche da `test_f7_cli_stats.py::test_unencodable_characters_never_crash_the_command`, che fissa la
      codifica da sé.
  - **Passati ad altre fasi:**
    - **F8** (verifica di completezza alla chiusura di M1):
      - `split3` da file (dalla chiusura di F7, «Passati ad altre fasi»);
      - AE non addestrati (D-69): nella sezione F8, «Dipendenze», da implementare dopo M1 e prima di F8;
      - modello di previsione (D-71, D-74): nella chiusura di F7 e nella sezione F8, «Dipendenze»;
      - golden F6 allineata alla regola di F7 (chiusura di F7);
      - docstring di `src/pipeline/legacy_export.py` (chiusura di F7);
      - *(aggiunto ora)* controllo dei non finiti: lo scostamento dichiarato su `raw_mean` (D-72 C15, caso
        `n_nonfinite_cal` > 0) non si è presentato su 1-1 e 3-8. Va controllato con la tabella `fits` nell'esperimento
        legacy contro canonico sulle 6 macchine dev, e nel report va tenuto tra le correzioni dichiarate (nota da F6
        nella sezione F8).
    - **F10:** numero di thread per i run ufficiali (fissato o registrato; nota su D-17).
    - **Dalla chiusura di F7, punti per M1:** chiusi. Lock rigenerato (D-55); confronto sui CSV per macchina e sui
      JSON, non sul summary; thread registrati. La verifica con `n_nonfinite_cal` è chiusa come «assente» e passa a F8.
  - **Decisioni approvate durante M1:** D1–D6 e I1–I4 dell'utente, registrate come note di rimando su D-14, D-17, D-18,
    D-50, D-55 e D-73; nessuna nuova voce `D-xx`.
  - **Decisioni APERTE toccate e rimaste aperte:** nessuna.

### F8 — Nuovo protocollo canonico (cambiamento numerico dichiarato)
- **Obiettivo:** passare a `split3` 60/20/20 [D-05], nessun contesto all'inizio del test nell'analisi primaria e contesto da cal solo nella variante di RQ1a [D-03, D-32, D-20], maschere secondo D-20 e D-22, normalizer e soglie su cal secondo D-23 con storia cal → test [D-04, D-21], PCA solo su fit [D-06], `max_epochs`/patience secondo D-39, `train_step` secondo D-38, preprocessing secondo D-40.
- **File:**
  - da modificare: `configs/base.yaml` (valori canonici);
  - nuovi: `configs/experiments/protocol_switch_dev.yaml`, test di leakage end-to-end;
  - la lista delle macchine di sviluppo è già in `configs/dev_machines.yaml` da F0 [D-51].
- **Dipendenze:** M1; decisioni D-38, D-39, D-40; implementazione, dopo M1 e prima di F8, degli AE non addestrati
  (D-69, con i punti aperti della voce: schema, seed nel `fit_key`, BatchNorm, controllo di parità) e del modello di
  previsione (D-71, punti aperti in D-74). *(rev. 2026-11-16: prima «M1; decisioni D-38, D-39, D-40».)*
- **Nota (2026-11-16, da F7, decisioni D-69, D-71, D-74):** quando il modello di previsione esiste (un solo punto
  previsto per finestra, solo `ts_agg = last`, D-74 punti a–c):
  - `analyze` / `analysis_grid` non produce righe offline (`ts_agg` `mean`, `max`) per il modello di previsione;
  - `src/stats/plan.py` lo rifiuta nei blocchi con `window_time_agg` o `ts_agg` a più di un livello;
  - test per entrambi i punti.
  Gli AE non addestrati si dichiarano baseline descrittive nel piano statistico (D-73).
- **Nota (2026-11-16, proposta di Claude Project in D-69, NON ancora approvata):** *(rev. 2026-11-20: approvata con
  modifiche, nota del 2026-11-20 su D-69: nomi distinti `*_untrained` al posto dell'opzione `untrained: true`, non
  realizzabile; `batch_size` di scoring nel `fit_key`; validatore di appaiamento. Il testo sotto resta come storico.)*
  Se approvata, ogni AE non
  addestrato è l'AE addestrato selezionato in F9 con zero epoche: stessa configurazione, stessa inizializzazione,
  stessi seed, BatchNorm con le statistiche iniziali. Si realizza con un'opzione esplicita (per esempio
  `untrained: true`), non con `max_epochs: 0`. Le classi storiche `Untrained…` con `init_std` non si usano per i run.
  Test da aggiungere:
  - stesso seed → pesi identici a quelli dell'AE addestrato prima della prima epoca;
  - il trainer gestisce zero epoche e la selezione dell'epoca su es non ha nulla da ripristinare;
  - cambiare solo le impostazioni di addestramento (per esempio il learning rate) lascia invariato il `fit_key`
    dell'AE non addestrato;
  - il ruolo dichiarato nella config si riscontra con `descriptive_baselines` del piano statistico (D-73).
- **Test (su dati sintetici):**
  - perturbare es, cal o test lascia invariati scaler e PCA;
  - perturbare il test lascia invariati normalizer e soglie;
  - la modalità `legacy2` resta riproducibile, verificata con lo snapshot.
- **Esperimento (lo lanci tu):** confronto legacy vs canonico sulle 6 macchine dev, con report delle differenze.
- **Completata quando:** report approvato da te.
- **Rischi:** nell'analisi primaria la maschera del test coincide con quella storica [W−1, T−W+1) [D-20, D-22] e cal coincide con la validation storica [D-23], quindi il confronto legacy vs canonico avviene sugli stessi timestamp. La variante con contesto cambia la maschera e va confrontata a parte.
- **Impatto sui risultati:** **sì, dichiarato e documentato.**
- **Nota (2026-11-12, da F6):** nel report legacy contro canonico si tengono separati due tipi di differenza:
  - **fattori di protocollo della letteratura** (PA, soglia oracolo, proiezione offline, metrica, contesto all'inizio
    del test): si decompongono e se ne misura l'effetto (RQ1a, D-12, D-42);
  - **correzioni di bug idiosincratici della pipeline storica** (per esempio la validation usata tre volte, P1, e il
    conteggio per righe della storia in `rules.py`, D-72 C1): si correggono e si dichiarano, con l'impatto misurato
    (`CLAUDE.md` §5), ma non diventano livelli di un fattore né entrano nella decomposizione. Elenco delle correzioni
    dichiarate finora: P1 (validation usata tre volte); storia per righe di `rules.py` (D-72 C1); NaN nel supporto di
    cal scartato in silenzio dalla soglia statica dello storico (`raw_mean`), ora riga NaN contata (D-72 C15, passo
    6.0).
- **Nota (2026-11-19, da M1):** il caso `n_nonfinite_cal` > 0 non si è presentato su `machine-1-1` e `machine-3-8`.
  Nell'esperimento legacy contro canonico sulle 6 macchine dev si controlla con la tabella `fits`; se compare, le righe
  `raw_mean` con `nan_reason = nonfinite_residual_cal` vanno riportate come correzione dichiarata (D-72 C15).
- **Nota (2026-11-20, prerequisiti di F8; utente, su raccomandazione di Claude Project).** Decisioni: D-54 e D-75
  approvate; D-77 nuova (caricamento delle config); note del 2026-11-20 su D-10, D-40, D-69, D-71, D-73, D-74. Restano
  APERTE, senza valori predefiniti nel codice (valori provvisori solo nei test sintetici): D-38, D-39, D-40, D-41;
  D-74 per budget e griglie. Nei passi 0–7 `configs/base.yaml` non cambia.
  - **Regole per ogni passo:** dopo ogni passo snapshot F1, golden, suite completa, riepilogo completo, poi stop per la
    revisione dell'utente; nessun commit senza richiesta; se un fatto nel codice contraddice le decisioni, ci si ferma
    e lo si segnala prima di procedere.
  - **Passi:**
    - **0 — registro e piano** (solo documenti): le voci sopra e questa nota;
    - **1 — golden di F6 e docstring** (nota su D-73): golden di `tests/test_f6_analyze_golden.py` con il criterio
      `identity_outputs` di F7 (`tests/test_f7_golden.py`), rigenerata una volta nel nuovo formato; docstring di
      `src/pipeline/legacy_export.py` corretto dopo aver verificato nell'ambiente smd_m1 quale handler registra davvero
      l'import di `main.py`. Nessun effetto sui numeri;
    - **2 — caricamento delle config** (D-77): blocco `split` atomico (A1) e valori storici espliciti in
      `parity_repro.yaml` (strategia (i)). Test: `config_sha256` e `fit_key` di `parity_repro` uguali a quelli del run
      M1 nel registro dello store (sola lettura); indipendenza (D-77, revisione del passo 0): con copie temporanee di
      `base.yaml` e dei file di `configs/models/` modificate, `config_sha256` e `fit_key` di `parity_repro` restano
      quelli di M1, e i file veri non si toccano; un esperimento `split3` e uno `legacy2` si caricano da file; una
      chiave del modo non attivo in `split` è un errore; `tests/test_f7_cli_e2e.py:114` capovolto in «ora si carica»,
      senza renderlo più permissivo;
    - **3 — AE non addestrati** (D-69 e D-73, note del 2026-11-20): `conv_autoencoder_untrained`,
      `lstm_autoencoder_untrained`. Test: pesi = pesi iniziali dell'AE addestrato con lo stesso seed; percorso a zero
      epoche (storia vuota, `best_epoch` `None`, nessun ripristino, modello in eval); `fit_key` con seed e `batch_size` di
      scoring (cambiare solo il learning rate → stesso `fit_key`; cambiare il `batch_size` di scoring → `fit_key`
      diverso); validatore (`params` o `batch_size` non appaiati → errore); `plan.py` (`*_untrained` rifiutato fuori da
      `descriptive_baselines`, native ancora ammessi lì); snapshot F1 e golden invariati;
    - **4 — modello di previsione** (D-74, nota del 2026-11-20): `lstm_forecaster`, `ForecastTrainer`. Test: leakage
      (l'ultimo punto della finestra non cambia la previsione, un punto tra i primi W−1 sì); forme dei residui e
      `residual_aggs = ("last",)`; determinismo a seed fissato; `analysis_grid` senza righe offline, contate in
      `analysis.json`; export legacy che rifiuta il modello; `PyTorchTrainer` invariato (snapshot F1 e golden degli AE
      identici); golden rigenerata solo per l'aumento di `ANALYSIS_CODE_VERSION`, con le sole differenze di identità;
    - **5 — regole di `plan.py` per il modello di previsione** (D-74): rifiuto nei blocchi di RQ1a con `ts_agg` ≠
      `[last]`; rifiuto nei blocchi di RQ1b con più di un `window_time_agg`; solo `window_time_agg: last` e
      `ts_agg: last` dove compare (RQ2, RQ4, `descriptive`); riscontro di `descriptive_baselines` con lo schema. Un
      test per ogni rifiuto;
    - **6 — prova sintetica completa** `split3` da file con tutti i modelli (`fit` → `analyze` → `stats`), D-16;
    - **7 — verifica di D-40** (nota del 2026-11-20 su D-40): script in sola lettura, solo le 6 macchine dev, uscite in
      `reports_local/d40/`; numeri senza conclusioni sulla decisione.
  - **Stato (2026-11-21): prerequisiti completati (passi 0–7).** Commit su `main`: 0 `723c9c2`, 1 `f179254`,
    2 `b2f7ec7`, 3 `51aaeae`, 4 `80ec3de`, 5 `7c72508`, 6 `1027870`; 7 = il commit «F8 passo 7» che contiene questa
    nota (un commit non può citare il proprio hash). Esiti del passo 7 nelle note del 2026-11-21 su D-40 e D-39 (solo
    fatti); D-38, D-39, D-40 restano APERTE e `configs/base.yaml` è invariato.
  - **Dopo il passo 7, F8 vera e propria:** decisioni D-38, D-39, D-40; valori canonici in `configs/base.yaml`;
    `configs/experiments/protocol_switch_dev.yaml`; esperimento legacy contro canonico sulle 6 macchine dev (lo lancia
    l'utente), con il controllo dei non finiti (nota del 2026-11-19 sopra) e le correzioni dichiarate tenute separate
    dai fattori di protocollo (nota del 2026-11-12).
  - **Tempi (D-54):** consegna entro il 2027-01-31, chiusura del progetto nella prima metà di dicembre 2026; l'opzione 1
    di D-75 si fa solo se F11 è chiusa entro il 2026-12-10.
- **Nota (2026-11-24, F8 vera e propria; utente, su raccomandazione di Claude Project).**
  - **Misure preliminari** (sola lettura, solo le 6 macchine dev, nessuna label, nessun training; uscite in
    `reports_local/f8_pre/`, esclusa da git): pavimento τ dello scaler su fit 60% (misura 1a) e finestre di training
    con `train_step` = 5 (misura 1c). Esiti nelle voci D-38 e D-40.
  - **Decisioni:** D-38 (`train_step` = 5), D-39 (100 / 10, stesso blocco `training` per il modello di previsione),
    D-40 (minmax su fit, `clip: null`, `min_range` = τ = 0,02, criterio scritto dopo la misura 1a) approvate; note
    del 2026-11-24 su D-37, D-41 e D-57 (le voci restano APERTE).
  - **Valori storici senza voce nel registro:** W = 100, `learning_rate` (1e−3), `weight_decay` (1e−5), `min_delta`
    (1e−6) e `batch_size` (64) restano fissi ai valori storici; non sono tarati, salvo quelli che entreranno nella
    griglia di D-41. Da dichiarare in tesi.
  - **Scostamenti approvati dal punto «File» di F8:**
    - `configs/experiments/protocol_switch_dev/v0…v3.yaml` (4 file) al posto di un solo
      `protocol_switch_dev.yaml`: lo schema ammette un solo `split`, `preprocessing` e `training` per esperimento;
    - `scripts/protocol_switch_report.py`, in sola lettura sullo store, con uscita in una cartella nuova;
    - due piani di analisi: primario per V0–V3 (scenario `primary`), con contesto solo per V1 (`test_context` non è
      ammesso in `legacy2`, D-60).
  - **Esperimento `protocol_switch_dev`** (solo le 6 macchine dev; 3 seed, 42 per primo; lo lancia l'utente). Catena
    di varianti con un cambio per passaggio, ordine fissato prima dei run:

    | variante | split | τ | epoche (max / patience) |
    |---|---|---|---|
    | V0 legacy | `legacy2` | 0 | 40 / 5 |
    | V3 | `split3` | 0 | 40 / 5 |
    | V2 | `split3` | 0 | 100 / 10 |
    | V1 canonico | `split3` | 0,02 | 100 / 10 |

    - modelli: `pca`, `pca_pointwise`, `conv_autoencoder`, `lstm_autoencoder`, `conv_autoencoder_untrained`,
      `lstm_autoencoder_untrained`, `lstm_forecaster`, `rolling_zscore`, `input_norm`; tutti i valori espliciti nei
      file (come `parity_repro`, D-77), iperparametri storici di M1; `lstm_forecaster.hidden_dim` = 64, provvisorio
      (D-41);
    - il `fit_key` non contiene il nome dell'esperimento: in V3 i modelli non addestrati si riusano da V2 (Δ = 0
      atteso per costruzione, controllo), e V0 riusa i fit di M1 su machine-1-1 e machine-3-8 con seed 42 (da
      verificare con `status`).
  - **Report: quattro tipi di differenza:**
    - fattori di protocollo della letteratura (PA, oracolo, proiezione offline, metrica, contesto): contrasti dentro
      ogni variante, confrontati tra V0 e V1;
    - correzioni dichiarate: P1 (V0 → V3); D-72 C15 (righe `nan_reason = nonfinite_residual_cal` e `n_nonfinite_cal`
      della tabella `fits`, nota del 2026-11-19); D-72 C1 non è osservabile in questo esperimento (`main.py` non ha mai
      usato `rules.py`, entrambe le modalità usano `thresholds.py`, il piano non ha soglie mobili perché D-42 è
      APERTA): nessun numero storico interessato. C1 resta tra le correzioni dichiarate e si misura quando le soglie
      mobili entreranno nelle analisi (livelli di D-42);
    - cambi di protocollo approvati: fit 80% → 60% (D-05, D-65), V0 → V3;
    - cambi di design dichiarati: D-39 (V3 → V2), D-40 (V2 → V1); D-38 resta al valore storico.
  - **Da dichiarare nel report:** per gli AE il passaggio V0 → V3 confonde P1 e riduzione dei dati; per i modelli
    deterministici dà solo l'effetto dei dati; l'esito della catena dipende dall'ordine dei passaggi. Si riportano
    anche i fit fermati al tetto per variante (criterio di D-39), le finestre di training e i tempi dai `meta.json`.
  - **Test da adattare al passo 1:** `tests/test_legacy_config_equivalence.py` punta a una copia storica di
    `base.yaml` in `tests/fixtures/`, senza diventare più permissivo; `tests/test_f7_cli_e2e.py:115` si rovescia
    (esperimento `legacy2` sopra una base `split3`) con la stessa forza.
  - **`CLAUDE.md`:** al passo 1 una riga di fatto in §3.3 (protocollo canonico in `configs/base.yaml`);
    aggiornamento completo dopo il report di F8 (passo 6).
  - **Da fare al passo 6 (revisione del passo 2, utente, 2026-11-24):** una riga in D-40 sul pavimento in float32:
    il range su fit è float32, quindi il denominatore effettivo è float32(τ) = 0,0199999996 e la scala vale
    50,0000011; limite esatto |z| ≤ 1/float32(τ) (fatto rilevato da `tests/test_f8_leakage_e2e.py`).
  - **Passi** (regola abituale: dopo ogni passo snapshot F1, golden, suite completa, riepilogo completo, poi stop per
    la revisione dell'utente; nessun commit senza richiesta):
    - **0 — registro e piano** (solo documenti): questa nota e le voci sopra;
    - **1 — `configs/base.yaml` canonico:** `split` → `{mode: split3, fractions: [0.6, 0.2, 0.2]}` (D-05, D-65, D-77
      A1); `preprocessing.min_range` → 0,02 (D-40); `training.max_epochs` / `patience` → 100 / 10 (D-39); tolti i
      commenti «provvisorio» di D-38, D-39, D-40; i due test sopra; un test che fissa i valori canonici;
      `parity_repro` identico a M1 (`tests/test_f8_config_loading.py` invariato); la riga di `CLAUDE.md` §3.3;
    - **2 — test di leakage end-to-end** (sintetici, D-16; config canonica con τ > 0 e tutti i modelli): perturbare
      es, cal o test lascia invariati scaler e PCA; perturbare il test lascia invariati normalizer, soglie statiche e
      storia di cal delle soglie mobili; perturbare es cambia solo l'early stopping; un sensore a range esattamente
      nullo ha scala 1/τ; `legacy2` riproducibile (snapshot F1);
    - **3 — config V0–V3 e i due piani di analisi**, con test di caricamento e delle relazioni tra `fit_key`;
    - **4 — `scripts/protocol_switch_report.py`** con test sintetici;
    - **5 — run** di `fit` e `analyze` delle 4 varianti (li lancia l'utente), poi il report;
    - **6 — report di F8** da approvare; poi `CLAUDE.md` completo e «Ordine consigliato e parallelismi».

### F9 — Tuning e sensibilità (esperimenti esplorativi)
- **Obiettivo:** scegliere le configurazioni globali per famiglia, con budget pari, sulle 6 macchine dev.
- **File (nuovi):** griglie in `configs/models/*.yaml`, `configs/experiments/tuning.yaml`, `configs/analysis/tuning.yaml`; ulteriori esperimenti esplorativi (preprocessing, epoche fisse, PCA su fit+es [esplorativo, D-06]) sempre sulle sole macchine dev.
- **Dipendenze:** F8; decisioni D-41 e D-46.
- **Nota (2026-11-16, da F6 passo 6.7, durata dell'analisi):** misura su dati sintetici e stima per l'esperimento
  ufficiale circa 35 minuti a un thread con 5 seed (circa 22 con 3), contro i circa 15 stimati; il costo principale è
  `roll_robust_z`. Con il primo piano di tuning si misura la durata reale e si decide allora, registrando la scelta,
  tra:
  - ottimizzazioni: z di `roll_robust_z` (e finestra mobile) calcolato una volta per combinazione e riusato tra i
    livelli; verifica di `residuals.npz` una volta per fit, sotto lock, riusata per tutti i suoi array (la verifica
    non si toglie);
  - parallelizzazione per macchina.
  Ogni ottimizzazione deve lasciare le tabelle identiche bit a bit (prova golden del passo 8 di F6).
- **Test:** un test sulle config verifica che il budget sia pari per costruzione; lo stadio di fit rifiuta le macchine di valutazione negli esperimenti esplorativi, salvo un flag esplicito che viene registrato.
- **Completata quando:** configurazioni selezionate salvate, con motivazione in `docs/decisioni.md`.
- **Rischi:** tentazione di guardare le macchine di valutazione, impedita dal controllo sopra. Stima del costo: circa 7 ore di CPU con `train_step=5`, da verificare.
- **Impatto sui risultati:** sceglie gli iperparametri finali. Questi risultati sono **esplorativi**, non della tesi.

### F10 — Congelamento del piano di analisi
- **Obiettivo:** fissare RQ, δ e analisi prima di vedere i risultati ufficiali.
- **File (nuovi)**, ciascuno con il suo hash nel registro, più la sezione corrispondente in `docs/decisioni.md`, congelati
  in quest'ordine: config dell'esperimento → piano di analisi → piano statistico:
  - `configs/experiments/official.yaml` (config dell'esperimento): D-09 (`experiment.seeds`, 5 seed), modelli e
    iperparametri selezionati in F9 (D-41), preprocessing (D-40), macchine di valutazione (D-08); hash =
    `config_sha256`;
  - le config degli altri esperimenti dichiarati nel piano statistico ufficiale: quelli che fanno da livelli di preproc
    nei blocchi di RQ1b (D-73 P5, nota del passo 1-ter). Si scrivono e si congelano qui, ciascuna con il suo
    `config_sha256` nel registro, insieme alla config dell'esperimento primario, prima del piano di analisi.
    *(rev. 2026-11-16: prima una sola config di esperimento.)*
  - `configs/analysis/official.yaml` (piano di analisi, D-72 C8): solo ciò che `analyze` calcola, cioè la parte di
    D-42 sui livelli calcolati (`scenarios`, compreso `test_context` di D-20 e D-32; `window_time_aggs`; `scorings`;
    `ts_aggs`; `thresholds` con livelli e `n_past`; `pa`), con i vincoli di D-24 e D-60 verificati dal validatore;
    nessun `legacy_export` (solo `legacy2`);
  - `configs/stats/official.yaml` (piano statistico, D-73 P5): D-30 (δ), D-36 (α, B ≥ 10.000, famiglia
    confermativa), D-37 (PCA di riferimento), D-42 (blocchi e livelli letti, configurazione canonica per τ, versione
    principale di η²), D-43 (descrittori), D-44 (regole di esclusione), più gli esperimenti attesi con il
    `config_sha256` della config dell'esperimento e il `plan_sha256` dell'analisi (D-73 P6), quindi si congela per
    ultimo.
  *(rev. 2026-11-16: prima un solo file `configs/analysis/official.yaml` con D-30, D-36, D-37, D-42, D-43, D-44 e
  D-09; la config dell'esperimento era un file nuovo di F11. Decisione dell'utente del 2026-11-16, nota su D-09.)*
- **Dipendenze:** F9; tutte le decisioni del gruppo C; D-34 per VUS-PR.
- **Test:** i comandi `cli` con `kind: official` rifiutano di partire se il file che li governa non è registrato: `fit`
  la config dell'esperimento, `analyze` il piano di analisi (e la config), `stats` il piano statistico (e gli hash che
  dichiara); inoltre se l'albero git è modificato o se tra le macchine compare una di sviluppo. *(rev. 2026-11-16: prima
  «il piano», al singolare, senza la config dell'esperimento.)*
- **Nota (2026-11-16, da F7, D-73 P6):** dopo il congelamento, modificare `configs/base.yaml` o i percorsi delle config
  (`paths.store_root`, `data.data_dir`) invalida i `config_sha256` dichiarati nel piano statistico.
- **Completata quando:** piano congelato in un commit (lo fai tu o lo chiedi tu).
- **Impatto sui risultati:** nessuno.

### F11 — Run ufficiali (esperimenti ufficiali, li lanci tu)
- **Obiettivo:** risultati della tesi sulle 22 macchine di valutazione.
- **File:** nessuno nuovo; si eseguono solo la config dell'esperimento e i due piani congelati in F10. *(rev.
  2026-11-16: prima «File (nuovi): `configs/experiments/official.yaml`», ora scritta e congelata in F10, nota su D-09.)*
- **Esecuzione:** `fit` → `analyze` → `stats` → `report`. Stima: circa 6 ore di CPU per i fit.
- **Completata quando:** IC, decisioni RQ2 e analisi RQ1, RQ3 e RQ4 sono generati dal piano congelato e da un commit noto; il registro è aggiornato.
- **Rischi:** un fit fallito si gestisce secondo la regola di esclusione congelata, non decidendo dopo.
- **Impatto sui risultati:** sono i risultati.

### F12 — Pulizia (solo con la tua approvazione)
- **Obiettivo:** un'unica pipeline e documentazione allineata.
- **File:** `main.py`, `aggregate_results.py`, `src/data/dataset.py`, `scripts/compare_npz.py`, `scripts/confronta_run.py` (deprecazione o rimozione); `architettura_repo_v1.md` (se esiste fuori dal repository: archiviato; nel repository non è presente); `CLAUDE.md` (aggiornato).
- **Dipendenze:** M1 superata; F11 completata o almeno F8.
- **Test:** suite completa; lo snapshot legacy si archivia o si rimuove insieme a `main.py`.
- **Impatto sui risultati:** nessuno.
- **Nota (2026-11-24, da F8 passo 2, utente):** messaggio di log degli AE non addestrati: «up to 0 epochs (early
  stopping, patience=5)», dove 5 è il default storico di `PyTorchTrainer` (il trainer riceve solo `epochs` = 0).
  Nessun effetto (zero epoche, nessun early stopping): correggere solo il testo.

---

## Ordine consigliato e parallelismi

```
F0 → F1 → F2 ─┬→ F3 → F4 → F5 → F6 ─┐
              └→ F7 (in parallelo) ──┴→ M1 → F8 → F9 → F10 → F11 → F12
```

Vincoli aggiuntivi: F3 richiede D-51 (D-31 è sostituita da D-32); F7 richiede D-36 e D-25 (entrambe approvate il 2026-11-16).

- **Si può interrompere in qualunque punto:** fino a M1 nessun risultato cambia e la pipeline storica resta intatta. Ogni fase è un branch e un commit separato, su tua richiesta.
- **`CLAUDE.md` va aggiornato tre volte:** in F0 (riconciliazione, D-02), dopo M1 (la nuova pipeline diventa quella di riferimento) e dopo F8 (il protocollo canonico cambia). Ogni nuova decisione si registra prima in `docs/decisioni.md`.
- **Dimensioni relative:** F1, F4 e F6 sono le fasi più delicate; F2 e F7 sono le più lunghe da scrivere ma le meno rischiose; F3 e F5 stanno nel mezzo.

## Cosa serve per partire

F0 è chiusa: le decisioni del gruppo A sono tutte approvate (D-32, D-33, D-34, D-50, D-51, D-55, D-56), così come D-52 e D-53, e i file di F0 sono presenti. Si può iniziare F1. Prima di M1 va rigenerato il lock da un ambiente virtuale dedicato (D-55). Prima di F7 servono D-36 e D-25. Il gruppo B serve da F8, il gruppo C da F10.