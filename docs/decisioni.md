# Registro delle decisioni

Fonte di livello 1 della gerarchia dei documenti (vedi `CLAUDE.md` §3.2): per i punti che registra prevale su
`CLAUDE.md`, `piano_tesi_v1.md`, `proposta_architettura_target.md` e `piano_operativo.md`.

**Stati**
- `APPROVATA`: vincolante. Si cambia solo con una nuova voce che la sostituisce (mai modificando il testo storico).
- `APERTA`: da decidere. Il codice non deve fissarla con un default implicito.
- `APERTA — DIPENDE DALL'UTENTE`: richiede informazioni o azioni che solo l'utente può fornire.
- `SOSTITUITA da D-xx`: storico.

**Formato di ogni voce:** ID · titolo · stato · data · decisione (o opzioni) · motivazione · serve entro · documenti toccati.

---

## A. Governance e documenti

### D-01 · Gerarchia dei documenti — APPROVATA · 2026-11-09
- **Decisione:** ordine di precedenza: (1) `docs/decisioni.md`; (2) `CLAUDE.md`; (3) `piano_tesi_v1.md`;
  (4) `proposta_architettura_target.md` (solo parti approvate); (5) `piano_operativo.md` (prevale sul §12 della proposta).
  `architettura_repo_v1.md` è superato. Per "come funziona oggi" la fonte è il codice. Un conflitto non registrato qui
  va segnalato, non risolto in silenzio.
- **Motivazione:** i documenti si contraddicevano (audit del 2026-11-09, C3–C5).
- **Documenti:** tutti.

### D-02 · Riconciliazione dei documenti prima di F0 — APPROVATA · 2026-11-09
- **Decisione:** `CLAUDE.md` viene aggiornato in F0 (non dopo M1); nella proposta e nel piano si aggiungono note di stato
  e rimandi a questo registro.
- **Motivazione:** senza riconciliazione, già da F3 l'implementazione violerebbe regole dichiarate inderogabili (C1, C4).
- **Documenti:** `CLAUDE.md`, proposta, piano.

### D-13 · Piano di migrazione di riferimento — APPROVATA · 2026-11-09
- **Decisione:** vale `piano_operativo.md` (F0–F12 + M1). Il §12 della proposta è superato. `main.py` resta congelato:
  la nuova pipeline cresce accanto a quella storica; ai moduli condivisi si fanno solo aggiunte.
- **Motivazione:** il riferimento storico non si può rompere per errore; la parità si dimostra confrontando due pipeline (C5).
- **Documenti:** proposta §1, §12; `CLAUDE.md` §3.

---

## B. Metodologia — decisioni approvate

### D-03 · Finestre: confini e contesto causale — APPROVATA · 2026-11-09
- **Decisione:** le finestre di addestramento (fit) e quelle della loss di early stopping (es) non attraversano mai un
  confine tra segmenti. Le finestre di scoring possono includere come contesto gli ultimi W−1 punti del segmento
  immediatamente precedente. In modalità `legacy2` niente contesto. Quali segmenti ricevono il contesto: D-31.
- **Motivazione:** senza contesto si perdono i primi W−1 punti del test, eventi compresi (P4); il contesto è passato,
  non etichettato e non usato per alcun fit.
- **Condizione:** l'applicazione al primo tratto del test dipende dall'esito della verifica di contiguità (D-32).
- **Serve entro:** F3.
- **Documenti:** `CLAUDE.md` §5; proposta §1, §2, §4.2, §10; piano F3.
- **Nota di rimando (2026-11-10):** integrata da D-20 (contesto iniziale del test e maschera) e D-22 (coda del test),
  entrambe subordinate a D-32. La «Condizione» sul primo tratto del test è risolta da D-32: nell'analisi primaria
  nessun contesto all'inizio del test; il contesto da cal è solo una variante di RQ1a. Il testo sopra resta invariato.

### D-04 · Il segmento es non entra nell'analisi — APPROVATA · 2026-11-09
- **Decisione:** la storia delle soglie mobili parte da cal (sequenza cal → test), per tutti i modelli. Il segmento es non
  si usa per normalizer, soglie statiche o mobili, né metriche. I residui di es si salvano solo come diagnostica.
- **Motivazione:** per gli AE es ha scelto l'epoca, quindi i suoi residui sono ottimistici; usarli per le soglie
  ripropone P1. Vincolo rispettabile senza perdite: il train più corto di SMD ha 23.687 righe, quindi cal ≈ 4.737
  punti > `n_past` massimo previsto (2.880).
- **Serve entro:** F6.
- **Documenti:** `CLAUDE.md` §5; proposta §2, §3, §4.6, §5.4; piano F6.

### D-05 · Split cronologico 60/20/20 — APPROVATA · 2026-11-09
- **Decisione:** fit 60% / es 20% / cal 20% del train originale, in ordine cronologico; frazioni configurabili.
  Un'eventuale analisi di sensibilità sulle proporzioni è D-46.
- **Documenti:** `CLAUDE.md` §5; proposta §4.1.

### D-06 · PCA fittata solo su fit — APPROVATA · 2026-11-09
- **Decisione:** nel protocollo canonico PCA e PCA puntuale si fittano solo su fit (stessi dati da cui gli AE apprendono
  i parametri). PCA su fit+es resta disponibile solo come esperimento esplorativo sulle macchine di sviluppo.
- **Documenti:** proposta §4.1; piano F8, F9.

### D-07 · Metriche: primaria e secondarie — APPROVATA · 2026-11-09
- **Decisione:** primaria AP normalizzata (AP − p)/(1 − p). Secondarie ROC-AUC e VUS-PR. F1 e metriche con soglia solo per RQ4.
  Quando e come implementare e validare VUS-PR, e l'elenco delle altre metriche: D-34.
- **Nota:** la decisione 11 del piano operativo ("δ e metrica primaria") si riduce a δ (D-30).
- **Documenti:** `CLAUDE.md` §1; proposta §4.7; piano gruppo A punto 3, gruppo C punto 11.

### D-08 · 6 macchine di sviluppo + 22 di valutazione — APPROVATA · 2026-11-09
- **Decisione:** tuning ed esperimenti esplorativi solo sulle 6 macchine di sviluppo (con label); run ufficiali sulle 22
  di valutazione. Leave-one-group-out scartato. Identità delle 6 macchine: D-51.
- **Documenti:** proposta §4.5, §11; piano gruppo B punto 10.

### D-09 · 5 seed nella valutazione — APPROVATA · 2026-11-09
- **Decisione:** 5 seed per gli AE nei run ufficiali; modelli deterministici una sola volta. Seed per il tuning: D-41.
- **Documenti:** proposta §4.5; piano gruppo C punto 16.

### D-10 · OC-SVM, Isolation Forest, VAE — APPROVATA · 2026-11-09
- **Decisione:** restano nel registro dei modelli e nei test; sono esclusi dai run ufficiali e dal tuning.
- **Documenti:** `CLAUDE.md` §6; proposta §11.

### D-11 · Normalizer robusto come variante — APPROVATA · 2026-11-09
- **Decisione:** si aggiunge un normalizer robusto (mediana/IQR) come variante, accanto a quello storico. Se entra nel budget
  di tuning come opzione di scoring: D-41.
- **Motivazione:** errori a coda pesante e possibile contaminazione del train da anomalie non etichettate (da dichiarare tra i limiti).
- **Documenti:** proposta §4.4; piano gruppo A punto 4, F6.

### D-12 · RQ1 divisa in RQ1a e RQ1b — APPROVATA · 2026-11-09
- **Decisione:** RQ1a = quanto cambiano numeri e ranking cambiando *come si misura* (fattori di misura: PA, oracolo,
  proiezione offline, metrica), a parità di modello addestrato. RQ1b = sensibilità dei modelli alle scelte di *design*
  (scaler, scoring, aggregazione, politica di soglia). Livelli dei fattori: D-42.
- **Documenti:** `CLAUDE.md` §1; proposta §4.9; `piano_tesi_v1.md` (rivisto da questa voce).

### D-32 · Confine train → test — APPROVATA · 2026-11-09
- **Decisione** (formulazione rev. 2):
  - **Base documentale:** secondo il README ufficiale di OmniAnomaly, per ogni macchina il train è la prima metà della
    serie e il test la seconda. La contiguità tra fine del train e inizio del test è quindi plausibile ma non
    verificabile; nessuna conclusione primaria dipende da essa.
  - **Controlli strutturali bloccanti nel manifest**, per ogni macchina: differenza tra le lunghezze di train e test ≤ 1;
    38 sensori; lunghezza delle label = lunghezza del test; valori in [0, 1]; hash dei file.
  - **Analisi primaria:** nessuna finestra di scoring usa il contesto all'inizio del test, per tutte le macchine.
    I primi W−1 punti del test sono esclusi dalla maschera comune di valutazione per tutti i modelli.
  - **Variante RQ1a:** contesto da cal all'inizio del test, senza nuovi fit; l'effetto si riporta rispetto all'analisi primaria.
  - **Confini rigidi:** tra fit ed es; nello scoring di cal ed es; nelle finestre usate per il fit; in modalità `legacy2`.
    Mai finestre tra macchine diverse né oltre la fine del test.
  - **Soglie mobili:** la storia parte da cal (D-04), indipendentemente dall'assunzione di contiguità.
  - **Diagnostica di confine:** facoltativa e solo descrittiva, senza ruolo decisionale.
- **Limiti dichiarati:** la regolarità del campionamento e l'assenza di buchi interni alle serie non sono verificabili.
  La normalizzazione dei dati su train+test è una limitazione separata.
- **Motivazione:** le conclusioni primarie non devono dipendere da un'assunzione non verificabile; il suo effetto si
  misura come variante di RQ1a invece di essere assunto.
- **Collegamenti:** D-03 (contesto nello scoring), D-04 (storia delle soglie), D-31 (segmenti con contesto), D-42 (livelli di RQ1a).
- **Serve entro:** F0.
- **Documenti:** `CLAUDE.md` §5; proposta §1, §2, §4.2, §10, §11; piano F0, F3, F6.

### D-20 · Contesto iniziale del test e maschera di valutazione — APPROVATA · 2026-11-09
- **Decisione:**
  - analisi primaria: nessun contesto all'inizio del test;
  - i primi W−1 punti del test sono esclusi dalla maschera comune di valutazione;
  - variante RQ1a: contesto da cal all'inizio del test, senza nuovi fit;
  - per ogni macchina tutti i modelli sono valutati sulla stessa maschera.
- **Precisazione (2026-11-09):** la maschera è specifica dello scenario e comune a tutti i modelli all'interno dello
  scenario: nell'analisi primaria i primi W−1 punti del test sono esclusi; nella variante con contesto sono inclusi.
  La coda del test è disciplinata da D-22.
- **Rapporto con altre voci:** integra D-03, il cui testo resta invariato, ed è subordinata a D-32: in caso di
  divergenza prevale D-32.
- **Motivazione:** rende operativa D-32 per finestre e maschere; la stessa maschera per tutti i modelli mantiene i
  confronti appaiati sugli stessi timestamp.
- **Collegamenti:** D-03, D-32, D-31 (sostituita), D-21, D-22, D-42 (livelli di RQ1a).
- **Serve entro:** F3.
- **Documenti:** `CLAUDE.md` §5; proposta §1, §2, §4.2, §10; piano F3, F6, F8.

### D-21 · Soglie mobili: punti privi di score — APPROVATA · 2026-11-09
- **Decisione:** nella sequenza cal → test usata dalle soglie mobili (D-04):
  - i punti privi di score (ad esempio i primi W−1 punti del test nell'analisi primaria, D-20) non aggiornano la storia
    della soglia mobile;
  - la soglia conserva la storia accumulata in cal;
  - l'aggiornamento riprende quando sono disponibili gli score del test.
- **Precisazione (2026-11-09):** `n_past` indica gli ultimi `n_past` score validi disponibili, anche se non
  corrispondono a timestamp consecutivi. I punti senza score non aggiornano la storia.
- **Rapporto con altre voci:** integra D-04, il cui testo resta invariato.
- **Motivazione:** la storia delle soglie resta definita anche dove manca lo score, senza riempire il vuoto con valori
  inventati e senza escludere un tratto iniziale del test.
- **Collegamenti:** D-04, D-20, D-32.
- **Serve entro:** F6.
- **Documenti:** proposta §4.6, §5.1; piano F6.

### D-22 · Coda del test esclusa dalla maschera comune — APPROVATA · 2026-11-09
- **Decisione:** gli ultimi W−1 punti del test sono esclusi dalla maschera comune di valutazione per tutti i modelli e
  per tutti gli scenari (analisi primaria e varianti di RQ1a, qualunque `timestamp_agg`), anche quando alcuni modelli
  producono score validi in quei punti.
- **Rapporto con altre voci:** integra D-20, che disciplina l'inizio del test; coerente con D-32 (nessuna finestra oltre
  la fine del test).
- **Motivazione:** con `timestamp_agg` offline (`mean`, `max`) gli ultimi W−1 punti sono coperti da meno finestre e il
  loro score ha una distribuzione diversa; escluderli per tutti mantiene gli scenari confrontabili sugli stessi punti
  finali. È la stessa regola della pipeline storica (`common_support_mask` con `trim_edges=True`), quindi `legacy2` non
  cambia.
- **Costo:** W−1 punti per macchina, eventuali anomalie comprese; da dichiarare tra i limiti.
- **Collegamenti:** D-12 (RQ1a), D-20, D-32, D-42.
- **Serve entro:** F3.
- **Documenti:** `CLAUDE.md` §5; proposta §4.2; piano F3, F6.

### D-23 · Supporto di calibrazione in cal — APPROVATA · 2026-11-10
- **Decisione:**
  - **Normalizer degli errori** (storico e robusto, D-11): fittato sui residui di tutte le finestre di scoring di cal
    (interne a cal, senza contesto, D-32), a livello di finestra, prima della proiezione sui timestamp.
  - **Soglie statiche e baseline random:** percentile degli score per timestamp di cal sulla maschera
    M_cal = [W−1, T_cal−W+1), uguale per tutti i modelli, per ogni `timestamp_agg` e per entrambi gli scenari di RQ1a
    (D-20, D-22).
  - **Storia delle soglie mobili:** tutti gli score validi di cal in ordine temporale, coda inclusa, poi il test (D-04, D-21).
  - **Calibrazione di `roll_robust_z`:** percentile dello z su M_cal ∩ {timestamp con almeno `n_past` score validi precedenti}.
  - In `legacy2` coincide con la pipeline storica: cal inizia allo stesso indice della validation storica
    (`round(0,8·n)`) in tutte le 28 macchine (verificato con `split_bounds` sulle 28 lunghezze reali).
  - Il numero di punti di calibrazione per macchina e configurazione si registra nella tabella lunga.
- **Motivazione:** le finestre di cal sono tutte complete, quindi il normalizer non ha motivo di scartarne; le soglie
  statiche usano la stessa regola di maschera del test, quindi il quantile è confrontabile per ogni `timestamp_agg`;
  il supporto non dipende dalla proiezione e non si confonde con essa in RQ1a; parità con lo storico per costruzione.
- **Limiti dichiarati:** normalizer e soglie stimati sullo stesso segmento (coerente con D-04): la quota di allarmi su cal
  è esatta per costruzione. Con `n_past` alto la calibrazione di `roll_robust_z` dispone di pochi punti
  (1.659 sulla macchina più corta con `n_past` = 2.880; vedi D-42).
- **Collegamenti:** D-04, D-11, D-20, D-21, D-22, D-24, D-32, D-42.
- **Serve entro:** F6.
- **Documenti:** proposta §4.4, §4.6; piano F6.

### D-24 · Soglie mobili solo con proiezione causale — APPROVATA · 2026-11-10
- **Decisione:** le politiche di soglia mobile (`roll_quantile`, `roll_robust_z`) si applicano solo con
  `timestamp_agg: last`. Con proiezioni offline (`mean`, `max`) si usano solo soglie statiche e metriche senza soglia.
- **Motivazione:** con proiezioni offline lo score al tempo t dipende da dati fino a t + W − 1; una soglia mobile
  applicata a quello score non sarebbe causale.
- **Conseguenza:** il disegno di RQ1a non è un fattoriale completo (manca la combinazione soglia mobile × proiezione
  offline); il metodo di decomposizione va deciso in D-25.
- **Collegamenti:** D-21, D-23, D-25, D-42.
- **Serve entro:** F6.
- **Documenti:** proposta §4.6, §4.8; piano F6.

### D-26 · Normalizzazione originale di SMD su train+test — APPROVATA · 2026-11-10 (fatto verificato)
- **Fatto verificato** (2026-11-10, sola lettura di `dataset/`, tolleranza 1e−6): sulle 28 macchine ci sono 1.064 sensori,
  di cui 175 costanti su train+test. Tutti gli 889 sensori non costanti hanno minimo esattamente 0 e massimo
  esattamente 1 sull'**unione** train+test; sul solo train questo vale per 186 sensori, sul solo test per 498.
  Nessun valore esce da [0, 1].
- **Interpretazione:** i dati distribuiti sono normalizzati min-max per sensore e per macchina usando train e test
  insieme. Una normalizzazione sul solo train darebbe minimo 0 e massimo 1 sul train per tutti i sensori non costanti,
  ed è quindi esclusa. La procedura esatta degli autori non è documentata nel repository.
- **Conseguenze:**
  - è la «limitazione separata» citata da D-32: il range dei dati dipende anche dal test, per tutti i modelli allo
    stesso modo; va dichiarata tra i limiti della tesi e non è correggibile a valle;
  - lo scaler `none` non è un'opzione pulita (proposta §4.3, D-40);
  - il controllo strutturale «valori in [0, 1]» di D-32 è soddisfatto sui dati reali.
- **Collegamenti:** D-32, D-40.
- **Serve entro:** F0 (registrazione); dichiarazione nei limiti della tesi.
- **Documenti:** proposta §4.3, §11.

### D-34 · Metriche da implementare in F6 e trattamento di VUS-PR — APPROVATA · 2026-11-10
- **Decisione:**
  - **senza soglia:** AP, AP normalizzata (primaria, D-07), ROC-AUC, AP/p;
  - **con soglia (RQ4):** F1, precision, recall, FPR, falsi allarmi ogni 1.440 punti;
  - **per evento (RQ4):** event recall, ritardo di rilevamento in punti, segmenti di falso allarme;
  - **diagnostiche, sempre etichettate come tali:** celle 2×2 PA/oracolo, best-F1 oracolo;
  - **VUS-PR (secondaria, D-07):** non in F6. Passo separato prima di F10, convalidato contro valori di riferimento
    calcolati una sola volta con l'implementazione ufficiale in un ambiente separato e salvati come fixture di test.
    Nessuna nuova dipendenza nel progetto (D-18).
- **Nota:** "falsi allarmi al giorno" si può riportare nella tesi solo come "ogni 1.440 punti, cioè al giorno assumendo
  un punto al minuto": la frequenza non è verificabile (D-32).
- **Collegamenti:** D-07, D-18, D-32. **Serve entro:** F6; VUS-PR entro F10.
- **Documenti:** proposta §4.7; piano F6, F10.

### D-51 · Lista delle 6 macchine di sviluppo — APPROVATA · 2026-11-10
- **Decisione:**
  - **fisse:** `machine-1-1` e `machine-3-8` (già osservate: i loro numeri hanno motivato scelte di protocollo, D-52);
  - **estratte:** 1 macchina dal gruppo 1 (esclusa 1-1), 2 dal gruppo 2, 1 dal gruppo 3 (esclusa 3-8);
  - **procedura:** candidati di ogni gruppo ordinati per nome; `numpy.random.default_rng(20261110)`; estrazione senza
    reinserimento, nell'ordine gruppo 1, 2, 3; nessuna lettura di label, metriche o risultati;
  - script riproducibile `scripts/make_dev_split.py`; lista in `configs/dev_machines.yaml` con seed e procedura;
  - le altre 22 macchine sono di valutazione.
  - Ordinamento lessicografico dei nomi; la regola non si modifica dopo l'estrazione. Le macchine estratte hanno pochi
    eventi (2-8: 1; 3-11: 3): da considerare in D-41.
- **Motivazione:** chi è già stato osservato va tra le macchine di sviluppo, così le 22 di valutazione restano mai
  usate per scelte di protocollo; 2 macchine per gruppo.
- **Collegamenti:** D-08, D-52. **Serve entro:** F0 (usata dal manifest in F3).

### D-52 · Macchine usate per i numeri preliminari — APPROVATA · 2026-11-10
- **Decisione:** chiusa da D-51: `machine-1-1` e `machine-3-8` sono macchine di sviluppo.
- **Limite dichiarato:** l'assegnazione è avvenuta dopo averle osservate; va scritto nella tesi.

---

## C. Software e riproducibilità — decisioni approvate

### D-14 · Verifica di "nessun cambiamento numerico" — APPROVATA · 2026-11-09
- **Decisione:** fino a M1: snapshot sintetico della pipeline storica (F1) e test golden dei residui. A M1: confronto su
  dati reali lanciato dall'utente. Dopo M1: confronto tra run dello `store/`. `confronta_run.py` resta lo strumento per
  la pipeline storica.
- **Documenti:** `CLAUDE.md` §5; proposta §9; piano F1, M1.

### D-15 · Interfaccia di `fit` invariata, `FitInfo` come attributo — APPROVATA · 2026-11-09
- **Decisione:** `fit(X, y=None, X_val=None) -> self` non cambia. Le informazioni di training vanno in `self.fit_info_`
  (`FitInfo`). La nuova pipeline passa fit come `X` ed es come `X_val`. `residuals`, `save`, `load` sono aggiunte.
- **Motivazione:** pura aggiunta, nessuna modifica strutturale ai sensi di `CLAUDE.md` §8, test esistenti invariati (C7).
- **Documenti:** `CLAUDE.md` §6; proposta §5.1; piano F4.

### D-16 · Training sintetico nei test — APPROVATA · 2026-11-09
- **Decisione:** i test possono addestrare modelli minimi su dati sintetici generati in `tmp_path`; non leggono `dataset/`
  e non scrivono in `results/`, `store/` o altre cartelle del repository.
- **Documenti:** `CLAUDE.md` §4; piano F1.

### D-17 · Procedura di M1 e tolleranze — APPROVATA · 2026-11-09
- **Decisione:** PCA e baseline deterministiche: tolleranza zero. AE: prima si tenta l'identità bit a bit con lo stesso
  numero di thread dei run storici; se ci sono differenze, indagine preliminare documentata della causa, poi eventuale
  tolleranza motivata. Il numero di thread si registra nella provenance.
- **Documenti:** `CLAUDE.md` §5; piano M1.

### D-18 · Dipendenze — APPROVATA · 2026-11-09
- **Decisione:** ammesse quelle di `requirements.txt` (numpy, pandas, scipy, scikit-learn, torch, pyyaml, matplotlib,
  seaborn, tqdm). Nessuna nuova dipendenza senza approvazione. `pyarrow` non si aggiunge.
- **Documenti:** `CLAUDE.md` §6; proposta §3, §10, §11.

### D-19 · Formato iniziale dello store: CSV.gz — APPROVATA · 2026-11-09
- **Decisione:** tabelle lunghe e registro in CSV (tabelle compresse `.csv.gz`). Lettura e scrittura isolate in
  `src/pipeline/store.py` / `table_schema.py`, così un passaggio futuro a parquet resta una modifica locale.
- **Documenti:** proposta §5.2, §5.4, §10, §11; piano gruppo A punto 5, F2.

### D-33 · Aggregazioni dei residui da salvare — APPROVATA · 2026-11-10
- **Decisione:** lo stadio di fit salva i residui per sensore con `window_time_agg` ∈ {`last`, `mean`}. `max` non si salva.
- **Motivazione:** `last` e `mean` sono i livelli del fattore di design previsti; `max` aumenterebbe lo spazio di circa
  il 50% e, se servisse, si rigenera dai checkpoint con `residual_cache` senza nuovi fit.
- **Collegamenti:** D-19, D-42. **Serve entro:** F2 (schema, `fit_key`), F4–F5.
- **Documenti:** proposta §4.4, §5.4, §8; piano F2, F4, F5.

### D-50 · Run storici di riferimento per M1 — APPROVATA · 2026-11-10
- **Decisione:** il riferimento per M1 è la coppia `results/results_tests/results_repro_last_01` /
  `results_repro_mean_01`. `results_repro_last` e `results_repro_mean` restano solo come storico.
- **Approvazione:** utente, 2026-11-10, su raccomandazione di Claude Project.
- **Fatti verificati (2026-11-10):** `results/` contiene solo `results_tests/`, con quattro run tracciati da git, tutti
  su `machine-1-1` e `machine-3-8` con seed 42:
  - `results_repro_last_01` e `results_repro_mean_01`: aggiunti nel commit `efdacc6` (2026-11-09), insieme al nuovo
    `scaler.py`; le config attuali `configs/config_tests/repro_{last,mean}.yaml` puntano a queste cartelle;
  - `results_repro_last` e `results_repro_mean`: aggiunti nel commit `843da6a` (2026-11-05), con lo scaler precedente.
  Il codice di calcolo (`src/`, `main.py`, `aggregate_results.py`, `configs/`) è invariato da `efdacc6` a `e3b28f0`.
- **Motivazione:** i run `*_01` sono prodotti con lo scaler attuale e con lo stesso codice di calcolo della pipeline
  storica congelata (D-13), quindi sono il confronto corretto per la parità di M1.
- **Limite:** le versioni delle librerie usate per produrre i run non sono registrate negli output; in caso di
  differenze a M1 si applica la procedura di D-17.
- **Collegamenti:** D-13, D-14, D-17, D-55.
- **Serve entro:** M1.
- **Documenti:** `CLAUDE.md` §4; piano gruppo A punto 6, F0, M1.

### D-55 · Ambiente di riferimento e file di lock — APPROVATA · 2026-11-10
- **Decisione:** ambiente di riferimento = ambiente Windows dell'utente con **Python 3.13.9**, lo stesso per test, M1 e
  run ufficiali. `requirements-lock.txt` generato con `python -m pip freeze` da quell'ambiente.
- **Nota:** lock provvisorio, generato dall'installazione globale di Python (contiene pacchetti estranei al progetto).
  Prima di M1: ambiente virtuale dedicato e lock rigenerato da lì, registrato come aggiornamento di questa voce.
- **Collegamenti:** D-17, D-18.

### D-56 · Posizione di `store/` e budget di disco — APPROVATA · 2026-11-10
- **Decisione:** `store/` nella radice del progetto, esclusa da git con `.gitignore`. Budget di disco: 5–10 GB.

### D-53 · File tracciati da git (`.pyc`, dataset) — APPROVATA · 2026-11-10
- **Decisione:** i `.pyc` non sono tracciati: i 35 file già tracciati sono tolti dal tracciamento con
  `git rm -r --cached` sulle cartelle `__pycache__` (restano sul disco) e `.gitignore` esclude `__pycache__/` e `*.pyc`.
  `dataset/` resta tracciato (sola lettura, riproducibilità).
- **Approvazione:** utente, 2026-11-10, su raccomandazione di Claude Project.
- **Collegamenti:** D-55, D-56.

---

## D. Decisioni aperte (metodologiche)

### D-25 · Metodo di decomposizione della varianza per RQ1a — APERTA
- **Conflitto:** `piano_tesi_v1.md` §2 prevede un modello misto
  (`metrica ~ modello * fattori + (1|macchina) + (1|gruppo)`); la proposta §4.8 prevede somme dei quadrati su disegno
  bilanciato, che con D-24 non è più completo.
- **Opzioni:** (a) decomposizione a blocchi, separata per proiezione causale e offline; (b) modello lineare per disegno
  sbilanciato (somme dei quadrati di tipo II/III) con numpy/scipy; (c) modello misto, che richiede statsmodels:
  nuova dipendenza da approvare (D-18).
- **Serve entro:** F7.

### D-30 · Margine δ — APERTA
- **Opzioni:** (a) δ = 0,05 in AP normalizzata; (b) δ ricavato solo dalle macchine di sviluppo con una regola dichiarata;
  (c) altro valore motivato. Criterio scartato: δ ≥ spread del protocollo sugli stessi dati (circolare).
- **Vincolo:** va fissato prima dei run ufficiali e congelato in `configs/analysis/official.yaml`.
- **Serve entro:** F10. Il codice di F7 deve trattare δ come parametro, senza valore predefinito.

### D-36 · Metodo degli intervalli di confidenza e famiglia confermativa — APERTA
- **Contrasto non risolto:** `piano_tesi_v1.md` prevede bootstrap gerarchico (cluster = gruppo); la proposta raccomanda
  bootstrap stratificato per gruppo (con 3 cluster il gerarchico ha 10 ricampionamenti distinti). Da decidere anche:
  famiglia confermativa {Conv vs PCA, LSTM vs PCA}, α, correzione di Holm.
- **Serve entro:** F7 (non F10): F7 implementa il bootstrap. In alternativa F7 implementa entrambi i metodi e la scelta
  si congela in F10.

### D-37 · PCA di riferimento per RQ2 — APERTA
- **Opzioni:** PCA a finestre (stesso input degli AE, raccomandata) oppure la migliore tra le due PCA scelta sulle
  macchine di sviluppo. Distinta da D-06 (dati di fit).
- **Serve entro:** F10.

### D-38 · `train_step` — APERTA
- **Opzioni:** 5 (storico) oppure 1; in ogni caso identico per tutti i modelli. **Serve entro:** F8.

### D-39 · `max_epochs` e patience — APERTA
- **Proposta:** 100 e 10. **Serve entro:** F8.

### D-40 · Preprocessing canonico — APERTA
- **Opzioni:** storico (minmax su fit, senza clip né pavimento; raccomandato) oppure con pavimento o clip.
  **Serve entro:** F8.

### D-41 · Budget di tuning — APERTA
- **Da decidere:** K per famiglia (proposta 8), griglie, criterio di selezione (proposta: AP normalizzata media sulle 6
  macchine di sviluppo, una configurazione globale per famiglia), seed nel tuning (proposta 3), se scoring e normalizer
  robusto rientrano nel budget. **Serve entro:** F9.

### D-42 · Livelli dei fattori per RQ1a e RQ1b — APERTA · **Serve entro:** F10.
- **Nota (2026-11-10, la voce resta APERTA):**
  - **Raccomandazione per i livelli di `n_past` di `roll_robust_z`:** escludere 2.880 dai livelli ufficiali oppure
    limitarlo agli esperimenti esplorativi (circa 8 superamenti per un percentile al 99,5% su 1.659 punti autocorrelati).
  - Già fissato da D-32: fattore "contesto all'inizio del test: sì/no". Vincolo da D-24: soglie mobili solo con `last`.

### D-43 · Descrittori per RQ3 (massimo 4) e ipotesi — APERTA · **Serve entro:** F10.
### D-44 · Regole di esclusione (fit fallito, macchina con un solo evento) — APERTA · **Serve entro:** F10.

### D-45 · Secondo dataset — APERTA
- **Vincolo:** deve avere più entità (PSM escluso; ASD candidato da verificare). Il piano congelato per SMD si riapplica
  invariato. **Serve entro:** da scegliere presto; condiziona la forza delle conclusioni.

### D-46 · Sensibilità delle proporzioni dello split — APERTA
- **Opzioni:** nessuna; oppure una sola variante (es. 70/15/15) sulle macchine di sviluppo in F9. **Serve entro:** F9.

---

## E. Decisioni aperte che dipendono dall'utente

### D-54 · Scadenze e tempo disponibile — APERTA — DIPENDE DALL'UTENTE
- Determinano se F9 e il secondo dataset sono realistici.

---

## F. Voci sostituite (storico)

### D-31 · Quali segmenti ricevono il contesto nello scoring — SOSTITUITA da D-32 · 2026-11-09
- **Stato:** sostituita da D-32 (2026-11-09). D-32 disciplina ora il contesto iniziale del test e i segmenti; la gestione
  operativa del contesto iniziale del test e della maschera è in D-20. Il testo seguente resta come riferimento storico.
- **Problema:** con `window_time_agg: mean`, le prime W−1 finestre di cal con contesto includerebbero residui di es, che
  per gli AE sono ottimistici; entrerebbero così nel normalizer e nelle soglie, in contrasto con D-04.
- **Opzioni:** (a) contesto solo per il test (da cal); cal ed es senza contesto (si perdono W−1 punti di cal su ≈4.700);
  (b) contesto per tutti i segmenti di scoring, dichiarando l'effetto; (c) contesto per cal solo con `window_time_agg: last`.
- **Raccomandazione:** (a).
- **Serve entro:** F3.
