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
- **Nota di rimando (2026-11-12, F5, il testo sopra resta invariato):** in `legacy2` non esiste un array di residui
  `es`: es e cal sono lo stesso segmento (`es_is_cal`, doppio uso P1 dichiarato, D-63) e lo stadio di fit salva solo
  `cal` e `test`. I residui `es__<agg>`, marcati `diagnostic_only`, esistono solo in `split3`; `residual_cache` rifiuta
  di leggerli senza un flag esplicito di diagnostica (D-70).

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
- **Nota di rimando (2026-11-16, F7, il testo sopra resta invariato):** i seed sono `experiment.seeds` della config
  dell'esperimento, non di un piano. Decisione dell'utente (2026-11-16, su raccomandazione di Claude Project):
  `configs/experiments/official.yaml` (seed, modelli e iperparametri selezionati in F9, preprocessing, macchine) si
  scrive e si congela in F10 con il suo hash nel registro; F11 la esegue soltanto. Ordine di congelamento: config
  dell'esperimento → piano di analisi → piano statistico (che dichiara gli hash dei primi due, D-73 P6).

### D-10 · OC-SVM, Isolation Forest, VAE — APPROVATA · 2026-11-09
- **Decisione:** restano nel registro dei modelli e nei test; sono esclusi dai run ufficiali e dal tuning.
- **Documenti:** `CLAUDE.md` §6; proposta §11.
- **Nota di rimando (2026-11-16, il testo sopra resta invariato):** il perimetro dei run ufficiali si estende con gli
  AE non addestrati come baseline descrittiva (D-69, approvata) e con un modello di previsione (D-71 approvata, punti
  aperti in D-74); un modello recente è un'eventuale estensione separata (D-75, aperta).

### D-11 · Normalizer robusto come variante — APPROVATA · 2026-11-09
- **Decisione:** si aggiunge un normalizer robusto (mediana/IQR) come variante, accanto a quello storico. Se entra nel budget
  di tuning come opzione di scoring: D-41.
- **Motivazione:** errori a coda pesante e possibile contaminazione del train da anomalie non etichettate (da dichiarare tra i limiti).
- **Documenti:** proposta §4.4; piano gruppo A punto 4, F6.
- **Nota di rimando (2026-11-12, F6, il testo sopra resta invariato):** la definizione operativa (centro, scala,
  pavimento) e la registrazione dei sensori con la scala sul pavimento sono in D-72 (C5).

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
- **Nota di rimando (2026-11-16, F7, il testo sopra resta invariato):** la «Conseguenza» vale solo se la politica di
  soglia fosse un fattore di RQ1a. Per D-12 (verificato sul testo) è un fattore di design (RQ1b), che si legge a
  `ts_agg = last`; RQ1a a soglia statica fissa è un disegno completo. Il non incrocio reale è la metrica. Decomposizione
  approvata in D-25.

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
- **Nota di rimando (2026-11-12, F6, il testo sopra resta invariato):** le definizioni operative di «falsi allarmi
  ogni 1.440 punti» e del ritardo di rilevamento, la formula unica dell'F1, le righe prodotte per ogni politica di
  soglia e le celle 2×2 sono in D-72 (C2, C3, C4, V10). La dicitura «FA/giorno» della proposta (§5.1, §9, §12 Fase 4)
  non si usa: vale la nota sopra («ogni 1.440 punti»).

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

### D-60 · Contesto all'inizio del test solo con `step` = 1 — APPROVATA · 2026-11-10
- **Decisione:** la variante con contesto (D-20, D-32) è definita solo con `step` = 1: le finestre del test con contesto
  usano esattamente gli ultimi W−1 punti di cal. Con `step` > 1 la richiesta del contesto è un errore esplicito.
  Lo scenario con contesto è ammesso solo in `split3` (in `legacy2` confini rigidi, D-32).
- **Motivazione:** con `step` > 1 la griglia delle finestre con contesto (inizi −(W−1) + k·step) non contiene in
  generale quella senza contesto (inizi k·step): le finestre senza contesto non sarebbero più un sottoinsieme di quelle
  con contesto, contro il contratto dei residui del test (`FIT_CODE_VERSION`, residui con le W−1 finestre di contesto).
  Il valore di base è `step` = 1 e nessuna RQ richiede il caso generale. Alternativa scartata: griglia allineata sulle
  fini delle finestre, con contesto di ⌊(W−1)/step⌋·step punti.
- **Rapporto con altre voci:** integra D-03 e D-20, il cui testo resta invariato.
- **Collegamenti:** D-03, D-20, D-32, D-42. **Serve entro:** F3.
- **Documenti:** piano F3; proposta §4.2.

### D-65 · Segmento di fit dello scaler in `split3` — APPROVATA · 2026-11-10
- **Decisione:** in `split3` lo scaler (qualunque tipo e parametri, D-40) si fitta **solo su fit**; es, cal e test
  ricevono solo `transform`. In `legacy2` vale la regola storica (D-63).
- **Motivazione:**
  - coerenza con D-06: lo scaler è un parametro appreso, come le componenti della PCA, e si stima sugli stessi dati;
  - fit + cal escluso: cal deve restare non visto da ogni componente fittato (D-23, P1);
  - fit + es scartato per simmetria con D-06.
- **Interazione con D-40 (non indipendente):** con lo scaler su fit (60% del train) gli intervalli stimati per sensore
  sono ≤ di quelli sulla parte di train storica (80%, `legacy2`), quindi l'amplificazione 1/range dei sensori quasi costanti può
  aumentare. La verifica in sola lettura proposta in D-40 sulle 6 macchine di sviluppo deve misurare gli intervalli
  anche su fit (60%), oltre che sulla parte di train storica (80%).
- **Incoerenza documentale trovata (2026-11-10):** `CLAUDE.md` §5 («fit → apprendere i parametri (AE, PCA, scaler)»)
  attribuiva la regola a D-05, il cui testo non cita lo scaler. Corretta in `CLAUDE.md` §5 con il rimando a D-65:
  è solo un riferimento, il protocollo non cambia.
- **Opzioni scartate:** (b) fit + es; (c) fit + es + cal.
- **Serve entro:** F3 (ramo `split3` di `fit_scaler`), F8.
- **Collegamenti:** D-04, D-05, D-06, D-23, D-40, D-63.
- **Documenti:** `CLAUDE.md` §5; piano F3, F8.

### D-25 · Metodo di decomposizione della varianza per RQ1a e RQ1b — APPROVATA · 2026-11-16
- **Testo della voce APERTA (storico, invariato):**
  - *Conflitto:* `piano_tesi_v1.md` §2 prevede un modello misto
    (`metrica ~ modello * fattori + (1|macchina) + (1|gruppo)`); la proposta §4.8 prevede somme dei quadrati su disegno
    bilanciato, che con D-24 non è più completo.
  - *Opzioni:* (a) decomposizione a blocchi, separata per proiezione causale e offline; (b) modello lineare per disegno
    sbilanciato (somme dei quadrati di tipo II/III) con numpy/scipy; (c) modello misto, che richiede statsmodels:
    nuova dipendenza da approvare (D-18).
  - *Serve entro:* F7.
- **Premessa (fatto verificato sul testo di D-12, 2026-11-16):** la politica di soglia è un fattore di *design* (RQ1b),
  l'oracolo un fattore di *misura* (RQ1a). («Aggregazione», tra i fattori di design di D-12, è l'aggregazione
  dell'errore dentro la finestra, `window_time_agg`, non la proiezione `timestamp_agg`: proposta §1 punto 3 e §4.4,
  dove `window_time_agg` e `sensor_agg` sono di design e `timestamp_agg` è di misura; D-12 elenca la «proiezione
  offline» tra i fattori di RQ1a.) Quindi:
  - RQ1a a soglia statica calibrata fissa: {statica, oracolo} × PA × `ts_agg` × scenario è un disegno **completo**;
  - RQ1b si legge a `ts_agg = last` (proiezione canonica): politica (mobili comprese) × scoring è **completo**;
  - il non incrocio reale è la **metrica**: le metriche senza soglia non hanno soglia né PA e le scale sono diverse,
    quindi «metrica» non è un fattore di una somma dei quadrati; si tratta con τ di Kendall e inversioni di segno (D-73).
  La «Conseguenza» di D-24 riguarda solo il caso, escluso da D-12, della politica di soglia come fattore di RQ1a.
- **Decisione:** opzione (a), **somme dei quadrati classiche a blocchi bilanciati, una risposta per metrica**:
  - blocchi RQ1a: modello × `ts_agg` × scenario × {statica canonica, oracolo} × PA per `f1`; modello × `ts_agg` ×
    scenario per le metriche senza soglia. Blocchi RQ1b a `ts_agg = last`, scenario `primary`: modello × scoring
    (`window_time_agg`, `sensor_agg`, `normalizer`) × politica (livello, `n_past`) (× scaler, tra analisi diverse).
    Livelli: D-42. I blocchi modello × scoring × politica comprendono solo i modelli con la griglia di scoring
    completa; i modelli `native` entrano in RQ1b solo con il fattore politica di soglia (e scaler), in un blocco
    separato (fatto verificato: per i modelli `native_only` `analysis_grid.fit_scores` produce la sola combinazione
    `none` / `native` / `none`);
  - la macchina è un fattore di blocco; i seed sono mediati prima (unità = macchina); una sola osservazione per cella
    e macchina; un disegno sbilanciato dato a un blocco è un errore, mai un adattamento silenzioso;
  - **interazioni modello × fattore in evidenza** nelle uscite, non solo gli effetti principali: un fattore che sposta
    tutti i modelli allo stesso modo (per esempio il PA) può avere una quota grande senza cambiare il confronto tra
    modelli, che è la domanda della tesi;
  - quote **η² = SS_fattore / SS_tot** (additive), in due versioni: con SS_tot complessivo e con SS_tot senza
    SS_macchina; il piano statistico dichiara quale si legge come principale (nessun default);
  - `stats.json` elenca per ogni blocco l'insieme delle macchine incluse: le esclusioni di D-44 possono far differire
    i blocchi.
- **Motivazione:** con la premessa sopra il disegno è completo dentro ogni metrica; la decomposizione è esatta (somma
  al totale), trasparente e senza nuove dipendenze. (b) dà quote non additive e dipendenti dalla codifica; (c) richiede
  statsmodels (D-18), quote non univoche per gli effetti fissi e un `(1|gruppo)` con 3 livelli non affidabile.
- **Costo dichiarato:** un'esclusione di D-44 toglie la macchina dall'intero blocco (bilanciamento).
- **Approvazione:** utente, 2026-11-16, su raccomandazione di Claude Project (piano F7).
- **Collegamenti:** D-12, D-24, D-42, D-44, D-73. **Serve entro:** F7.
- **Documenti:** piano F7; proposta §4.8 (superata su questo punto); `piano_tesi_v1.md` §2 (modello misto non adottato).

### D-36 · Metodo degli intervalli di confidenza e famiglia confermativa — APPROVATA · 2026-11-16
- **Testo della voce APERTA (storico, invariato):**
  - *Contrasto non risolto:* `piano_tesi_v1.md` prevede bootstrap gerarchico (cluster = gruppo); la proposta raccomanda
    bootstrap stratificato per gruppo (con 3 cluster il gerarchico ha 10 ricampionamenti distinti). Da decidere anche:
    famiglia confermativa {Conv vs PCA, LSTM vs PCA}, α, correzione di Holm.
  - *Serve entro:* F7 (non F10): F7 implementa il bootstrap. In alternativa F7 implementa entrambi i metodi e la scelta
    si congela in F10.
- **Decisione:**
  - **metodo di RQ2: (A) bootstrap delle macchine stratificato per gruppo** (numerosità dei gruppi conservate: 6/7/9
    fra le 22 di valutazione), statistica = media di Δ con pesi uguali per macchina, intervallo **BCa** con
    accelerazione jackknife leave-one-machine-out;
  - **(B) bootstrap gerarchico** (gruppi, poi macchine): implementato solo per documentare con la simulazione di
    copertura perché è scartato. **Non è un'alternativa selezionabile dopo aver visto i risultati:** non è un valore
    ammesso del piano statistico, si usa solo nelle simulazioni dei test;
  - **molteplicità: (M1)**, IC di Bonferroni al livello 1 − α/m, con la regola δ (D-30) applicata a quegli IC; un solo
    livello per i quattro esiti. Holm sui p-value (M2) non adottato. Nota: con (M1) il ramo «equivalente» richiede
    IC al 1 − α/m ⊂ [−δ, δ], quindi un'equivalenza più conservativa del TOST classico (che userebbe un IC al 1 − 2α);
  - **famiglia confermativa: parametro del piano statistico**, non una costante del codice; m si ricava dalla famiglia
    dichiarata. {Conv vs PCA, LSTM vs PCA} compare solo come esempio di configurazione. Motivo: il perimetro DL è
    aperto (D-71);
  - **α e B:** parametri senza default, congelati in F10. «B ≥ 10.000» per la BCa è una regola di validazione di
    `src/stats/plan.py` solo per `kind: official` (da F10); nei piani esplorativi B è un parametro libero, senza
    default, registrato in `stats.json`;
  - **sign-flip stratificato:** solo di supporto, verifica Δ = 0 (non δ); **non entra nella regola a quattro esiti**;
  - **bande di copertura della simulazione (fissate ora, non si allargano dopo):** caso normale [0,93; 0,97]; code
    pesanti e caso asimmetrico ≥ 0,90, con il valore ottenuto documentato. Se la copertura nel caso normale esce dalla
    banda (possibile: con strati da 6 macchine la varianza dentro lo strato è distorta di un fattore (n_h − 1)/n_h),
    è un risultato metodologico: ci si ferma e lo si riporta; né il test né il metodo si correggono senza una nuova
    voce di registro.
- **Limite dichiarato:** l'inferenza è condizionata ai 3 gruppi di SMD (gruppo come effetto fisso, nessuna
  generalizzazione ad altri gruppi); vedi D-45 (secondo dataset).
- **Motivazione:** con 3 cluster il gerarchico ha 10 multinsiemi distinti di gruppi e IC instabili; lo stratificato è
  valido con 3 strati; Bonferroni è coerente con una regola basata sugli IC; famiglia, α e B dichiarati prima evitano
  scelte a posteriori.
- **Approvazione:** utente, 2026-11-16, su raccomandazione di Claude Project (piano F7).
- **Collegamenti:** D-30, D-37, D-45, D-71, D-73. **Serve entro:** F7 (implementazione); α, B e famiglia entro F10.
- **Documenti:** piano F7, F10; proposta §1 punto 4, §4.8, §10 (superate su questi punti); `piano_tesi_v1.md` §2
  (bootstrap gerarchico non adottato).
- **Nota di rimando (2026-11-16, il testo sopra resta invariato):**
  - **famiglia confermativa prevista** (D-71): {Conv vs PCA, LSTM vs PCA, previsione vs PCA}, m = 3, congelata in F10
    come parametro del piano statistico; la molteplicità resta (M1), IC di Bonferroni al 1 − α/3. «Holm su tre
    confronti» non è adottato: contraddice la regola a quattro esiti basata sugli IC;
  - **terminologia:** «famiglia confermativa» (questa voce, insieme dei confronti su cui si corregge) ≠ «famiglia di
    modelli» (D-71, unità delle conclusioni); la correzione è unica sui tre confronti, non una per famiglia di modelli;
  - **limite aggiornato:** l'inferenza è condizionata ai 3 gruppi di SMD e non c'è un secondo dataset (D-45): RQ2 è
    confermativa rispetto al protocollo congelato, su dati già osservati con altri protocolli.

### D-45 · Secondo dataset — APPROVATA · 2026-11-16
- **Testo della voce APERTA (storico, invariato):**
  - **Vincolo:** deve avere più entità (PSM escluso; ASD candidato da verificare). Il piano congelato per SMD si riapplica
    invariato. **Serve entro:** da scegliere presto; condiziona la forza delle conclusioni.
- **Decisione:** nessun secondo dataset in questa tesi, per ora. Limite dichiarato.
- **Conseguenza:** la proposta §4.9 considera il secondo dataset necessario per un'affermazione confermativa, perché SMD
  è già stato osservato con i protocolli storici. Senza di esso RQ2 resta pre-registrata e congelata (F10), ma nella
  tesi va descritta come **«confermativa rispetto al protocollo congelato, su dati già osservati con altri
  protocolli»**. L'inferenza resta condizionata ai 3 gruppi di SMD (D-36).
- **Approvazione:** utente, 2026-11-16 (discussione sul perimetro della tesi, su raccomandazione di Claude Project).
- **Collegamenti:** D-08, D-36, D-52, D-73 (P13).
- **Documenti:** proposta §4.9 (nota), §7 punto 4 (replica non prevista); piano gruppo C punto 20.

### D-69 · AE non addestrati (baseline descrittiva) — APPROVATA · 2026-11-16
- **Testo della voce APERTA (storico, invariato):**
  - **Fatti verificati (2026-11-11):** `UntrainedLSTMAEDetector` e `UntrainedConvAEDetector` sono nel registro di `main.py`
    (`untrained_lstm_ae`, `untrained_conv_ae`) ma in nessuna config di `configs/` e in nessun run di D-50
    (`results_repro_last_01` / `results_repro_mean_01` contengono solo `pca`, `pca_pointwise`, `conv_autoencoder`,
    `lstm_autoencoder` e `random_baseline`). Non sono in `schema.MODEL_SCHEMAS`: la nuova pipeline oggi non li esegue.
  - **Rimandati da F4** (decisione dell'utente, 2026-11-11). Se una config li includesse prima della chiusura di questa
    voce, quella config non avrebbe parità verificata con la pipeline storica.
  - **Da decidere:**
    - chiavi di schema: `init_std`, `channels` / `hidden_dim`, `latent_dim`;
    - seed nel `fit_key` (modelli non deterministici: i pesi dipendono dal seed);
    - statistiche di BatchNorm del Conv-AE: in eval non vengono mai aggiornate (media 0, varianza 1); scelta da dichiarare.
  - **Serve entro:** F8.
  - **Collegamenti:** D-10, D-33, D-67, D-68.
  - **Nota (2026-11-12, F6, la voce resta APERTA):** proposta di Claude (non approvata): includere gli AE non addestrati
    come baseline che separa il contributo dell'architettura (pesi casuali, stesso input e stesso scoring) da quello
    dell'apprendimento, utile per interpretare Δ(DL − PCA) in RQ2–RQ3. La decisione spetta all'utente, entro F8; se
    approvata, servono le scelte elencate sopra e un controllo di parità dedicato (oggi nessuna config li usa). Vedi
    anche D-71.
- **Decisione:** gli AE non addestrati (Conv, LSTM) entrano nei run come **baseline descrittiva**, fuori dalla famiglia
  confermativa. Separano il contributo dell'architettura (pesi casuali, stesso input e stesso scoring) da quello
  dell'apprendimento, per interpretare Δ(DL − PCA).
- **Trattamento in F7:** come la baseline random (D-73 P7): righe di riferimento e descrittive, **mai** livelli del
  fattore modello in RQ1a, RQ1b e RQ4, mai nella famiglia confermativa né come riferimento di RQ2. Finché lo schema non
  esiste, il ruolo si dichiara nel piano statistico (chiave da aggiungere al passo 1-bis di F7, D-73).
- **Restano da fissare prima di F8** (punti della voce aperta): chiavi di schema (`init_std`, `channels` /
  `hidden_dim`, `latent_dim`); seed nel `fit_key` (pesi dipendenti dal seed); trattamento del BatchNorm del Conv-AE (in
  eval le statistiche non si aggiornano mai). Serve anche un controllo di parità dedicato (nessuna config storica li
  usa).
- **Ordine:** implementazione dopo M1, prima di F8.
- **Approvazione:** utente, 2026-11-16 (discussione sul perimetro della tesi, su raccomandazione di Claude Project).
- **Collegamenti:** D-10, D-33, D-36, D-67, D-68, D-71, D-73.
- **Documenti:** piano F8.
- **Nota (2026-11-16, il testo sopra resta invariato): proposta di Claude Project sui punti da fissare prima di F8,
  NON ancora approvata.**
  - **Fatti verificati nel codice (2026-11-16):**
    - le classi storiche `UntrainedLSTMAEDetector` e `UntrainedConvAEDetector` (`src/models/deep/untrained_ae.py`)
      reinizializzano i pesi con una normale di deviazione `init_std` (0,02 se assente), diversa dall'inizializzazione
      degli AE addestrati;
    - lo schema rifiuta `training.max_epochs: 0` (intero ≥ 1, `src/utils/schema.py`);
    - con zero epoche il ciclo di `PyTorchTrainer.fit` non viene eseguito e non c'è uno stato migliore da ripristinare;
    - il seed entra nel `fit_key` dei modelli non deterministici (`config.fit_payload`); il blocco `training` vi entra
      solo per i modelli con `trained` vero.
  - **Proposta:**
    - **stessa configurazione:** ogni AE non addestrato ha la stessa architettura, gli stessi iperparametri e lo stesso
      preprocessing dell'AE addestrato selezionato in F9, così l'unica differenza è l'addestramento; nessun tuning
      dedicato;
    - **inizializzazione:** lo stesso schema dell'AE addestrato. Le classi storiche `Untrained…` con `init_std` non si
      usano per i run;
    - **seed:** gli stessi dell'AE addestrato (D-09). La variabilità dovuta ai pesi casuali si misura, e il confronto
      con l'AE addestrato è appaiato per seed; il costo è trascurabile;
    - **realizzazione, da valutare in F8:** l'AE non addestrato è l'AE addestrato con zero epoche, così i pesi
      coincidono per costruzione con quelli iniziali dell'AE addestrato con lo stesso seed. Meglio un'opzione
      esplicita (per esempio `untrained: true`, che imposta zero epoche) di `max_epochs: 0`: rende il ruolo leggibile
      dalla config (riscontro con `descriptive_baselines`, D-73) ed evita conflitti con il validatore e con il tuning.
      Da verificare:
      - che il trainer e il modello gestiscano zero epoche da capo a fondo (storia vuota, stato `eval`);
      - che la selezione dell'epoca su es (split3) non abbia nulla da scegliere né da ripristinare;
      - che il seed entri nel `fit_key`;
      - test: stesso seed → pesi identici a quelli dell'AE addestrato prima della prima epoca;
    - **`fit_key`:** le impostazioni di addestramento (learning rate, batch, `max_epochs`…) **non** entrano nel
      `fit_key` dell'AE non addestrato: con zero epoche non influenzano i pesi, che dipendono solo da architettura e
      seed. È coerente con la regola esistente: il blocco `training` entra solo per i modelli con `trained` vero, e
      l'AE non addestrato ha `trained` falso. Due AE non addestrati con pesi identici hanno la stessa chiave.
      L'abbinamento con l'AE addestrato passa dalla config (stessa architettura, stesso seed), non dal `fit_key`. Test
      in F8: cambiare solo il learning rate → stesso `fit_key`;
    - **BatchNorm del Conv-AE:** raccomandate le statistiche iniziali (media 0, varianza 1), come nell'AE addestrato
      all'epoca 0. Alternativa scartata: ricalcolare le statistiche con un passaggio sui dati di train senza aggiornare
      i pesi, perché sarebbe già una forma di adattamento ai dati e il modello non sarebbe più «quello prima
      dell'addestramento». Se approvata, va registrata come scelta motivata;
    - **nella tesi:** una riga descrittiva nella tabella di RQ2 (fuori dalla famiglia confermativa, senza esito della
      regola a quattro esiti, etichettata come tale) e un paragrafo di interpretazione.
  - **Effetto sull'elenco «Restano da fissare prima di F8», se approvata:**
    - `init_std` non serve più;
    - `channels` / `hidden_dim` e `latent_dim` coincidono con quelli dell'AE addestrato selezionato;
    - seed nel `fit_key` e BatchNorm sono risolti dai punti sopra;
    - resta da fissare la chiave che dichiara il ruolo (per esempio `untrained: true`) e il controllo di parità
      dedicato.

### D-71 · Perimetro dei modelli DL — APPROVATA · 2026-11-16
- **Testo della voce APERTA (storico, invariato):**
  - **Fatto verificato (D-68, `schema.MODEL_SCHEMAS`):** i modelli profondi del nucleo sono solo autoencoder
    ricostruttivi (`conv_autoencoder`, `lstm_autoencoder`); VAE, OC-SVM e Isolation Forest sono esclusi dai run
    ufficiali (D-10); gli AE non addestrati sono rimandati (D-69).
  - **Problema:** una conclusione formulata su «il deep learning» generalizzerebbe oltre la famiglia di modelli valutata.
  - **Opzioni:**
    - (a) dichiarare le conclusioni limitate ai modelli ricostruttivi (Conv-AE, LSTM-AE) in tutte le RQ;
    - (b) aggiungere un modello recente come riferimento esterno; richiede una voce che integri D-10 (perimetro dei run
      ufficiali) e una scelta del modello, del budget e della parità di trattamento (D-41).
  - **Proposta:** (a) in ogni caso; (b) da valutare con il relatore dopo M1.
  - **Serve entro:** F8.
  - **Collegamenti:** D-10, D-41, D-68, D-69.
  - **Nota (2026-11-16, F7, la voce resta APERTA):** la famiglia confermativa di RQ2 è un parametro del piano statistico,
    non una costante del codice (D-36, D-73): un eventuale modello aggiunto con l'opzione (b) entra nella famiglia senza
    modificare il codice, e m segue la famiglia dichiarata.
- **Decisione:**
  - **(a)** le conclusioni si dichiarano **per famiglia di modelli** (ricostruzione: Conv-AE, LSTM-AE; previsione), mai
    su «il deep learning» in generale;
  - si aggiunge un **modello di previsione** (per esempio un LSTM); forma della previsione, scoring e proiezione sono
    decisi in D-74, che integra D-10;
  - un modello recente (opzione (b) della voce aperta) è una voce a parte, D-75, da decidere a fine M1.
- **Famiglia confermativa prevista di RQ2:** {Conv vs PCA, LSTM vs PCA, previsione vs PCA}, m = 3, congelata in F10
  come parametro del piano statistico (D-36, D-73).
- **Molteplicità:** resta D-36 (M1), IC di Bonferroni al 1 − α/m con m = 3. La proposta «Holm su tre confronti»
  **non** è adottata: contraddice D-36 approvata, la cui regola a quattro esiti si basa sugli IC.
- **Terminologia:** «famiglia di modelli» (questa voce: unità delle conclusioni, per esempio la ricostruzione) ≠
  «famiglia confermativa» (D-36: insieme dei confronti su cui si corregge). La correzione è **unica** sui tre confronti,
  non una per famiglia di modelli.
- **Ordine:** implementazione del modello di previsione dopo M1, prima di F8.
- **Approvazione:** utente, 2026-11-16 (discussione sul perimetro della tesi, su raccomandazione di Claude Project).
- **Collegamenti:** D-10, D-36, D-41, D-68, D-69, D-73, D-74, D-75.
- **Documenti:** piano F8.

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
- **Nota di rimando (2026-11-11, F4, il testo sopra resta invariato):**
  - `rolling_zscore` non ha residui per sensore nel senso di D-33: il suo |z| per sensore esiste solo per l'ultimo
    punto della finestra, e nella pipeline storica ignora `window_time_agg` (usa solo `score_anomalies`). Dichiara
    `residual_aggs = ("last",)`; chiedere `mean` è un errore. Il payload del `fit_key` non cambia (registra le
    aggregazioni richieste, `RESIDUAL_AGGS`).
  - Coerenza con `table_schema.py` (invariato): `rolling_zscore` è un modello senza residui per sensore
    (`sensor_agg = native`, `window_time_agg = none`, `normalizer = none`). Nella cache dei residui la chiave resta
    `last`; l'attributo di classe `native_only` (`True` per `rolling_zscore`, `False` per gli altri modelli del
    nucleo) dice a F5–F6 di applicare la convenzione none/native. Un test verifica che `native_only` sia vero
    esattamente per i modelli che `table_schema` tratta come privi di residui per sensore. Nessun nuovo livello di fattore.
  - Approvazione: utente, 2026-11-11, su raccomandazione di Claude Project (piano F4, C2).

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

### D-58 · Spec del dataset e controlli strutturali parametrici — APPROVATA · 2026-11-10
- **Decisione:**
  - ogni dataset ha uno spec `configs/datasets/<nome>.yaml`, individuato da `data.dataset`; nel codice non ci sono
    costanti del dataset (numero di sensori, macchine, gruppi, lunghezze);
  - campi: `name`, `layout` (per ora solo `smd_txt` = `loader.load_smd_machine`), `n_features`, `groups`
    (nome del gruppo come stringa → elenco delle macchine; una macchina in un solo gruppo), `max_len_diff` (intero ≥ 0 o null), `value_range` ([lo, hi] o null), `label_values`,
    `split_file` (file dev/eval o null), `hashes_file` (riferimento degli hash, D-61, o null);
  - lo spec e il file dev/eval passano da un validatore rigoroso con le stesse regole di F2 (chiave sconosciuta, campo
    mancante o tipo errato → `ConfigError`); il caricamento YAML dello spec rifiuta le chiavi duplicate;
  - i controlli strutturali di D-32 si applicano con i valori dello spec. Per SMD: `max_len_diff` 1, `n_features` 38,
    28 macchine in 3 gruppi (8/9/11), `value_range` [0, 1], hash uguali al riferimento. Il testo di D-32 resta invariato;
  - controlli aggiunti per ogni dataset: valori finiti; label in `label_values`; lunghezza delle label = lunghezza del test;
  - inventario su tutto il dataset (file presenti = macchine dello spec; dev ∪ eval = macchine, dev ∩ eval = ∅);
    controlli di contenuto per macchina prima di ogni uso;
  - i controlli sono bloccanti: un fallimento ferma la pipeline con l'elenco completo dei fallimenti; nessuna macchina
    viene esclusa in silenzio (le esclusioni restano a D-44).
- **Motivazione:** D-45 richiede di riapplicare invariato il piano a un secondo dataset; i numeri di SMD in D-32 diventano
  valori di uno spec, verificabili anche sul dataset sintetico di F1 e su un secondo dataset sintetico.
- **Collegamenti:** D-26, D-32, D-44, D-45, D-51, D-59, D-61. **Serve entro:** F3.
- **Documenti:** piano F3; `CLAUDE.md` §5 (al prossimo aggiornamento).
- **Nota di rimando (2026-11-12, F5, il testo sopra resta invariato):** nello stadio di fit i controlli bloccanti
  si eseguono **prima del primo fit**, per tutte le macchine selezionate: validazione della config, `manifest.load`
  (hash e contenuto) e `segment_bounds`. Un errore lì ferma il run con codice di uscita 2 e l'elenco completo dei
  fallimenti, prima che si scriva qualunque fit. Sono isolati per `fit_key` solo gli errori di modello (D-70).

### D-59 · `data_version` del fit e versione delle label — APPROVATA · 2026-11-10
- **Decisione:**
  - hash di file = sha256 dei byte dopo CRLF → LF (`provenance.file_sha256`), file letti in modalità binaria;
  - `data_version(m)` = `"dv1-"` + `canonical_hash({manifest_format, dataset, machine, files: {train, test}})[:20]`:
    solo **train e test**, nessun percorso assoluto, data di modifica o sistema operativo. È la `data_version` del `fit_key`;
  - `labels_version(m)` = `"lv1-"` + hash del file delle label: **non** entra in `data_version` né nel `fit_key`; è
    registrata nel manifest. Lo stadio di analisi (F6) la ricalcola, la confronta con quella del manifest e la scrive nei
    propri metadati; una differenza è un errore. F7 non mescola analisi con `labels_version` diverse per la stessa macchina;
  - `dataset_version` = hash delle `data_version` e `labels_version` ordinate: per provenance e registro, non per il `fit_key`;
  - nessuna cache degli hash.
- **Test richiesti:** cambiare le label non cambia `data_version` né `fit_key` (cambia `labels_version`); stesso
  contenuto in LF e CRLF → stesso hash; cartella spostata → stesso hash; un byte cambiato → hash diverso; una macchina
  modificata non cambia le altre; valore golden sul dataset sintetico.
- **Motivazione:** checkpoint e residui non dipendono dalle label, quindi una correzione delle label non deve invalidare
  i fit; il controllo delle label spetta all'analisi, l'unico stadio che le usa. La normalizzazione dei fine riga serve
  perché sulla macchina di riferimento `core.autocrlf=true` e il repository non ha `.gitattributes` (verificato 2026-11-10).
- **Collegamenti:** D-14, D-19, D-58, D-61. **Serve entro:** F3 (manifest), F6 (controllo delle label).
- **Documenti:** piano F3, F6, F7.
- **Nota di rimando (2026-11-12, F6, il testo sopra resta invariato):** come lo stadio di analisi ricalcola e
  confronta `labels_version`, e dove la registra, è fissato in D-72 (C13).

### D-61 · Hash di riferimento dei file di SMD — APPROVATA · 2026-11-10
- **Decisione:** `scripts/make_data_hashes.py` legge `dataset/` in sola lettura e scrive
  `configs/datasets/smd_sha256.yaml` (hash di train, test e label di ogni macchina, con la regola di D-59). Se il file
  esiste con un contenuto diverso si ferma (sovrascrittura solo con `--force`); `--check` verifica senza scrivere.
  Lo script viene scritto in F3 ma lo **lancia l'utente**, insieme ai test `real_data` (passo 7 del piano F3).
  Finché il riferimento manca, il manifest di SMD si ferma con errore.
- **Motivazione:** il controllo «hash dei file» di D-32 è bloccante solo con un riferimento fissato.
- **Collegamenti:** D-32, D-58, D-59, D-62. **Serve entro:** passo 7 di F3; prima di M1.
- **Nota di rimando (2026-11-10, il testo sopra resta invariato):**
  - **Eseguito:** l'utente ha generato `configs/datasets/smd_sha256.yaml` (84 file, 28 macchine) e lo ha verificato con
    `--check`; i test `real_data` sono passati (3, inclusa la parità di `prepare_machine_data` su `machine-1-1`).
  - **Fatto verificato (storia git, sola lettura):** `dataset/` è stato toccato da un solo commit, `e32b1bc`
    (2026-11-05 14:54 +0100), precedente ai commit dei run di D-50 (`843da6a` 2026-11-05 17:26 per `results_repro_*`,
    `efdacc6` 2026-11-09 12:01 per `results_repro_*_01`). Gli 84 blob di `dataset/` in `e32b1bc`, `efdacc6` e `HEAD`
    coincidono con `smd_sha256.yaml` (regola di D-59); l'albero di lavoro di `dataset/` è pulito.
  - **Limite per M1:** gli output storici non registrano hash dei dati né provenance (nessun hash nei file di testo di
    `results/results_tests/`), e la data di esecuzione dei run non è registrata (solo quella del commit). Non è quindi
    verificabile che al momento dei run l'albero di lavoro di `dataset/` coincidesse con il commit. Se M1 trova
    differenze, i dati restano un'ipotesi da escludere con la procedura di D-17, non un fatto già escluso.

### D-62 · Test `real_data` esclusi dalla suite di default — APPROVATA · 2026-11-10
- **Decisione:** `tests/conftest.py` aggiunge l'opzione `--real-data`; senza l'opzione i test marcati `real_data` sono
  saltati con un motivo esplicito. I test `real_data` leggono `dataset/` solo in modalità binaria o con il loader e
  verificano che `git status dataset/` e le date di modifica dei file siano uguali prima e dopo.
- **Fatto verificato (2026-11-10):** nessun test è oggi marcato `real_data` (`pytest -m real_data`: 0 selezionati,
  527 deselezionati) e nessun test legge `dataset/`; la modifica non salta alcun test oggi in esecuzione (527 passati).
- **Motivazione:** `pytest.ini` dichiarava i test `real_data` esclusi dalla suite di default, ma niente lo imponeva.
- **Collegamenti:** D-16, D-61. **Serve entro:** F3.

### D-63 · Segmenti e scaler in `legacy2`; finestre della loss di early stopping — APPROVATA · 2026-11-10
- **Decisione:**
  - `legacy2`: taglio = lunghezza del primo pezzo di `protocol.chronological_split` (`int(round(n·(1 − val_ratio)))`),
    riusato senza modificarlo; fit = [0, cut), es = cal = [cut, n), con il doppio uso dichiarato (`es_is_cal`, P1);
  - `split3`: `preprocessing/split.py::split_bounds(n, fractions, min_len=W)`, riusato senza modificarlo; cal deve avere
    una maschera non vuota (T_cal ≥ 2W − 1);
  - `fit_scaler` fa parte di F3. In `legacy2` riproduce esattamente `main.prepare_machine_data`: `SMDScaler(scaler_type,
    clip, min_range)` fittato su `raw_train[:cut]`, poi `transform` di validation e test. In `split3` vale D-65;
  - `train_fraction` < 1 di `main.py` non è supportato dalla nuova pipeline ed è fuori dal perimetro della parità
    (non usato dai run di D-50). Le run storiche con `train_fraction` < 1 (`train_fraction` 0,05–0,50 e varianti AE
    al 5%) esistono fuori da questo repository, nel progetto precedente (cartella `exercises/SMD` del tirocinio), e non
    sono riproducibili dalla nuova pipeline. Quando la
    quantità di dati di training entrerà nel protocollo, la regola storica (ultime righe della parte di train, almeno
    2W) non si eredita in automatico: servirà una voce nuova;
  - finestre della loss di early stopping: regola storica `score_windows(es, step)[::max(1, train_step // step)]` in
    entrambe le modalità; con `step` = 1 coincide con il passo `train_step`.
- **Motivazione:** parità esatta con la pipeline storica; `split_bounds` non rappresenta lo split in due parti.
  Per `val_ratio` = 0,2 i due tagli coincidono per ogni n (la parte frazionaria di 4n/5 non è mai 0,5), ma per altri
  rapporti l'uguaglianza non è garantita.
- **Collegamenti:** D-05, D-13, D-23, D-40, D-65. **Serve entro:** F3.

### D-64 · Maschere `legacy2` e lotti di scoring — APPROVATA · 2026-11-10
- **Decisione:**
  - `eval_mask` ha un ramo `legacy2` esplicito che delega a `protocol.common_support_mask(T, [(W, step)])`, come
    `main.py` per test e validation; i rami `split3` (`primary`, `test_context`) sono implementati a parte. Un test fissa
    il ramo `legacy2`: cambiare la maschera primaria di `split3` non lo tocca;
  - **F3:** `iter_batches` restituisce copie contigue, scrivibili, in float32; `batch_size` è obbligatorio, senza
    default. In F3 non c'è altro: nessuna regola per modello;
  - **F4:** in `legacy2` il lotto esterno di scoring deve essere allineato al blocco interno che `main.py` usa per ogni
    modello, così ogni chiamata BLAS o torch vede le stesse matrici (identità bit a bit in F4 e M1, D-17). Ogni modello
    espone la dimensione del proprio blocco storico come attributo (aggiunta: `_CHUNK` resta dov'è) e la pipeline genera
    un errore in `legacy2` se il lotto è incompatibile. Contenuto della regola (confermato dall'utente il 2026-11-10):
    - Conv-AE e LSTM-AE: multiplo di `batch_size` della config storica (64 nei run di D-50);
    - PCA a finestre: multiplo di `PCADetector._CHUNK` (4.096);
    - PCA puntuale: con `mean` multiplo di `PointwisePCADetector._CHUNK` (2.048); con `last` un solo blocco;
    - `rolling_zscore` e `input_norm`: un solo blocco.
- **Fatto verificato (2026-11-10):** per PCA e PCA puntuale il blocco di scoring di `main.py` non è una chiave di
  config ma una costante di classe (`_CHUNK`).
- **Collegamenti:** D-15, D-17, D-20, D-22, D-23. **Serve entro:** F3 (`eval_mask`, `iter_batches`), F4 (regola dei lotti).
- **Nota di rimando (2026-11-11, F4, il testo sopra resta invariato):**
  - la regola dei lotti è implementata in `src/models/batching.py` (file nuovo); il blocco storico di ogni modello è
    esposto dalla proprietà `legacy_score_block` ({aggregazione: intero, oppure `None` = un solo blocco}), che legge
    `_CHUNK` o `batch_size` senza copiarli;
  - un lotto è compatibile se `batch_size` è multiplo del blocco storico oppure se `batch_size` ≥ numero di finestre
    del segmento (un solo lotto, equivalente alla chiamata di `main.py`); con "un solo blocco" vale solo la seconda
    condizione. Un lotto incompatibile è un errore;
  - **regola applicata in entrambe le modalità da F4** (`legacy2` e `split3`): in `legacy2` per la parità, in `split3`
    per prudenza, perché la dimensione del lotto di scoring è fuori dal `fit_key` e i residui non devono dipenderne.
    F5 può allentarla solo con una decisione esplicita;
  - approvazione: utente, 2026-11-11, su raccomandazione di Claude Project (piano F4, C7; estensione a `split3`
    decisa dall'utente il 2026-11-11, passo 2).
- **Nota di rimando (2026-11-11, F4 passo 3, il testo sopra resta invariato): layout dei lotti.**
  - **Fatto verificato:** `create_sliding_windows` (`main.py`) fa `as_strided(...).astype(np.float32)`, e `astype`
    conserva l'ordine dei passi: le finestre storiche non sono C-contigue (lungo la finestra i valori sono contigui,
    passi `(80, 4, ...)` con 6 sensori). `iter_batches` (F3) produce copie C-contigue. Con valori identici,
    `mean(axis=1)` di numpy somma in un altro ordine e cambia l'ultimo bit: i residui di `input_norm` (`mean`) e del
    rolling z-score differivano da quelli storici. Le due PCA non ne risentono (copiano in ordine C con `reshape`).
    Gli array scalati di F3 hanno già i passi di quelli di `main.py`.
  - **Decisione** (utente, 2026-11-11, su raccomandazione di Claude Project): `compute_residuals` costruisce i lotti come
    `create_sliding_windows` (`view[a:b].astype(np.float32)`, ordine dei passi conservato), in entrambe le modalità,
    così in F8 la differenza tra `legacy2` e `split3` non contiene un effetto di layout. `iter_batches` resta invariato
    e non è usato per lo scoring dei residui.
  - **Limite dei controlli esistenti:** il golden T0 e la parità di F3 confrontano i valori, non i passi, quindi non
    rilevano differenze di layout. La verifica del layout è affidata a `tests/test_f4_layout.py`: passi dei lotti di
    `compute_residuals` contro le finestre di `main.prepare_machine_data`, e passi degli array scalati di F3 contro
    quelli di `main.py` per tutti gli `scaler_type`.
- **Nota di rimando (2026-11-12, F5, il testo sopra resta invariato): lotti nello stadio di fit.** Nessun allentamento
  della regola dei lotti in `split3`: lo stadio di fit la applica in entrambe le modalità tramite
  `batching.compute_residuals`. Lotto predefinito: un solo lotto per segmento (stessa chiamata di `main.py`, sempre
  compatibile); un lotto esplicito viene validato da `check_legacy_batch`. La dimensione del lotto resta fuori dal
  `fit_key` e si registra in `meta.json` (D-70). Approvazione: utente, 2026-11-12 (piano F5, V8).

### D-66 · Tipo numerico dei residui — APPROVATA · 2026-11-11
- **Decisione:** `residuals()` restituisce il tipo nativo di `score_features` del modello, senza conversioni. Il formato
  dei residui di F5 deve conservare quel tipo, e `meta.json` lo registra per ogni modello.
- **Fatto verificato (golden T0, `tests/golden/models_pre_f4.json`):** `score_features` restituisce float32 per Conv-AE
  e LSTM-AE e float64 per PCA, PCA puntuale e `input_norm`; il |z| del rolling z-score è float32.
- **Motivazione:** la proposta §5.4 prevedeva residui in float32; per i modelli in float64 la conversione cambierebbe gli
  score e violerebbe la tolleranza zero su PCA e baseline in M1 (D-17). L'aumento di spazio riguarda solo modelli
  deterministici, che girano una volta per macchina.
- **Approvazione:** utente, 2026-11-11, su raccomandazione di Claude Project (piano F4, C3).
- **Collegamenti:** D-17, D-19, D-33, D-56. **Serve entro:** F4 (interfaccia), F5 (formato e `meta.json`).
- **Documenti:** proposta §5.4 (superata su questo punto); piano F4, F5.

### D-67 · Checkpoint dei modelli — APPROVATA · 2026-11-11
- **Decisione:**
  - ogni checkpoint è una cartella con `checkpoint.json` (`checkpoint_format` 1, `fit_code_version`, nome nel registro,
    `class_path`, config del modello, architettura per gli AE, `FitInfo`, versioni di python, numpy, torch e
    scikit-learn, thread di torch) più lo stato:
    - AE → `model.pt` (`state_dict`, buffer di BatchNorm compresi), letto con `torch.load(weights_only=True)`;
    - PCA e PCA puntuale → `.npz` con tutti gli attributi fittati (`*_`), letto con `allow_pickle=False`, più
      `get_params()` in JSON; l'oggetto `PCA` di scikit-learn viene ricostruito (la PCA puntuale aggiunge `mean_` e `std_`);
    - rolling z-score e `input_norm` → solo la config.
    Nessun pickle;
  - il solver effettivo della PCA è letto da un attributo privato di scikit-learn: il checkpoint lo annota;
  - `save` rifiuta una cartella non vuota; `load` rifiuta una `fit_code_version` diversa, un formato sconosciuto, una
    classe diversa e versioni di python, numpy, torch o scikit-learn diverse da quelle registrate (confronto esatto,
    coerente con il lock di D-55). `allow_version_mismatch=True` forza il caricamento: emette un warning e imposta
    sull'istanza `loaded_with_version_mismatch = True`, che F5 riporta in `meta.json`;
  - `FIT_CODE_VERSION` resta 1: nessun artefatto è stato prodotto prima di F4, e F4 completa il contratto descritto
    dalla versione 1;
  - **campi non deterministici:** `fit_seconds` e ogni misura di tempo non entrano in alcun hash (golden R1, confronti di
    `checkpoint.json`, `fit_key`, futuro `meta.json`).
- **Test richiesti:** salva → ricarica → residui identici bit a bit per ogni modello, segmento e aggregazione; rifiuto e
  forzatura del controllo di versione; il golden R1 non cambia se cambia solo `fit_seconds`; `residuals` chiamato due
  volte dà risultati identici e non modifica `state_dict` (buffer di BatchNorm compresi) né lo stato del generatore
  di torch.
- **Motivazione:** la proposta §5.4 prevedeva per la PCA solo componenti e media, insufficienti per ricostruire
  `transform`/`inverse_transform` di scikit-learn 1.9 bit a bit; senza pickle il caricamento non esegue codice e non
  dipende dalla struttura interna degli oggetti; il controllo di versione rende esplicita ogni deroga al lock.
- **Approvazione:** utente, 2026-11-11, su raccomandazione di Claude Project (piano F4, §5, C8, P3, P4).
- **Collegamenti:** D-14, D-15, D-17, D-55, D-57, D-66. **Serve entro:** F4.
- **Documenti:** proposta §5.4; piano F4, F5.
- **Nota (2026-11-11, F4 passo 4, il testo sopra resta invariato):**
  - **fatto verificato:** `score_features` degli AE (invariato) fa avanzare il generatore globale di torch, perché il
    `DataLoader` ne estrae un seme a ogni iterazione anche con `shuffle=False` e `num_workers=0`; il seme non entra nel
    calcolo. `residuals` degli AE esegue `score_features` dentro `torch.random.fork_rng`, che ripristina il generatore:
    i residui non cambiano e la voce «non modifica lo stato del generatore di torch» è rispettata. Nella pipeline
    storica l'avanzamento non ha effetti sui numeri, perché `set_seed` precede ogni fit;
  - **pesi salvati:** con early stopping attivo il trainer ricarica i pesi dell'epoca migliore (`best_epoch`); il
    checkpoint salva lo `state_dict` di `self.model`, che è la stessa rete del trainer usata da `score_features`
    (verificato con uno scenario di early stopping forzato);
  - **device** (utente, 2026-11-11): il checkpoint di un AE registra il device effettivo del fit (`trainer.device`);
    `load` ricostruisce il trainer con la stessa regola di `fit` e rifiuta se il device scelto è diverso da quello
    registrato (non disponibile o diverso), salvo `allow_version_mismatch=True` (warning e
    `loaded_with_version_mismatch = True`);
  - **baseline random (verificato in sola lettura, `main.run_machine`):** usa un generatore numpy proprio,
    `np.random.default_rng(seed)`, creato all'inizio di ogni seed subito dopo `set_seed(seed)` e prima di qualunque
    modello di quel seed. Non usa il generatore globale di torch né quello globale di numpy, quindi il suo valore
    storico non dipende dai semi estratti dal `DataLoader` nello scoring degli AE. Tra lo scoring di un AE e il
    `set_seed` successivo ci sono solo metriche e soglie deterministiche.
- **Nota di rimando (2026-11-12, F5, il testo sopra resta invariato):**
  - **`loaded_with_version_mismatch` e `meta.json`:** `meta.json` si scrive una sola volta, al fit, con
    `loaded_with_version_mismatch: false` (nello stadio di fit il modello non viene mai caricato). Il caricamento di un
    checkpoint avviene solo nella rigenerazione dei residui, e il suo esito si registra in `residuals.json` (D-70).
  - **Regola degli hash:** la verifica degli sha256 dei residui contro `meta.json` è sempre obbligatoria, anche con
    `allow_version_mismatch=True`. La forzatura permette solo il `load`; hash diversi sono un errore, senza eccezioni.
    Quindi `loaded_with_version_mismatch: true` significa «caricato con versioni diverse, residui identici bit a bit».
  - **Ordine di `set_seed` (fatto verificato in sola lettura, `main.run_machine`):** `set_seed` reimposta i generatori
    di `random`, numpy globale e torch; tra `main.py:330` e `model.fit` (`main.py:340`) nulla consuma un generatore; la
    costruzione del modello (`main.py:328`) avviene prima e non lo usa (le reti si costruiscono in `fit`); la baseline
    random usa un generatore proprio (`main.py:313`). La chiamata di `main.py:310` non ha quindi effetti sul fit: lo
    stadio di fit replica solo quella di `main.py:330` (costruzione → finestre → `set_seed(seed)` → `fit`, nessun
    `load` in mezzo). Va dimostrato con un test sullo stato dei generatori (torch, numpy globale, `random`) catturato
    all'ingresso di `fit`, uguale a quello subito dopo un `set_seed(seed)` nuovo. Per i modelli deterministici il valore
    passato è `seeds[0]`, come `main.py:321`, e un test verifica che i loro residui non cambino tra `set_seed(42)` e
    `set_seed(7)` (giustifica il seed assente dal `fit_key`).
  - Approvazione: utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, C3, C7).
- **Nota di rimando (2026-11-12, F5 passo 4, il testo sopra resta invariato):** il `fit_info` di `checkpoint.json` usa
  la serializzazione dei non finiti di `FitInfo` descritta nella nota «F5 passo 4» di D-70 (`null`, più
  `nonfinite_fields` per i campi scalari). Con valori finiti il `fit_info` di `checkpoint.json` è identico a prima
  (stesse chiavi, stessi valori); `CHECKPOINT_FORMAT` non cambia.
### D-68 · Interfaccia dei modelli in F4 — APPROVATA · 2026-11-11
- **Decisione:**
  - **registro:** `src/models/registry.py` è uno strato sottile sopra `schema.MODEL_SCHEMAS`, che resta l'unica fonte;
    aggiunge `NUCLEUS`, `get(name)` e `load_checkpoint(dir)`. `main.py` non si tocca (D-13); un test verifica che le
    classi coincidano con `main.MODEL_REGISTRY`;
  - **perimetro:** nucleo = `pca`, `pca_pointwise`, `conv_autoencoder`, `lstm_autoencoder`, `rolling_zscore`,
    `input_norm`. VAE, OC-SVM e Isolation Forest (D-10) sollevano `NotImplementedError` su `residuals`, `save` e `load`;
    AE non addestrati: D-69. Il perimetro si legge dagli attributi: `residual_aggs` vale `()` in `base.py` e lo
    dichiara solo chi implementa `residuals` (`("last", "mean")` nei modelli del nucleo, `("last",)` per `rolling_zscore`);
  - **residui:** `residuals(X, aggs)` lavora su un solo lotto in memoria; il ciclo sui lotti e la regola di D-64 sono in
    `src/models/batching.py`. Nessun corpo di `score_anomalies`, `score_features`, `get_error_matrix` o
    `get_reconstruction_errors` cambia;
  - **`FitInfo`** nell'attributo `fit_info_`, firma di `fit` invariata (D-15); alias `max_epochs` (nota su D-39);
  - **golden pre-F4 (T0):** hash di ingressi e uscite di `score_anomalies` e `score_features(last|mean)` dei 6 modelli,
    generato su `d2d8478` prima di ogni modifica a `src/` (commit `79ea2cf`), perché lo snapshot F1 non copre
    `score_anomalies` di PCA, PCA puntuale, AE e `input_norm`.
- **Approvazione:** utente, 2026-11-11, su raccomandazione di Claude Project (piano F4, C1, C6, C7, C9).
- **Collegamenti:** D-10, D-13, D-15, D-33, D-39, D-64, D-66, D-67, D-69. **Serve entro:** F4.
- **Documenti:** proposta §5 (riga `models/registry.py`: «spostato da `main.py`» superato); piano F4.

### D-70 · Stadio di fit: store, `meta.json`, residui — APPROVATA · 2026-11-12
- **Decisione:**
  - **Layout dello store** (modifica strutturale, `CLAUDE.md` §8; D-56 fissa solo posizione e budget):
    - `store/fits/<fit_key>/`: `meta.json`, `checkpoint/` (cartella di D-67), `scaler.npz`, `residuals.npz`,
      `residuals.json`;
    - `store/runs/exploratory/<esperimento>/<run_id>/`: `run.log`, `fit_summary.csv`;
    - `store/tmp/` (scritture in corso), `store/registry.csv`, `store/.lock`.
    `proposta_architettura_target.md:252` (`model.pt` / `pca.npz` direttamente in `fits/<key>/`) è superata su
    questo punto da D-67 (checkpoint come cartella, `save` rifiuta una cartella non vuota) e da questa voce.
  - **Residui:**
    - nomi `<segmento>__<agg>`; aggregazioni = `RESIDUAL_AGGS` ∩ `residual_aggs` del modello (per il rolling z-score
      solo `last`, nota su D-33);
    - `legacy2`: `cal` e `test` (nessun `es`: es = cal, nota su D-04);
    - `split3`: anche `es` (`diagnostic_only`, D-04) e, solo con `step` = 1, `test_ctx_head` = le W−1 finestre del test
      che iniziano in cal (D-20, D-60), calcolate con una **chiamata separata** di `compute_residuals`, così i residui
      primari del test sono identici nei due scenari di RQ1a;
    - formato `np.savez` non compresso, tipo nativo (D-66), lettura con `allow_pickle=False`;
    - gli indici dei timestamp non si salvano come array: `meta.json` registra `n_timestamps`, W, `step`, `context` e
      `n_windows`, da cui un test ricostruisce esattamente `WindowSet.starts` / `ends`.
  - **`meta.json` e `residuals.json`:**
    - `meta.json` si scrive una sola volta, al fit, con `loaded_with_version_mismatch: false` e il record di ogni
      residuo (forma, dtype, sha256, `n_nonfinite`, parametri della finestra, `diagnostic_only`); confronti solo sulla
      forma senza tempi, provenance e `created_by` (`meta_hashable`, coerente con D-67);
    - `residuals.json` descrive il file presente sul disco: origine (`fit` / `regenerated`),
      `loaded_with_version_mismatch`, firma d'ambiente;
    - la verifica degli sha256 contro `meta.json` è **sempre** obbligatoria, anche con `allow_version_mismatch=True`:
      la forzatura permette solo il `load`; hash diversi sono un errore, senza eccezioni (nota su D-67).
  - **Aggiunte a `store.py` (V4 allargata in modo esplicito).** Elenco chiuso:
    - `replace_fit_cache_file` (scrittura atomica, ammessa **solo** per `residuals.npz` e `residuals.json`);
    - `orphan_tmp()` (cartelle temporanee e cestini orfani);
    - acquisizione e rilascio del lock (vedi sotto);
    - valore `interrupted` in `RUN_STATUSES` (`store.py:36`), accettato da `finish_run` (`store.py:200`) e da
      `append_registry` (`store.py:217`).
    Nessun metodo esistente cambia comportamento per i valori già ammessi. `replace_fit_cache_file` è un'eccezione
    limitata alla regola «un fit completo non si sovrascrive mai» (`store.py:12`), ammessa perché riguarda solo file
    derivati, rigenerabili dal checkpoint e verificati con hash contro `meta.json`; `meta.json`, `checkpoint/` e
    `scaler.npz` non si riscrivono mai. **Per il passo 2:** se un test o lo schema di F2 fissa i valori ammessi per lo
    stato nel registro, si aggiorna insieme a `RUN_STATUSES`, dichiarandolo.
  - **Controlli preliminari e fallimenti isolati:** prima del primo fit, per tutte le macchine selezionate:
    validazione della config, `manifest.load` e `segment_bounds`; un errore blocca il run (codice 2, nota su D-58).
    Sono isolati per `fit_key` solo gli errori di modello (fit, `save`, residui, serializzazione): `meta.json` con
    `status: failed`, tipo, messaggio, traceback e fase. Un fit `failed` si ripete al rilancio; la sostituzione avviene
    con il meccanismo esistente di `store.write_fit` (`store.py:134-153`): il nuovo fit si scrive per intero in una
    cartella temporanea nuova (quindi `save` vede sempre `tmp/<uuid>/checkpoint` vuota), poi la vecchia
    `fits/<key>` si sposta nel cestino temporaneo, la nuova prende il suo posto con `os.replace` e il cestino si
    cancella. Un fit `complete` non si ripete mai: `write_fit` continua a rifiutarne la sostituzione (`FitExists`,
    `store.py:130`) e il meccanismo con il cestino vale solo per i fit `failed`. Un processo morto tra lo spostamento
    nel cestino e `os.replace` lascia un cestino orfano e nessun `fits/<key>`: è accettabile (il compito si rifà al
    rilancio), ma `orphan_tmp` e `status` elencano anche i cestini orfani.
  - **Valori non finiti** (sostituisce la proposta C11 del piano F5): una loss non finita **non** rende il fit
    `failed` (sarebbe una regola di esclusione implicita, D-44 aperta, e romperebbe la parità con `main.py`, che in quel
    caso produce comunque residui). Il fit resta `complete` con `nonfinite_loss: true`; i valori non finiti si
    serializzano in modo esplicito (`null` più il flag), così `json.dump(allow_nan=False)` non fallisce; per ogni
    residuo si registra `n_nonfinite`. Nessuna esclusione (nota su D-44).
  - **`kind: official`:** `cli fit` lo rifiuta («non disponibile prima di F10»); le protezioni dei run ufficiali
    arrivano in F9–F10.
  - **Interruzione distinta dal fallimento:** `KeyboardInterrupt` non è un fallimento isolato: si propaga, `tmp/`
    viene ripulita, il registro riceve una riga `interrupted` (non `failed`) e il codice di uscita è 130. Serve a F11:
    la regola di esclusione (D-44) dovrà distinguere un modello fallito da un run interrotto. **Richiede** di
    aggiungere `interrupted` a `RUN_STATUSES` (`store.py:36`) e ai valori ammessi da `finish_run` (`store.py:200`):
    estensione dei valori ammessi, nessun altro comportamento cambia.
  - **`status` in sola lettura:** non crea cartelle né file; elenca anche cartelle temporanee e cestini orfani.
  - **Lock di scrittura:** `store/.lock` creato con `O_EXCL` all'apertura del run e rimosso alla fine, anche su
    eccezione. Contenuto: pid, host, ora di avvio, `run_id` (file locale e temporaneo, non entra nella provenance). Se
    esiste, `fit` rifiuta di partire e ne mostra il contenuto.
  - **Pulizia (`clean-tmp`):** prende il lock e poi pulisce; poiché un processo di fit tiene il lock per tutto il run,
    non può cancellare il `tmp/` di un processo attivo. Se il lock è occupato rifiuta e ne mostra il contenuto.
    `clean-tmp --break-stale-lock` rimuove il lock solo se l'host coincide con quello corrente e il pid non è attivo
    (host diverso: rifiuta), poi prende il lock e pulisce; la rottura del lock si annota nel registro.
  - **Test richiesti (lock e pulizia):** secondo run concorrente rifiutato; lock attivo → `clean-tmp` rifiuta; lock di un
    pid morto → `clean-tmp` senza opzione rifiuta, con `--break-stale-lock` pulisce e annota la rottura nel registro;
    lock di un host diverso → `--break-stale-lock` rifiuta.
- **Motivazione:** artefatti indirizzati per contenuto, scritti in modo atomico e verificabili con hash; residui
  rigenerabili senza nuovi fit e senza possibilità di sostituirli con valori diversi; `split3` si implementa già in F5
  per completare il contratto della versione 1 di `FIT_CODE_VERSION`, che include le W−1 finestre del test con contesto
  (D-20, D-60, `schema.py:23-24`), e per testare subito gli invarianti anti-leakage. I fallimenti di dati restano
  bloccanti (D-58), quelli di modello isolati; interruzioni e fallimenti si distinguono per F11.
- **Approvazione:** utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, V1–V6, C11 sostituita, A1–A3).
- **Collegamenti:** D-04, D-20, D-33, D-44, D-56, D-58, D-60, D-64, D-66, D-67. **Serve entro:** F5.
- **Documenti:** proposta §5.4, §5.5, §8 (superate sui punti sopra); piano F5.
- **Nota (2026-11-12, F5 passo 3, il testo sopra resta invariato):**
  - **Estensione dell'elenco chiuso di V4:** `begin_run` riceve un parametro facoltativo `run_id=None`; con `None` il
    comportamento è invariato (id generato internamente, `store.py:349`). Un `run_id` passato si valida con
    `_check_name` e, se la cartella del run o una riga di registro con quell'id esistono già, `begin_run` lo rifiuta
    con un errore esplicito, senza scrivere nulla. Motivo: il lock va preso **prima** di `begin_run` (un secondo run
    si rifiuta senza toccare lo store) e deve contenere il `run_id` del run; `run_fit` genera l'id, prende il lock con
    quell'id, poi chiama `begin_run(..., run_id=id)`. Alternative scartate: `begin_run` prima del lock; lock con id
    provvisorio riscritto dopo; id del lock distinto dal `run_id`.
  - **Device nel `fit_key`:** il `fit_key` usa il device **richiesto** (`cfg.device`, `config.fit_payload`), non
    quello effettivo, che sceglie il trainer (`trainer.py:30`, fallback storico usato da `main.py`, non si tocca) e
    che il checkpoint registra (D-67). Oggi coincidono per costruzione (`schema.DEVICES = ("cpu",)`). **Vincolo:**
    prima di ammettere `cuda` in `schema.DEVICES`, i controlli preliminari dovranno rifiutare un device richiesto e
    non disponibile (nessun fallback silenzioso), così richiesto = effettivo. **Protezione già in F5:** prima di
    `write_fit`, se il device effettivo del modello fittato differisce da `payload["device"]`, il compito è un errore
    esplicito e non entra in cache.
  - Approvazione: utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, passo 3).
- **Nota (2026-11-12, F5 passo 4, il testo sopra resta invariato):**
  - **Serializzazione dei non finiti in `FitInfo`** (modifica di `src/models/base.py`, fuori dal perimetro di
    `fit.py`, necessaria: `_write_checkpoint` scrive con `allow_nan=False`, quindi senza questa modifica una loss non
    finita avrebbe fatto fallire `save` e reso il fit `failed`, contro la regola sui valori non finiti sopra).
    `to_dict`: elementi non finiti delle loss → `null`; campi scalari (`explained_variance`, `fit_seconds`) non finiti →
    `null` più il loro nome in `nonfinite_fields`, chiave scritta solo se non vuota. `from_dict`: `null` nelle loss →
    NaN; NaN solo nei campi scalari elencati; ogni altro `null` resta «assente» (`None`, per esempio
    `explained_variance` di un AE). NaN e ±inf diventano tutti `null`: la distinzione non si conserva (campi solo
    diagnostici). Il formato è invariato per i valori finiti (nessuna chiave in più, round-trip identico).
    `meta.json` registra `nonfinite_loss` (flag sulle loss). Alternative scartate: `null` → NaN per ogni campo scalare
    (avrebbe trasformato in NaN i campi assenti di tutti i modelli non-PCA); scalari non finiti riletti come `None`
    (perdita d'informazione).
  - **`run_fit` valida al proprio interno (firma cambiata rispetto al passo 3):** riceve la specifica grezza (mappa
    già unita) o un percorso YAML (`config.load_experiment`), non una `ExperimentConfig`, e come primo passo chiama
    `validate_experiment`, prima di lock, `begin_run` e di qualunque scrittura; un errore di validazione è bloccante
    (codice 2, passo 6) e lo store non viene creato. Una `ExperimentConfig` passata direttamente è rifiutata. Il corpo
    su config validata è privato (`_run_fit`) e i test non lo usano; le loro varianti modificano la specifica grezza.
    Motivo: `ExperimentConfig` è una dataclass senza controlli propri e si costruisce anche a mano o con
    `dataclasses.replace`; verificato che config costruite così (`device: tpu`, seed vuoti, `kind` errato) passavano
    senza errori o lasciavano una cartella di store vuota.
  - **Lacuna di F2 rilevata, da decidere a parte (schema non modificato):** `validate_experiment` accetta
    `experiment.name` con caratteri non ammessi in un percorso (es. `a/b`): `schema._str` controlla solo la stringa non
    vuota, e il nome è rifiutato più tardi da `begin_run`, dopo il lock (cartella di store creata vuota). Test relativo
    marcato `xfail` (strict) con il motivo. `experiment.seeds` vuoto invece è già rifiutato dallo schema.
  - Approvazione: utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, passo 4).
- **Nota (2026-11-12, F5 passo 4-bis, il testo sopra resta invariato):** lacuna di F2 su `experiment.name` chiusa.
  `validate_experiment` applica a `experiment.name` la stessa regola dei nomi di percorso dello store; la regola sta
  in un solo punto, `schema.is_path_name`, e `store._check_name` la riusa (prima la regex era in `store.py`; lo schema
  non può importare lo store, che importa lo schema). **Regola rafforzata** rispetto a quella storica dello store
  (`re.match` con `^[A-Za-z0-9_.-]+$`, che accettava un a capo finale, `.` e `..`): `re.fullmatch` su
  `[A-Za-z0-9_.-]+`; nessun punto finale (copre `.`, `..`, `abc.`; Windows toglie il punto finale); nessun nome
  riservato di Windows (CON, PRN, AUX, NUL, COM1–COM9, LPT1–LPT9), senza distinzione di maiuscole e anche con
  estensione (`com1.txt`); il punto iniziale resta ammesso (`.hidden`). Vale per `experiment.name`, `run_id` e
  `stage`. Un nome non valido è ora un `ConfigError` della validazione, prima di qualunque scrittura; il test `xfail`
  del passo 4 è diventato un test normale. Verificato: nessuna config in `configs/` definisce `experiment.name`; tutti
  i nomi, `run_id` e stage usati nei test e in `src/` rispettano la regola rafforzata; `new_run_id` produce nomi
  validi (test su 5.000 generazioni). Nessun effetto sui `fit_key` (il nome non entra nel payload).
  - Approvazione: utente, 2026-11-12 (piano F5, passo 4-bis).
- **Nota (2026-11-12, F5 passo 5, il testo sopra resta invariato):** `get_residuals` e `regenerate` in
  `src/pipeline/residual_cache.py`.
  - **Regola: letture senza lock, scritture solo con il gettone del lock del run.** Il gettone è il dizionario
    restituito da `store.lock(run_id)`; `regenerate` verifica che `store.read_lock()` e il gettone coincidano in tutti
    i `LOCK_FIELDS` all'inizio e subito prima di scrivere, e rifiuta senza scrivere se non coincidono o se il gettone
    manca. **Campi del lock** (estensione del contenuto fissato sopra: pid, host, ora di avvio, `run_id`): si aggiunge
    `nonce` (uuid4 generato a ogni acquisizione), così due acquisizioni con lo stesso `run_id` danno gettoni diversi
    anche con la stessa `started_utc`. Fatto misurato: `started_utc` ha formato a microsecondi e passo reale di 2–3 µs
    su questa macchina (nessun valore ripetuto in 200.000 chiamate consecutive); il nonce non dipende dalla
    risoluzione dell'orologio. `is_stale` e `clean-tmp` non usano il nonce. `get_residuals` con i file presenti legge senza lock; con un file mancante e senza lock dà l'errore
    «residui mancanti: rigenerazione richiede il lock del run», senza acquisire nulla. `store.replace_fit_cache_file`
    resta invariato (l'elenco chiuso di V4 non cambia). Alternative scartate: lock del processo corrente riconosciuto
    dal solo pid (non distingue un lock rilasciato e ripreso); acquisizione implicita del lock in una funzione di
    lettura.
  - **Scostamento dalla firma del piano:** `get_residuals(store, key, segment, agg, allow_diagnostic=False, *, lock,
    data_dir, datasets_dir)` e `regenerate(store, key, data_dir, datasets_dir, lock, allow_version_mismatch=False)`.
    Oltre a `lock`, servono `data_dir` e `datasets_dir`: il `fit_payload` registra dataset, macchina e `data_version`
    ma non la posizione dei dati (che non deve entrare nella chiave). Sono parametri di sola localizzazione, non una
    config dell'esperimento: split, preprocessing e finestre vengono solo dal `fit_payload` di `meta.json`.
  - **Verifiche di `regenerate`** (oltre a quelle del piano): `canonical_hash(fit_payload)` uguale alla chiave;
    `checkpoint.json` uguale al record `checkpoint.hashable_sha256` di `meta.json`; `scaler.npz` uguale al record di
    `meta.json` e allo scaler ricostruito; lotto di scoring = `scoring.score_batch_requested` registrato. «Mancante» =
    `residuals.npz` o `residuals.json` assente. In lettura: `npz_sha256` del file, insieme degli array uguale a
    `meta.json`, sha256 dell'array richiesto. I generatori globali sono salvati e ripristinati attorno a `load` e
    `compute_all` (`load` ricostruisce la rete e consuma torch, D-67).
  - Approvazione: utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, passo 5, opzione C del lock).
- **Nota (2026-11-12, F5 passo 6, il testo sopra resta invariato):** CLI `src/cli.py`
  (`python -m src.cli {fit,status,clean-tmp} <esperimento.yaml> [--datasets-dir]`, `main(argv) -> int`).
  - **Codici di uscita:** 0 tutto complete o skipped; 1 almeno un fit failed (fit) o un errore di integrità
    (`status --verify`); 2 config o dati non validi (validazione, controlli preliminari, `kind: official`, file di
    esperimento inesistente); 3 store occupato (lock attivo, rimasto senza `--break-stale-lock`, di un altro host,
    illeggibile); 130 interruzione.
  - **`status`** in sola lettura (nessuna cartella né file, nemmeno lo store se manca): per compito complete /
    failed (fase) / missing, residui presenti o mancanti con origine, voci di `orphan_tmp`, stato del lock (libero,
    occupato, rimasto secondo `is_stale`, illeggibile). `--verify` legge i residui presenti con `get_residuals` senza
    lock (quindi senza mai rigenerare) e dà codice 1 con l'elenco delle chiavi in caso di errore di integrità.
  - **`clean-tmp`** prende il lock con un `run_id` di `new_run_id`, rilegge `orphan_tmp` sotto lock e rimuove tutte
    le voci (file, scritture parziali, cestini, scritture complete non installate); non promuove mai un `tmp/` a
    `fits/<key>`; `--dry-run` elenca senza lock e senza cancellare; su uno store inesistente non crea nulla.
  - **Annotazione della rottura del lock (forma minima, nessun nuovo valore di stato):** ogni `clean-tmp` effettivo
    apre un run con il meccanismo esistente (`begin_run`/`finish_run`, `stage = "clean-tmp"`, `kind` ed esperimento
    della config, commit e `dirty` come ogni riga): riga `started` con nota `break-stale-lock: <JSON del lock
    rimosso>` (pid, host, `started_utc`, `run_id`, `nonce`) quando il lock è stato rotto, vuota altrimenti; riga
    `completed` con nota «rimosse N voci di tmp/». Nella cartella del run (`runs/<kind>/<esperimento>/<run_id>/`)
    `clean-tmp` scrive `clean_summary.csv` (colonne `kind`, `path` relativo allo store, `fit_key`, `status`,
    `detail`): una riga per voce rimossa con la classificazione di `orphan_tmp`, più una riga `lock_broken` (`path` =
    `.lock`, `detail` = JSON del lock rimosso) se il lock è stato rotto; con zero voci il file contiene solo
    l'intestazione; il file si scrive anche se la pulizia si interrompe a metà (voci già rimosse). `--dry-run` non apre
    alcun run e non scrive alcun file. `--break-stale-lock` rompe il lock solo se l'host coincide e `is_stale` è vero; host diverso, processo
    attivo o lock illeggibile → rifiuto (codice 3); per il lock illeggibile il messaggio spiega la rimozione manuale
    dopo aver verificato che nessun processo sia attivo.
  - Approvazione: utente, 2026-11-12, su raccomandazione di Claude Project (piano F5, passo 6).
- **Nota (2026-11-12, F5 passo 7, il testo sopra resta invariato):** precisazioni emerse alla chiusura di F5.
  - **(a) Interruzione:** con `KeyboardInterrupt` si rimuove solo la cartella temporanea del compito in corso
    (`write_fit` la cancella su qualunque eccezione); le voci orfane preesistenti in `tmp/` restano fino a `clean-tmp`.
    Il testo sopra («`tmp/` viene ripulita») va letto in questo senso.
  - **(b) Fasi isolate:** oltre a fit, `save`, residui e serializzazione, è isolata anche la fase `device_check`
    (`DeviceMismatch`, passo 4): `meta.json` failed con `phase: device_check`, gli altri compiti proseguono.
  - Il test sulla ricostruzione di `WindowSet.starts` / `ends` dai record di `meta.json` (C9) è stato aggiunto alla
    chiusura (`tests/test_f5_regenerate.py`).
  - Approvazione: utente, 2026-11-12 (piano F5, passo 7).

### D-72 · Stadio di analisi (F6) — APPROVATA · 2026-11-12
- **Decisione** (gli identificativi C1–C15, V10–V13 e A/B rimandano al piano F6 e alle modifiche dell'utente):
  - **Perimetro:** 6a parità esatta con `main.py` dai residui dello store; 6b metriche di D-34 (VUS-PR esclusa), soglie
    statiche e mobili, normalizer robusto, scenari `primary` / `test_context`, baseline random, tabelle `free` / `thr` /
    `fits`, comando `analyze`; modalità `legacy2` e `split3`. `kind: official` rifiutato fino a F10, come in `fit`.
    Fuori: VUS-PR, statistica (F7), file di M1, parallelizzazione (solo se la misura la giustifica). `main.py`,
    `protocol.py`, `protocols.py` e `rules.py` non si modificano.
  - **C1 — `rules.py`:** resta invariato fino a F12 (nessuna ri-esportazione: ne cambierebbe la semantica e
    indebolirebbe `tests/test_rules.py`, contro `CLAUDE.md` §5). `thresholds.py` è un modulo nuovo. Un test dimostra
    che con storia vuota e score mancanti solo iniziali `thresholds.py` coincide bit a bit con `rules.py`. Fatto
    verificato (2026-11-12, script temporaneo): `rules.rolling_quantile_threshold` conta per righe; coincide con la
    regola «ultimi `n_past` score validi» (D-21) se i NaN sono solo iniziali, non con un buco interno, che è proprio la
    giunzione cal → test dell'analisi primaria.
    - **Regola `burnin`:** non passa in `thresholds.py` e non fa parte della griglia. Motivo: calibra la soglia sui
      primi punti del test, mentre D-04, D-21 e D-23 fissano cal come unico supporto di calibrazione e di storia
      iniziale. Resta disponibile in `rules.py` fino a F12.
  - **C2 — formula unica dell'F1 e celle 2×2:** tutte le metriche con soglia usano i conteggi e la formula di
    `protocol._prf` (2PR/(P+R)); le celle non vengono da `protocols.py`, che resta invariato. Cella C = soglia statica,
    `pa` falso; D = statica, `pa` vero; B = oracolo, `pa` falso = `best_f1` di `evaluate_timestamps`; A = oracolo,
    `pa` vero = ricerca esatta sui massimi dei segmenti (l'algoritmo di `EventRegion`) con i conteggi e la stessa
    formula. Fatto verificato (2026-11-12, script temporaneo, 2.000 casi casuali): `protocols.four_cells` differisce
    nell'ultimo bit da `protocol.py` nel 38% dei casi per C e nel 33% per D, quindi riusarlo avrebbe cambiato le
    metriche storiche.
    - **Tolleranze delle proprietà:** A ≥ D e PA ≥ no-PA a tolleranza zero (stessa formula). B ≥ C a tolleranza zero:
      verificato con uno script temporaneo (2026-11-12, scikit-learn 1.9.0) su 20.000 serie casuali, 14.975 con pari
      merito (valori arrotondati, pochi valori distinti, blocchi di pari merito al massimo), 3,29 milioni di confronti
      a soglie uguali a tutti i valori osservati, a quantili casuali e fuori intervallo: 0 violazioni, minimo di B − C
      = 0,0; alla soglia `best_f1_threshold` l'F1 di `_prf` è identico bit a bit a `best_f1` in tutti i casi. Motivo:
      `precision_recall_curve` calcola P e R con le stesse divisioni tra interi, e `max(P+R, 1e-12)` coincide con P+R
      quando P+R > 0. Il test definitivo, con gli stessi casi, si scrive al passo 3; se fallisse, la tolleranza
      effettiva si dichiara qui con la causa prima di proseguire.
    - **Esito del test definitivo (passo 3, 2026-11-13):** tolleranza zero confermata per A ≥ D, B ≥ C e PA ≥ no-PA;
      nella suite 2.000 serie casuali con pari merito (test marcato `slow`, resta nella suite predefinita); verifica
      una tantum su 20.000 serie (19.986 con pari merito), 1.014.946 confronti, 0 violazioni, minimi di B − C, A − D
      e D − C = 0,0.
    - Lo scarto rispetto a `protocols.four_cells` è documentato da un test (atteso ≤ 1e−12 relativo).
    - **FPR (passo 3):** `fpr` = fp / (n · (1 − p)), la stessa espressione di `FPR_at_thr` in `main.make_summary_row`,
      per la parità bit a bit con lo storico. È matematicamente uguale a fp / n_neg ma può differire nell'ultimo bit:
      non va «corretta».
  - **C3 — falsi allarmi ogni 1.440 punti** (`false_alarms_per_1440`) = `fa_segments · 1440 / n_eval_points`;
    segmento di falso allarme come nello storico (segmento di allarme che non tocca nessun punto anomalo);
    `n_eval_points` comprende i punti anomali. Il nome dice «ogni 1.440 punti», mai «al giorno» (D-34).
  - **C4 — ritardo di rilevamento** (`detection_delay`, in punti) = media, sui soli eventi rilevati, di (indice del
    primo allarme dentro l'evento − inizio dell'evento); eventi della regione mascherata; NaN se nessun evento è
    rilevato; solo con `pa` falso. Nessun ritardo censurato, nessuna mediana aggiuntiva. **Avvertenze per F7:**
    - il ritardo dipende dal rilevamento (media sui soli eventi rilevati) e non è confrontabile da solo tra modelli:
      F7 lo presenta sempre insieme a `events_detected` / `event_recall`;
    - nell'analisi primaria un evento tagliato dalla maschera inizia al bordo della maschera, quindi il ritardo è
      sottostimato rispetto a `test_context`: il confronto del ritardo tra scenari va dichiarato come distorto.
  - **C5 — normalizer robusto (D-11):** centro = mediana, scala = IQR / 1,349 per sensore sulle finestre di cal, con lo
    stesso pavimento del normalizer storico (`min_scale_ratio` · media delle scale + 1e−12). `analysis.json` registra,
    per ogni fit e ogni normalizer, quanti sensori hanno la scala sul pavimento e quanti la scala grezza esattamente 0
    (serve a interpretare il fattore normalizer di RQ1b). Chiave (passo 6, approvata dall'utente il 2026-11-13):
    (fit, `window_time_agg`, normalizer), perché il normalizer si fitta separatamente su `cal__last` e `cal__mean`.
    - **Costante (passo 5, 2026-11-13):** 1,349 resta fissa e con nome (`IQR_TO_SIGMA`, C14). L'invarianza di metriche
      senza soglia e decisioni rispetto a un fattore c > 0 sulla costante è verificata empiricamente (bit a bit nei
      casi provati, c ∈ {2; 0,5; 3,7; 1.000}, aggregazioni mean, max e top-k) ma non garantita in generale: il termine
      1e−12 del pavimento non scala con c, quindi le scale sono proporzionali solo entro 1e−12·(c + 1).
  - **C6 — non finiti nella tabella `fits`** (modifica dello schema di F2): tre colonne `nonfinite_loss` (bool),
    `n_nonfinite_cal` (int), `n_nonfinite_test` (int, residui del test comprese le finestre di `test_ctx_head`,
    dichiarato nella docstring dello schema). Definizione (passo 6, approvata dall'utente il 2026-11-13): numero di
    elementi non finiti, somma su tutti i record `cal__*` (rispettivamente `test__*` e `test_ctx_head__*`) di
    `meta.json`; la colonna descrive il fit, non l'analisi; null per i fit `failed`. La somma sulle aggregazioni conta
    più volte lo stesso valore non finito: è un indicatore (> 0), non una misura di gravità; il dettaglio per record è
    in `meta.json`. **`nonfinite_loss` dei fit `failed` = null** (passo 6.1, approvato dall'utente il 2026-11-13):
    un fit fallito non ha `fit_info`; `False` affermerebbe una loss finita senza saperlo, e un filtro
    «`nonfinite_loss == False`» in F7 farebbe passare i fit falliti come puliti. Lo schema ammette per questo i bool
    nullabili (dtype `boolean`); i bool non nullabili restano invariati. Motivo: D-44 deve distinguere un non finito in cal (normalizer e soglia
    compromessi per tutto il test) da uno nel test (solo alcuni punti) leggendo solo le tabelle. Nessuna esclusione.
  - **C7 — esportazione legacy:** `src/pipeline/legacy_export.py` importa da `main.py`, in modo tardivo e solo nel suo
    percorso, le funzioni pure `evaluate_variant`, `save_scores` e `make_summary_row`; si rimuove in F12 con `main.py`.
    **Test richiesti (B10):** in un processo separato, importare `legacy_export` ed eseguire il percorso legacy non
    cambia `torch.get_num_threads`, lo stato del generatore globale di numpy e di `random`, gli handler del logging e
    il backend di matplotlib (`main.py` lo importa tramite `src/utils/visualization.py`); in `split3` senza
    esportazione legacy `main.py` non viene mai importato.
  - **C8 — piano di analisi:** nuovo tipo di config `configs/analysis/<nome>.yaml`, validato in
    `src/utils/analysis_plan.py` con le regole di F2 (chiave sconosciuta, campo mancante o tipo errato → errore;
    chiavi YAML duplicate rifiutate). Chiavi: `scenarios`, `window_time_aggs`, `scorings` (coppie esplicite
    `sensor_agg`, `normalizer`, non un prodotto cartesiano), `ts_aggs`, `thresholds` (statiche: percentili; mobili:
    politica, livello, `n_past`), `pa`, `legacy_export` (facoltativo). Nessun valore predefinito (D-42). Regole di
    coerenza: soglie mobili solo con `last` (D-24); `test_context` solo in `split3` con `step` = 1 (D-60); modelli
    `native_only` solo con la convenzione none/native.
    - **Precisazioni (passo 1, approvate dall'utente il 2026-11-13):**
      1. `native` non si dichiara negli `scorings`: per i modelli `native_only` l'analisi applica da sola la
         convenzione none/native; `analysis.json` registra quali modelli producono righe native (passo 6);
      2. `legacy_export` è ammesso solo con split `legacy2` (la parità con `main.py` ha senso solo lì);
      3. `level` di `roll_quantile` e `roll_robust_z` è un percentile in (0, 100], con la semantica di `rules.py`
         (quantile `level`/100 della storia; percentile su cal del robust z); il test «storia vuota = `rules.py`» del
         passo 4 copre la conversione percentile → quantile anche con livelli non interi (es. 99,5);
      4. soglie mobili con `ts_aggs` senza `last` sono un errore, non un salto silenzioso; con `last` e livelli offline
         le righe mobili esistono solo per `last`.
      Inoltre: ogni famiglia di soglie è obbligatoria e si dichiara vuota se non usata (`[]`, `false`); livelli
      ripetuti in qualunque lista (anche dopo la normalizzazione, es. `98` e `98.0`) sono rifiutati; `plan_sha256` è
      l'hash canonico del piano in forma normalizzata (liste ordinate, livelli float), quindi non dipende da ordine,
      commenti e formato del file.
  - **C9 — test su es:** per i modelli che non usano `X_val`, perturbare es lascia invariati end-to-end normalizer,
    soglie e metriche; per gli AE in `split3` es sceglie l'epoca (comportamento atteso, F5), quindi il test verifica
    con una spia che l'analisi non legga mai i residui `es__*`.
  - **C10 — proiezioni offline:** lo snapshot F1 e i run di M1 coprono solo `timestamp_agg: last`; un test confronta
    dal vivo l'esportazione legacy con `main.run_experiment` sui dati sintetici con `mean` e `max` (D-16).
  - **C11 — ruolo delle macchine:** un dataset senza divisione dev/eval (`split_file: null`, D-58) è rifiutato nei
    controlli preliminari di `analyze`; `machine_role` resta obbligatorio.
  - **C12 — varianti non calcolabili:** l'esportazione legacy replica il salto di `main.py` (parità); la tabella lunga
    scrive righe con `value` NaN, contate in `analysis.json`. Nessuna esclusione (D-44).
    - **Colonna `nan_reason` in `free` e `thr` (passo 6.3, approvata dall'utente il 2026-11-13; modifica dello schema
      di F2, come C6):** motivo per riga dei NaN da cause note: `nonfinite_residual_cal`, `nonfinite_residual_test`,
      `nonfinite_residual_test_ctx_head` (score di finestra non finito, controllato prima della proiezione),
      `projection_nan_cal`, `projection_nan_test` (NaN dopo la proiezione fuori dalla geometria; difesa, con finestre
      finite non deve mai comparire), `region_without_both_classes`, `nonfinite_score`. Null se `value` è finito oppure
      NaN per definizione (`detection_delay` senza eventi rilevati, C4). Regole dello schema: `value` NaN con motivo
      null solo per `detection_delay`; motivo non null richiede `value` NaN; motivo fuori elenco rifiutato. I conteggi
      per motivo di `analysis.json` coincidono con quelli della colonna. Storia insufficiente (C15) ed errori inattesi
      non diventano righe NaN: si propagano. Motivo: F7 legge solo le tabelle e, per applicare D-44, deve distinguere
      riga per riga i NaN da residui non finiti da quelli dovuti alla regione o definiti da C4.
  - **C13 — `labels_version` (D-59):** l'hash si calcola sui byte delle label effettivamente caricati per le metriche e
    si confronta con la voce del manifest (e con il riferimento degli hash, se il dataset lo ha); una differenza è un
    errore bloccante. `labels_version` per macchina va in `analysis.json`.
  - **C14 — costanti:** pavimento del rolling z (0,05 · |mediana| + 1e−12), costante 1,349, `min_scale_ratio` 0,05 e
    1e−12 dei normalizer sono costanti con nome, uguali allo storico, registrate in `analysis.json`; diventano chiavi
    del piano solo se lo richiede D-42. Anche k = 5 dell'aggregazione `top5` (passo 6.2, verificato): è un letterale di
    `main.build_score_variants`, senza chiave di config (il `top_k` di `configs/models/baselines.yaml` è l'iperparametro
    del rolling z-score, nel `fit_key`); costante con nome `TOP5_K` in `analysis_grid.py`, uguale a
    `legacy_export.LEGACY_TOPK`, registrata in `analysis.json` con le altre.
  - **C15 — storia di cal:** errore bloccante nei controlli preliminari se cal ha meno di `n_past` score validi, o se
    il supporto di calibrazione di `roll_robust_z` (D-23) è vuoto. Mai soglie NaN in silenzio.
    - **Non finiti (passo 6.0, approvato dall'utente il 2026-11-13; cambia il comportamento dei passi 3 e 4):**
      `metrics.eval_region` rifiuta qualunque score non finito nella regione valutata (prima solo i NaN: un `inf`
      produceva F1 ed eventi finiti), e `thresholds.py` rifiuta score non finiti (`±inf`) nel supporto di cal, nella
      storia delle soglie mobili e nello score valutato (prima un `inf` dava una soglia `inf`, cioè decisioni finite).
      L'errore diventa una riga NaN contata (C12). Motivo: D-44 resta aperta e nessun numero finito deve nascere in
      silenzio da un non finito. Parità con lo storico invariata: `evaluate_timestamps` falliva già (scikit-learn
      rifiuta gli `inf`) e `main.py` saltava la variante.
    - **Scostamento dichiarato (passo 6.0, approvato dall'utente il 2026-11-13):** per la variante senza normalizer
      (`raw_mean`, cioè `mean`/`none`) un residuo NaN in cal dà uno score NaN nel supporto M_cal; lo storico lo
      scartava in silenzio (`threshold_from_validation` ignora i NaN) e otteneva una soglia finita, ora la tabella
      lunga ha una riga NaN contata, coerente con D-44 aperta. L'esportazione legacy non cambia (usa
      `main.evaluate_variant`). È una correzione dichiarata (nota della sezione F8 del piano), da verificare a M1 con
      `n_nonfinite_cal` e da trattare come caso dichiarato nel controllo incrociato tabella lunga = summary legacy (6.6).
    - **`roll_robust_z` (passo 4, 2026-11-13):** la decisione è z_t ≥ τ, con τ calibrata sullo z di cal; la soglia non
      si riporta sullo score grezzo (la conversione inversa non è esatta in virgola mobile). Le metriche con soglia di
      questa politica si calcolano sulle decisioni in z; le metriche senza soglia restano sullo score grezzo.
  - **V10 — righe per politica di soglia:**
    - `pa` falso: per ogni politica del piano (`static_pct`, `roll_quantile`, `roll_robust_z`) tutte le metriche con
      soglia e per evento; per l'oracolo solo `f1`;
    - `pa` vero: per ogni politica elencata nel piano con `pa` vero, oracolo compreso, solo `f1`;
    - tutte le righe con `pa` vero e tutte quelle con oracolo hanno `diagnostic` vero (D-34);
    - le celle 2×2 sono casi particolari (C2); `free.best_f1_oracle` coincide esattamente con la riga oracolo, `pa`
      falso, `f1`;
    - quali combinazioni compaiono lo decide il piano di analisi (D-42 resta APERTA, nessun default).
  - **V11 — `n_cal_points`:** `free`: finestre di cal del normalizer (0 senza normalizer); `static_pct`: |M_cal|;
    `roll_robust_z`: dimensione del supporto di calibrazione (D-23); `roll_quantile`: `n_past`; oracolo: 0.
  - **V12 — convenzioni:** baseline random con `config_id = "random_baseline"`, `deterministic` falso, convenzione
    none/native, righe replicate identiche per ogni `ts_agg` (e per ogni altro livello che lo schema richieda);
    `preproc` = hash del blocco di preprocessing `[:12]`; `seed` dei modelli deterministici = `set_seed_value`.
    **Le repliche della baseline random non sono osservazioni indipendenti:** F7 le deduplica prima di qualunque
    aggregazione.
    - **Baseline random nella tabella `thr` (passo 6, approvato dall'utente il 2026-11-13):** riceve tutte le politiche
      del piano, PA e oracolo compresi, come ogni modello (è il caso che motiva RQ1a: l'F1 con PA del random). Le
      repliche esistono solo tra `ts_agg` (lo score random non dipende dalla proiezione; le soglie mobili solo con
      `last`); nello scenario `test_context` le righe random sono calcoli distinti (cambia la maschera), non repliche.
  - **V13 e B11 — scritture e lock:** `analyze` tiene il lock dello store per tutto il run (con il gettone legge e, se
    servono, rigenera i residui; `fit` e `analyze` sono mutuamente esclusivi). Tutti i file dell'analisi si scrivono
    in una cartella temporanea dentro il run, poi rinominata in modo atomico in quella definitiva
    `runs/<kind>/<esperimento>/analyses/<analysis_id>/` (rev. passo 0: la cartella definitiva è indicizzata da
    `analysis_id`, non dal `run_id`, che è unico per run; `analyses` non può coincidere con un `run_id`, che ha la forma
    ora UTC + 8 caratteri casuali; rev. passo 2: `analyses` è anche un `run_id` riservato, rifiutato da `lock` e
    `begin_run` senza distinzione di maiuscole). Il run conserva log e `run.json`, con l'`analysis_id` prodotto.
    `analysis.json` è il marcatore di completamento, scritto per ultimo. Una cartella temporanea rimasta dopo
    un'interruzione non blocca il nuovo lancio; la elencano `status` e `orphan_analysis_tmp` e la rimuove `clean-tmp`
    (estensione della nota «F5 passo 6» su D-70, passo 7), con test. Una cartella definitiva esistente e completa con
    lo stesso `analysis_id` causa un rifiuto esplicito (codice 2) con un messaggio che indica la cartella esistente,
    mai una sovrascrittura.
    - **Completezza (rev. passo 2, approvata dall'utente il 2026-11-13):** un'analisi è completa se e solo se esiste la
      cartella pubblicata `analyses/<analysis_id>/` con `analysis.json`; F7 legge solo quelle, indipendentemente dallo
      stato del run. Se dopo la pubblicazione `run.json` o il registro non si aggiornano: avviso nel log, `status`
      segnala l'incongruenza (passo 7), codice di uscita 0 (l'analisi è completa).
  - **B11 — rinomina atomica su Windows (rev. passo 0):** `os.rename` della cartella temporanea solo se la
    destinazione non esiste (controllo esplicito prima della rinomina, sotto lock: su POSIX `os.rename` sostituirebbe
    in silenzio una cartella vuota). Su `PermissionError` (antivirus o indicizzatore che tengono aperto un file) al
    massimo `PUBLISH_RENAME_ATTEMPTS` = 5 tentativi, con attese crescenti `PUBLISH_RENAME_WAITS_S` = (0,2, 0,4, 0,8,
    1,6) s tra un tentativo e il successivo, circa 3 s in tutto (costanti con nome in `store.py`; fissate al passo 2,
    approvate dall'utente il 2026-11-13). Motivo: i blocchi temporanei di antivirus e indicizzatore su Windows durano
    di solito meno di un secondo, ma non sempre; con 5 × 0,2 s fissi la finestra coperta è di 1 s, con il raddoppio
    di 3 s a parità di tentativi, con un costo trascurabile rispetto alla durata di un'analisi. Poi errore esplicito: codice di uscita 4, run chiuso `failed`, cartella
    temporanea lasciata per `clean-tmp`. Mai una copia come ripiego, mai un risultato parziale nella cartella
    definitiva. Test: rinomina che fallisce in modo simulato sempre (errore esplicito, nessuna cartella definitiva,
    temporanea intatta) e solo al primo tentativo (analisi pubblicata, contenuto identico).
  - **B12 — versione del codice di analisi e identità dell'analisi:** costante `ANALYSIS_CODE_VERSION`, analoga a
    `FIT_CODE_VERSION`. Rev. passo 0: `analysis_id` = sha256 della serializzazione canonica di
    - `plan_sha256`;
    - `ANALYSIS_CODE_VERSION`;
    - per ogni fit usato, in ordine di `fit_key`: `fit_key`, `status` e, per i fit non `failed`, gli sha256 per array
      dei residui effettivamente letti (identità dei residui di F5, record di `meta.json`, mai `npz_sha256`);
    - `labels_version` per macchina, in ordine di macchina;
    - hash del manifest (`canonical_hash` di `Manifest.to_dict()`);
    - (passo 1, approvato dall'utente il 2026-11-13) le pseudo-chiavi `random_baseline_key` di ogni (macchina, seed)
      della baseline random, ordinate: la baseline non ha fit nello store, e senza di esse un esperimento di soli
      modelli deterministici con seed diversi avrebbe lo stesso id con righe della baseline diverse;
    - (passo 1, dal controllo sistematico di copertura; rev. passo 2) per ogni fit `meta_sha256` = hash dei soli campi
      di `meta.json` elencati in `META_ID_FIELDS` (elenco esplicito, non per esclusione): `meta_format`, `fit_payload`,
      `fit_code_version`, `fit_key`, `machine`, `dataset`, `group`, `data_version`, `status`, `error` (tipo, messaggio,
      fase), `nonfinite_loss`, `loaded_with_version_mismatch`, `model`, `seed` (`set_seed_value`: colonna `seed` dei
      deterministici per V12), `segments`, `windows`, `scoring`, `device`, `fit_info` (epoche, componenti, loss,
      campioni), `checkpoint`, `scaler`, `residuals`. Fuori dall'id, in `META_MEASURED_FIELDS`: `timings`,
      `fit_info.fit_seconds`, `provenance`, `created_by` e `error.traceback` (percorsi locali e numeri di riga:
      diagnostica). Un campo di `meta.json` in nessuno dei due elenchi è un errore esplicito (codice 2). **Le colonne dei
      tempi della tabella `fits` (`fit_sec`) sono misure non riproducibili, escluse dal significato dell'id:** due
      analisi con lo stesso id hanno gli stessi numeri salvo i tempi. `fit_key` verificato contro l'hash del
      `fit_payload` registrato;
    - (passo 1, dal controllo sistematico di copertura) ambiente dell'analisi: versione di Python, versioni di numpy,
      scipy, pandas e scikit-learn, sha256 del lock delle dipendenze (le metriche dipendono dalle librerie);

    troncato a 12 caratteri. `analysis.json` registra in chiaro tutti i componenti dell'hash, la costante e il commit
    git. Motivo: `fit_key` identifica la richiesta, non il risultato; un fit DL rifatto con lo stesso `fit_key` (in un altro store, su
    un'altra macchina fisica o con un altro numero di thread) può dare residui diversi, e una correzione delle label o
    delle metriche cambia i numeri a parità di piano. L'id deve garantire «stessi ingressi e stesso codice, quindi
    stessi numeri». Ordine dei passi (nessuna lettura dei residui serve all'id: gli sha256 sono in `meta.json`):
    1. senza lock (letture ammesse da F5): validazione, controlli preliminari, manifest e `labels_version`, lettura dei
       `meta.json`, insieme dei residui da leggere secondo il piano, id provvisorio; se la cartella definitiva esiste ed è
       completa, rifiuto anticipato (codice 2);
    2. lock e `begin_run`;
    3. sotto lock, nuova lettura dei `meta.json` e nuovo calcolo dell'id, che è quello definitivo (un fit `failed` può
       essere stato rifatto nel frattempo); nuovo controllo della cartella definitiva (codice 2);
    4. ogni array letto passa da `get_residuals`, che lo verifica contro lo sha256 di `meta.json`; la rigenerazione
       rifiuta residui con sha256 diverso; quindi gli array letti sono esattamente quelli entrati nell'id.
  - **`ANALYSIS_CODE_VERSION` congelata a 1 (passo 8, 2026-11-16):** prova golden delle uscite di un'analisi
    sintetica deterministica (tabelle senza `fit_sec`, `analysis.json` senza `run_id`, `created_utc` e `git`) in
    `tests/golden/analyze_synth.json`, legata alla versione e all'ambiente: il test fallisce se le uscite cambiano senza
    aumentare la versione; la rigenerazione (`--write-golden`) rifiuta di sovrascrivere a parità di versione. Le analisi
    prodotte con il codice dei passi intermedi di F6 erano solo artefatti di test sintetici.
  - **B13 — test di leakage:** perturbare il test non cambia normalizer e soglie statiche; le soglie mobili restano
    invariate fino a t (test di causalità).
  - **Codici di uscita di `analyze`:** 0 completa; 1 almeno un fit `failed` (solo righe in `fits`); 2 config, piano o
    dati non validi, fit mancante nello store, `analysis_id` già presente, `kind: official`; 3 store occupato; 4
    pubblicazione non riuscita, nessun risultato (rinomina fallita dopo tutti i tentativi di B11; run chiuso `failed`,
    cartella temporanea lasciata per `clean-tmp`; aggiunto con la revisione del passo 0, 2026-11-13); 130 interruzione.
  - **File nuovi (rev. passo 6, approvata dall'utente il 2026-11-13):** oltre a `analyze.py`, `metrics.py`,
    `thresholds.py`, `analysis_plan.py` e `legacy_export.py`, anche `src/pipeline/analysis_grid.py` con le funzioni
    pure della 6b (nucleo di scoring per fit, righe `free` / `thr`, tabella `fits`); `analyze.py` resta orchestrazione.
  - **Aggiunte fuori dai file nuovi (elenco chiuso):** `scoring.py` (`RobustFeatureErrorNormalizer`); `table_schema.py`
    (tre colonne di `fits`, C6); `store.py` (rev. passo 2, funzioni reali: `analysis_dir`, `analysis_complete`,
    `new_analysis_tmp`, `write_analysis_file`, `write_table(..., dest_dir=...)`, `publish_analysis` con i tentativi di
    B11, `orphan_analysis_tmp` per `status` e `clean-tmp`; eccezioni `AnalysisExists`, `AnalysisPublishError`);
    `cli.py` (`analyze`, estensione di `clean-tmp`). Nessun metodo esistente cambia comportamento per i casi già
    ammessi, con un'eccezione approvata (passo 2): `lock` e `begin_run` rifiutano i `run_id` riservati
    (`RESERVED_RUN_IDS` = `analyses`, senza distinzione di maiuscole).
- **Motivazione:** metriche storiche invariate per costruzione (stesse funzioni e stessa formula), nuove metriche
  definite prima di vederne i valori, nessuna esclusione implicita mentre D-44 è aperta, analisi ripetibile e
  identificata dai suoi ingressi effettivi (piano, residui, label, manifest) e dalla versione del codice.
- **Approvazione:** utente, 2026-11-12, su raccomandazione di Claude Project (piano F6, C1–C15, V10–V13, modifiche
  A1–A9, B10–B13); revisione del passo 0 (ambito e formula di `analysis_id`, rinomina su Windows) approvata
  dall'utente il 2026-11-13.
- **Collegamenti:** D-04, D-11, D-20, D-21, D-22, D-23, D-24, D-34, D-42, D-44, D-59, D-60, D-70.
  **Serve entro:** F6.
- **Documenti:** piano F6 (testo corretto e nota); proposta §12 Fase 4 (nota), §5.1, §6, §9 (superate sui punti sopra,
  rimando di una riga in ciascuna); `CLAUDE.md` §3.4 (D-72 tra le approvate).

### D-73 · Stadio di statistica e report (F7) — APPROVATA · 2026-11-16
- **Decisione** (gli identificativi P1–P15 rimandano al piano F7):
  - **Perimetro:** piano statistico, lettura delle analisi pubblicate, D-44 parametrica, deduplicazione della baseline
    random, media sui seed, descrittive (media, std, mediana, ranghi), RQ2 (D-36), RQ1a/RQ1b (D-25), τ di Kendall e
    inversioni di segno, RQ3 con i soli descrittori ricavabili dalle tabelle, RQ4, comandi `stats` e `report`, test su
    tabelle simulate. **Fuori:** VUS-PR (D-34), descrittori dai dati (D-43), controllo E5 sulla deriva, congelamento
    e `kind: official` (F10), dati reali, statsmodels, modifiche a `aggregate_results.py`, `main.py`, `analyze` e
    allo schema delle tabelle, parallelizzazione.
  - **P1 — media sui seed:** F7 non usa `table_schema.mean_over_seeds`, che salta i NaN (media di pandas) e conta
    `n_seeds` senza considerarli, in contrasto con D-44. `src/stats` media con `skipna=False` dopo il filtro di D-44;
    `table_schema.py` resta invariato e un test di F7 documenta la differenza; la rimozione si propone in F12.
  - **P5 — piano statistico separato:** `configs/stats/<nome>.yaml`, validato da `src/stats/plan.py` con le regole di
    F2 (chiave sconosciuta, campo mancante o tipo errato → errore; chiavi YAML duplicate rifiutate; nessun valore
    predefinito), con `stats_plan_sha256` proprio. δ, α, B, seed del bootstrap, famiglia confermativa, riferimento di
    RQ2 (D-37), regole di D-44, blocchi e livelli (D-42) stanno qui, non nel piano di analisi (il cui hash entra in
    `analysis_id`). Ogni chiave di RQ è obbligatoria, `null` per non calcolarla. F10 congela entrambi i piani.
    - **Revisione (2026-11-16, passo 1, schema approvato dall'utente):**
      - sezioni di primo livello: `stats` (`name`, `kind`, `machine_role`), `inputs`, `exclusions`, `descriptive`,
        `rq2`, `rq1a`, `rq1b`, `rq3`, `rq4`; un piano con tutte le RQ `null` è rifiutato; `rq3` richiede `rq2`;
      - **fissati per definizione, senza chiave nel piano:** RQ2 scenario `primary` e `ts_agg = last`; RQ1b e RQ4
        scenario `primary`, `ts_agg = last`, `pa` falso (il PA è diagnostico, D-34). Un loro cambiamento è un cambiamento
        del codice (`STATS_CODE_VERSION`);
      - `eta2_principal` (`total` | `without_machine`) in ciascuna sezione che decompone (`rq1a`, `rq1b`, `rq4`);
        ogni blocco dichiara `tau_reference` (combinazione canonica dei livelli, P11) e `delta_reference` (modello
        di riferimento per le inversioni di segno); in `rq1b` un blocco con un solo modello ha entrambi `null`
        (esplicito), in `rq1a` servono almeno 2 modelli; almeno un fattore (modello compreso) con almeno 2 livelli;
      - i fattori dichiarabili sono tutti obbligatori per la tabella del blocco (un fattore con un solo livello è una
        condizione fissata, mai un default implicito): RQ1a `free` → `ts_agg`, `scenario`; RQ1a `thr` → `ts_agg`,
        `scenario`, `threshold` (solo `static_pct` | `oracle`), `pa`; RQ1b `free` → `scoring`, `preproc`; RQ1b `thr`
        → `scoring`, `threshold` (`static_pct` | `roll_quantile` | `roll_robust_z`), `preproc`; senza `scoring` per i
        blocchi di modelli `native` (D-25), che non si mescolano con quelli non `native`. Il validatore sa quali modelli
        sono `native` dall'attributo di classe `native_only` (`schema.model_class`); il passo 2 lo riscontra con
        `native_models` di `analysis.json`;
      - V10: in un blocco `thr` con oracolo o con `pa` vero tra i livelli la metrica deve essere `f1` (le altre
        metriche non hanno quelle righe);
      - **`detection_delay` non è mai una risposta decomposta** (RQ1a, RQ1b, RQ4): è NaN per definizione (C4) e
        decomporlo scarterebbe celle in silenzio. Si riporta solo in forma descrittiva (`descriptive` e la parte
        descrittiva di RQ4, `rq4.descriptive_metrics`), sempre insieme a `event_recall` (regola del validatore) e al
        numero di seed con almeno un evento rilevato;
      - con `kind: official`: `rq2.metric` = `ap_norm` (D-07) e `n_boot` ≥ 10.000 (D-36); con `exploratory` la
        metrica di RQ2 è libera tra quelle senza soglia non diagnostiche e `n_boot` è libero;
      - metriche diagnostiche (`best_f1_oracle`, righe con oracolo o `pa` vero) escluse da `descriptive`, RQ2, RQ1b e
        RQ4 (P13); in RQ1a l'oracolo e il PA sono livelli di fattori di misura;
      - **canonicalizzazione come `plan_sha256` di F6:** numeri reali normalizzati a float (99 e 99.0 sono lo stesso
        livello, per tipo e per hash), interi senza conversione (un float al posto di un intero è un errore), bool mai
        accettati come numeri; duplicati controllati sui valori normalizzati **prima** dell'ordinamento; liste senza
        ordine significativo ordinate; `stats_plan_sha256` = hash canonico della forma normalizzata.
        `STATS_CODE_VERSION` non entra in `stats_plan_sha256` (entra in `stats_id`, P6): un aggiornamento del codice
        non cambia l'hash di un piano congelato;
      - tutti i motivi di `NAN_REASONS` sono chiavi obbligatorie di `exclusions.nan_rows`.
  - **P6 — ingressi e uscite:**
    - gli `analysis_id` si passano da CLI; il piano statistico dichiara gli **esperimenti attesi** per nome e, per
      ciascuno, il `plan_sha256` dell'analisi atteso. Codice 2 per: ingresso di un esperimento non dichiarato;
      ingresso con `plan_sha256` diverso dal dichiarato; esperimento dichiarato e non fornito. Motivo: l'insieme degli
      ingressi è congelabile in F10, senza scelta a posteriori delle analisi da includere;
    - per ogni esperimento atteso il piano statistico dichiara anche l'hash della config dell'esperimento
      (`config_sha256` di `src/utils/config.py`); codice 2 se non corrisponde. Fatto verificato: oggi
      `config_sha256` è solo nel registro (`registry.csv`), che per V13 non fa parte della garanzia di completezza.
      Quindi `stats` lo calcola dal file dell'esperimento passato con `--input` e verifica che l'analisi derivi da
      quella config: nome e kind uguali a quelli di `analysis.json` e insieme dei `fit_key` di `analysis.json`
      (`components`) uguale a quello generato dalla config. Test al passo 2 (decisione dell'utente, 2026-11-16, vedi
      la nota su D-09);
    - **Revisione (2026-11-16, passo 1, approvata dall'utente):**
      - *Copertura di `fit_key` (fatto verificato in `config.fit_payload` e `_base_payload`):* entrano
        `FIT_CODE_VERSION`, `data_version` della macchina (hash di train e test, D-59), `dataset`, `machine`, `split`
        (modo e frazioni o `val_ratio`), `preprocessing` (`scaler_type`, `min_range`, `clip`), `windows`
        (`window_size`, `step`, `train_step`), modello (nome e `params`), `residual_aggs` (D-33), `device`,
        `training` (tutti e sei i campi) per i modelli addestrati, `seed` per i modelli non deterministici (la PCA con
        `random_state: from_seed` è non deterministica). Campi della config che non entrano, e perché non influenzano
        il fit: `experiment.name`, `experiment.kind`, `experiment.notes` (etichette; `kind: official` è solo un
        rifiuto); `experiment.seeds` come lista (entra il singolo seed dei modelli non deterministici; per i
        deterministici il seed non cambia il fit per costruzione, D-57, combinazioni incoerenti rifiutate dal
        validatore); `paths.store_root` (posizione); `data.data_dir` (posizione: il contenuto è identificato da
        `data_version`); `data.machines` (insieme delle macchine: ogni fit ha la sua `machine`); `training` per i
        modelli non addestrati (non lo usano); `models` fuori da `run_models` e `run_models` (selezione di cosa
        eseguire, non di come). Nessun campo che influenza il fit ne resta fuori. Fuori dalla config, quindi fuori da
        `fit_key`: numero di thread di torch e versioni delle librerie (provenance, D-17); per l'analisi li coprono
        gli sha256 dei residui in `analysis_id` (D-72 B12);
      - *riscontro con il registro:* se `registry.csv` contiene la riga del run dell'analisi (`run_id` di
        `analysis.json`) con un `config_sha256` diverso da quello calcolato → codice 2; se la riga manca → solo un
        avviso, registrato in `stats.json`. Il registro resta un riscontro, non una prova (V13). Test al passo 2;
    - si leggono solo le analisi complete (cartella pubblicata con `analysis.json`, D-72 V13); controlli bloccanti tra
      ingressi: stesso dataset, stesso hash del manifest, stessa `labels_version` per macchina, stesso ruolo delle
      macchine;
    - livelli e blocchi dichiarati nel piano statistico devono esistere nelle analisi in ingresso (D-42 è divisa tra
      i due piani: livelli calcolati nel piano di analisi, blocchi e livelli letti nel piano statistico); altrimenti
      codice 2. Test al passo 2;
    - uscite in `store/stats/<stats_id>/`, `stats.json` scritto per ultimo, pubblicazione atomica come D-72 B11, sotto
      lock, riga di registro con stage `stats`; `stats_id` = hash canonico di `stats_plan_sha256`, ingressi ordinati
      (kind, esperimento, `analysis_id`, sha256 di `analysis.json`), `STATS_CODE_VERSION` e ambiente (Python, numpy,
      scipy, pandas), troncato a 12. Stesso id già presente e completo → rifiuto (codice 2).
  - **P7 — baseline random:** le repliche tra `ts_agg` (D-72 V12) si verificano identiche bit a bit (altrimenti
    errore) e se ne tiene una; nello scenario `test_context` le righe sono calcoli distinti e non si deduplicano. Il
    random **non è un livello del fattore modello** (RQ1a, RQ1b, RQ2) e resta fuori da ranking e τ; compare solo come
    riferimento (gonfiamento del PA, celle 2×2) e nelle descrittive.
  - **P8 — forma operativa di D-44** (i valori restano APERTI):
    - ordine fisso: (1) stato del fit (`fits.status`, join su `fit_key`; la baseline random non ha riga in `fits` ed è
      trattata come «nessun fit, sempre `complete`»); (2) non finiti del fit (`nonfinite_loss`, `n_nonfinite_cal` > 0,
      `n_nonfinite_test` > 0), solo dopo (1) perché le colonne sono nullabili; (3) righe NaN per `nan_reason`;
      (4) macchine per numero di eventi, contato sullo scenario `primary`;
    - azioni: `error`, `exclude_fit`, `exclude_machine_pair` (la macchina esce dal confronto per tutti i modelli
      confrontati), `keep` (solo per i non finiti del test: il fit supera il passo 2; un'eventuale metrica NaN è poi
      trattata dal passo 3 con la regola del suo `nan_reason`). Mai un NaN che arriva a Δ senza una regola;
    - nessun default: il blocco delle esclusioni è obbligatorio; un caso incontrato senza regola dichiarata è un errore
      (codice 2);
    - `exclude_fit` può introdurre un bias di selezione se i fallimenti dipendono dal modello: colonna `n_seeds_used`
      per (macchina, modello) nelle uscite; in `stats.json` il conteggio delle esclusioni per regola, motivo e modello;
      se il piano dichiara `exclude_fit` deve dichiarare anche un'analisi di sensibilità con `exclude_machine_pair`,
      calcolata e riportata accanto, altrimenti il piano è rifiutato;
    - `detection_delay` NaN per definizione (D-72 C4) non è un'esclusione: per macchina si riportano la media sui seed
      in cui è definito, il numero di seed con almeno un evento rilevato ed `event_recall`.
    - **Revisione (2026-11-16, passo 1, approvata dall'utente) — portata delle esclusioni**, per le righe
      (`nan_rows`) e per coerenza per i casi a livello di fit:
      - `exclude_fit` esclude il fit da **tutta** l'inferenza, in ogni cella, mai da una sola cella: medie su seed
        diversi tra celle confonderebbero il fattore con il seed;
      - `exclude_machine_pair` esclude la macchina dall'**intera unità di inferenza** che contiene la cella: l'intero
        blocco in RQ1a, RQ1b e RQ4, l'intero confronto in RQ2 (per tutti i modelli confrontati).
      Azioni ammesse: `failed_fit`, `nonfinite_loss`, `nonfinite_cal` e ogni motivo di `nan_rows` → `error` |
      `exclude_fit` | `exclude_machine_pair`; `nonfinite_test` → anche `keep`; `min_events` (`n` ≥ 1, scenario
      `primary`) → `error` | `exclude_machine_pair`. `exclude_fit_sensitivity` (bool) è vero se e solo se compare
      almeno un `exclude_fit`. Test al passo 3: una riga NaN in una sola cella fa sparire il fit da tutte le celle,
      oppure la macchina da tutto il blocco.
  - **P9 — RQ3:** Spearman con IC bootstrap e interfaccia dei descrittori; in F7 solo i descrittori ricavabili dalle
    tabelle (`prevalence`, `n_events` dello scenario `primary`). I descrittori dai dati attendono D-43.
  - **P10 — RQ4:** decomposizione delle metriche con soglia in modello e politica (livello, `n_past`) a
    `ts_agg = last`; **«indicatore di scostamento cal → test»** = FPR del test − quota nominale (1 − livello/100) per le
    soglie statiche. Non è una misura di deriva: la quota su cal è esatta per costruzione (D-23). **Da fare, fuori da
    F7:** il controllo E5 (`piano_tesi_v1.md` §5: escludere i punti vicini agli eventi), che richiede nuove righe di
    analisi; da collocare con una voce futura prima della lettura di RQ4.
  - **P11 — τ di Kendall:** τ-b tra il ranking dei modelli nella configurazione canonica e in ogni altra configurazione,
    per macchina, con la **distribuzione completa** sulle macchine; τ-b sui ranking delle medie per macchina; quota di
    macchine in cui il segno di Δ si inverte. Tutto etichettato come esplorativo.
  - **P12 — parità con `aggregate_results.py`** (sull'esportazione legacy sintetica, in `tmp_path`, mai su `results/`):
    (1) `per_machine_means` sulle righe rilette con `float_precision="round_trip"` contro la media sui seed di F7, a
    tolleranza zero, sulle colonne AP, ROC_AUC, BestF1_oracle, F1 (statica, `pa` falso), EventRecall, FalseAlarmSeg;
    (2) `aggregate_results.main(..., out_dir=tmp)`: stringhe media ± std e AP_lift identiche; MeanDelta,
    vittorie/pareggi/sconfitte, N e p di Wilcoxon a tolleranza zero; uno scarto si diagnostica e l'unica causa
    ammessa, da documentare (D-17), è il parser predefinito di `pd.read_csv`; (3) con NaN presenti un test documenta
    la differenza voluta (D-44). Non si riproducono il ripiego su `raw_mean` per i modelli `native`, `AlertRate` e
    `InferSec` (assenti dalle tabelle).
  - **P13 — confermativo ed esplorativo:** ogni riga delle uscite ha `inference` ∈ {`confirmatory`, `exploratory`};
    `stats` rifiuta `kind: official` fino a F10 (come `fit` e `analyze`), quindi in F7 tutto è esplorativo; vietato
    mescolare `machine_role` dev ed eval nella stessa inferenza; le righe `diagnostic` non entrano mai negli ingressi di
    RQ2 / RQ1b, solo nelle tabelle diagnostiche etichettate.
  - **P14 — generatore:** B e seed del bootstrap sono parametri dichiarati senza default, registrati in `stats.json`.
    La regola B ≥ 10.000 (D-36) si applica in `plan.py` solo per `kind: official` (da F10); nei piani esplorativi B è
    libero.
  - **P15 — LaTeX:** writer minimale scritto a mano in `src/reporting/tables.py`. jinja2 è nel lock solo come
    dipendenza indiretta di torch e non è in `requirements.txt`: `DataFrame.to_latex` (e `to_markdown`, che richiede
    tabulate) non si usano (D-18).
  - **File:** nuovi `src/stats/{plan,inputs,aggregate,bootstrap,decisions,ranking,variance,descriptors,run}.py`,
    `src/reporting/{tables,figures,report}.py`; aggiunte a `src/pipeline/store.py` (cartella delle statistiche,
    completezza, pubblicazione con i tentativi di B11) e a `src/cli.py` (`stats <piano_stats.yaml> --input
    <esperimento.yaml> <analysis_id> [...]`, `report <cartella_stats> --out <cartella_nuova>` che rifiuta una cartella
    esistente, `status` con le statistiche pubblicate; codici di uscita 0/2/3/4/130).
- **Motivazione:** F7 legge solo artefatti pubblicati e identificati; ogni scelta metodologica è un parametro dichiarato
  e congelabile, nessuna esclusione o selezione implicita; le statistiche sono ripetibili e identificate dai loro
  ingressi.
- **Approvazione:** utente, 2026-11-16, su raccomandazione di Claude Project (piano F7, voci P1–P15 con le modifiche
  dell'utente a P6, P8, P10, P11).
- **Collegamenti:** D-07, D-08, D-12, D-18, D-19, D-23, D-25, D-30, D-34, D-36, D-37, D-42, D-43, D-44, D-71, D-72.
  **Serve entro:** F7.
- **Documenti:** piano F7; `CLAUDE.md` §3.4; proposta §5.1–5.3 (superate sui punti sopra).
- **Nota di rimando (2026-11-16, passo 0-bis, il testo sopra resta invariato):**
  - **P7 esteso:** gli AE non addestrati (D-69) hanno lo stesso trattamento della baseline random: righe di riferimento
    e descrittive, mai livelli del fattore modello in RQ1a, RQ1b e RQ4, mai nella famiglia confermativa né come
    riferimento di RQ2;
  - **P13 e D-45:** senza secondo dataset, le uscite di RQ2 con `inference = confirmatory` si descrivono nella tesi come
    «confermative rispetto al protocollo congelato, su dati già osservati con altri protocolli»; il report lo riporta
    accanto alle decisioni di RQ2.
- **Nota di rimando (2026-11-16, passo 1-bis, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **P5, nuova sezione `descriptive_baselines`:** lista obbligatoria (anche vuota) di `{model, config_id}`. Il ruolo
    di baseline descrittiva (AE non addestrati, D-69) si dichiara nel piano, perché lo schema dei modelli di D-69 non
    esiste ancora; il riscontro con le config dei modelli si aggiunge quando D-69 lo fissa. L'identità è la coppia
    `(model, config_id)`: un Conv-AE addestrato e uno non addestrato hanno lo stesso `model` e `config_id` diversi.
    Rifiutati: duplicati; `random_baseline` (ha il suo ruolo, P7); una baseline nella famiglia o come riferimento di
    RQ2, tra i modelli o come `delta_reference` di un blocco di RQ1a o RQ1b, tra i modelli di RQ4. `tau_reference`
    contiene solo livelli di fattori, mai modelli: il rifiuto lì è vuoto per costruzione;
  - **P5, livello degli IC:** il piano espone `m` = numero dei confronti della famiglia e il livello 1 − α/m
    (Bonferroni, D-36);
  - **P13, colonna `model_role`:** ogni riga delle uscite ha `model_role` ∈ {`compared`, `reference`,
    `descriptive_baseline`, `random_baseline`}, distinta da `inference` e dalle righe `diagnostic`.
  - **P13, assegnazione di `model_role` (utente, 2026-11-16):** per RQ e per blocco, non una volta per il piano:
    - RQ2: il riferimento è `reference`, i membri della famiglia sono `compared`;
    - blocchi di RQ1a, RQ1b e RQ4: il modello `delta_reference` è `reference`, gli altri sono `compared`;
    - le baseline descrittive (`descriptive_baseline`) e il random (`random_baseline`) tengono il loro ruolo ovunque;
    - il ruolo si scrive riga per riga, insieme alla RQ e al blocco (passi 4 e 7).
  - **P6, modelli non dichiarati (utente, 2026-11-16):** ogni modello `(model, config_id)` presente negli ingressi
    compare nel piano almeno una volta (famiglia o riferimento di RQ2, modelli di un blocco, modelli di RQ4,
    `descriptive_baselines`); altrimenti codice 2 con l'elenco dei modelli non dichiarati. Motivo: ignorarli in
    silenzio sarebbe un'esclusione implicita (D-44). Un modello che non serve a nessuna inferenza si dichiara in
    `descriptive_baselines`. Controllo e test nel passo 2.
  - **Modelli `native` come baseline descrittive (utente, 2026-11-16):** ammessi (le baseline semplici sono il caso
    d'uso); nessun rifiuto.

### D-53 · File tracciati da git (`.pyc`, dataset) — APPROVATA · 2026-11-10
- **Decisione:** i `.pyc` non sono tracciati: i 35 file già tracciati sono tolti dal tracciamento con
  `git rm -r --cached` sulle cartelle `__pycache__` (restano sul disco) e `.gitignore` esclude `__pycache__/` e `*.pyc`.
  `dataset/` resta tracciato (sola lettura, riproducibilità).
- **Approvazione:** utente, 2026-11-10, su raccomandazione di Claude Project.
- **Collegamenti:** D-55, D-56.

---

## D. Decisioni aperte (metodologiche)

### D-30 · Margine δ — APERTA
- **Opzioni:** (a) δ = 0,05 in AP normalizzata; (b) δ ricavato solo dalle macchine di sviluppo con una regola dichiarata;
  (c) altro valore motivato. Criterio scartato: δ ≥ spread del protocollo sugli stessi dati (circolare).
- **Vincolo:** va fissato prima dei run ufficiali e congelato in `configs/analysis/official.yaml`.
- **Serve entro:** F10. Il codice di F7 deve trattare δ come parametro, senza valore predefinito.
- **Nota di rimando (2026-11-16, F7, la voce resta APERTA):** δ è una chiave obbligatoria del piano statistico (D-73
  P5), senza default; δ ≤ 0 è un errore. La regola a quattro esiti si applica agli IC di Bonferroni al 1 − α/m
  (D-36): l'equivalenza è più conservativa del TOST classico. Il «Vincolo» sopra va letto così: δ si congela in
  `configs/stats/official.yaml` (D-73 P5), non in `configs/analysis/official.yaml`.

### D-37 · PCA di riferimento per RQ2 — APERTA
- **Opzioni:** PCA a finestre (stesso input degli AE, raccomandata) oppure la migliore tra le due PCA scelta sulle
  macchine di sviluppo. Distinta da D-06 (dati di fit).
- **Serve entro:** F10.

### D-38 · `train_step` — APERTA
- **Opzioni:** 5 (storico) oppure 1; in ogni caso identico per tutti i modelli. **Serve entro:** F8.

### D-39 · `max_epochs` e patience — APERTA
- **Proposta:** 100 e 10. **Serve entro:** F8.
- **Nota (2026-11-11, F4, la voce resta APERTA):** il trainer accetta `max_epochs` come alias opzionale di `epochs`
  (se assente non cambia nulla; entrambe presenti e diverse → `ValueError`). Fatto verificato: oggi nessun percorso
  passa entrambe le chiavi (`main.py` passa `config["models"][key]`, che in `repro_*.yaml` e nello snapshot F1 contiene
  solo `epochs`; la nuova pipeline traduce `training.max_epochs` in `epochs` in `schema.model_kwargs`). Il controllo
  tutela D-39: il valore scelto non può essere contraddetto in silenzio da una seconda chiave. Nessun valore fissato.

### D-40 · Preprocessing canonico — APERTA
- **Opzioni:** storico (minmax su fit, senza clip né pavimento; raccomandato) oppure con pavimento o clip.
  **Serve entro:** F8.
- **Nota (2026-11-10, la voce resta APERTA):**
  - **Osservazione su dati sintetici (F1):** nel banco di prova sintetico un sensore quasi costante nel train
    (deviazione 0,05 nel test) dopo lo scaler minmax senza pavimento (`min_range = 0`) arrivava a ±27 e dominava
    tutti gli score per finestra: PCA a finestre e AE avevano AUC e AP vicine al caso e le 4 varianti di scoring erano
    quasi identiche. Il generatore è stato ritarato (2,5 volte la deviazione del train, circa ±1,9) solo perché lo
    snapshot distinguesse i percorsi di scoring.
  - **Ipotesi (non verificata su SMD):** su alcune macchine reali l'amplificazione 1/range dei sensori quasi costanti
    nel train potrebbe penalizzare PCA a finestre e AE rispetto alle baseline. Se confermata, è un artefatto del
    protocollo pertinente alla domanda della tesi, e indebolisce la raccomandazione "storico, senza pavimento".
  - **Verifica proposta, prima di decidere D-40:** in sola lettura su `dataset/`, per ogni macchina e sensore, range
    nel train, range nel test e valore massimo assoluto dopo lo scaler storico; contare i sensori con valore massimo
    maggiore di 2 nel test. Nessun training. Gli esiti si guardano prima solo sulle 6 macchine di sviluppo [D-51].
  - **Collegamenti:** F1, F8, D-26.
- **Nota di rimando (2026-11-10, la voce resta APERTA):** con D-65 lo scaler in `split3` si fitta solo su fit (60%):
  la verifica proposta sopra misura gli intervalli anche su fit, non solo sulla parte di train storica (80%).

### D-41 · Budget di tuning — APERTA
- **Da decidere:** K per famiglia (proposta 8), griglie, criterio di selezione (proposta: AP normalizzata media sulle 6
  macchine di sviluppo, una configurazione globale per famiglia), seed nel tuning (proposta 3), se scoring e normalizer
  robusto rientrano nel budget. **Serve entro:** F9.

### D-42 · Livelli dei fattori per RQ1a e RQ1b — APERTA · **Serve entro:** F10.
- **Nota (2026-11-10, la voce resta APERTA):**
  - **Raccomandazione per i livelli di `n_past` di `roll_robust_z`:** escludere 2.880 dai livelli ufficiali oppure
    limitarlo agli esperimenti esplorativi (circa 8 superamenti per un percentile al 99,5% su 1.659 punti autocorrelati).
  - Già fissato da D-32: fattore "contesto all'inizio del test: sì/no". Vincolo da D-24: soglie mobili solo con `last`.
- **Nota (2026-11-12, F6, la voce resta APERTA):** in F6 i livelli dei fattori sono dati del piano di analisi
  (`configs/analysis/<nome>.yaml`, D-72 C8): nessun valore predefinito nel codice; le combinazioni presenti in tabella
  (scoring, soglie, `n_past`, PA) sono quelle elencate nel piano. I piani usati in F6 sono solo sintetici, nei test.
- **Nota (2026-11-16, F7, la voce resta APERTA):** i blocchi della decomposizione (D-25), i livelli letti da RQ1a /
  RQ1b, la configurazione canonica di riferimento per τ e la versione principale di η² sono dati del piano statistico
  (D-73 P5), senza default.

### D-43 · Descrittori per RQ3 (massimo 4) e ipotesi — APERTA · **Serve entro:** F10.
- **Nota (2026-11-16, F7, la voce resta APERTA):** F7 implementa Spearman con IC bootstrap e l'interfaccia dei
  descrittori, ma solo con quelli ricavabili dalle tabelle (`prevalence`, `n_events` dello scenario `primary`);
  i descrittori che richiedono i dati (deriva, quota fuori range, rango PCA) attendono questa voce (D-73 P9).
### D-44 · Regole di esclusione (fit fallito, macchina con un solo evento) — APERTA · **Serve entro:** F10.
- **Nota (2026-11-12, F5, la voce resta APERTA):** lo stadio di fit registra senza escludere: un fit con loss non
  finita resta `complete` con `nonfinite_loss: true`, e per ogni residuo si registra `n_nonfinite`; un errore di
  modello dà `status: failed`; un run interrotto dà la riga di registro `interrupted`, distinta da `failed` (D-70).
  Le tre situazioni restano distinguibili per la regola che si deciderà qui.
- **Nota (2026-11-12, F6, la voce resta APERTA):** lo stadio di analisi non esclude nulla (D-72):
  - la tabella `fits` riporta `status`, `nonfinite_loss`, `n_nonfinite_cal` e `n_nonfinite_test` (D-72 C6), così la
    regola si può applicare leggendo solo le tabelle: un non finito in cal compromette normalizer e soglia per tutto il
    test, uno nel test solo alcuni punti;
  - un fit `failed` compare solo in `fits`, senza righe di metriche;
  - una metrica non calcolabile (score non finiti nella maschera, regione senza positivi o senza negativi) dà una riga
    con `value` NaN, contata in `analysis.json`; la pipeline storica in quel caso saltava la variante, e solo
    l'esportazione legacy replica il salto (D-72 C12).
- **Nota (2026-11-13, F6 passo 5, la voce resta APERTA):** con un solo `inf` nei residui di cal il normalizer robusto
  dà centro e scala finiti (mediana e quartili lo assorbono), lo storico no (media e deviazione standard non finite).
  Un confronto tra normalizer su fit con `n_nonfinite_cal` > 0 mescolerebbe la robustezza dello stimatore con il
  trattamento dei non finiti: **D-44 va decisa prima della lettura di RQ1b.**
- **Nota (2026-11-16, F7, la voce resta APERTA):** forma operativa approvata in D-73 P8 (ordine fisso stato del fit →
  non finiti del fit → `nan_reason` → eventi sullo scenario `primary`; azioni `error`, `exclude_fit`,
  `exclude_machine_pair`, `keep`; nessun default; `exclude_fit` solo con sensibilità `exclude_machine_pair`). Restano
  da decidere i valori, cioè quale azione per ogni caso e la soglia sul numero di eventi.

### D-46 · Sensibilità delle proporzioni dello split — APERTA
- **Opzioni:** nessuna; oppure una sola variante (es. 70/15/15) sulle macchine di sviluppo in F9. **Serve entro:** F9.

### D-57 · PCA a finestre con `n_components` intero: determinismo — APERTA · 2026-11-10
- **Fatto verificato (codice, piano F2, K6):** con `n_components` intero, `PCADetector` usa `svd_solver="auto"`; con più
  di 1.000 feature (W·F = 3.800) e poche componenti scikit-learn sceglie il solver randomizzato, e `random_state` vale il
  seed (`setdefault` in `main.py`). Il modello è marcato deterministico, ma non lo è. Con `n_components` float (0,9,
  run storici e M1) il solver è `full` e il problema non si presenta. La PCA puntuale (38 feature) usa
  `covariance_eigh` ed è deterministica.
- **Impatto:** nessuno sui run storici e su M1. Rilevante per il `fit_key` (il seed è escluso per i modelli
  deterministici) e per D-41 se la griglia di tuning contiene `n_components` interi.
- **Opzioni:** (a) forzare `svd_solver="full"`: deterministica, costo maggiore, da misurare; (b) lasciare `auto` e
  trattare la PCA come non deterministica quando `n_components` è intero (seed nel `fit_key`, varianza tra seed da
  riportare).
- **Raccomandazione:** (a). La PCA è il riferimento di RQ2 (D-37): una varianza tra seed dovuta solo al solver
  aggiungerebbe rumore al confronto senza un significato metodologico.
- **In F2:** `random_state` esplicito; il validatore rifiuta la combinazione incoerente.
- **In F4 (rev. 2026-11-11):** F4 non corregge il solver; `PCADetector` resta invariato e il validatore F2 continua a
  rifiutare la combinazione incoerente. Il checkpoint (D-67) salva le componenti fittate, quindi i residui dopo `load`
  non dipendono dal solver; `FitInfo` registra il solver richiesto e quello effettivo. Nessun default nuovo per
  `n_components` o `svd_solver`.
- **Serve entro:** prima di D-41 (griglia di F9).
- **Revisione (2026-11-11):** le righe «In F2» e «Serve entro» sono state corrette su decisione dell'utente (piano F4,
  C4); prima dicevano «Correzione del modello in F4» e «F4 (implementazione), D-41 (griglia)». La voce resta APERTA.
- **Collegamenti:** D-37, D-41, `CLAUDE.md` §5.

### D-74 · Modello di previsione (integra D-10) — APERTA · 2026-11-16
- **Origine:** D-71 (approvata): si aggiunge un modello di previsione ai run ufficiali, con conclusioni dichiarate per
  famiglia di modelli. Integra D-10 (perimetro dei run ufficiali), il cui testo resta invariato.
- **Punti approvati (utente, 2026-11-16, su raccomandazione di Claude Project):**
  - **a) Forma della previsione:** un solo punto previsto per finestra (dai primi W−1 punti si prevede l'ultimo).
    Motivo: forma classica dei detector a previsione, semplice, distinta in modo netto dalla ricostruzione. La
    previsione a più passi o sequence-to-sequence è scartata: darebbe un modello molto vicino a una ricostruzione e più
    parametri da tarare.
  - **b) Conseguenza di (a), non una decisione:** c'è un solo residuo per finestra (vettore per sensore), quindi
    `window_time_agg` è degenerato (`last` = `mean`). `sensor_agg` e `normalizer` restano fattori veri.
  - **c) Proiezione:** il modello di previsione usa solo `ts_agg = last`. Motivo: il residuo riguarda il solo punto
    previsto (l'ultimo della finestra); con `ts_agg` offline (`mean`, `max`) all'istante t finirebbero errori di
    finestre che prevedono altri istanti, anche successivi a t: lo score non sarebbe interpretabile né causale. In RQ1a
    il modello entra solo in blocchi con `ts_agg` a un solo livello (`last`); gli altri fattori di misura (scenario,
    statica contro oracolo, PA) restano.
  - **d) Conseguenze da dichiarare nella tesi:**
    - in RQ1a e RQ1b il modello di previsione sta in blocchi senza i fattori `ts_agg` e `window_time_agg` (come i
      `native` per lo scoring);
    - nel tuning (D-41) gli AE scelgono anche `window_time_agg` e il modello di previsione no: non è una disparità di
      budget, perché l'opzione non esiste, ma va dichiarato.
- **Restano APERTI, da decidere prima di F8:** schema (chiavi, iperparametri); budget di tuning e parità di
  trattamento (D-41).
- **Nota per F8:** quando il modello di previsione esiste:
  - `analyze` / `analysis_grid` non produce righe offline (`ts_agg` `mean`, `max`) per il modello di previsione;
  - `src/stats/plan.py` lo rifiuta nei blocchi con `window_time_agg` o `ts_agg` a più di un livello;
  - test per entrambi i punti.
- **Ordine:** implementazione dopo M1, prima di F8. **Serve entro:** F8.
- **Collegamenti:** D-10, D-33, D-36, D-41, D-71.

### D-75 · Modello recente come estensione (es. Anomaly Transformer) — APERTA · 2026-11-16
- **Origine:** opzione (b) di D-71, separata dalla decisione approvata.
- **Decisione da prendere:** a fine M1.
- **Vincoli se approvata:** estensione con piano di analisi e piano statistico **congelati a parte**; non entra mai
  nella famiglia confermativa del piano principale; righe etichettate a parte; non deve ritardare l'analisi
  principale. Integrerebbe D-10 (perimetro) e D-41 (budget e parità di trattamento).
- **Collegamenti:** D-10, D-36, D-41, D-71.

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
