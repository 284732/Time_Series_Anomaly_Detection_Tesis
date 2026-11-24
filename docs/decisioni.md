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

### D-54 · Scadenze e tempo disponibile — APPROVATA · 2026-11-20
- **Testo della voce APERTA — DIPENDE DALL'UTENTE (storico, invariato):**
  - Determinano se F9 e il secondo dataset sono realistici.
- **Decisione:** consegna della tesi entro il **2027-01-31**; obiettivo di chiusura del progetto nella **prima metà di
  dicembre 2026**.
- **Conseguenze:**
  - il secondo dataset resta escluso (D-45, decisione sul perimetro del 2026-11-16);
  - condiziona l'opzione 1 di D-75 (solo se F11 è chiusa entro il 2026-12-10).
- **Approvazione:** utente, 2026-11-20, su raccomandazione di Claude Project.
- **Collegamenti:** D-45, D-75.
- **Documenti:** piano F8 (ordine dei passi 0–7), F9–F11.

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
- **Nota di rimando (2026-11-20, il testo sopra resta invariato):** D-75 è approvata (nessun modello recente nei run
  principali; estensione solo alle condizioni di D-75); gli AE non addestrati hanno i nomi
  `conv_autoencoder_untrained` e `lstm_autoencoder_untrained` (D-69, nota del 2026-11-20); il modello di previsione è
  `lstm_forecaster` (D-74, nota del 2026-11-20).

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
- **Nota di rimando (2026-11-17, F7 passo 8, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **Termini:** macchina come fattore di blocco; tutti gli effetti principali dei fattori fissi e tutte le loro
    interazioni fino all'ordine massimo; residuo = interazioni con la macchina (macchina × trattamento), accorpate.
    Nel disegno bilanciato i termini sono ortogonali: SS_tot = SS_macchina + Σ SS_termini + SS_residuo esattamente, e
    le quote non dipendono dalla codifica. Nelle uscite vengono prima le interazioni modello × fattore di primo ordine,
    poi gli effetti principali, poi le interazioni di ordine superiore. Un fattore con un solo livello nel blocco non
    ha termini.
  - **Nessuna inferenza sulle somme dei quadrati:** niente test F né p-value. Ogni macchina è misurata in tutte le
    celle (disegno a misure ripetute), quindi l'F con il residuo macchina × trattamento richiede sfericità e
    omogeneità non plausibili; i test sarebbero decine per blocco; tutto è esplorativo. Solo quote descrittive, con i
    gradi di libertà come informazione sul disegno.
  - **Indice R** (decisione dell'utente), per ogni blocco con almeno 2 modelli: R = Σ SS(termini fissi che contengono
    il modello e almeno un altro fattore) / [SS(modello) + Σ SS(termini fissi che contengono il modello e almeno un
    altro fattore)]. È la quota della variabilità legata ai modelli che dipende dai fattori del blocco. Esclude
    macchina e residuo. È descrittivo ed esplorativo, e si riporta accanto alla tabella dei termini.
  - **Stati espliciti, con criteri esatti** (la SS calcolata di valori identici può essere un residuo di
    arrotondamento non nullo):
    - η² non definite (`zero_total`) se tutti i valori del blocco sono identici;
    - η²_without_machine non definita se ogni macchina è costante;
    - R `zero_denominator` se i valori sono identici tra i livelli del modello in ogni cella; R `single_model` se il
      blocco ha un solo modello.

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
- **Nota di rimando (2026-11-17, F7 passo 5, il testo sopra resta invariato; utente):**
  - **regola a quattro esiti** (`piano_tesi_v1.md` §2, con gli IC di questa voce), con Δ = metrica(modello) −
    metrica(riferimento) e [L, U] l'IC al 1 − α/m:
    - superiore se L > δ;
    - equivalente se −δ ≤ L e U ≤ δ (IC ⊂ [−δ, δ], estremi compresi);
    - inferiore se U < −δ;
    - inconcludente altrimenti.

    I confini di superiore e inferiore sono stretti;
  - **un Δ con IC tutto sopra 0 che attraversa δ (0 < L ≤ δ < U) è *inconcludente*:** è una scelta voluta, perché la
    regola chiede rilevanza pratica (più di δ), non solo una differenza diversa da zero. Lo stesso vale,
    specularmente, sotto lo 0;
  - un IC NaN o non finito, L > U, δ ≤ 0, α fuori da (0, 1), m < 1 sono errori, mai un esito;
  - Holm non si implementa (M2 non adottato).
- **Nota di rimando (2026-11-17, F7 passo 6, il testo sopra resta invariato; fatto registrato su richiesta
  dell'utente):**
  - **Fatto (simulazione, nessun dato reale visto):** la BCa stratificata del metodo (A) non rispetta le bande di
    copertura fissate da questa voce.
    - Disegno: gruppi 6/7/9 (22 macchine), Δᵢ = μ + γ_g + σ_g·εᵢ, con effetti di gruppo fissi e varianza diversa per
      gruppo (σ = 0,5 / 1 / 2). Il valore vero è θ = μ + Σₕ (nₕ/n)·γₕ.
    - Suite: R = B = 2.000 al 95%; R = 1.000 e B = 10.000 al livello operativo 1 − 0,05/3.

    | caso | 95% (banda) | 1 − 0,05/3 (senza banda) |
    |---|---|---|
    | normale | 0,903 (banda [0,93; 0,97]: fuori) | 0,937 |
    | t₃ | 0,856 (banda ≥ 0,90: fuori) | 0,925 |
    | lognormale | 0,848 (banda ≥ 0,90: fuori) | 0,901 |
    | gerarchico (B), normale, solo documentazione | 0,976 | — |
  - **Diagnosi** (stesso disegno, R = 20.000):
    - un intervallo normale con la varianza nello strato distorta ((nₕ − 1)/nₕ, come il bootstrap) copre 0,907,
      come la BCa;
    - con la varianza corretta copre 0,926;
    - con la varianza corretta e i gradi di libertà di Welch–Satterthwaite copre 0,949;
    - la BCa nel caso omoschedastico copre 0,916.

    Le cause sono due: la distorsione (nₕ − 1)/nₕ della varianza nello strato e i pochi gradi di libertà effettivi
    (il gruppo più variabile domina l'errore standard). Il codice non è in errore.
  - **Conseguenza:** il metodo (A) **non è utilizzabile così com'è**. Gli IC di RQ2 sarebbero troppo stretti e gli
    esiti della regola a quattro esiti troppo frequenti.
  - **Il cambio di metodo si decide solo su simulazioni**, prima dei run ufficiali: nessun dato reale è stato visto.
    La nuova voce che rivede questa decisione si scrive dopo il confronto dei metodi candidati, con l'approvazione
    dell'utente.
- **Nota (2026-11-17, F7 passo 6, criterio di scelta fissato PRIMA di eseguire le simulazioni dei metodi candidati;
  utente, dettagli marcati «[Claude]» fissati da Claude Project e da confermare):**
  - **Metodi candidati:**
    - (1) **BCa stratificata riscalata:** in ogni strato xᵢ′ = x̄ₕ + √(nₕ/(nₕ − 1))·(xᵢ − x̄ₕ), poi la BCa
      stratificata invariata su x′; il jackknife si calcola su x′ [Claude];
    - (2) **bootstrap-t stratificato:** SE² = Σₕ (nₕ/n)² sₕ²/nₕ, con sₕ² corretta (ddof = 1); t*_b = (θ*_b − θ̂)/SE*_b;
      IC = [θ̂ − q₁₋q(t*)·SE, θ̂ − q_q(t*)·SE], con i quantili lineari di `np.percentile`. Una replica con SE*_b = 0 →
      errore, mai un ripiego; nella simulazione si conta come mancata copertura e si riporta il numero di casi.
      Repliche con SE* quasi nullo: nessuna soglia e nessuno scarto (sarebbe un ripiego silenzioso); l'instabilità si
      misura. Stabilità: simulazione aggiuntiva con 3 gruppi di uguale numerosità nₕ ∈ {2, 3, 4, 5, 6}, caso normale
      eteroschedastico, 95%, R = 1.000, B = 2.000, riportando copertura, larghezza media e numero di repliche con
      SE* = 0 [Claude];
    - (3) **intervallo t di Welch–Satterthwaite** sulla media stratificata, con varianza corretta per strato:
      θ̂ ± t_{ν, 1−q}·SE, con ν = (Σ cₕ)² / Σ (cₕ²/(nₕ − 1)) e cₕ = (nₕ/n)² sₕ²/nₕ. Nessun bootstrap.
  - **Disegno:** quello della nota precedente (6/7/9, eteroschedastico, casi normale, t₃ e lognormale, θ = μ +
    Σₕ (nₕ/n)γₕ), con gli stessi seed per tutti i metodi (stessi dati; stessi indici per (1) e (2)).
    - Al 95%: R = B = 2.000.
    - A 1 − 0,05/3: R = 1.000 e B = 10.000 (B non conta per (3)).
    - Si riportano copertura, errore Monte Carlo √(c(1 − c)/R) e larghezza media.
  - **Criterio:**
    - a) bande di questa voce al 95%: normale in [0,93; 0,97]; t₃ e lognormale ≥ 0,90;
    - b) tra i metodi che le rispettano vince la copertura più vicina a 0,9833 al livello operativo. [Claude]
      Distanza di un metodo = il **caso peggiore** sui tre casi, max |c − 0,9833|;
    - c) a parità vince il più semplice: (3), poi (1), poi (2). [Claude] «Parità»: differenza tra le distanze
      ≤ 2·√(SE₁² + SE₂²), con SE l'errore Monte Carlo del caso che determina ciascuna distanza;
    - se nessun metodo rispetta le bande: ci si ferma e si riporta.
  - **Esito (2026-11-17):** solo il metodo (2), bootstrap-t stratificato, rispetta le bande, nella suite e nella prova
    una tantum (R = B = 10.000). È adottato con D-76, che rivede il metodo (A). I dettagli marcati «[Claude]» non
    hanno influito: un solo metodo superava il punto a).

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
- **Nota (2026-11-17, F7 passo 7, proposta di Claude Project NON ancora approvata):** quando questa voce sarà
  realizzata, per un AE non addestrato il confronto descrittivo più informativo è con il proprio AE addestrato (stessa
  architettura, stesso seed), non solo con il riferimento di RQ2. Oggi RQ2 riporta per le baseline descrittive solo
  il Δ rispetto al riferimento (D-73, nota del passo 7). Il confronto appaiato «addestrato − non addestrato» si progetta
  allora.
- **Nota di rimando (2026-11-20, prerequisiti di F8, il testo sopra resta invariato; utente, su raccomandazione di
  Claude Project): dettagli approvati.**
  - **Fatti verificati nel codice (2026-11-20):**
    - un esperimento ammette un solo blocco per nome di modello (`src/utils/schema.py:349`, `:377`), e RQ2 legge solo
      dal `primary_experiment` (D-73): l'opzione «`untrained: true`» della proposta del 2026-11-16 non permetterebbe
      di avere l'AE addestrato e quello non addestrato nello stesso esperimento, quindi non è realizzabile;
    - le classi storiche inizializzano **tutti** i parametri con N(0, `init_std`²), compresi γ e β del BatchNorm
      (`src/models/deep/untrained_ae.py:30-34`): con γ ≈ 0 l'uscita è quasi costante e lo score è di fatto l'energia
      dell'input;
    - con zero epoche il ciclo di `PyTorchTrainer.fit` non viene eseguito, nessuno stato viene ripristinato,
      `stopped_epoch` = 0 e `best_epoch` = `None`; la rete si costruisce dentro `fit`, dopo `set_seed`
      (`src/pipeline/fit.py:113`) e prima di qualunque consumo del generatore da parte del DataLoader (lettura del
      codice, da dimostrare con un test al passo 3);
    - con `trained` falso `schema.model_kwargs` non passa `batch_size` (`schema.py:390-394`): lo scoring userebbe il
      64 implicito di `config.get("batch_size", 64)`, che decide il blocco interno di scoring (D-64) ed è fuori dal
      `fit_key`.
  - **Decisioni:**
    - **B1, nomi distinti:** `conv_autoencoder_untrained` e `lstm_autoencoder_untrained`, nuove voci di
      `schema.MODEL_SCHEMAS`. Sostituisce l'opzione «`untrained: true`» della proposta del 2026-11-16;
    - **zero epoche:** sottoclassi dei detector addestrati; inizializzazione standard di PyTorch, la stessa dell'AE
      addestrato; BatchNorm con le statistiche iniziali (γ = 1, β = 0, media 0, varianza 1), **scelta da dichiarare**
      nella tesi;
    - **B3 scartata** (classi storiche con `init_std`): non isola l'apprendimento (fatto sopra). In tesi si cita la
      differenza con Kim et al. (2022), Case 3. Le classi storiche restano nel registro di `main.py` (D-13) e in
      `OUTSIDE_NUCLEUS`;
    - **seed nel `fit_key`:** sì (modello non deterministico);
    - **correzione della proposta del 2026-11-16:** il `batch_size` usato nello scoring entra nel `fit_key` come blocco
      esplicito (per esempio `scoring`; nome della chiave fissato al passo 3), perché decide il blocco interno di
      scoring (D-64) e può cambiare i residui. Gli altri parametri di training (learning rate, `max_epochs`, patience,
      …) restano fuori;
    - **validatore:** quando un AE e il suo `*_untrained` sono entrambi in `run_models`, `params` identici e `batch_size`
      di scoring uguale a `training.batch_size` dell'AE addestrato; altrimenti `ConfigError`;
    - **controllo di parità dedicato:** test «pesi dell'AE non addestrato = pesi iniziali dell'AE addestrato con lo
      stesso seed, prima della prima epoca» (non esistono run storici di confronto);
    - **confronto appaiato «addestrato − non addestrato»** (nota del 2026-11-17): approvato nel principio come riga
      descrittiva; si progetta nel piano statistico in F10. Non è un prerequisito del codice.
  - **Realizzazione (passo 3 di F8, da rivedere alla revisione del passo):**
    - chiave di config del lotto di scoring: `score_batch_size` nei `params` di `*_untrained` (obbligatoria, intero
      ≥ 1, nessun default); `ModelSchema.untrained_of` collega ogni `*_untrained` al suo AE;
    - `fit_key`: blocco esplicito `scoring: {batch_size}` **in aggiunta** ai `params` completi. Scostamento dal
      primo disegno (togliere `score_batch_size` dai `params` del payload): `analysis_grid.fits_row` ricalcola
      `config_id` dai `params` del `fit_payload` con la formula di `config.config_id`, e i due `config_id` dello
      stesso fit sarebbero diversi tra le tabelle `free` e `fits` (fatto rilevato dal test d'integrazione);
    - classi `ConvAutoencoderUntrainedDetector` e `LSTMAutoencoderUntrainedDetector` in
      `src/models/deep/ae_untrained.py`, sottoclassi dei detector addestrati (`conv_ae.py`, `lstm_ae.py` invariati):
      `epochs` = 0 esplicito (il trainer altrimenti userebbe il default storico 40), un valore diverso da 0 è un errore,
      controllo dell'esito a zero epoche e rete in eval dopo `fit`;
    - `src/models/registry.py`: `NEW_PIPELINE_ONLY` (nomi senza corrispettivo in `main.py`) e `LEGACY_NUCLEUS`; i test
      ancorati alla pipeline storica (golden T0 e R1, registro di `main.py`) li escludono per nome, con le asserzioni
      sui modelli storici invariate.
  - **Nota per F10 (revisione del passo 3, utente, 2026-11-20):** nella config dell'esperimento ufficiale ogni
    `*_untrained` compare sempre insieme al suo AE addestrato in `run_models`, così il validatore di appaiamento si
    applica sempre. Da verificare al congelamento (F10); nessun codice ora.
  - **Effetto sull'elenco «Restano da fissare prima di F8»:** chiavi di schema (stesse dell'AE addestrato, `init_std`
    non serve), seed nel `fit_key`, BatchNorm e controllo di parità sono fissati; la regola di `plan.py` è in D-73
    (nota del 2026-11-20).
  - **Documenti:** piano F8 (passo 3).

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
- **Nota di rimando (2026-11-20, il testo sopra resta invariato):** D-75 (modello recente) è approvata: opzione 0 nei
  run principali, opzione 1 condizionata ai tempi (D-54); il perimetro della famiglia confermativa (m = 3) non cambia.
  Schema e implementazione del modello di previsione: nota del 2026-11-20 su D-74.

### D-76 · Metodo degli IC di RQ2: bootstrap-t stratificato — APPROVATA · 2026-11-17
- **Rivede:** D-36 per il solo metodo (A). Il resto di D-36 resta invariato: Bonferroni 1 − α/m, m dalla famiglia
  dichiarata, α e B parametri del piano, B ≥ 10.000 per `kind: official`, gerarchico solo nelle simulazioni, sign-flip
  di supporto.
- **Metodo di RQ2: bootstrap-t stratificato.**
  - Δᵢ per macchina; θ̂ = media di Δ con pesi uguali.
  - SE² = Σₕ (nₕ/n)²·sₕ²/nₕ, con sₕ² la varianza nello strato con `ddof = 1`.
  - Repliche: in ogni strato nₕ macchine con reinserimento. Indici generati una sola volta da `rq2.ci.seed` e
    condivisi da tutti i membri della famiglia: `np.random.default_rng(seed)`, gruppi in ordine di etichetta,
    macchine in ordine di nome, `rng.integers(0, nₕ, size=(B, nₕ))` per gruppo.
  - Per ogni replica θ*_b e SE*_b, con la stessa formula; t*_b = (θ*_b − θ̂)/SE*_b.
  - IC a code uguali: [θ̂ − t*₍₁₋q₎·SE, θ̂ − t*₍q₎·SE], con q = α/(2m). I quantili usano l'interpolazione lineare di
    `np.percentile`.
- **Nessun ripiego silenzioso:**
  - **SE osservato = 0** (Δ costanti dentro ogni strato, compreso il caso di Δ tutti uguali): IC degenere [θ̂, θ̂],
    stato `degenerate`, registrato. In questo caso tutte le repliche sono identiche e ogni SE* = 0, per costruzione.
    È l'effetto concreto del limite dello stratificato, che ignora la variabilità tra gruppi.
  - **SE osservato > 0 e almeno una replica con SE*_b = 0** → errore. Nessuna soglia e nessuno scarto delle repliche
    con SE* piccolo.
  - «SE = 0» (osservato o di una replica) si riconosce con il criterio esatto equivalente: valori identici dentro ogni
    strato, per uguaglianza. Non si usa la varianza calcolata, che per valori identici in float può dare un residuo
    di arrotondamento non nullo (fatto verificato da un test).
  - Δ, SE o quantili non finiti → errore.
  - **Casi noti:** con Δ costanti in più gruppi (per esempio due modelli con risultati identici su molte macchine) il
    bootstrap-t può incontrare repliche con SE* = 0 e dà errore per costruzione. Se accade sui dati reali, la
    decisione su come procedere richiede una nuova voce.
- **`min_machines_per_group`:** il minimo imposto da `plan.py` sale a **5**, la soglia di stabilità misurata. Con
  nₕ = 4 si sono avute 9 ripetizioni su 1.000 con SE* = 0; con nₕ ≥ 5, nessuna. Se dopo le esclusioni un gruppo
  scende sotto il minimo dichiarato, **RQ2 è «non calcolabile»** e si riporta come tale, con il gruppo e il
  conteggio. Nessun esito, nessun cambio di metodo dopo.
- **Macchine di sviluppo:** hanno 2 macchine per gruppo (D-51), quindi nessuna inferenza è possibile. `rq2` e `rq3`
  non nulle con `machine_role: dev` → errore in `plan.py`. Sulle macchine di sviluppo valgono le descrittive.
- **Valore del piano:** `rq2.ci.method` ammette solo `stratified_bootstrap_t`; `stratified_bca` è rifiutato, con un
  messaggio che rimanda a questa voce.
- **Evidenza:** `scripts/sim_ci_methods.py` e `docs/sim_ci_methods.json` (profili `suite` e `once`, con ambiente e
  data). Evidenza rigenerata con lo stesso criterio del codice per «SE* = 0»: le coperture del disegno 6/7/9 sono
  identiche; nello studio di stabilità cambia solo nₕ = 3 (947 ripetizioni su 1.000 con SE* = 0 invece di 726);
  nₕ = 4, 5, 6 restano 9, 0, 0 e la soglia 5 resta valida.
  - Disegno 6/7/9 eteroschedastico; casi normale, t₃ e lognormale; θ = μ + Σ(nₕ/n)γₕ.
  - Il criterio di scelta è stato fissato prima di eseguire (nota del 2026-11-17 su D-36): bande al 95%; poi la
    vicinanza a 1 − 0,05/3; a parità, il più semplice.
  - Solo (2) rispetta le bande. Prova una tantum (R = B = 10.000, larghezza media tra parentesi):

    | metodo | 95% normale | 95% t₃ | 95% lognormale | bande | 98,3% normale | 98,3% t₃ | 98,3% lognormale |
    |---|---|---|---|---|---|---|---|
    | (A) BCa | 0,895 (1,10) | 0,862 (1,03) | 0,846 (1,05) | no | 0,942 (1,34) | 0,920 (1,26) | 0,905 (1,29) |
    | (1) BCa riscalata | 0,913 (1,17) | 0,885 (1,10) | 0,868 (1,12) | no | 0,956 (1,43) | 0,940 (1,35) | 0,920 (1,38) |
    | (2) bootstrap-t | 0,951 (1,35) | 0,932 (1,33) | 0,917 (1,85) | sì | 0,982 (1,76) | 0,976 (1,68) | 0,956 (2,46) |
    | (3) Welch | 0,950 (1,30) | 0,958 (1,18) | 0,865 (1,13) | no | 0,984 (1,67) | 0,987 (1,52) | 0,915 (1,46) |
- **Limite dichiarato:**
  - nel caso asimmetrico la copertura resta sotto il nominale: 0,917 al 95% e 0,956 a 1 − 0,05/3;
  - gli intervalli sono più larghi (lognormale 1,85 contro 1,13 di Welch al 95%);
  - l'inferenza resta condizionata ai 3 gruppi di SMD.
- **RQ3:** non eredita il metodo. Il metodo dell'IC della correlazione di Spearman si decide al passo 9, con una
  propria simulazione; fino ad allora `plan.py` rifiuta `rq3` non nulla, con un messaggio che rimanda al passo 9.
- **Approvazione:** utente, 2026-11-17, su raccomandazione di Claude Project, con il criterio fissato prima.
- **Collegamenti:** D-36, D-44, D-51, D-73.

### D-75 · Modello recente come estensione (es. Anomaly Transformer) — APPROVATA · 2026-11-20
- **Testo della voce APERTA (storico, invariato):**
  - **Origine:** opzione (b) di D-71, separata dalla decisione approvata.
  - **Decisione da prendere:** a fine M1.
  - **Vincoli se approvata:** estensione con piano di analisi e piano statistico **congelati a parte**; non entra mai
    nella famiglia confermativa del piano principale; righe etichettate a parte; non deve ritardare l'analisi
    principale. Integrerebbe D-10 (perimetro) e D-41 (budget e parità di trattamento).
  - **Collegamenti:** D-10, D-36, D-41, D-71.
- **Decisione:**
  - **Opzione 0, adesso:** nessun modello recente nei run principali; il limite si dichiara nella tesi (D-71 (a):
    conclusioni per famiglia di modelli).
  - **Opzione 1, condizionata:** un modello recente (per esempio Anomaly Transformer) con gli iperparametri
    dell'articolo, **senza tuning**; solo i fattori di misura di RQ1a e le descrittive; piano di analisi e piano
    statistico a parte, **scritti dopo F11 e dichiarati come tali**; righe etichettate a parte, mai nella famiglia
    confermativa. Si fa **solo se F11 è chiusa entro il 2026-12-10**, con un tetto di **3 giorni di lavoro**;
    altrimenti va tra gli sviluppi futuri, senza riaprire la decisione.
  - **Opzione 2 scartata:** tuning con budget pari (D-41), incompatibile con i tempi (D-54).
- **Conseguenze:**
  - il perimetro dei run principali (D-10, D-71) non cambia;
  - con l'opzione 1 il piano dell'estensione non è pre-registrato prima dei risultati principali: va dichiarato
    esplicitamente nella tesi, e le sue righe restano descrittive.
- **Approvazione:** utente, 2026-11-20, su raccomandazione di Claude Project.
- **Collegamenti:** D-10, D-36, D-41, D-54, D-71.
- **Documenti:** piano F8 (nota del 2026-11-20), F11.

### D-38 · `train_step` — APPROVATA · 2026-11-24
- **Testo della voce APERTA (storico, invariato):**
  - **Opzioni:** 5 (storico) oppure 1; in ogni caso identico per tutti i modelli. **Serve entro:** F8.
- **Decisione:** `train_step` = 5, identico per tutti i modelli che si addestrano su finestre: Conv-AE, LSTM-AE, PCA a
  finestre, modello di previsione (`lstm_forecaster`). La PCA puntuale si fitta su tutti i punti di fit
  (`fit_on_points`, `src/pipeline/fit.py:98-102`), quindi `train_step` non la riguarda; `rolling_zscore` e
  `input_norm` non apprendono parametri. Lo scoring resta a ogni istante (`step` = 1, D-60). Finestre della loss di
  early stopping: regola di D-63 (`score_windows(es, step)[::max(1, train_step // step)]`, solo dentro es).
- **Motivazione:** con passo 5 le finestre (W = 100) si sovrappongono già al 95%; il passo 1 costerebbe circa 5 volte
  per epoca, soprattutto in F9, in F11 e nei rilanci.
- **Limite da dichiarare (misura del 2026-11-24, sola lettura, solo le 6 macchine dev, `reports_local/f8_pre/`):** con
  il fit al 60% le finestre di training sono 2.824–3.430 per macchina (1-1 3.398; 1-7 2.824; 2-6 3.430; 2-8 2.825;
  3-11 3.424; 3-8 3.425), sempre meno di W·F = 3.800 (rapporto 0,74–0,90); il train più corto di SMD (23.687 righe)
  dà 2.823 finestre. La covarianza campionaria della PCA a finestre ha quindi rango ≤ N − 1 < W·F. Con lo storico
  (train all'80%) erano 3.772–4.579, già sotto 3.800 su 1-7 e 2-8.
- **Conseguenza:** nella griglia di F9 un `n_components` intero va limitato a ≤ N − 1 (note del 2026-11-24 su D-41 e
  D-57).
- **Effetto sui numeri:** nessuno, è il valore storico.
- **Approvazione:** utente, 2026-11-24, su raccomandazione di Claude Project.
- **Collegamenti:** D-03, D-41, D-57, D-60, D-63.
- **Documenti:** piano F8 (nota del 2026-11-24); `configs/base.yaml` (passo 1 di F8).

### D-39 · `max_epochs` e patience — APPROVATA · 2026-11-24
- **Testo della voce APERTA (storico, invariato):**
  - **Proposta:** 100 e 10. **Serve entro:** F8.
  - **Nota (2026-11-11, F4, la voce resta APERTA):** il trainer accetta `max_epochs` come alias opzionale di `epochs`
    (se assente non cambia nulla; entrambe presenti e diverse → `ValueError`). Fatto verificato: oggi nessun percorso
    passa entrambe le chiavi (`main.py` passa `config["models"][key]`, che in `repro_*.yaml` e nello snapshot F1 contiene
    solo `epochs`; la nuova pipeline traduce `training.max_epochs` in `epochs` in `schema.model_kwargs`). Il controllo
    tutela D-39: il valore scelto non può essere contraddetto in silenzio da una seconda chiave. Nessun valore fissato.
  - **Nota (2026-11-21, F8 passo 7, la voce resta APERTA): epoche dei run esistenti, solo fatti.** Sola lettura dello
    store (`store/fits/*/meta.json`) e di `results/results_tests/*/summary_timestamp_metrics.csv` (colonna `Epochs` =
    `stopped_epoch`); nessun training. Tutti i run sono solo su machine-1-1 e machine-3-8 (dev, D-52), seed 42, con
    `max_epochs` 40 e patience 5. Regola del trainer (`src/training/trainer.py`): stop per early stopping quando
    `stop − best ≥ patience`, altrimenti al tetto di `max_epochs`.

    | run | conv 1-1 | lstm 1-1 | conv 3-8 | lstm 3-8 |
    |---|---|---|---|---|
    | store M1, `parity_repro` (best / stop) | 39 / 40, tetto | 8 / 13, ES | 17 / 22, ES | 21 / 26, ES |
    | `results_repro_last_01`, `_mean_01` (stop, `efdacc6`) | 40 | 13 | 22 | 26 |
    | `results_repro_last`, `_mean` (stop, `843da6a`) | 40 | 19 | 22 | 21 |

    - **Fatto (M1):** `best_epoch` e `stopped_epoch` sono registrati, quindi la causa dello stop è certa (conv 1-1 al
      tetto di 40; gli altri tre per early stopping).
    - **Run storici:** solo `stopped_epoch`. Uno stop prima di 40 è early stopping (best = stop − 5). Uno stop a 40 non
      si distingue da un early stopping proprio all'epoca 40: per `_01` si risolve con M1 (parità esatta,
      `docs/m1_parita.md`), quindi è il tetto; per `843da6a` resta un'**inferenza**.
    - Le epoche dell'LSTM differiscono tra `843da6a` e `efdacc6`, che differiscono per lo scaler (D-50).
    - Tabella: `reports_local/d39/epochs.csv` (esclusa da git).
- **Decisione:** `max_epochs` = 100, `patience` = 10. Lo stesso blocco `training` vale per Conv-AE, LSTM-AE e modello
  di previsione. Fatto verificato nel codice (2026-11-24): `lstm_forecaster` ha `trained` vero
  (`src/utils/schema.py:182-183`), quindi `schema.model_kwargs` gli passa lo stesso `cfg.training` degli AE
  (`schema.py:432-436`) e il blocco entra nel suo `fit_key` (`src/utils/config.py:223-227`); `ForecastTrainer` rende
  obbligatorie quelle chiavi (`src/training/forecast_trainer.py:23-44`) ed eredita `fit` di `PyTorchTrainer` (early
  stopping e ripristino dell'epoca migliore, `src/training/trainer.py:80-132`). Gli AE non addestrati (D-69) restano
  fuori: zero epoche; per loro vale solo il vincolo `score_batch_size` = `training.batch_size` (`schema.py:350`).
- **Fuori da questa voce:** `min_delta` (1e−6), `learning_rate` (1e−3), `weight_decay` (1e−5) e `batch_size` (64)
  restano ai valori storici (nota del 2026-11-24 nel piano F8; griglia di D-41).
- **Conteggio dei fit fermati al tetto (report di F8):** un fit si è fermato al tetto se `stopped_epoch` =
  `max_epochs` e `stopped_epoch` − `best_epoch` < `patience` (regola del trainer: `stopped_epoch` da solo non
  distingue il tetto da un early stopping proprio all'ultima epoca); un `best_epoch` nullo (val loss mai migliorata,
  per esempio non finita) si conta a parte. Le colonne `best_epoch` e `stopped_epoch` sono nella tabella `fits`.
- **Vincolo:** il valore non si cambia dopo aver visto i risultati di F8.
- **Natura del cambiamento:** cambio di design dichiarato (storico 40 / 5); tocca solo i modelli addestrati, quindi può
  spostare Δ(DL − PCA). Si isola nel passaggio V3 → V2 dell'esperimento di F8 (nota del 2026-11-24 nel piano).
- **Approvazione:** utente, 2026-11-24, su raccomandazione di Claude Project.
- **Collegamenti:** D-41, D-69, D-74.
- **Documenti:** piano F8 (nota del 2026-11-24); `configs/base.yaml` (passo 1 di F8).

### D-40 · Preprocessing canonico — APPROVATA · 2026-11-24
- **Testo della voce APERTA (storico, invariato):**
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
  - **Nota (2026-11-20, la voce resta APERTA; utente, su raccomandazione di Claude Project): verifica autorizzata.**
    - **Perimetro:** sola lettura su `dataset/`; nessun training; nessuna label; **solo le 6 macchine di sviluppo**
      (`configs/dev_machines.yaml`, D-51), mai le 22 di valutazione.
    - **Misure, per macchina e sensore:** range su fit (60%) e sul train storico (80%); range nel test; valore massimo
      assoluto nel test dopo lo scaler storico (minmax senza clip né pavimento); conteggio dei sensori con massimo
      maggiore di 2 nel test.
    - **Uscite:** script in sola lettura, risultati in `reports_local/d40/` (esclusa da git); i numeri si riportano
      senza conclusioni sulla decisione, che resta dell'utente.
    - **Documenti:** piano F8 (passo 7).
  - **Nota (2026-11-21, F8 passo 7, la voce resta APERTA): esito della verifica, solo fatti, nessuna decisione.**
    - **Come:** `scripts/verify_d40.py` (test: `tests/test_f8_d40_check.py`); uscite in `reports_local/d40/` (esclusa da
      git: `per_sensor.csv`, `per_machine.csv`, `worst_cases.csv`, `meta.json`, `summary.md`). Hash del dataset
      verificati prima del run (84 file coincidenti). Solo `train/` e `test/` delle 6 macchine dev; nessuna label, nessun
      training. Confini da `src/data/segments.py::segment_bounds`: fit = 60% (`split3`, frazioni D-05), train storico =
      80% (`legacy2`, `val_ratio` 0,2), W = 100. Scaler storico `SMDScaler` minmax, `min_range = 0`, nessun clip; con
      range esattamente nullo lo scaler usa scala 1.
    - **Soglia dichiarata:** «quasi nullo» = range sul segmento di fit dello scaler < 0,01 (amplificazione ≥ 100×); il
      range esattamente nullo è contato a parte. Massimo assoluto contato oltre 2.
    - **Per macchina** (ogni cella: scaler su fit 60% / scaler su train 80%):

      | macchina | range nullo | quasi nullo (< 0,01) | \|max\| > 2 nel test | \|max\| peggiore (sensore) | sensori con range fit < range train80 |
      |---|---|---|---|---|---|
      | 1-1 | 8 / 8 | 10 / 10 | 24 / 24 | 787 (33) / 787 (33) | 20 |
      | 1-7 | 4 / 4 | 4 / 4 | 2 / 2 | 12,6 (9) / 12,6 (9) | 6 |
      | 2-6 | 8 / 8 | 10 / 10 | 14 / 14 | 113 (25) / 112 (23) | 7 |
      | 2-8 | 7 / 7 | 12 / 12 | 27 / 27 | 497 (29) / 333 (33) | 10 |
      | 3-11 | 9 / 9 | 11 / 11 | 12 / 12 | 922 (9) / 922 (9) | 18 |
      | 3-8 | 8 / 8 | 8 / 8 | 17 / 16 | 89 (31) / 26 (31) | 14 |

    - **Casi peggiori** (range nel test ≈ 1 in tutti): 3-11 s9 (range su fit 0,0011; \|max\| 922 / 922); 1-1 s33
      (0,0013; 787 / 787); 2-8 s29 (0,0020; 497 / 332); 1-1 s32 (0,0023; 431 / 431); 2-8 s33 (0,0030; 333 / 333). Elenco
      completo (5 per macchina) in `worst_cases.csv`.
    - **Altre soglie** (fit / train80; ordine 1-1, 1-7, 2-6, 2-8, 3-11, 3-8): \|max\| > 10: 6/5, 1/1, 4/4, 14/14, 6/6,
      1/1; \|max\| > 100: 2/2, 0/0, 2/2, 5/5, 2/2, 0/0. Range < 0,001 coincide con il range nullo su tutte le macchine.
    - **Sensori quasi nulli ma non nulli** (uguali con i due scaler): 1-1 {32, 33}; 2-6 {23, 25}; 2-8 {6, 29, 31, 32, 33};
      3-11 {5, 9}; nessuno su 1-7 e 3-8.
    - **I 44 sensori a range nullo** (gli stessi su fit e su train80):
      - 1-1 {4, 7, 16, 17, 26, 28, 36, 37}; 1-7 {7, 26, 36, 37}; 2-6 {4, 7, 16, 17, 26, 28, 36, 37};
        2-8 {7, 16, 17, 26, 28, 36, 37}; 3-11 {7, 10, 16, 17, 24, 26, 28, 31, 37}; 3-8 {4, 7, 16, 17, 26, 28, 36, 37};
      - 36 restano costanti anche nel test; 8 sono costanti nel train ma nel test coprono un range di 1 e dopo lo
        scaler arrivano a \|max\| = 1 (scala 1): 1-1 {26, 28}, 2-8 {16, 17, 28}, 3-11 {10, 31}, 3-8 {28}.
- **Decisione:** scaler `minmax` fittato solo su fit (D-65), `clip: null`, pavimento sul denominatore `min_range` =
  τ = **0,02** (amplificazione massima 1/τ = 50).
- **Criterio di τ:** «il τ più grande tra {0,01; 0,02; 0,05; 0,1} che, su ogni macchina dev, tocca al massimo 1
  sensore con range su fit ≥ 0,01». Misura del 2026-11-24 (sola lettura, `reports_local/f8_pre/`): sensori con range
  su fit in [0,01; τ), nell'ordine 1-1, 1-7, 2-6, 2-8, 3-11, 3-8: τ = 0,02 → 1, 0, 1, 0, 1, 1; τ = 0,05 → 3, 0, 1, 5,
  4, 1; τ = 0,1 → 4, 1, 2, 9, 4, 1. Il criterio dà τ = 0,02.
- **Da dichiarare sulla scelta di τ:**
  - il criterio è stato scritto **dopo** la misura del 2026-11-24, con i dati della nota del 2026-11-21 già noti;
  - usa solo i range su fit delle 6 macchine dev: nessuna label, nessun valore del test, nessuna prestazione;
  - τ è espresso nelle unità del dataset distribuito, già normalizzato min-max su train+test (D-26).
- **Sensori a range nullo:** si tiene la transizione continua: con τ > 0 il loro denominatore è τ, non 1
  (`src/preprocessing/scaler.py:69-70`, verificato il 2026-11-24). Per ogni range in [0, τ) il denominatore vale τ.
- **Effetti da dichiarare (misura del 2026-11-24):**
  - poiché i dati sono in [0, 1] (D-26), |z| ≤ 1/max(range su fit, τ) ≤ 1/τ = 50 per costruzione;
  - |max| nel test: da 787, 12,6, 113, 497, 922, 89,2 (storico, τ = 0) a 50, 12,6, 50, 50, 50, 50;
  - gli 8 sensori a range nullo su fit ma variabili nel test (nota del 2026-11-21: 1-1 {26, 28}, 2-8 {16, 17, 28},
    3-11 {10, 31}, 3-8 {28}) passano da |max| 1 a 1/τ = 50 e su 4 macchine (1-1, 2-8, 3-11, 3-8) raggiungono il
    massimo del test;
  - il numero di sensori con |max| > 2 nel test aumenta: da 24, 2, 14, 27, 12, 17 a 26, 2, 14, 30, 14, 18; con
    |max| > 10 da 6, 1, 4, 14, 6, 1 a 8, 1, 4, 17, 8, 2; con |max| > 100 da 2, 0, 2, 5, 2, 0 a 0 ovunque.
- **Motivazione (corretta rispetto al testo dell'utente del 2026-11-24, su verifica nel codice):** con lo scaler
  storico (`min_range` = 0) la verifica del 2026-11-21 mostra amplificazioni fino a circa 900× su sensori quasi
  costanti. L'amplificazione decide lo score, nelle finestre colpite, dei modelli che usano direttamente i dati
  scalati: AE (addestrati e non), modello di previsione, PCA a finestre, `rolling_zscore` (`eps` in unità scalate),
  `input_norm`. L'effetto sul confronto di RQ2 ha direzione ignota. La **PCA puntuale** ristandardizza internamente
  ogni sensore con la deviazione standard su fit (`src/models/traditional/pca_pointwise.py:31-35`, `:43`): per i
  sensori con range positivo è invariante rispetto allo scaler minmax (a meno degli arrotondamenti float32), quindi τ
  non la cambia e la sua amplificazione interna 1/std resta; τ agisce su di lei solo sui sensori a range nullo
  (std < 1e−8 → 1), che passano da 1 a 1/τ. Conseguenza per D-37: nota del 2026-11-24.
- **Alternative scartate:**
  - scala 1 solo per il range esattamente nullo: reintroduce la discontinuità (range 0 → scala 1, range ε → 1/ε) che
    crea l'amplificazione storica;
  - rimozione dei sensori costanti su fit: cambio strutturale di F (W·F, schema dei residui), e scarta sensori che
    variano nel test;
  - storico senza pavimento come canonico: resta come variante misurata (sotto).
- **Lo storico (`min_range` = 0)** resta come variante misurata nell'esperimento di F8 (passaggio V2 → V1) e come
  livello candidato del fattore scaler di RQ1b (D-42 resta APERTA). Per `standard` e `robust` `min_range` ha unità
  diverse (deviazione standard, IQR / 1,349; `scaler.py:68-69`): τ = 0,02 vale per `minmax` e non si trasferisce in
  automatico a quei livelli (D-42).
- **Natura del cambiamento:** cambio di design dichiarato; tocca tutti i modelli (la PCA puntuale solo sui sensori a
  range nullo).
- **Approvazione:** utente, 2026-11-24, su raccomandazione di Claude Project (criterio e valore di τ fissati
  dall'utente dopo la misura del 2026-11-24, prima di qualunque codice).
- **Collegamenti:** D-26, D-37, D-42, D-65.
- **Documenti:** piano F8 (nota del 2026-11-24); `configs/base.yaml` (passo 1 di F8).

---

## C. Software e riproducibilità — decisioni approvate

### D-14 · Verifica di "nessun cambiamento numerico" — APPROVATA · 2026-11-09
- **Decisione:** fino a M1: snapshot sintetico della pipeline storica (F1) e test golden dei residui. A M1: confronto su
  dati reali lanciato dall'utente. Dopo M1: confronto tra run dello `store/`. `confronta_run.py` resta lo strumento per
  la pipeline storica.
- **Documenti:** `CLAUDE.md` §5; proposta §9; piano F1, M1.
- **Nota di rimando (2026-11-19, chiusura di M1, il testo sopra resta invariato):** M1 superata il 2026-11-19 con
  parità esatta su entrambe le varianti di D-50. Fit e analisi sul commit `4d8d248`; run di fit
  `20261119T144234-c54fd9f8`; analisi `be0a03233647` (aggregazione nella finestra `last`, proiezione `last`) e
  `a02e1de4a1e5` (aggregazione nella finestra `mean`, proiezione `last`). Dettagli in `docs/m1_parita.md`. Da qui la
  verifica di «nessun cambiamento numerico» è il confronto tra due run dello `store/` con stessa config e stessi seed;
  snapshot F1 e golden restano nella suite come controlli sintetici.

### D-15 · Interfaccia di `fit` invariata, `FitInfo` come attributo — APPROVATA · 2026-11-09
- **Decisione:** `fit(X, y=None, X_val=None) -> self` non cambia. Le informazioni di training vanno in `self.fit_info_`
  (`FitInfo`). La nuova pipeline passa fit come `X` ed es come `X_val`. `residuals`, `save`, `load` sono aggiunte.
- **Motivazione:** pura aggiunta, nessuna modifica strutturale ai sensi di `CLAUDE.md` §8, test esistenti invariati (C7).
- **Documenti:** `CLAUDE.md` §6; proposta §5.1; piano F4.

### D-16 · Training sintetico nei test — APPROVATA · 2026-11-09
- **Decisione:** i test possono addestrare modelli minimi su dati sintetici generati in `tmp_path`; non leggono `dataset/`
  e non scrivono in `results/`, `store/` o altre cartelle del repository.
- **Documenti:** `CLAUDE.md` §4; piano F1.
- **Nota di rimando (2026-11-19, chiusura di M1, il testo sopra resta invariato; modifica di test esistenti approvata
  dall'utente):** il controllo su `store/` del repository passa da «`store/` assente» a «`store/` non modificato dai
  test».
  - **Motivo:** dopo M1 `store/` esiste legittimamente (D-56) e contiene i run di M1. Il vecchio controllo falliva
    sempre e non verificava più l'intento di questa voce.
  - **Come:** in `tests/conftest.py` un'impronta di sessione presa all'inizio (`tests/fixtures/store_guard.py`:
    percorsi relativi, cartelle, dimensione e `mtime_ns` dei file), confrontata alla fine dei 6 test che prima
    controllavano l'assenza: `test_no_store_in_repository` (rinominato `test_store_not_modified`) in
    `test_f5_cli.py`, `test_f5_failures.py`, `test_f5_fit.py` e `test_f5_regenerate.py`, più i due test di fumo con sottoprocesso di F5 e F6. Se `store/`
    manca all'inizio, il test verifica che manchi ancora: in quel caso il controllo coincide con il precedente.
  - **Verifica:** prove su radici temporanee in `tests/test_store_guard.py` (file nuovo, cartella vuota nuova, file
    riscritto con lo stesso contenuto, dimensione cambiata a `mtime` invariato, file cancellato, `store/` creato quando
    mancava, `store/` rimosso). Le prove di difetto sono fatte sulla guardia e sulla fixture, mai su `store/`.

### D-17 · Procedura di M1 e tolleranze — APPROVATA · 2026-11-09
- **Decisione:** PCA e baseline deterministiche: tolleranza zero. AE: prima si tenta l'identità bit a bit con lo stesso
  numero di thread dei run storici; se ci sono differenze, indagine preliminare documentata della causa, poi eventuale
  tolleranza motivata. Il numero di thread si registra nella provenance.
- **Documenti:** `CLAUDE.md` §5; piano M1.
- **Nota di rimando (2026-11-19, M1, il testo sopra resta invariato):**
  - **Thread in M1** (decisione D1 dell'utente, 2026-11-19, su raccomandazione di Claude Project): thread predefiniti,
    non fissati a 1; prima di ogni comando si verifica che nessuna variabile `*_NUM_THREADS` sia impostata, e il numero
    di thread si registra nella provenance di ogni fit (`scripts/parity_report.py` lo controlla e lo riporta). Motivo:
    `main.py` non fissa i thread, quindi i run storici hanno usato i valori predefiniti; fissarli a 1 introdurrebbe una
    differenza rispetto allo storico. La nota di F7 «fissato o registrato» riguarda le tabelle diagnostiche di
    `stats`, che M1 non esegue: si sceglie «registrato», nessun conflitto con questa voce. Il valore per i run
    ufficiali si decide in F10.
  - **Valori predefiniti misurati su questo PC alla data di M1** (2026-11-19, ambiente `C:\Users\predi\venvs\smd_m1`,
    uguali nell'installazione globale): torch 4 thread intra-op e 4 inter-op; OpenBLAS 8 e 8 (`libscipy_openblas`
    0.3.30 di numpy e 0.3.29.dev di scipy); OpenMP `libiomp` 1 e 4 (due pool); `vcomp` 8.
  - **Limite** (D-50, D-61): per i run `_01` non è registrato su quale macchina e con quali thread siano stati prodotti;
    resta un limite dichiarato.
  - **Regola D5** (decisione dell'utente, 2026-11-19): se storico ≠ riesecuzione di `main.py` ma riesecuzione = nuova
    pipeline bit a bit, M1 è superata come «parità nell'ambiente di riferimento», con una voce di registro. Per PCA, PCA
    puntuale e baseline random una differenza storico ≠ riesecuzione va comunque diagnosticata (prima i dati, con
    `scripts/make_data_hashes.py --check`) e l'accettazione richiede l'approvazione esplicita dell'utente (tolleranza
    zero di questa voce).
- **Nota di rimando (2026-11-19, chiusura di M1, il testo sopra resta invariato):** identità bit a bit ottenuta per
  tutti i modelli, AE compresi, con i thread predefiniti registrati sopra (uguali in tutti gli 8 fit). Questo vale per
  storico contro nuovo, storico contro riesecuzione e riesecuzione contro nuovo, epoche comprese. Nessuna tolleranza
  fissata, nessuna diagnosi necessaria; la regola D5 non si è applicata. Il valore dei thread per i run ufficiali resta
  da decidere in F10. Dettagli in `docs/m1_parita.md`.

### D-18 · Dipendenze — APPROVATA · 2026-11-09
- **Decisione:** ammesse quelle di `requirements.txt` (numpy, pandas, scipy, scikit-learn, torch, pyyaml, matplotlib,
  seaborn, tqdm). Nessuna nuova dipendenza senza approvazione. `pyarrow` non si aggiunge.
- **Documenti:** `CLAUDE.md` §6; proposta §3, §10, §11.
- **Nota di rimando (2026-11-19, M1, il testo sopra resta invariato; decisione D4 dell'utente):** due pacchetti usati
  dal progetto non sono in `requirements.txt` e si dichiarano qui, senza aggiungere dipendenze nuove:
  - `pytest`: strumento di sviluppo per la suite (non usato dal codice di calcolo); è nel lock (9.1.1);
  - `threadpoolctl`: importato direttamente da `src/utils/provenance.py` per registrare i pool BLAS/OpenMP; è già una
    dipendenza di scikit-learn ed è nel lock (3.6.0).
  `requirements.txt` resta invariato.

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
- **Nota di rimando (2026-11-19, chiusura di M1, il testo sopra resta invariato):** riferimento riprodotto esattamente,
  con tolleranza zero, dalla nuova pipeline (analisi `be0a03233647` e `a02e1de4a1e5`) e anche dalla riesecuzione di
  `main.py` nell'ambiente dedicato (`reports_local/m1/main_rerun_{last,mean}/`). Su 70 file per confronto non c'è
  nessuna differenza, salvo le esclusioni fissate prima dei run. Il limite sopra (versioni, thread e macchina dei run
  `_01` non registrati) resta dichiarato, con portata ridotta: lo storico coincide bit a bit con la riesecuzione.
  Dettagli in `docs/m1_parita.md`.

### D-55 · Ambiente di riferimento e file di lock — APPROVATA · 2026-11-10
- **Decisione:** ambiente di riferimento = ambiente Windows dell'utente con **Python 3.13.9**, lo stesso per test, M1 e
  run ufficiali. `requirements-lock.txt` generato con `python -m pip freeze` da quell'ambiente.
- **Nota:** lock provvisorio, generato dall'installazione globale di Python (contiene pacchetti estranei al progetto).
  Prima di M1: ambiente virtuale dedicato e lock rigenerato da lì, registrato come aggiornamento di questa voce.
- **Collegamenti:** D-17, D-18.
- **Nota di rimando (2026-11-19, M1, il testo sopra resta invariato): lock rigenerato dall'ambiente dedicato.**
  - **Ambiente:** `C:\Users\predi\venvs\smd_m1`, fuori dal repository (la provenance conta anche i file non
    tracciati), Python 3.13.9, creato con `python -m venv` dall'installazione globale (decisioni D4 e D6
    dell'utente, 2026-11-19).
  - **Installazione:** `pip install -r requirements.txt pytest -c requirements-lock.txt` (vincoli dal lock
    provvisorio: stesse versioni dell'ambiente usato finora). Nuovo lock scritto in byte dal freeze
    (`subprocess.check_output([sys.executable, "-m", "pip", "freeze"])`, niente `>` di PowerShell);
    `provenance.file_sha256` = `6e09688d74c4b22a23036c23b79bd826eeafceefcb93d56ea9f73d60d790ac59`.
  - **Criterio del confronto dei lock (fissato prima):** numpy, scipy, pandas, scikit-learn, torch, threadpoolctl e le
    loro librerie native con versioni identiche; ammesse solo la scomparsa dei pacchetti estranei al progetto e la
    comparsa di pytest e delle sue dipendenze; qualsiasi altra differenza ferma M1.
  - **Esito:** 49 pacchetti estranei tolti, nessuno aggiunto, nessuna versione cambiata (37 pacchetti identici;
    `pytest` era già nel lock). Diff approvato dall'utente il 2026-11-19.
  - **Librerie native** (controllo proposto da Claude, approvato dall'utente): sha256 di tutti i `.dll` e `.pyd` di
    numpy, scipy, scikit-learn, torch, pandas e threadpoolctl confrontati tra installazione globale e ambiente
    dedicato: identici (le sole 4 differenze sono file di pacchetti estranei assenti dall'ambiente dedicato:
    `charset_normalizer`, `cbor2`, `zmq`); `threadpool_info()` e thread di torch identici. Script e risultati in
    `reports_local/m1/env_check/` (`native_check.py`, `glob.json`, `venv.json`), esclusa da git.
  - **Effetto:** cambia `lock_sha256`, quindi la firma d'ambiente (`provenance.environment_signature`): i fit di M1 si
    fanno dopo il commit del nuovo lock.

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
- **Nota di rimando (2026-11-16, passo 1-ter e piano del passo 2, il testo sopra resta invariato; utente, su
  raccomandazione di Claude Project):**
  - **Fatto verificato (`config.config_id`, `config.fit_payload`, `_base_payload`):** `config_id` è l'hash di nome e
    `params` del modello. Del `fit_payload` restano fuori da `config_id`:
    - `fit_code_version`, `data_version`, `dataset`, `machine`;
    - `split`, `preprocessing`, `windows`, `residual_aggs`, `device`;
    - `training`, per i modelli addestrati;
    - `seed`, per i modelli non deterministici.

    Lo stesso `(model, config_id)` può quindi comparire in più esperimenti con fit diversi.
  - **P5, `primary_experiment`:** chiave obbligatoria, uno degli esperimenti di `inputs`. RQ2, RQ1a, RQ4 e le
    descrittive (comprese le baseline descrittive) leggono solo da quell'esperimento. Qui si fissa il preprocessing
    canonico (D-40), congelato in F10.
  - **P5, livelli di preproc di RQ1b:** sono **nomi di esperimenti** di `inputs` (non hash del preprocessing); ogni
    blocco di RQ1b contiene l'esperimento primario tra i suoi livelli (verificato in `plan.py`). L'id del
    preprocessing di ogni livello si legge dalle tabelle al passo 2 e si registra in `stats.json`.
  - **P5/P6, nessun ingresso senza ruolo:** un esperimento di `inputs` che non è né `primary_experiment` né un livello
    di preproc in un blocco di RQ1b è rifiutato in `plan.py`. È la stessa logica dei modelli non dichiarati: niente
    entra negli ingressi senza un uso dichiarato.
  - **P6, controlli del passo 2:**
    - (i) dentro un esperimento, ogni `(model, config_id, machine, seed)` corrisponde a un solo fit;
    - (ii) gli esperimenti che fanno da livelli di preproc in un blocco di RQ1b hanno `fit_payload` identici a parte
      `preprocessing`: stessi split, finestre, training, `residual_aggs`, device, macchine, seed e `data_version`,
      per gli stessi `(model, config_id)`. Altrimenti codice 2 con i campi che differiscono, perché le altre
      differenze si confonderebbero con l'effetto dello scaler; due livelli con lo stesso preprocessing → codice 2;
    - (iii) l'esperimento primario tra i livelli di preproc di ogni blocco di RQ1b: già in `plan.py`.
  - **P6, `stats_id` e `stats.json`:**
    - gli sha256 dei file delle tabelle letti si registrano in `stats.json` ed entrano nei componenti di `stats_id`;
    - `analysis_code_version` è uguale tra gli ingressi, altrimenti codice 2; la versione corrente si registra
      soltanto;
    - con `stats.kind = official` tutti gli ingressi sono `official`; con `exploratory` qualunque ingresso è
      ammesso, e `stats.json` registra il kind di ciascuno.
  - **Ricostruzione dei `fit_key`:** una funzione pura estratta da `fit.plan_fits` (macchine e `data_version` come
    argomenti), usata sia da `plan_fits` sia dallo stadio di statistica; snapshot F1 e golden invariati. Un eventuale
    helper in `src/data/manifest.py` per risolvere le macchine dallo spec senza leggere i dati è un'aggiunta.
- **Nota di rimando (2026-11-16, passo 2, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **P6, ingresso senza fit:** un'analisi senza fit nei `components` (al più la baseline random) → codice 2: non
    contribuisce a nessuna RQ e il dataset non è verificabile.
  - **Fatto verificato (`fit.plan_tasks`, `fit.set_seed_value`, `analyze._tables`; tabelle di un'analisi sintetica):**
    - un modello deterministico ha **un solo fit per macchina**, con il seed `experiment.seeds[0]`. Il `fit_key` non
      contiene il seed, quindi cambiare `experiment.seeds` non cambia i suoi `fit_key`; cambia solo le chiavi della
      baseline random;
    - nelle tabelle `free` e `thr` ha **una sola riga per cella**, con `seed = seeds[0]` e `deterministic = true`
      (colonna già presente). Non ci sono repliche per seed da deduplicare;
    - la baseline random ha una riga per seed, con valori diversi tra i seed: sono repliche vere. La replica identica
      bit a bit di V12 riguarda le copie tra livelli di `ts_agg`, non i seed.
  - **Conseguenze per i passi 3 e 4:**
    - per i modelli deterministici `n_seeds = 1`. La deviazione standard sui seed è **non applicabile**: si dichiara
      con un valore esplicito nelle uscite, distinto da un NaN da filtrare o da un'anomalia. La forma (colonna o
      valore) si fissa al passo 4;
    - per un modello deterministico `exclude_fit` toglie l'unico fit, quindi equivale a escludere la macchina per quel
      modello. La sensibilità di D-44 (P8, analisi con `exclude_machine_pair`) lo dichiara.
- **Nota di rimando (2026-11-16, passo 3, P8 e P7, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **Unità di inferenza** (portata di `exclude_machine_pair`):
    - `rq2`: il confronto intero;
    - `rq1a/<blocco>`: il blocco;
    - `rq1b/<blocco>`: il blocco, su tutti gli esperimenti dei suoi livelli di preproc;
    - `rq4`: il blocco intero, con lo stesso insieme di macchine per tutte le metriche;
    - `descriptive/<i>`: ogni voce delle descrittive è la sua unità, e riporta il proprio numero di macchine.
  - **Celle non usate da nessuna unità né descrittiva:** le regole di `nan_rows` non si applicano e non generano
    esclusioni. I loro NaN si contano in `stats.json` per motivo e modello, come informazione («NaN in celle non
    usate»), non come esclusioni. I casi a livello di fit (`failed_fit`, `nonfinite_*`) si applicano sempre,
    indipendentemente dalle celle.
  - **Principio dei seed:** dentro ogni unità l'insieme dei seed di un (model, config_id, machine) è lo stesso in
    tutte le celle (verificato; altrimenti errore). Quindi:
    - `exclude_fit` toglie il fit da ogni cella in cui compare (globale anche per le descrittive);
    - nelle unità che attraversano più esperimenti (blocchi di RQ1b con il fattore preproc) toglie lo stesso
      (model, config_id, machine, seed) da tutti gli esperimenti del blocco;
    - le unità che leggono solo `primary_experiment` non sono toccate dai fit esclusi negli altri esperimenti.
  - **Baseline random (P7):** dopo la deduplicazione tra `ts_agg` (solo scenario `primary`, verifica bit a bit) si tiene
    la riga con `ts_agg = last`, e il numero di repliche rimosse va nel registro. La random è nelle unità delle
    descrittive e, come riferimento, nei blocchi di RQ1a; non è un livello del fattore modello.
  - **Propagazione a senso unico (P7/P8, utente):**
    - un'esclusione causata da un modello confrontato o di riferimento (riferimento di RQ2, `delta_reference`) toglie
      la macchina dall'unità per tutti, random e baseline descrittive comprese;
    - un caso della baseline random o di una baseline descrittiva (NaN di una riga, oppure caso a livello di fit) non
      toglie mai macchine né fit ai modelli dell'unità. Toglie solo le righe di quella baseline per quella macchina:
      dalle unità della cella per un NaN di riga, da tutte le sue unità per un caso a livello di fit. Si registra con
      la sua regola e il suo motivo e con azione effettiva `reference_only`, qualunque sia l'azione dichiarata salvo
      `error`, che ferma comunque;
    - vale anche per la baseline random nei blocchi di RQ1a.
  - **Etichetta `diagnostic` verificata, non presunta:** una riga deve avere `diagnostic` vero esattamente quando ha
    oracolo, PA vero o una metrica diagnostica (`best_f1_oracle`); altrimenti errore. Le unità di RQ2, RQ1b, RQ4 e
    delle descrittive escludono il PA solo attraverso questa etichetta, che quindi si controlla.
  - **Errori aggiuntivi:**
    - un NaN con `nan_reason` nullo su una metrica diversa da `detection_delay`, in una cella usata (NaN senza
      motivo);
    - un'unità senza macchine dopo le esclusioni;
    - `n_events` diversi tra le righe dello scenario `primary` della stessa macchina.
  - **Ordine dentro il passo 2 di P8:** `nonfinite_loss`, poi `nonfinite_cal`, poi `nonfinite_test`; vale il primo
    caso trovato.
  - **Casi `error`:** si segnalano tutti, anche quando un'altra esclusione dello stesso passo toglierebbe la stessa
    riga.
  - **Conteggi:** il filtro restituisce il numero di macchine per unità e per gruppo, prima e dopo le esclusioni, da
    registrare in `stats.json`. La soglia minima per gruppo del bootstrap stratificato si fissa al passo 6.
- **Nota di rimando (2026-11-17, passo 4, P1, P8, P12, P13, il testo sopra resta invariato; utente, su raccomandazione
  di Claude Project):**
  - **Deviazione standard sui seed:** `ddof = 1`. Lo stato `seed_sd_status` vale:
    - `computed`;
    - `not_applicable` per i modelli deterministici;
    - `single_seed` per un modello non deterministico rimasto con un solo seed dopo le esclusioni.

    La tabella per macchina ha la colonna `deterministic`.
  - **`detection_delay`:** media sui seed in cui è definito, `n_seeds_defined`. Senza eventi rilevati il valore è vuoto
    con stato `no_detected_event`: non è un'esclusione né un errore.
  - **`model_role` nelle descrittive:** si usano i ruoli di RQ2, oppure `compared` per tutti se il piano non ha RQ2.
    `reference` vale solo per l'identità completa del riferimento di RQ2 (model, `config_id` e terna di scoring); le
    altre varianti di scoring dello stesso modello sono `compared` (correzione dell'utente). Random e baseline
    descrittive tengono il loro ruolo. RQ4 non ha `delta_reference`, quindi i suoi modelli sono tutti `compared`.
  - **`inference`:** `confirmatory` solo per le righe di RQ2 con ruolo `compared` o `reference` e `kind: official`;
    tutto il resto è `exploratory`.
  - **Uscite:** una tabella lunga per macchina (unità, `rq`, `block`, `model_role`, `inference`, identità, cella,
    `value_mean`, `n_seeds_used`, `n_seeds_defined`, `seed_sd`, `seed_sd_status`, `delay_status`, `prevalence`,
    `n_events`), più le descrittive. Sono gli ingressi dei passi 5–9.
  - **Descrittive complete** (perimetro di F7, `piano_tesi_v1` §3.6), per voce e per modello (variante di scoring
    compresa):
    - media, deviazione standard (`ddof = 1`), mediana, `n_machines`;
    - ranghi per macchina, solo tra i modelli `compared` e `reference` (random e baseline descrittive esclusi, P7);
      rango medio per i pari merito, pareggi esatti; riassunto con il rango medio sulle macchine e
      `n_machines_ranked`;
    - verso dei ranghi: più basso è meglio per `fpr`, `false_alarms_per_1440`, `fa_segments`, `detection_delay`; più
      alto è meglio per le altre metriche.
  - **`detection_delay` nelle descrittive:** nessuna esclusione implicita delle macchine senza eventi rilevati.
    Accanto alla media sulle macchine in cui il ritardo è definito si riportano `n_machines_defined` su `n_machines`
    e la media di `event_recall` della voce con le stesse condizioni.
  - **Parità di livello 2 (P12):**
    - MeanDelta, vittorie/pareggi/sconfitte, N e Wilcoxon si confrontano solo sui modelli non `native`: il ripiego
      storico su `raw_mean` è una differenza voluta;
    - la tolleranza 1e-4 sui pareggi esiste solo nelle funzioni di parità `legacy_*`, etichettate come tali; non entra
      in nessuna uscita metodologica di F7 (verificato da un test sul sorgente).
  - **Fatto verificato (P12):** `aggregate_results.main` rilegge i CSV per macchina con il parser predefinito di
    `pd.read_csv`, che può perdere l'ultima cifra. MeanDelta coincide a tolleranza zero con le funzioni di
    `aggregate_results` applicate alle righe lette con `round_trip`. L'uscita di `main` coincide a tolleranza zero con
    le stesse funzioni applicate alle righe lette con il parser predefinito. Lo scarto residuo è quindi attribuito al
    solo parser (D-17), non tollerato.
  - **Fatto verificato e avvertenza di interpretazione: `input_norm`** (`src/models/traditional/input_norm.py:38-52`):
    - il residuo per sensore è `x²` sui dati scalati, cioè l'errore di un modello che ricostruisce sempre zero:
      l'ultimo istante della finestra con `window_time_agg = last`, la media sulla finestra con `mean`.
      L'aggregazione sui sensori e il normalizer sono quelli dello scoring della pipeline;
    - ha la griglia completa di `window_time_agg`, quindi in RQ1 sta nei blocchi normali (non è `native`);
    - con lo scaler standard misura la distanza dalla media del train; con lo scaler min-max misura la vicinanza al
      massimo del train. Nel fattore scaler di RQ1b la sua sensibilità è quindi attesa: dipende dalla definizione
      dello score, non dal protocollo.
- **Nota di rimando (2026-11-17, passo 5, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **`src/stats/decisions.py`:** regola a quattro esiti come nella nota del 2026-11-17 su D-36 (estremi compresi per
    «equivalente», confini stretti per superiore e inferiore, codici `superior`, `equivalent`, `inferior`,
    `inconclusive`). Errore, non `inconclusive`, per IC NaN o non finiti.
  - **Verso della metrica verificato:** la regola si applica solo a metriche «più alto è meglio» (`HIGHER_IS_BETTER`,
    che contiene tutte le metriche ammesse in RQ2); un'altra metrica → errore.
  - **m dal piano, non dalle righe:** m è la dimensione della famiglia dichiarata nel piano statistico. Un membro
    dichiarato e assente dalle righe → errore; una riga di un modello non dichiarato → errore; un IC calcolato a un
    livello diverso da 1 − α/m → errore. Motivo: se m si ricavasse dai dati, un membro mancante restringerebbe gli IC
    dopo aver visto i risultati.
  - **Riga di decisione ricontrollabile da sola:** identità del modello (model, `config_id`, terna di scoring), esito,
    L, U, livello dell'IC (1 − α/m), δ, m, α e metrica.
- **Nota di rimando (2026-11-17, passo 6a, P5, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **`min_machines_per_group`:** chiave obbligatoria in `rq2` e in `rq3`, intera, ≥ 2, senza default. Uno strato da
    una macchina non varia nel ricampionamento; la soglia minima potrà salire dopo la scelta del metodo degli IC
    (nota del 2026-11-17 su D-36). Dopo i filtri, un gruppo sotto soglia è un errore, non un adattamento; il conteggio
    per gruppo del passo 3 si riporta nelle uscite.
  - **`sign_flip`:** booleano (calcolarlo o no). Il test è esatto, per enumerazione dei 2ⁿ vettori di segni, con
    n ≤ 24 (altrimenti errore). È di solo supporto e non entra nella regola a quattro esiti. Le chiavi `n_perm` e
    `seed` sono tolte. L'aggettivo «stratificato» di D-36 non si applica: con segni indipendenti per macchina gli
    strati non entrano. Ipotesi nulla: i Δᵢ sono indipendenti e ciascuno è simmetrico attorno a 0.
  - **Fuori da F7, registrata come possibile aggiunta:** la sensibilità a due livelli (macchine, poi seed), che viene
    dalla proposta §5 e non dal testo di D-36. Il bootstrap gerarchico resta solo nelle simulazioni dei test, mai come
    valore del piano.
- **Nota di rimando (2026-11-17, passo 7, il testo sopra resta invariato; utente, su raccomandazione di Claude
  Project):**
  - **Righe descrittive di RQ2 (P5):** due chiavi obbligatorie in `rq2`:
    - `baselines`: lista, anche vuota, di modelli con la terna di scoring, che devono essere baseline descrittive
      dichiarate;
    - `random_baseline`: booleano.

    Le loro righe entrano nell'unità `rq2` del passo 3 come righe di riferimento, con la regola a senso unico. In RQ2
    sono solo descrittive: Δ medio rispetto al riferimento, mediana, conteggi dei segni, numero di macchine; nessun
    IC, nessun esito, sempre `exploratory`.
  - **File (P15):** si aggiunge `src/stats/rq2.py`, l'assemblaggio di RQ2. Flusso: unità `rq2` filtrata → tabella per
    macchina → Δ per macchina e membro (solo dall'esperimento primario, appaiato per macchina, sulle macchine comuni a
    tutti i membri) → bootstrap-t → regola a quattro esiti → righe di uscita.
  - **Uscite di RQ2:**
    - una riga per modello. Il riferimento ha `statistic = metric_mean`. Membri e baseline hanno
      `statistic = delta_mean`, con θ̂, mediana e `n_pos`, `n_neg`, `n_zero` (macchine con Δ > 0, < 0, = 0 esatto,
      come descrittive). Per i membri, anche SE, IC, livello, esito, δ, m, α, sign-flip (`support_only`), stato e
      motivo;
    - una **tabella lunga di Δ** per (membro o baseline, macchina): Δ, gruppo, valori di membro e riferimento, stato
      (incluso o escluso, con il motivo). Serve a RQ3 (passo 9) e al report (passo 11). La media sulle macchine
      incluse coincide con θ̂;
    - il blocco RQ2 di `stats.json`: metodo, B, seed, generatore, livello, m, δ, α, metrica, riferimento, famiglia,
      baseline, sign-flip, macchine per gruppo prima e dopo, numero di macchine, esclusioni, stato e motivo.
  - **RQ2 «non calcolabile»:** nessuna decisione. Le righe dei membri hanno stato `not_computable`, il motivo e IC ed
    esito vuoti; le descrittive restano.
  - **Wilcoxon non si calcola in RQ2.** Motivo: il sign-flip esatto ha già il ruolo di supporto, e due p-value di
    supporto sullo stesso confronto invitano a scegliere il più comodo. Wilcoxon resta solo nella parità con
    `aggregate_results.py` (P12). La frase «Wilcoxon solo come statistica descrittiva» non è nel piano operativo:
    compare nella proposta §5, dove è marcata come superata.
- **Nota di rimando (2026-11-17, passo 8, P11 e P15, il testo sopra resta invariato; utente, su raccomandazione di
  Claude Project):**
  - **File (P15):** si aggiunge `src/stats/rq1.py`, l'assemblaggio dei blocchi di RQ1a e RQ1b. `variance.py`
    contiene la decomposizione (D-25, nota del passo 8) e servirà anche a RQ4; `ranking.py` contiene τ-b e
    inversioni.
  - **Righe dei blocchi:** solo i modelli `compared` e `reference`; baseline random e descrittive escluse (P7).
    Bilanciamento verificato: ogni combinazione macchina × modello × livelli compare esattamente una volta, altrimenti
    errore. In RQ1b il fattore preproc ha come livelli gli esperimenti (colonna `experiment`).
  - **τ-b (P11):**
    - per macchina, tra il ranking dei modelli nella configurazione canonica c₀ (`tau_reference`) e in ogni altra
      configurazione, con `scipy.stats.kendalltau(variant="b")`, che tratta i pari merito;
    - distribuzione completa sulle macchine in tabella, con riassunti (media, mediana, minimo, quartili, numero di
      macchine definite e non definite);
    - τ-b sui ranking delle medie sulle macchine;
    - un ranking tutto in pari merito dà lo stato `undefined_all_tied`, mai scartato;
    - una configurazione è una combinazione dei livelli dei fattori del blocco con almeno 2 livelli, modello escluso.
  - **Inversioni di segno:** Δ(c) = valore(modello, c) − valore(`delta_reference`, c) per macchina.
    - Inversione se segno(Δ(c₀)) · segno(Δ(c)) < 0, cioè segni strettamente opposti.
    - Δ = 0 esatto in c₀ o in c = pareggio, contato a parte; altrimenti segno conservato.
    - Uscite: per (modello, configurazione) i conteggi e le quote di macchine; per modello la quota di
      configurazioni in cui si inverte il segno del Δ medio.
  - **Blocchi con un solo modello (RQ1b):** nessun termine del modello, R `single_model`, nessun τ e nessuna
    inversione.
  - **Uscite di RQ1a e RQ1b:** tutte `exploratory`. `stats.json` riporta per blocco fattori e livelli, η² principale,
    stati, R, macchine incluse e numero per gruppo, c₀.
- **Nota (2026-11-17, passo 9, scritta PRIMA di eseguire la simulazione; utente, su raccomandazione di Claude
  Project):**
  - **RQ3, statistica principale: ρ entro gruppo aggregata (ρ_W).**
    - Ranghi calcolati dentro ogni gruppo, con ranghi medi per i pari merito.
    - uₕᵢ = rangoₕᵢ/(nₕ + 1) − 1/2.
    - ρ_W = correlazione di Pearson degli u su tutte le macchine.
    - Chiave obbligatoria `rq3.statistic: within_group`.
    - ρ su tutte le macchine e ρ per gruppo sono solo descrittivi.
    - Motivo: con 6/7/9 macchine, se descrittore e Δ si spostano insieme tra gruppi, la ρ su tutte le macchine misura
      soprattutto 3 medie di gruppo (paradosso di Simpson); ρ_W misura l'associazione tra macchine a parità di
      gruppo, coerente con l'inferenza condizionata ai gruppi (D-36, D-76).
  - **Metodo dell'IC di ρ_W: criterio di scelta fissato prima di eseguire.**
    - Candidati:
      - (1) Fisher z con l'errore standard di Bonett–Wright e n_eff = n − G = 19 (22 macchine, 3 gruppi):
        z = atanh(ρ̂), SE = √((1 + ρ̂²/2)/(n_eff − 3)), IC = [tanh(z − z₁₋q·SE), tanh(z + z₁₋q·SE)];
      - (3) bootstrap percentile stratificato per gruppo (stesso generatore di D-76), con ρ_W ricalcolata su ogni
        replica.

      Il bootstrap non stratificato (2) è escluso per ρ_W, perché mescolerebbe i gruppi. Per documentazione si
      simula anche la ρ su tutte le macchine con (1) a n = 22, (2) e (3).
    - Script `scripts/sim_spearman_ci.py` (autonomo, seed fissi); risultati in `docs/sim_spearman_ci.json`.
    - Disegno: n = 22 (6/7/9), una coppia (X, Y) per macchina, livello 95%, R = 2.000, B = 2.000. Casi:
      - (a) indipendenza;
      - (b) copula gaussiana con ρ_S = 0,5 dentro i gruppi, r = 2·sin(π·0,5/6);
      - (c) come (b) con marginali t₃ (ρ_S non cambia, perché dipende solo dalla copula);
      - (d) X discreta con molti pari merito (8 classi, come `n_events`), Y latente correlata come in (b);
      - (e) solo confondimento di gruppo: spostamenti di gruppo in X e in Y, indipendenza dentro il gruppo.
    - **Valore vero = ρ_S di popolazione** (la quantità di interesse), non l'atteso della statistica:
      - per ρ_W, il ρ_S dentro il gruppo: 0 in (a) ed (e); 0,5 esatto in (b) e (c); stimato su 10⁶ punti in (d);
      - per la ρ su tutte le macchine, il ρ_S della miscela con le quote 6/7/9, stimato su 10⁶ punti.

      La Spearman su pochi punti è distorta verso 0: con ρ_S = 0,5 l'atteso entro gruppo è circa 0,44–0,45 con
      nₕ = 6–9. La distorsione pesa sulla copertura ed è parte di ciò che si misura.
    - Per ogni caso si riportano copertura, errore MC √(c(1 − c)/R), distorsione media della stima e larghezza
      media.
    - **Criterio** (applicato a ρ_W):
      - a) bande al 95%: caso (a) in [0,93; 0,97]; casi (b)–(e) ≥ 0,90;
      - b) tra i metodi che le rispettano vince la copertura più vicina a 0,95 nel caso peggiore, max |c − 0,95|;
      - c) a parità (differenza tra le distanze ≤ 2·√(SE₁² + SE₂²)) vince il più semplice: (1), poi (3).
    - **Se nessun metodo rispetta le bande:** RQ3 riporta la stima puntuale entro gruppo e i ρ per gruppo SENZA IC,
      con stato e avviso espliciti; nessun cambio di statistica dopo i risultati. In quel caso ci si ferma e si
      riporta prima di implementare.
    - **Esito (2026-11-17, dopo l'esecuzione; approvato dall'utente):** `docs/sim_spearman_ci.json`, R = B = 2.000.
      Valori veri: 0; 0,5; 0,5; 0,493; 0 (entro gruppo). Errore MC circa 0,004.

      | caso | (1) Fisher, n_eff = 19: copertura / distorsione / larghezza | (3) bootstrap stratificato: copertura / distorsione / larghezza |
      |---|---|---|
      | a) indipendenza | 0,962 / −0,004 / 0,878 | 0,966 / −0,004 / 0,922 |
      | b) gaussiana, ρ_S = 0,5 | 0,958 / −0,054 / 0,761 | 0,967 / −0,054 / 0,816 |
      | c) t₃, ρ_S = 0,5 | 0,958 / −0,054 / 0,761 | 0,967 / −0,054 / 0,816 |
      | d) pari merito | 0,962 / −0,044 / 0,759 | 0,965 / −0,044 / 0,799 |
      | e) confondimento di gruppo | 0,962 / −0,004 / 0,878 | 0,966 / −0,004 / 0,922 |

      - (c) coincide con (b) e, per ρ_W, (e) con (a): è atteso, perché i ranghi non cambiano con le trasformazioni
        monotone e con gli spostamenti di gruppo.
      - a) Entrambi i metodi rispettano le bande.
      - b) Caso peggiore, max |c − 0,95|: Fisher 0,012, bootstrap stratificato 0,0165.
      - c) Differenza 0,0045 ≤ 2·√(0,0043² + 0,0040²) = 0,0117: parità, vince il più semplice.
      - **Scelta: (1) Fisher z con l'errore standard di Bonett–Wright e n_eff = n − G.**
      - Per documentazione: la ρ su tutte le macchine, nel caso (e), stima 0,62 con IC lontani da 0, mentre
        l'associazione dentro i gruppi è nulla (paradosso di Simpson).
    - **Potenza (avvertenza da riportare nel report del passo 11, accanto ai risultati di RQ3):** con n = 22 e G = 3
      le larghezze medie degli IC sono circa 0,76–0,88. Quindi |ρ_W| < ~0,4 avrà tipicamente un IC che comprende 0, e
      un IC che comprende 0 non è prova di assenza di associazione.
    - **Piano:** `rq3.ci` contiene solo `method: fisher_bonett_wright`; `n_boot` e `seed` sono rifiutate (nessun
      parametro senza effetto in un piano congelabile). `rq3.statistic: within_group` è obbligatoria.
  - **Molteplicità di RQ3:** numero di correlazioni = membri della famiglia × descrittori; nessuna correzione, perché
    esplorativo; il numero è scritto nelle uscite. Si riporta anche la correlazione tra i due descrittori.
  - **File (P15):** si aggiungono `src/stats/descriptors.py` (descrittori per macchina e correlazioni),
    `src/stats/rq3.py` e `src/stats/rq4.py` (assemblaggi).
  - **P10, indicatore di scostamento cal → test:** per macchina, modello e soglia statica di livello L, FPR del test
    − (1 − L/100). L'FPR è quello della riga `thr` (scenario `primary`, `ts_agg = last`, `pa` falso), cioè i falsi
    positivi diviso i punti normali della regione valutata (`metrics.thr_metrics`). È descrittivo.
    - `plan.py`: `offset_indicator: true` richiede `fpr` tra le metriche di `rq4`.
    - **Limite:** la quota nominale 1 − L/100 si riferisce a tutti i punti di cal (senza label, possono contenere
      anomalie); l'FPR del test si riferisce ai soli punti normali. L'indicatore è quindi uno scostamento
      approssimato, da leggere insieme al limite E5 (`piano_tesi_v1.md` §5).
- **Nota di rimando (2026-11-18, passo 10, P6, P8, P13, il testo sopra resta invariato; utente, su raccomandazione di
  Claude Project):**
  - **Flusso di `stats` (`src/stats/run.py`):**
    1. validazione del piano;
    2. rifiuto di `kind: official` fino a F10;
    3. `load_inputs`;
    4. `stats_id` provvisorio, con rifiuto se già pubblicato (prima del lock, senza scritture);
    5. lock, ingressi riletti e `stats_id` definitivo sotto lock;
    6. `begin_run` con stage `stats` e `config_sha256` vuoto (gli esperimenti sono più di uno; gli hash delle config
       sono in `stats.json`);
    7. filtri, tabella per macchina, descrittive e RQ secondo il piano (`null` = saltata);
    8. tabelle, `run_info.json` e per ultimo `stats.json` (marcatore);
    9. rinomina atomica in `store/stats/<stats_id>/` con i tentativi di B11;
    10. `finish_run`.

    Su eccezione: registro `failed`, oppure `interrupted` per un'interruzione; nessuna pubblicazione; cartella
    temporanea lasciata per `clean-tmp`.
  - **Codici di uscita di `stats`:**
    - 0: statistiche pubblicate, anche con RQ «non calcolabili» (la CLI stampa un messaggio per ciascuna, con il
      motivo);
    - 2: piano, ingressi o dati non validi per il piano (tutti gli errori dei moduli di `src/stats`), statistiche già
      presenti, `official` prima di F10;
    - 3: store occupato; 4: pubblicazione non riuscita; 130: interruzione;
    - 1 non si usa.
  - **Determinismo:** stessi ingressi, piano e ambiente danno lo stesso `stats_id` e file pubblicati identici byte per
    byte, salvo `run_info.json` (`run_id`, data, git, cartella dello store), che resta escluso dal confronto e da
    `stats_id`.
    - Fine riga `\n` esplicito in tutti i CSV e JSON; gzip con `mtime = 0`; JSON canonico; float scritti nella forma
      più corta che si rilegge esatta.
    - Nessun timestamp né percorso assoluto nei file confrontati.
  - **Thread:** lo stadio di statistica non importa torch, quindi il numero di thread di torch non si registra; le
    analisi in ingresso non lo registrano (`analysis_environment`).
    - Per non importare torch, `FitTask`, `set_seed_value`, `plan_tasks` e `validated_config` sono stati spostati in
      `src/pipeline/fit_plan.py` (senza torch); `fit.py` li reimporta con gli stessi nomi; nessun cambiamento
      numerico (snapshot F1 e golden invariati).
  - **Sensibilità P8:** nello stesso run, in `sensitivity/`. Tutte le sue uscite hanno `analysis_variant =
    sensitivity` e sono sempre `exploratory`, anche con `kind: official`: mai due esiti confermativi per lo stesso
    confronto. Le uscite principali hanno `analysis_variant = main`.
  - **Registro:** le righe `stats` hanno `config_sha256` vuoto. Il riscontro di `inputs.py` e `status` filtrano per
    stage e non leggono mai una riga `stats` come una riga di analisi.
  - **`plan.py`:** `stats.name` è diverso dai nomi degli esperimenti di `inputs`, perché i run di statistica stanno
    in `runs/<kind>/<stats.name>/`.
  - **Test d'integrazione** fit → analyze → stats con 15 macchine di valutazione (5 per gruppo) e due esperimenti:
    circa 23 s (fit 3, analyze 15, stats 5). Sta nella suite predefinita, marcato `slow`.
  - **Niente BLAS:** nei moduli di `src/stats` non ci sono `@`, `dot`, `matmul`, `einsum`, `inner`, `tensordot`,
    `vdot` o `np.linalg`. Il loro risultato può cambiare nell'ultimo bit con il numero di thread o con la libreria
    BLAS della macchina; somme e medie di numpy non dipendono dai thread.
    - L'unica occorrenza trovata era il prodotto `signs @ d` del sign-flip esatto (`bootstrap.py`), ora
      `(signs * d).sum(axis=1)`.
    - Un test sul sorgente (`tests/test_f7_no_blas.py`, via `ast`) lo verifica; `lstsq` resta solo nei test.
  - **Test del run** (`tests/test_f7_run.py`): i test che leggono soltanto il risultato condividono un run di modulo;
    a fine modulo l'hash della sua cartella deve essere invariato. I test che scrivono lavorano su copie proprie.
- **Nota di rimando (2026-11-18, passo 11, P13, P15, il testo sopra resta invariato; utente, su raccomandazione di
  Claude Project):** comando `report` (`src/reporting/{source,tables,notices,figures,report}.py`).
  - **Il report non calcola.** Legge solo `store/stats/<stats_id>/` e prima di formattare verifica, nell'ordine:
    - posizione sotto `store/stats/`;
    - `stats.json` presente e `stats_format` supportato;
    - `stats_id` uguale al nome della cartella;
    - file elencati presenti e con lo sha256 dichiarato, calcolato sugli stessi byte che si leggono;
    - nessun file non elencato, salvo `stats.json` e `run_info.json`.

    Garanzia doppia, decisa dall'utente:
    - (a) test sul sorgente: nessun import di `src.stats`, scipy, statsmodels, sklearn o torch; nessuna chiamata di
      aggregazione (`mean`, `sum`, `quantile`, `groupby`, `rank`, …);
    - (b) test sulle uscite: ogni numero di `.md` e `.tex` è la formattazione di un valore pubblicato; i CSV del
      report sono sottoinsiemi di righe e colonne delle tabelle pubblicate; i dati delle figure sono valori
      pubblicati.
  - **Contenuto:** solo ciò che il piano richiede; una RQ saltata ha la riga «Non richiesta dal piano statistico.».
    Le sezioni sono:
    - intestazione, con ingressi e stato delle RQ;
    - descrittive per modello (media, sd, mediana, rango medio, n macchine; per `detection_delay` anche n definite
      ed `event_recall`);
    - RQ2, con le righe descrittive (riferimento, random, baseline descrittive) nella stessa tabella, separate e
      senza esito;
    - RQ1a e RQ1b (η² con le interazioni modello × fattore in grassetto, R, distribuzione di τ, inversioni);
    - RQ3, senza retta di regressione;
    - RQ4, con l'indicatore di scostamento;
    - sensibilità P8, in una sezione propria etichettata come esplorativa e senza figure;
    - «Dati esclusi o non usati»: registro, conteggi, NaN in celle non usate, repliche random rimosse, macchine per
      unità, macchine escluse da RQ2.

    La figura di RQ2 mostra Δ per macchina e l'IC dell'effetto aggregato (decisione dell'utente).
  - **Numeri:**
    - cifre fisse in `tables.DIGITS` e in `report.json`: 3 per valori, ρ, R, τ e quote; 1 per η² in percentuale;
    - p del sign-flip in notazione scientifica con 3 cifre significative sotto 0,001.
    - Arrotondamento vicino a δ: se un estremo dell'IC, arrotondato, coincide con ±δ senza esserlo, si mostra con
      le cifre necessarie a distinguerlo, fino alla forma esatta. È formattazione, non calcolo. Test: nessuna riga
      mostrata contraddice il proprio esito.
    - δ e i parametri del piano si mostrano nella forma esatta più corta.
  - **Separatore decimale:** `report … --decimal comma|point`, valore iniziale `comma`.
    - Vale per `.md`, `.tex` e per le etichette degli assi delle figure.
    - I CSV restano sempre col punto e a precisione piena.
    - In LaTeX la virgola è `{,}` e i numeri sono in `$…$`.
    - Nella prosa e nelle didascalie si localizzano solo i numeri decimali isolati: restano invariati quelli dentro
      etichette JSON, versioni e nomi.

    L'opzione si registra in `report.json`. Nessun generatore `.docx`.
  - **Avvertenze:** testi fissi in italiano, in `notices.py`, ciascuna con la sua condizione: risultati esplorativi,
    D-45, C4, D-34, potenza di RQ3, P8, RQ non calcolabili con il motivo di `stats.json`. Frasi vietate in tutte le
    uscite: «vince», «non vince», «il deep learning», «significativ», «dimostra», «prova che».
  - **Figure:** matplotlib con canvas Agg, senza pyplot.
    - Stile fisso: default di matplotlib più `figures.RC`, in un `rc_context`; si ignora `matplotlibrc`
      dell'utente.
    - Font DejaVu Sans incluso in matplotlib.
    - Metadati di data e versione rimossi da PDF e PNG; byte identici tra processi con `PYTHONHASHSEED` diversi.
    - Gruppi con la palette categoriale validata e marcatori distinti.
  - **Uscita:**
    - cartella nuova fuori dallo store, rifiutata se esiste, se manca la cartella madre o se sta nello store;
    - scrittura in una cartella temporanea accanto a `--out` e rinomina; nessuna riga di registro;
    - codici di uscita: 0; 2 (ingresso o uscita non ammessi); 4 (scrittura); 130 (interruzione, cartella
      temporanea rimossa).
  - **`status` di RQ4.**
    - Fatto verificato: il blocco `rq4` di `stats.json` non aveva la chiave `status`, a differenza delle altre RQ.
    - `run.py` ora la scrive (`computed`), anche nella sensibilità.
    - `STATS_CODE_VERSION` passa da 1 a 2: lo stesso `stats_id` deve sempre corrispondere agli stessi byte (approvato dall'utente).
    - Il report rifiuta, con codice 2, un blocco RQ senza `status` o con uno stato sconosciuto; nessuno stato si
      presume. Ammessi: `computed`, `computable`, `not_computable`, `skipped`.
  - **Percorsi lunghi su Windows.**
    - Fatto verificato: con la radice dei test di pytest, il file `.part` di una tabella diagnostica della
      sensibilità arrivava a 259 caratteri, uno sotto il limite di 260.
    - `store.py`:
      - `MAX_STATS_PATH_CHARS = 250`;
      - `check_stats_paths` controlla, per ogni file, il percorso nella cartella temporanea, quello del `.part` e
        quello pubblicato;
      - `run.py` chiama il controllo prima di creare la cartella temporanea, quindi nessuna scrittura parziale;
      - `write_stats_file` ripete il controllo prima di ogni scrittura.
    - Errore `StatsPathTooLong`, con il percorso e il suggerimento di abbreviare i nomi o spostare lo store;
      codice di uscita 2.
    - La cartella temporanea nasce ora dopo il calcolo: un'interruzione durante il calcolo non lascia cartelle
      orfane.
    - `plan.py`: `stats.name`, i nomi dei blocchi di RQ1a e RQ1b e i nomi degli esperimenti dichiarati in `inputs`
      (che entrano nei nomi delle tabelle diagnostiche) solo `[a-z0-9_]`, al più 32 caratteri, oltre alle regole sui
      nomi dei percorsi (nomi riservati di Windows). Un nome troppo lungo emerge alla validazione del piano; lo
      schema degli esperimenti non cambia.
  - **ρ per gruppo (RQ3):** il testo JSON pubblicato si mostra come «gruppo g: valore» a 3 cifre, con il separatore
    scelto; è solo formattazione. Il test di tracciabilità verifica che ogni valore mostrato sia l'arrotondamento di
    quello pubblicato. Il CSV del report conserva il testo pubblicato intero.
  - **Sha256 nella prosa:** abbreviati a 12 caratteri. I valori completi sono in `report.json`
    (`stats_plan_sha256`, `stats_json_sha256`, sha di ogni file).
  - **Ambiente sintetico** (`tests/fixtures/synth_many.py`).
    - Numero (da 2 a 6), lunghezza, posizione e tipo degli eventi variano per macchina, anche dentro i gruppi, con
      un generatore separato da quello dei segnali; quindi RQ3 è calcolata.
    - Lo snapshot F1 non cambia: usa `synth_smd`.
  - **Separatore nelle figure:** approvato dall'utente.
  - **Impaginazione LaTeX:** rimandata a quando si deciderà il formato della tesi. Le tabelle larghe (RQ2: 16
    colonne) e le etichette lunghe delle configurazioni non sono ancora adattate alla pagina.
- **Nota di rimando (2026-11-18, passo 11-bis, P8, P10, il testo sopra resta invariato; utente, dalla lettura del
  report d'esempio):** correzioni prima della golden del passo 12, perché la golden non congeli testi sbagliati.
  - **Avvertenza P8** (`notices.py`): il testo diceva «fit con residui di calibrazione non finiti esclusi», ed era
    sbagliato. Ora dice che la sensibilità ripete tutte le statistiche con ogni azione `exclude_fit` trasformata in
    `exclude_machine_pair` (exclude_fit → exclude_machine_pair, D-73 P8).
  - **Avvertenza di potenza di RQ3** (`rq3.py`): non attribuisce più i numeri della simulazione al campione corrente.
    - Le larghezze 0,76–0,88 vengono dalla simulazione del disegno a 22 macchine (6/7/9, `docs/sim_spearman_ci.json`).
    - Con meno macchine gli intervalli sono più larghi.
    - Nessun «n = …» del campione corrente.
    - Un test verifica che i numeri citati coincidano con quelli del file della simulazione.
  - **Titolo RQ4 nel report:** «RQ4: soglie e scostamento cal → test». L'indicatore non misura la deriva (P10).
  - **Sensibilità senza casi.**
    - `stats.json` pubblica `sensitivity.n_exclude_fit_cases`: le righe del registro principale con azione
      `exclude_fit`.
    - Il report legge questo conteggio e non lo ricalcola. Con 0 casi, la sezione di sensibilità e la sua parte in
      «Dati esclusi o non usati» contengono solo l'avvertenza e la frase «Nessun caso exclude_fit: la sensibilità
      coincide con l'analisi principale per costruzione», senza tabelle.
    - Il report rifiuta, con codice 2, una sensibilità senza conteggio valido.
  - **`STATS_CODE_VERSION` = 3**, perché cambiano l'avvertenza di RQ3 e `stats.json`.
  - **Report d'esempio:** si generano in `reports_local/`, esclusa da git.
- **Nota di rimando (2026-11-18, passo 12 A+B, il testo sopra resta invariato; piano approvato dall'utente):**
  golden e prova completa.
  - **Golden** (`tests/test_f7_golden.py`, `tests/golden/{stats,report}_synth.json`), sullo stesso ambiente
    sintetico del report.
    - Statistiche, legate a `STATS_CODE_VERSION`: sha256 di ogni tabella pubblicata e di `stats.json` normalizzato.
    - Report, legato a `REPORT_CODE_VERSION` e `STATS_CODE_VERSION`: `.md`, `.tex` e `.csv` con separatore
      `comma`, più `report.json` normalizzato. Figure escluse (decisione dell'utente). L'ambiente registrato include
      matplotlib.
    - Rigenerazione solo con `python tests/test_f7_golden.py --write-golden`, che rifiuta uscite cambiate con la
      stessa versione.
  - **Fatti verificati per la normalizzazione:** costruito due volte in cartelle diverse, l'ambiente dà tabelle
    identiche byte per byte. Di `stats.json` cambiano solo i campi che dipendono:
    - dai percorsi assoluti (`config_sha256` li include): `stats_id`, `stats_plan_sha256`, `components`,
      `config_sha256` e `analysis_json_sha256` degli ingressi;
    - dalle misure del run: lo sha256 della tabella `fits` degli ingressi, che contiene `fit_sec`.

    L'elenco è in `RUN_DEPENDENT_KEYS` ed è registrato nella golden.
    - La normalizzazione dei percorsi assoluti vale solo per l'ambiente dei test, che scrive percorsi assoluti nelle
      config.
    - Fatto verificato: con le config reali i percorsi sono relativi (`base.yaml`: `data.data_dir: dataset`;
      specifiche dei dataset: percorsi relativi alla loro cartella). Lo stesso esperimento caricato da due copie di
      `configs/` in cartelle diverse dà lo stesso `config_sha256`.
    - Limite: `paths.store_root` è aggiunto da ogni esperimento e lo schema accetta anche un percorso assoluto (non
      esistono ancora esperimenti reali in `configs/experiments/`). Proposta, non applicata: rifiutare i percorsi
      assoluti negli esperimenti `official` (F10).
  - **`stats_id` senza tempi misurati (`STATS_CODE_VERSION` = 4).**
    - L'identità si calcola su proiezioni: tabella `fits` senza `fit_sec`, l'unica colonna di durata dello schema
      (un test lo verifica su tutte le tabelle); `analysis.json` senza `run_id`, `created_utc` e `git`. Su
      `analysis.json` si applica lo stesso principio: un'analisi rifatta con risultati identici dà lo stesso
      `stats_id`.
    - L'elenco è dichiarato in `stats.json` (`identity_exclusions`). Le tabelle e gli sha256 completi restano in
      `stats.json` come provenance.
    - Test: `fit_sec` diversi con risultati identici → stesso `stats_id`; un risultato diverso in `fits` → `stats_id`
      diverso; identità del run diversa in `analysis.json` → stesso `stats_id`, un suo risultato diverso → diverso.
    - Prove di difetto: 4 su 4.
    - La golden si genera con un solo thread, come i test: con più thread PCA (BLAS) cambia le ultime cifre delle
      tabelle diagnostiche. Il primo tentativo senza `single_thread()` era sbagliato; è stato rigenerato.
  - **Prova completa da riga di comando** (`tests/test_f7_cli_e2e.py`, `slow`): esperimenti come file YAML uniti a
    `base.yaml` e ai modelli; `stats` → `status` → `report` → secondo `report` rifiutato (codici 0, 0, 0, 2).
  - **Fatto verificato, limite fino a F8:** un esperimento `split3` scritto su file non si carica finché
    `configs/base.yaml` dichiara `split.val_ratio` (`legacy2`). L'unione profonda non toglie chiavi e lo schema di
    `split3` rifiuta `val_ratio`.
    - Per questo la prova completa usa `legacy2` (solo `primary`; `test_context` richiede `split3`, D-60).
    - Un test documenta il limite; va rivisto quando F8 cambia `base.yaml`.
  - **Difetto trovato e corretto** (`src/cli.py`): su una pipe cp1252 di Windows, un avviso con «ρ» faceva uscire
    `stats` con codice 1 dopo la pubblicazione. Ora `main` imposta `errors="backslashreplace"` su stdout e stderr,
    con la stessa codifica. Test e prova di difetto.
- **Nota di chiusura (2026-11-18, F7 completata, il testo sopra resta invariato):** riepilogo in
  `piano_operativo.md`, sezione F7, «Stato: completata».
  - **File effettivi:**
    - `src/stats/{plan,inputs,filters,aggregate,decisions,bootstrap,rq2,variance,ranking,rq1,descriptors,rq3,rq4,
      run}.py`;
    - `src/pipeline/fit_plan.py`, più aggiunte a `src/pipeline/store.py` e `src/cli.py`;
    - `src/reporting/{source,tables,notices,figures,report}.py`.

    Rispetto a P15 ci sono in più `rq1`–`rq4`, `run`, `fit_plan`, `source` e `notices`.
  - **Versioni finali:** `STATS_CODE_VERSION` = 4, `STATS_FORMAT` = 1, `REPORT_CODE_VERSION` = 1. Golden in
    `tests/golden/{stats,report}_synth.json`.
  - **Decisioni approvate durante F7:** D-25, D-36, D-45, D-69 (dettagli aperti, proposta registrata in nota), D-71,
    D-73, D-76.
  - **Decisioni toccate e rimaste aperte:** D-30, D-37, D-41, D-42, D-43, D-44, D-74, D-75.

    Stato di ogni voce verificato sul registro alla chiusura. Nessuna voce APERTA è risolta da un default implicito:
    δ, i livelli dei fattori, i descrittori e le regole di esclusione sono parametri dichiarati nel piano statistico,
    senza valori predefiniti.
  - **Scostamento:** nessun piano in `configs/stats/` in F7; un piano reale richiede δ (D-30) e i valori di D-44.
- **Nota di rimando (2026-11-19, M1, il testo sopra resta invariato): regola delle golden di F7 resa più permissiva in
  un solo caso.** Modifica di `tests/test_f7_golden.py` approvata dall'utente il 2026-11-19 (opzione A, su
  raccomandazione di Claude Project).
  - **Perché:** il lock rigenerato per M1 (nota su D-55) cambia `lock_sha256`, quindi gli `analysis_id`, che compaiono
    nelle tabelle diagnostiche delle statistiche e nei file del report che elencano gli ingressi. Le golden di F7
    fallivano senza alcun cambiamento numerico, e la regola precedente («si rigenera solo dopo aver aumentato
    `STATS_CODE_VERSION` o `REPORT_CODE_VERSION`») avrebbe registrato come cambio di codice un cambio d'ambiente.
  - **Cosa cambia:**
    - l'ambiente registrato nelle golden di F7 contiene anche `lock_sha256`, come la golden di F6;
    - le golden registrano anche le uscite in forma d'identità (`identity_outputs`): sha256 calcolati dopo aver
      sostituito con segnaposto gli identificativi che derivano dall'ambiente (`lock_sha256`, `analysis_id` degli
      ingressi; gli sha256 delle tabelle d'ingresso e della mappa `files` sono sostituiti dallo sha256 delle stesse
      tabelle in forma d'identità, così il contenuto resta controllato);
    - riscrittura a parità di versione: ammessa se le uscite sono identiche; rifiutata se l'ambiente è lo stesso e le
      uscite sono diverse (come prima); con l'ambiente cambiato, ammessa solo se le uscite differiscono soltanto nei
      campi d'identità (`identity_outputs` uguali), altrimenti rifiutata, anche con una golden senza `identity_outputs`.
  - **Verifica prima della rigenerazione (punto 1 dell'utente):** uscite ricostruite con il lock vecchio (hash
    identici alle golden registrate) e con il lock nuovo; dopo la sostituzione di `lock_sha256`, dei tre
    `analysis_id`, di `stats_id` e degli sha256 che ne dipendono, coincidono byte per byte nella forma registrata dalle
    golden: F6 `free`, `thr`, `analysis.json` (e `fits` identica); F7 le 8 tabelle `diagnostic_integ_*` e `stats.json`
    normalizzato (le altre tabelle identiche); report `report.md`, `report.tex`, `tables/inputs.md`,
    `tables/inputs.tex` (gli altri file identici). Esito completo e script (`gbuild.py` costruisce le uscite con il
    lock attivo, `gverify.py` sostituisce e confronta, `iddiff.py` confronta la forma d'identità con la golden,
    `mutate_golden2.py` sono le prove di difetto) in `reports_local/m1/golden_check/` (`gverify_out.txt`), esclusa da
    git.
  - **Rigenerazione:** F7 in due passi con la regola nuova (lock vecchio: uscite identiche, si aggiungono lock e
    `identity_outputs`; lock nuovo: ambiente cambiato, forma d'identità uguale → ammessa); F6 con la sua regola.
  - **Segnalato, non modificato:** la golden di F6 (`tests/test_f6_analyze_golden.py`) ammette la rigenerazione con
    l'ambiente cambiato senza il controllo della forma d'identità; per M1 il controllo è stato fatto a parte (sopra).
    Allinearla richiede l'approvazione dell'utente.
  - **Documenti:** `CLAUDE.md` §7; piano, nota di M1.
- **Nota di rimando (2026-11-20, prerequisiti di F8, il testo sopra resta invariato; utente, su raccomandazione di
  Claude Project):**
  - **P5, `descriptive_baselines` (nota del passo 1-bis):** l'affermazione «un Conv-AE addestrato e uno non addestrato
    hanno lo stesso `model` e `config_id` diversi» non vale più: con D-69 (nota del 2026-11-20) hanno **nomi di modello
    distinti** (`conv_autoencoder` e `conv_autoencoder_untrained`; lo stesso per l'LSTM). L'identità resta la coppia
    `(model, config_id)`.
  - **Regola in `src/stats/plan.py` (passo 3 di F8):** un modello `*_untrained` può comparire **solo** in
    `descriptive_baselines`: mai nella famiglia o come riferimento di RQ2, mai come `delta_reference` o tra i modelli
    di un blocco di RQ1a, RQ1b o RQ4. È il riscontro con lo schema dei modelli rimandato al passo 1-bis.
  - **Invariato (decisione del 2026-11-16):** `descriptive_baselines` resta aperta anche ad altri modelli, compresi i
    native.
  - **Golden di F6 (punto «Segnalato, non modificato» sopra):** allineamento alla regola di F7 (`identity_outputs`)
    approvato; si fa al passo 1 di F8.
  - **Scostamento dal criterio di F7 (passo 1 di F8, approvato dall'utente alla revisione):** la forma d'identità
    della golden di F6 (`tests/test_f6_analyze_golden.py`) sostituisce **campi precisi** con un segnaposto (la colonna
    `analysis_id` di `free` e `thr`; `analysis_id` e `components.environment.lock_sha256` di `analysis.json`; `fits`
    non ne contiene), invece della sostituzione testuale di F7. Motivo: nasconde esattamente ciò che deriva
    dall'ambiente; un campo nuovo che dipende dall'ambiente e non è elencato fa rifiutare la riscrittura, quindi il
    criterio sbaglia nella direzione più severa. La regola di riscrittura è la stessa di F7: un test con 7 casi
    verifica lo stesso esito di `write_one` di `tests/test_f7_golden.py`. Golden rigenerata una volta: `outputs`
    invariati, aggiunto solo `identity_outputs`.
  - **Documenti:** piano F8 (passi 1, 3, 5).
- **Nota (2026-11-20, F8 passo 4, golden di F7; utente, alla revisione del passo, su raccomandazione di Claude
  Project, opzione (b)):**
  - **Problema:** le statistiche e il report registrano l'`analysis_code_version` degli ingressi (`stats.json`,
    `inputs[]`; tabella degli ingressi del report). Con `ANALYSIS_CODE_VERSION` 1 → 2 (D-74) il valore cambia anche se
    le statistiche non cambiano, e la regola di F7 rifiutava la riscrittura a parità di `STATS_CODE_VERSION` /
    `REPORT_CODE_VERSION` e di ambiente. Il caso si ripresenterà a ogni aumento di `ANALYSIS_CODE_VERSION` (per esempio
    VUS-PR in F10). Aumentare le versioni di statistiche e report è scartato: il loro codice non è cambiato.
  - **Criterio (`tests/test_f7_golden.py`):**
    - la golden registra la versione dell'analisi degli ingressi (`input_analysis_code_versions`,
      {esperimento: versione});
    - la forma d'identità la nasconde come `analysis_id` (segnaposto `<analysis_code_version>`): per struttura in
      `stats.json`; nel report solo nella cella che segue il segnaposto dell'`analysis_id` dell'ingresso, mai per valore
      altrove;
    - **caso ammesso nuovo** a parità di versione: versione degli ingressi diversa da quella registrata **e** uscite
      identiche in forma d'identità; qualunque altra differenza, comprese quelle delle tabelle statistiche, è un
      rifiuto;
    - versione degli ingressi uguale o non registrata: regole di F7 invariate.
  - **Test:** versione cambiata e solo la versione diversa → ammessa; versione cambiata e un valore di tabella diverso →
    rifiutata; versione uguale e uscite diverse → rifiutata; golden senza versione registrata → regole precedenti;
    sulle uscite vere, cambiare solo la versione lascia uguale la forma d'identità e cambiare un altro campo no. Prove
    di difetto (5 mutazioni in memoria, tutte rilevate): caso nuovo senza controllo della forma d'identità; versione
    considerata sempre cambiata; versione non nascosta nel report; versione non nascosta in `stats.json`;
    sostituzione per valore non ancorata.
  - **Rigenerazione, in due passi** (come la nota di M1): (1) con `ANALYSIS_CODE_VERSION` = 1 impostata solo in
    memoria, uscite identiche a quelle registrate, si aggiungono la versione degli ingressi e la nuova forma d'identità
    (cambiata solo in `stats.json` normalizzato e in `report.md`, `report.tex`, `tables/inputs.md`, `tables/inputs.tex`);
    (2) con la versione 2 il caso nuovo: forma d'identità identica, cambia solo la versione degli ingressi (1 → 2).
  - **Documenti:** piano F8 (passo 4).

### D-53 · File tracciati da git (`.pyc`, dataset) — APPROVATA · 2026-11-10
- **Decisione:** i `.pyc` non sono tracciati: i 35 file già tracciati sono tolti dal tracciamento con
  `git rm -r --cached` sulle cartelle `__pycache__` (restano sul disco) e `.gitignore` esclude `__pycache__/` e `*.pyc`.
  `dataset/` resta tracciato (sola lettura, riproducibilità).
- **Approvazione:** utente, 2026-11-10, su raccomandazione di Claude Project.
- **Collegamenti:** D-55, D-56.

### D-77 · Caricamento delle config: blocco `split` atomico e valori storici espliciti — APPROVATA · 2026-11-20
- **Problema (fatti verificati nel codice, 2026-11-20):**
  - `merge_layers` (`src/utils/config.py:74-89`) fa un'unione profonda e non toglie mai chiavi;
  - `_validate_split` (`src/utils/schema.py:279-293`) accetta solo le chiavi del modo attivo (`legacy2`:
    `mode`, `val_ratio`; `split3`: `mode`, `fractions`);
  - `configs/base.yaml` dichiara `legacy2` con `val_ratio`, quindi un esperimento `split3` scritto su file viene
    rifiutato (documentato da `tests/test_f7_cli_e2e.py:114`). Il problema è simmetrico: quando F8 metterà `split3`
    in `base.yaml`, un esperimento `legacy2` non si caricherebbe più;
  - `config_sha256` è l'hash della config risolta (`config.py:234-237`), non dei file; il `fit_key` usa solo i campi
    del modo attivo (`schema.py:228-231`).
- **Decisione:**
  - **A1:** `split` è l'**unico blocco atomico** dell'unione dei livelli: se un livello lo dichiara, sostituisce per
    intero quello dei livelli precedenti; tutte le altre mappe restano in unione profonda. La regola si documenta nel
    codice (`merge_layers`, `load_experiment`). Lo schema non cambia: una chiave del modo non attivo dentro `split`
    resta un errore (nessuna chiave ignorata);
  - **strategia (i):** `configs/experiments/parity_repro.yaml` dichiara esplicitamente tutti i valori storici, così
    resta indipendente dai valori canonici che F8 metterà in `base.yaml` (piano F8, «File»). Nei passi 0–7 di F8
    `base.yaml` non cambia.
  - **«Tutti i valori storici» (revisione del passo 0, utente, 2026-11-20):** `parity_repro.yaml` non eredita
    **alcun** valore che entri in `config_sha256` o nel `fit_key`: `split`, `preprocessing`, `windows`, `training`,
    `params` dei modelli in `run_models`, `device`, blocco `data` e percorsi. Motivo: F9 scriverà le griglie in
    `configs/models/*.yaml` e F8 i valori canonici in `base.yaml`.
- **Vincolo di verifica (passo 2):**
  - `config_sha256` e `fit_key` di `parity_repro` dopo la modifica uguali a quelli del run M1 nel registro dello store
    (letto in sola lettura);
  - **indipendenza:** con copie temporanee di `base.yaml` e dei file di `configs/models/` modificate (valori diversi
    qualunque), `config_sha256` e `fit_key` di `parity_repro` restano quelli di M1. I file veri non si toccano;
  - un esperimento `split3` e uno `legacy2` si caricano da file; una chiave del modo non attivo è un errore;
  - `tests/test_f7_cli_e2e.py:114` si capovolge in «ora si carica» senza diventare più permissivo.
- **Limite dell'indipendenza e sua copertura (passo 2, approvato alla revisione):** l'indipendenza di `parity_repro`
  vale per i **valori**. Una **chiave** nuova in un blocco che `parity_repro` dichiara arriverebbe comunque con
  l'unione profonda, ma:
  - se è aggiunta solo in `base.yaml` o in `configs/models/`, lo schema la rifiuta come chiave sconosciuta (errore);
  - se è aggiunta anche allo schema, entra in `parity_repro`, cambia `config_sha256` (e, se tocca il fit, i
    `fit_key`) e fa fallire il test di uguaglianza con M1 (`tests/test_f8_config_loading.py`).

  In nessun caso il cambiamento è silenzioso.
- **Alternative scartate:** A2 (`split` tolto da `base.yaml`), A3 (lo schema ignora le chiavi dell'altro modo: chiave
  ignorata in silenzio), A4 (`null` come cancellazione: ambiguo, `clip: null` è un valore), A5 (due livelli base:
  nuova chiave o nuovo flag di CLI).
- **Impatto sui risultati:** nessuno; le config esistenti risolvono alla stessa config, quindi `config_sha256` e
  `fit_key` restano invariati (da dimostrare con il test sopra).
- **Approvazione:** utente, 2026-11-20, su raccomandazione di Claude Project.
- **Collegamenti:** D-05, D-14, D-50, D-63, D-70, D-73 (P6).
- **Documenti:** piano F8 (passo 2).

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
- **Nota di rimando (2026-11-24, da D-40, la voce resta APERTA):** la PCA puntuale ristandardizza internamente ogni
  sensore con la deviazione standard su fit (`src/models/traditional/pca_pointwise.py:31-35`), quindi il pavimento
  dello scaler (τ = 0,02, D-40) su di lei agisce solo sui sensori a range nullo, mentre agisce su tutti i sensori quasi
  costanti per gli AE, il modello di previsione e la PCA a finestre. Con la PCA puntuale come riferimento di RQ2 il
  pavimento sarebbe asimmetrico tra DL e riferimento; con la PCA a finestre no. Da considerare nella decisione.

### D-41 · Budget di tuning — APERTA
- **Da decidere:** K per famiglia (proposta 8), griglie, criterio di selezione (proposta: AP normalizzata media sulle 6
  macchine di sviluppo, una configurazione globale per famiglia), seed nel tuning (proposta 3), se scoring e normalizer
  robusto rientrano nel budget. **Serve entro:** F9.
- **Nota (2026-11-24, da D-38, la voce resta APERTA):** nella griglia di F9 un `n_components` intero della PCA a
  finestre va limitato a ≤ N − 1, con N = finestre di training su fit (`train_step` = 5, fit 60%): 2.824–3.430 sulle 6
  macchine dev, minimo su SMD 2.823 (train più corto, 23.687 righe). Misura in sola lettura, `reports_local/f8_pre/`.
  Vedi anche D-57.
- **Nota (2026-11-24, la voce resta APERTA):** W = 100, `learning_rate`, `weight_decay`, `min_delta` e `batch_size`
  sono valori storici fissi senza voce nel registro (fuori da D-39); non sono tarati, salvo quelli che entreranno nella
  griglia di questa voce. Da dichiarare in tesi (nota del 2026-11-24 nel piano F8).

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
- **Nota (2026-11-17, D-76, la voce resta APERTA):** le azioni che tolgono macchine intere (`exclude_machine_pair`, e
  `exclude_fit` per i modelli deterministici) riducono la possibilità di calcolare RQ2. Con il minimo a 5 macchine per
  gruppo (D-76), il gruppo da 6 delle macchine di valutazione ne può perdere una sola. Da considerare quando si
  fissano i valori di questa voce.

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
- **Nota (2026-11-24, da D-38, la voce resta APERTA):** con `train_step` = 5 e fit al 60% la PCA a finestre ha meno
  finestre di training che feature (N = 2.824–3.430 sulle dev, minimo SMD 2.823, contro W·F = 3.800): un
  `n_components` intero va limitato a ≤ N − 1 nella griglia di F9 (D-41). Il vincolo vale qualunque sia l'opzione
  scelta qui (solver `full` o `auto`).

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
- **Nota (2026-11-20, prerequisiti di F8; utente, su raccomandazione di Claude Project): schema e implementazione
  approvati. La voce resta APERTA solo per budget e griglie di tuning (D-41).**
  - **Fatti verificati nel codice (2026-11-20):**
    - `PyTorchTrainer` è solo ricostruttivo: la loss è `criterion(recon_x, batch)` sulla finestra intera (`train_epoch`,
      `evaluate`) e `get_error_matrix` confronta con tutta la finestra (`src/training/trainer.py:196-211`);
    - un modello può dichiarare una sola aggregazione dei residui (precedente: `rolling_zscore`,
      `residual_aggs = ("last",)`), e lo stadio di fit salva solo quelle dichiarate
      (`src/pipeline/residual_cache.py:65-67`);
    - `analysis_grid.residual_names` chiede tutte le `window_time_aggs` del piano per ogni modello non native e
      `fit_scores` cicla su tutti i `ts_aggs` (`src/pipeline/analysis_grid.py:96-106`, `:203`, `:219`);
    - `window_time_agg = none` è riservato ai modelli native (`src/pipeline/table_schema.py:220-224`);
    - **chiavi dell'LSTM-AE:** lo schema di `lstm_autoencoder` ha `hidden_dim` e `latent_dim`
      (`src/utils/schema.py:163-164`). `hidden_dim` ha lo stesso nome e lo stesso significato (dimensione dello stato
      nascosto dell'LSTM). **`num_layers` non è una chiave dell'LSTM-AE:** esiste solo come argomento interno di
      `LSTMEncoder` e `LSTMDecoder`, con lo stesso nome e lo stesso significato di `nn.LSTM`, ed è fisso a 1 perché
      `LSTMAutoencoderNet` non lo passa (`src/models/deep/lstm_ae.py:18`, `:37`, `:60-63`). Segnalato all'utente al
      passo 0: con `num_layers` come chiave il modello di previsione avrebbe un iperparametro che l'LSTM-AE non espone
      (asimmetria dello spazio di tuning, D-41). Deciso alla revisione del passo 0 (punto «chiavi» sotto).
  - **Decisioni:**
    - **nome e architettura:** `lstm_forecaster`: LSTM sui primi W−1 punti della finestra più una testa lineare verso
      gli F sensori; orizzonte 1 fisso (non è una chiave); blocco `training` condiviso con gli AE;
    - **chiavi (revisione del passo 0, utente, 2026-11-20):** solo `{hidden_dim}`, con lo stesso nome e lo stesso
      significato dell'LSTM-AE; **un solo strato** LSTM, come l'LSTM-AE (`num_layers` fisso a 1, non è una chiave).
      Motivo: stessa architettura di base dell'LSTM-AE e nessun iperparametro di tuning in più rispetto all'AE (D-41);
      l'unica differenza resta ricostruzione contro previsione. Scartate: `{hidden_dim, num_layers}` (disparità di
      tuning) ed esporre `num_layers` nell'LSTM-AE (cambio di schema dell'AE, rischio per `config_id` e per la parità
      di M1). **Da dichiarare in tesi.** Nessun valore predefinito nel codice; valori provvisori solo nei test
      sintetici (D-41 aperta);
    - **trainer (T2):** `ForecastTrainer`, sottoclasse di `PyTorchTrainer`. `PyTorchTrainer` resta intatto e
      `FIT_CODE_VERSION` invariata;
    - **residui:** `residual_aggs = ("last",)`; nelle tabelle `window_time_agg = last`;
    - **analisi:** le righe offline (`ts_agg` `mean`, `max`) del modello sono **assenti**, non NaN; il loro conteggio si
      registra in `analysis.json`;
    - **export legacy:** rifiuta il modello (non esiste un corrispettivo storico);
    - **`ANALYSIS_CODE_VERSION` aumentata** (il contratto delle uscite cambia); la golden di F6 si rigenera con la
      regola di F7, ammessa solo se cambiano soltanto gli identificativi;
    - **test di leakage obbligatorio:** modificare l'ultimo punto della finestra non cambia la previsione, modificare un
      punto tra i primi W−1 la cambia.
  - **Conseguenza da dichiarare:** un blocco che mescola AE e previsione non stima l'effetto di `ts_agg` sugli AE. In
    F10 RQ1a avrà due blocchi: solo AE, con il fattore `ts_agg`; tutti i modelli, con `ts_agg = last`.
  - **Restano APERTI:** budget di tuning, griglie e parità di trattamento (D-41).
  - **Documenti:** piano F8 (passi 4, 5).
  - **Realizzazione (passo 4 di F8, da rivedere alla revisione del passo):**
    - `src/models/deep/lstm_forecaster.py`: `LSTMForecasterNet` (un `nn.LSTM` a un solo strato sui primi W−1 punti,
      testa `nn.Linear(hidden_dim, F)` dallo stato nascosto finale) e `LSTMForecasterDetector` (`residual_aggs =
      ("last",)`, `deterministic` falso, `native_only` falso); `hidden_dim`, `batch_size` e `device` obbligatori nel
      codice del modello (nessun default); checkpoint come gli AE (D-67), con il trainer di previsione ricostruito in
      `load`;
    - `src/training/forecast_trainer.py` (file nuovo, così `src/training/trainer.py` resta intatto anche come file):
      `ForecastTrainer(PyTorchTrainer)` ridefinisce `train_epoch`, `evaluate`, `get_error_matrix` (solo `last`) e
      `get_reconstruction_errors`; `fit` (early stopping, ripristino dell'epoca migliore) è quello ereditato. Le chiavi
      di training che `PyTorchTrainer` legge con un default storico sono obbligatorie;
    - `src/models/base.py`: attributo di classe `allowed_ts_aggs` (`None` = tutti i `ts_agg`; previsione: `("last",)`),
      aggiunta senza effetto sugli altri modelli;
    - analisi (`analysis_grid.model_levels`, `analyze`): per ogni fit non native i livelli sono i `window_time_agg` del
      piano salvati dal fit (`residual_aggs` di `meta.json`) e i `ts_agg` ammessi dalla classe. Le combinazioni tolte
      sono assenti e contate in `analysis.json`, chiave `absent_rows` {modello: {motivo: {free, thr}}}, sempre presente
      (vuota senza modello di previsione). **Scelta di realizzazione da rivedere:** oltre ai `ts_agg` offline
      (motivo `offline_ts_agg`) sono assenti e contate anche le righe con `window_time_agg = mean` (motivo
      `window_time_agg_degenerate`, conseguenza di (b): il residuo `mean` non esiste e sarebbe uguale a `last`);
    - controllo preliminare `analyze.check_models_for_plan` (`ConfigError`, codice 2): un piano senza `last` tra i
      `window_time_aggs` o i `ts_aggs` lascerebbe il modello senza righe ed è un errore; l'esportazione legacy rifiuta
      i modelli senza corrispettivo storico (sotto, revisione del passo);
    - `ANALYSIS_CODE_VERSION` 1 → 2; `registry.NEW_PIPELINE_ONLY` comprende `lstm_forecaster`.
  - **Revisione del passo 4 (utente, 2026-11-20):**
    - **approvate** le due scelte di realizzazione sopra: righe con `window_time_agg = mean` assenti e contate con il
      motivo `window_time_agg_degenerate`; piano senza `last` con il modello presente → errore preliminare;
    - **esportazione legacy, elenco di ammessi:** accetta solo i modelli di `registry.LEGACY_NUCLEUS`
      (`legacy_export.refused_models`); ogni modello futuro è rifiutato per costruzione. Oltre a `lstm_forecaster` sono
      ora rifiutati anche `conv_autoencoder_untrained` e `lstm_autoencoder_untrained` (D-69), che al passo 3 l'export
      accettava pur senza corrispettivo in `main.py`. Sostituisce il primo disegno (elenco `REFUSED_MODELS` dei soli
      rifiutati);
    - **`residual_aggs` nel `fit_payload` (invariato):** il campo del payload (`["last", "mean"]`, da
      `schema.RESIDUAL_AGGS`) è ciò che l'esperimento chiede ed entra nel `fit_key`; i residui salvati davvero sono
      quelli di `meta.json` (`model.residual_aggs` e i record `residuals`): per `lstm_forecaster` solo `last`, come per
      il rolling z-score. L'analisi legge `meta.json`. Cambiare il payload cambierebbe tutti i `fit_key` esistenti;
    - **golden di F6, correzione della regola del passo:** la regola «golden rigenerata solo se cambiano soltanto gli
      identificativi» era imprecisa, perché un aumento di versione cambia per forza il valore della versione. Regola
      applicata: tabelle `free`, `thr`, `fits` identiche in forma d'identità; in `analysis.json` cambiano solo
      `analysis_code_version` (1 → 2, nei due punti) e la chiave nuova `absent_rows: {}` (sempre presente, anche vuota),
      cioè il contratto che giustifica l'aumento di versione. Golden rigenerata una volta;
    - **golden di F7:** criterio nuovo nella nota del 2026-11-20 su D-73 (golden).
  - **Passo 5 di F8, regole di `src/stats/plan.py` (utente, 2026-11-20, decisioni prese prima di scrivere il codice,
    su raccomandazione di Claude Project; da rivedere alla revisione del passo):**
    - **riscontro con lo schema:** le regole si ricavano dagli attributi della classe (`residual_aggs`,
      `allowed_ts_aggs`; `plan.model_window_time_aggs`, `plan.model_ts_aggs`), mai dal nome `lstm_forecaster`; ogni
      modello futuro con le stesse restrizioni è coperto;
    - **scoring:** dovunque il piano dà uno scoring a un modello (RQ2 riferimento, famiglia e `rq2.baselines`, cioè le
      righe descrittive tratte da `descriptive_baselines`; modelli dei blocchi di RQ1a; modelli di RQ4) il
      `window_time_agg` deve avere righe: per il modello di previsione solo `last`;
    - **RQ1a:** un blocco con il modello di previsione ha `ts_agg = [last]` (rifiutati sia i livelli offline in più sia
      un solo livello offline);
    - **RQ1b:** un blocco con il modello di previsione ha solo livelli di `scoring` con `window_time_agg = last`: più
      di un `window_time_agg` e il solo `mean` sono due rifiuti distinti (il secondo va oltre la lettera del piano, «più
      di un `window_time_agg`», per decisione dell'utente: altrimenti il blocco fallirebbe più tardi, ai controlli sugli
      ingressi);
    - **`descriptive`:** nessuna regola. Le voci non nominano modelli e `plan.py` non vede l'esperimento: con `ts_agg`
      ≠ `last` il modello di previsione è assente dalla voce per costruzione, come le sue righe offline (`absent_rows`).
      **Da dichiarare in tesi** insieme alla conseguenza del punto (d);
    - **`descriptive_baselines`:** resta aperta a ogni modello (decisione del 2026-11-16), compreso quello di previsione;
    - **`STATS_CODE_VERSION` invariata** (4): la modifica aggiunge solo rifiuti, quindi ogni piano valido prima dà le
      stesse uscite (stesso criterio del passo 3); golden di F7 invariata;
    - **test:** `tests/test_f8_forecaster_plan.py`, un test per ogni rifiuto e i casi ammessi; prove di difetto (6
      mutazioni in memoria, tutte rilevate).

*(D-75 è stata approvata il 2026-11-20 ed è passata alla sezione B.)*
*(D-38, D-39 e D-40 sono state approvate il 2026-11-24 e sono passate alla sezione B.)*

---

## E. Decisioni aperte che dipendono dall'utente

Nessuna voce aperta. *(D-54 è stata chiusa il 2026-11-20 ed è passata alla sezione A.)*

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
