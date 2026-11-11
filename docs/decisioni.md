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

### D-43 · Descrittori per RQ3 (massimo 4) e ipotesi — APERTA · **Serve entro:** F10.
### D-44 · Regole di esclusione (fit fallito, macchina con un solo evento) — APERTA · **Serve entro:** F10.

### D-45 · Secondo dataset — APERTA
- **Vincolo:** deve avere più entità (PSM escluso; ASD candidato da verificare). Il piano congelato per SMD si riapplica
  invariato. **Serve entro:** da scegliere presto; condiziona la forza delle conclusioni.

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

### D-69 · AE non addestrati (baseline di sanità E0, `piano_tesi_v1.md` §5) — APERTA · 2026-11-11
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
