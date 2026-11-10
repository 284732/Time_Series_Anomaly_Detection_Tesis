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

---

## D. Decisioni aperte (metodologiche)

### D-30 · Margine δ — APERTA
- **Opzioni:** (a) δ = 0,05 in AP normalizzata; (b) δ ricavato solo dalle macchine di sviluppo con una regola dichiarata;
  (c) altro valore motivato. Criterio scartato: δ ≥ spread del protocollo sugli stessi dati (circolare).
- **Vincolo:** va fissato prima dei run ufficiali e congelato in `configs/analysis/official.yaml`.
- **Serve entro:** F10. Il codice di F7 deve trattare δ come parametro, senza valore predefinito.

### D-31 · Quali segmenti ricevono il contesto nello scoring — SOSTITUITA da D-32 · 2026-11-09
- **Stato:** sostituita da D-32 (2026-11-09). D-32 disciplina ora il contesto iniziale del test e i segmenti; la gestione
  operativa del contesto iniziale del test e della maschera è in D-20. Il testo seguente resta come riferimento storico.
- **Problema:** con `window_time_agg: mean`, le prime W−1 finestre di cal con contesto includerebbero residui di es, che
  per gli AE sono ottimistici; entrerebbero così nel normalizer e nelle soglie, in contrasto con D-04.
- **Opzioni:** (a) contesto solo per il test (da cal); cal ed es senza contesto (si perdono W−1 punti di cal su ≈4.700);
  (b) contesto per tutti i segmenti di scoring, dichiarando l'effetto; (c) contesto per cal solo con `window_time_agg: last`.
- **Raccomandazione:** (a).
- **Serve entro:** F3.

### D-33 · Aggregazioni dei residui da salvare — APERTA
- **Opzioni:** `last` e `mean` (raccomandato); `max` opzionale. Influisce sul formato di `residuals.npz` e su `fit_key`.
- **Serve entro:** F0 (gruppo A).

### D-34 · Metriche da implementare in F6 e trattamento di VUS-PR — APERTA
- **Da decidere:** se in F6 entrano metriche per evento, ritardo di rilevamento, falsi allarmi al giorno, celle 2×2
  (raccomandato: sì). Per VUS-PR (approvata come secondaria, D-07): quando implementarla e contro quale riferimento
  esterno validarla, senza nuove dipendenze (D-18). Proposta: implementazione separata dopo F6, con test contro valori
  pubblicati da un'implementazione di riferimento.
- **Serve entro:** F0 (gruppo A) per l'elenco; VUS-PR entro F10.

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
### D-43 · Descrittori per RQ3 (massimo 4) e ipotesi — APERTA · **Serve entro:** F10.
### D-44 · Regole di esclusione (fit fallito, macchina con un solo evento) — APERTA · **Serve entro:** F10.

### D-45 · Secondo dataset — APERTA
- **Vincolo:** deve avere più entità (PSM escluso; ASD candidato da verificare). Il piano congelato per SMD si riapplica
  invariato. **Serve entro:** da scegliere presto; condiziona la forza delle conclusioni.

### D-46 · Sensibilità delle proporzioni dello split — APERTA
- **Opzioni:** nessuna; oppure una sola variante (es. 70/15/15) sulle macchine di sviluppo in F9. **Serve entro:** F9.

---

## E. Decisioni aperte che dipendono dall'utente

### D-50 · Run storici di riferimento per M1 — APERTA — DIPENDE DALL'UTENTE
- Nel repository `results/` contiene solo `results_tests/`. Da confermare: esistenza locale dei run di riproduzione
  (`results_repro_*_01` nella proposta e nel piano, `results_repro_last` / `results_repro_mean` nel vecchio `CLAUDE.md`),
  nome esatto, config e commit che li hanno prodotti. Se non esistono: rigenerarli con `main.py` a un commit noto (li lancia l'utente).
- **Serve entro:** M1 (meglio in F0, per pianificare).

### D-51 · Lista delle 6 macchine di sviluppo — APERTA — DIPENDE DALL'UTENTE
- La proposta le dice "già estratte (seed 20261109)", ma nel repository non ce n'è traccia. Se la lista esiste, va
  registrata qui con procedura e seed. Se non esiste: estrarla in F0, 2 per gruppo, prima di guardare qualsiasi metrica.
- **Serve entro:** F3 (il manifest separa sviluppo e valutazione), non F8.

### D-52 · Macchine usate per i numeri preliminari della proposta — APERTA — DIPENDE DALL'UTENTE
- Verificare se `machine-1-1` e `machine-3-8` sono tra quelle di sviluppo. Se non lo sono, va dichiarato tra i limiti
  (scelte di protocollo motivate da metriche di macchine di valutazione).

### D-53 · File tracciati da git (`.pyc`, dataset) — APERTA — DIPENDE DALL'UTENTE
- 35 file `.pyc` sono tracciati. Il `.gitignore` di F0 non li rimuove dal tracciamento: `git rm --cached` è una scelta dell'utente.

### D-54 · Scadenze e tempo disponibile — APERTA — DIPENDE DALL'UTENTE
- Determinano se F9 e il secondo dataset sono realistici.

### D-55 · Ambiente di riferimento e file di lock — APERTA — DIPENDE DALL'UTENTE
- `requirements-lock.txt` va generato con `pip freeze` nell'ambiente Windows dell'utente, lo stesso che produrrà M1 e i
  run ufficiali. Da indicare: versione di Python e ambiente usato.

### D-56 · Posizione di `store/` e budget di disco — APERTA — DIPENDE DALL'UTENTE
- Stima della proposta: ~5–10 GB. **Serve entro:** F0.
