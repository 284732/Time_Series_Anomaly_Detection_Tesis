# Architettura del repository v1 (bozza, 2026-11-09)

Decisioni prese: (1) split cronologico del training 60% fit / 20% early stopping / 20% calibrazione, uguale per tutti i modelli; (2) sei macchine di sviluppo, due per gruppo server, sorteggio a seed fisso. Secondo dataset rimandato.

## 1. Layout

```
thesis-smd/
  configs/
    base.yaml                 # dati, split, budget, modelli
    eval.yaml                 # scoring, aggregazioni, soglie, metriche (nessun fit)
    tuning/                   # generate da tools/make_grid.py, non scritte a mano
  src/
    data/        loader.py, split.py, scaler.py, windows.py
    models/      base.py, pca.py, pca_pointwise.py, conv_ae.py, lstm_ae.py,
                 baselines/ (random, input_norm, untrained_ae, rolling_zscore)
    training/    trainer.py
    fit/         run_fit.py          # STADIO 1: fit + residui
    scoring/     residuals.py, scores.py     # STADIO 2
    thresholds/  policies.py                 # STADIO 3 (ex rules.py)
    metrics/     pointwise.py, protocols.py, events.py   # STADIO 4
    stats/       paired.py, equivalence.py, variance.py, descriptors.py  # STADIO 5
    utils/       config.py (hash), provenance.py, seed.py
  tools/         make_grid.py, make_dev_split.py
  tests/
  artifacts/     # fuori da git
    fit/<run_id>/<macchina>/<modello>__s<seed>.npz + meta.json
    tables/*.parquet
  figures/, thesis/
```

`run_id` = hash(config base + commit git + versione dati). Ogni artefatto porta `meta.json` con config completa, hash, commit, versioni librerie, tempo di fit.

## 2. Split e macchine

- Per ogni macchina: train originale → fit 60% / es 20% / calib 20% (ordine cronologico). Scaler fittato solo sul fit. Finestre mai a cavallo tra segmenti.
- PCA e baseline senza early stopping: fit solo sul 60%. Analisi di sensibilità: PCA su fit+es.
- Macchine di sviluppo (sorteggio numpy default_rng, seed 20261109, 2 per gruppo): `machine-1-3, machine-1-6, machine-2-7, machine-2-9, machine-3-5, machine-3-11`. Il tuning usa solo queste; la valutazione confrontativa le esclude (22 macchine). Nota: 1-3 e 3-11 risultano, dai vecchi notebook, tra le macchine con forte spostamento già nel validation. La lista resta quella sorteggiata (non si rimpiazza guardando i risultati); va solo dichiarato.

## 3. Stadio 1 — fit e residui (unico stadio costoso)

Input: macchina, modello, seed, config di training. Output per (macchina, modello, seed):

| Chiave npz | Forma | Contenuto |
|---|---|---|
| `res_es_{last,mean}` | (N_es, F) | residui per sensore, slice early stopping |
| `res_cal_{last,mean}` | (N_cal, F) | slice di calibrazione |
| `res_test_{last,mean}` | (N_test, F) | test |
| `window_size, step` | scalari | per la proiezione sui timestamp |
| `n_components` / `best_epoch` / `stopped_epoch` | scalari | complessità effettiva del modello |
| `fit_sec, infer_sec` | scalari | costo |

`{last,mean}` = `window_time_agg`: errore all'ultimo istante della finestra oppure medio sugli istanti. Float32, ~8 MB per macchina/modello/seed. Modelli senza residui per sensore (random) salvano uno score scalare con la stessa struttura.

Fattori che richiedono un nuovo fit: modello, iperparametri, scaler/clip, finestra, frazione di training, seed. Tutto il resto è a valle.

## 4. Stadi 2–5 (a valle, nessun fit)

- **S2 scoring**: da residui a score di finestra: raw mean, z per sensore (statistiche dalla calibrazione), top-k, max. Proiezione sui timestamp: last / mean / max (`timestamp_agg`).
- **S3 soglie**: percentile della calibrazione (95, 98, 99, 99.5), `burnin`, `roll_quantile`, `roll_robust_z` con `n_past` variabile (360, 720, 1440, 2880), oracolo.
- **S4 metriche**: AP, AP normalizzata per prevalenza, ROC-AUC, VUS-PR, F1 pointwise e PA (4 protocolli), metriche per evento, FPR per decile con e senza fascia di esclusione attorno agli eventi.
- **S5 statistica**: tabella lunga (macchina, gruppo, modello, seed, fattori di protocollo, metrica, valore) → delta appaiati con IC bootstrap gerarchico, equivalenza con δ, modello misto, τ di Kendall.

## 5. Config base (bozza)

```yaml
project: smd_thesis
data: {dir: dataset, window: 100, step: 1, train_step: 5}
split: {fit: 0.6, es: 0.2, calib: 0.2}        # cronologico, uguale per tutti
preprocessing: {scaler: minmax, clip: none}   # none | [0,1] | [-k,k]
seeds: [42, 43, 44]                           # i modelli deterministici girano una volta
device: cpu
machines:
  dev: [machine-1-3, machine-1-6, machine-2-7, machine-2-9, machine-3-5, machine-3-11]
  eval: rest                                  # le altre 22
training:
  max_epochs: 100
  early_stopping_patience: 10                 # budget canonico unico
  batch_size: 64
models:
  pca: {n_components: 0.9}
  pca_pointwise: {n_components: 0.9}
  conv_autoencoder: {latent_dim: 16, channels: [32, 64]}
  lstm_autoencoder: {latent_dim: 16, hidden_dim: 64}
baselines: [random, input_norm, untrained_conv_ae, untrained_lstm_ae, rolling_zscore]
provenance: {config_hash: true, git_commit: true, refuse_resume_on_mismatch: true}
```

Rimosso rispetto alla vecchia config: `timestamp_agg`, `window_time_agg`, `threshold_percentile` (ora in `eval.yaml`), `results_dir` per variante, `save_plots`, OC-SVM, Isolation Forest, VAE.

## 6. Test da scrivere / portare

Portare: test_protocol, test_rules, test_scoring, test_baselines, test_kim_baselines, test_models, early stopping.
Scrivere:
1. Lo scaler non vede es, calib e test.
2. Nessuna finestra attraversa i confini fit/es/calib/test.
3. `fit/es/calib` coprono tutto il train, in ordine, senza sovrapposizioni.
4. Oracolo esatto ≥ oracolo a griglia; oracolo ≥ F1 con soglia da calibrazione.
5. F1 con PA ≥ F1 senza PA a pari soglia.
6. Proiezione dei residui: `res_*_last` coincide con l'errore dell'ultimo istante, `res_*_mean` con la media sugli istanti, su un modello giocattolo.
7. Causalità end-to-end: modificare il futuro di una serie non cambia gli score passati con `window_time_agg` e `timestamp_agg` = last.
8. Riproducibilità: stesso run_id e seed → stessi residui (tolleranza); `resume` con hash diverso rifiuta.
9. Test di regressione: sulla PCA con seed 42 e config storica, AP/ROC-AUC entro tolleranza dai vecchi CSV (l'unico uso dei vecchi numeri).
10. Test statistici su dati sintetici: IC con copertura nominale, regola superiore/equivalente/inferiore con casi noti.

## 7. Ordine di costruzione

1. data + split + scaler + test 1–3. 2. PCA + baseline + S1 salvataggio residui + test 6. 3. S2–S4 con test 4, 5, 7. 4. Regressione contro i vecchi numeri (PCA, 28 macchine). 5. Solo dopo: AE, tuning sulle 6 macchine di sviluppo, run completo. 6. Statistica e figure.
