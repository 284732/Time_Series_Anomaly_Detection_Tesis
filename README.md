# Time Series Anomaly Detection su SMD — tesi

Confronto tra PCA e autoencoder (Conv-AE, LSTM-AE) per l'anomaly detection multivariata sul Server Machine Dataset (SMD),
con attenzione all'effetto delle scelte di valutazione sui risultati.

- Regole di lavoro, stato della pipeline e protezioni: [`CLAUDE.md`](CLAUDE.md).
- Decisioni approvate e aperte: [`docs/decisioni.md`](docs/decisioni.md) (prevale sugli altri documenti per i punti che registra).
- Domanda di ricerca: [`piano_tesi_v1.md`](piano_tesi_v1.md); architettura e piano di sviluppo:
  [`proposta_architettura_target.md`](proposta_architettura_target.md), [`piano_operativo.md`](piano_operativo.md).

## Ambiente

Ambiente di riferimento (D-55): Windows, **Python 3.13.9**, solo CPU. Le versioni esatte delle librerie sono in
[`requirements-lock.txt`](requirements-lock.txt); [`requirements.txt`](requirements.txt) elenca solo le dipendenze dirette.

```
python --version                                   # deve dare Python 3.13.9
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

## Test

Dalla radice del progetto:

```
python -m pytest -q
```

I test usano solo dati sintetici e non leggono `dataset/`. I `.pyc` e la cache di pytest sono esclusi da git
(`.gitignore`, D-53).

## Dati e risultati

- `dataset/`: SMD (28 macchine, `train/`, `test/`, `test_label/`), **sola lettura**.
- `results/`: risultati storici, **sola lettura**; il riferimento per la verifica di parità M1 è indicato in D-50.
- `configs/dev_machines.yaml`: le 6 macchine di sviluppo e le 22 di valutazione (D-51), generato da
  `python scripts/make_dev_split.py` (`--check` per verificarlo senza riscriverlo).

La pipeline storica si esegue con `python main.py <config>` e allena modelli sui dati reali: va lanciata solo
intenzionalmente (vedi `CLAUDE.md` §4).
