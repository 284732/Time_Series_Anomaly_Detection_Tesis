from pathlib import Path
import numpy as np

# Percorso del vecchio progetto
OLD = Path(r"..\SMD_aggiornato_v3\SMD")

# Percorso dei nuovi risultati
NEW = Path("results/results_tests")

MACCHINE = ["machine-1-1", "machine-3-8"]

CONFRONTI = [
    (
        "results_repro_last",
        [
            ("results_last_full", [
                "random_baseline",
                "pca",
                "conv_autoencoder",
                "lstm_autoencoder"
            ]),
            ("results_pointwise_last_new", ["pca_pointwise"]),
        ],
    ),
    (
        "results_repro_mean",
        [
            ("results", [
                "pca",
                "conv_autoencoder",
                "lstm_autoencoder"
            ]),
            ("results_pointwise_mean_new", ["pca_pointwise"]),
        ],
    ),
]

for nuova, coppie in CONFRONTI:

    for vecchia, modelli in coppie:

        for macchina in MACCHINE:

            cartella_nuova = NEW / nuova / "scores" / macchina

            for f in sorted(cartella_nuova.glob("*_scores.npz")):

                modello = f.name.split("__")[0]

                if modello not in modelli:
                    continue

                vecchio_file = (
                    OLD / vecchia / "scores" / macchina / f.name
                )

                if not vecchio_file.exists():
                    print(f"MANCA {vecchio_file}")
                    continue

                a = np.load(f)
                b = np.load(vecchio_file)

                # Confronto score
                val_ts = np.array_equal(
                    a["val_ts"], b["val_ts"], equal_nan=True
                )

                test_ts = np.array_equal(
                    a["test_ts"], b["test_ts"], equal_nan=True
                )

                # Confronto maschere
                val_mask = np.array_equal(
                    a["val_mask"], b["val_mask"]
                )

                test_mask = np.array_equal(
                    a["test_mask"], b["test_mask"]
                )

                # Confronto threshold
                threshold = np.array_equal(
                    a["threshold"], b["threshold"]
                )

                uguali = all([
                    val_ts,
                    test_ts,
                    val_mask,
                    test_mask,
                    threshold
                ])

                print(
                    f"{'IDENTICO' if uguali else 'DIVERSO'} "
                    f"{vecchia:28s} "
                    f"{macchina} "
                    f"{f.name}"
                )