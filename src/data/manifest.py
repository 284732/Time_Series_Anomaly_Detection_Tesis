"""
Manifest del dataset (F3): spec, inventario, controlli strutturali bloccanti, versioni dei dati.

Regole (docs/decisioni.md):
  * nessuna costante del dataset nel codice: macchine, gruppi, sensori e controlli vengono dallo spec
    ``configs/datasets/<nome>.yaml`` [D-58];
  * spec, file dev/eval e riferimento degli hash passano da un validatore rigoroso (chiave sconosciuta, campo
    mancante o tipo errato → ``ConfigError``) e da un lettore YAML che rifiuta le chiavi duplicate [D-58];
  * controlli bloccanti: un fallimento ferma tutto con l'elenco completo; nessuna esclusione silenziosa [D-32, D-58];
  * ``data_version`` = hash di train e test (entra nel ``fit_key``); ``labels_version`` = hash delle label, a parte,
    controllato dall'analisi [D-59]. Hash dei file = sha256 dei byte con CRLF → LF [D-59];
  * i file di ``data_dir`` si leggono solo: hash in modalità binaria, valori con il loader esistente.

Uso:
    spec = load_dataset_spec("configs/datasets/smd.yaml")
    man = build_manifest(spec, "dataset")      # inventario subito, hash e contenuto per macchina
    train, test, labels = man.load("machine-1-1")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loader import load_smd_machine
from src.utils.config import canonical_hash, load_yaml_strict
from src.utils.provenance import file_sha256
from src.utils.schema import ConfigError, _check_keys, _choice, _int, _real, _str

MANIFEST_FORMAT = 1
VERSION_LEN = 20                     # caratteri esadecimali delle versioni
FIT_FILE_KINDS = ("train", "test")   # entrano in data_version [D-59]
LABEL_FILE_KIND = "test_label"       # entra solo in labels_version [D-59]
FILE_KINDS = FIT_FILE_KINDS + (LABEL_FILE_KIND,)
ROLES = ("dev", "eval")


class ManifestError(RuntimeError):
    """Controllo strutturale fallito: la pipeline si ferma [D-32, D-58]."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("controlli del manifest falliti:\n  - " + "\n  - ".join(self.failures))


# ---------------------------------------------------------------- layout dei file
@dataclass(frozen=True)
class Layout:
    """Posizione dei file di una macchina e funzione di lettura (train, test, label)."""
    patterns: Mapping[str, str]                                   # tipo di file -> "cartella/{m}.txt"
    loader: Callable[[Union[str, Path], str], Tuple[np.ndarray, np.ndarray, np.ndarray]]


LAYOUTS: Dict[str, Layout] = {
    # Struttura SMD: train/, test/, test_label/ con <macchina>.txt (loader storico, invariato).
    "smd_txt": Layout({"train": "train/{m}.txt", "test": "test/{m}.txt", "test_label": "test_label/{m}.txt"},
                      load_smd_machine),
}


# ---------------------------------------------------------------- spec del dataset
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    layout: str
    n_features: int
    machines: Mapping[str, str]                    # nome -> gruppo, nell'ordine dello spec
    max_len_diff: Optional[int]                    # null = controllo non applicato
    value_range: Optional[Tuple[float, float]]     # null = controllo non applicato
    label_values: Tuple[int, ...]
    split_file: Optional[Path]                     # file dev/eval (assoluto) o None
    hashes_file: Optional[Path]                    # riferimento degli hash (assoluto) o None [D-61]
    source: Optional[Path] = None                  # file dello spec, solo informativo

    @property
    def groups(self) -> Dict[str, Tuple[str, ...]]:
        out: Dict[str, List[str]] = {}
        for m, g in self.machines.items():
            out.setdefault(g, []).append(m)
        return {g: tuple(ms) for g, ms in out.items()}


SPEC_KEYS = ("name", "layout", "n_features", "groups", "max_len_diff", "value_range", "label_values",
             "split_file", "hashes_file")


def _optional_path(v: Any, path: str, base: Path) -> Optional[Path]:
    if v is None:
        return None
    return (base / _str(v, path)).resolve()


def validate_dataset_spec(raw: Any, base_dir: Union[str, Path], source: Optional[Path] = None) -> DatasetSpec:
    """Valida uno spec già letto. I percorsi relativi si risolvono rispetto a ``base_dir`` (cartella dello spec)."""
    base = Path(base_dir)
    d = _check_keys(raw, "dataset_spec", SPEC_KEYS)
    name = _str(d["name"], "dataset_spec.name")
    layout = _choice(tuple(LAYOUTS))(d["layout"], "dataset_spec.layout")
    n_features = _int(1)(d["n_features"], "dataset_spec.n_features")

    groups = d["groups"]
    if not isinstance(groups, Mapping) or not groups:
        raise ConfigError(f"dataset_spec.groups: atteso un blocco non vuoto gruppo -> [macchine], trovato {groups!r}")
    machines: Dict[str, str] = {}
    for g, ms in groups.items():
        gp = f"dataset_spec.groups[{g!r}]"
        if not isinstance(g, str) or not g:
            raise ConfigError(f"{gp}: il nome del gruppo deve essere una stringa non vuota (tra virgolette in YAML)")
        if not isinstance(ms, (list, tuple)) or not ms:
            raise ConfigError(f"{gp}: attesa una lista non vuota di macchine, trovato {ms!r}")
        for i, m in enumerate(ms):
            m = _str(m, f"{gp}[{i}]")
            if m in machines:
                raise ConfigError(f"{gp}: macchina {m!r} ripetuta")
            machines[m] = g

    mld = d["max_len_diff"]
    max_len_diff = None if mld is None else _int(0)(mld, "dataset_spec.max_len_diff")

    vr = d["value_range"]
    value_range = None
    if vr is not None:
        if not isinstance(vr, (list, tuple)) or len(vr) != 2:
            raise ConfigError(f"dataset_spec.value_range: atteso null o [lo, hi], trovato {vr!r}")
        lo, hi = (_real()(x, f"dataset_spec.value_range[{i}]") for i, x in enumerate(vr))
        if not lo < hi:
            raise ConfigError(f"dataset_spec.value_range: serve lo < hi, trovato {vr!r}")
        value_range = (lo, hi)

    lv = d["label_values"]
    if not isinstance(lv, (list, tuple)) or not lv:
        raise ConfigError(f"dataset_spec.label_values: attesa una lista non vuota di interi, trovato {lv!r}")
    label_values = tuple(_int()(x, f"dataset_spec.label_values[{i}]") for i, x in enumerate(lv))
    if len(set(label_values)) != len(label_values):
        raise ConfigError("dataset_spec.label_values: valori ripetuti")

    return DatasetSpec(
        name=name, layout=layout, n_features=n_features, machines=machines, max_len_diff=max_len_diff,
        value_range=value_range, label_values=label_values,
        split_file=_optional_path(d["split_file"], "dataset_spec.split_file", base),
        hashes_file=_optional_path(d["hashes_file"], "dataset_spec.hashes_file", base),
        source=source,
    )


def load_dataset_spec(path: Union[str, Path]) -> DatasetSpec:
    """Legge e valida ``configs/datasets/<nome>.yaml`` (chiavi duplicate rifiutate)."""
    path = Path(path).resolve()
    return validate_dataset_spec(load_yaml_strict(path), path.parent, source=path)


def spec_path_for(dataset: str, datasets_dir: Union[str, Path]) -> Path:
    """Spec del dataset indicato da ``data.dataset`` nella config dell'esperimento."""
    return Path(datasets_dir) / f"{_str(dataset, 'data.dataset')}.yaml"


# ---------------------------------------------------------------- file dev/eval e riferimento degli hash
SPLIT_KEYS_REQUIRED = ("dev", "eval")
SPLIT_KEYS_OPTIONAL = ("seed", "procedure", "numpy_version")   # metadati di make_dev_split.py (D-51)


def load_role_split(path: Union[str, Path]) -> Dict[str, Tuple[str, ...]]:
    """File dev/eval (es. ``configs/dev_machines.yaml``) validato con le regole di F2."""
    d = _check_keys(load_yaml_strict(path), f"{path}", SPLIT_KEYS_REQUIRED, SPLIT_KEYS_OPTIONAL)
    out = {}
    for role in ROLES:
        ms = d[role]
        if not isinstance(ms, (list, tuple)):
            raise ConfigError(f"{path}.{role}: attesa una lista di macchine, trovato {ms!r}")
        names = tuple(_str(m, f"{path}.{role}[{i}]") for i, m in enumerate(ms))
        if len(set(names)) != len(names):
            raise ConfigError(f"{path}.{role}: macchine ripetute")
        out[role] = names
    return out


HASHES_KEYS = ("format", "dataset", "rule", "files")
HASH_RULE = "sha256 dei byte dopo CRLF -> LF"


def load_reference_hashes(path: Union[str, Path], dataset: str) -> Dict[str, str]:
    """Riferimento degli hash [D-61]: ``files`` = {percorso relativo posix: sha256}."""
    d = _check_keys(load_yaml_strict(path), f"{path}", HASHES_KEYS)
    if d["format"] != MANIFEST_FORMAT:
        raise ConfigError(f"{path}.format: atteso {MANIFEST_FORMAT}, trovato {d['format']!r}")
    if d["dataset"] != dataset:
        raise ConfigError(f"{path}.dataset: atteso {dataset!r}, trovato {d['dataset']!r}")
    if d["rule"] != HASH_RULE:
        raise ConfigError(f"{path}.rule: atteso {HASH_RULE!r}, trovato {d['rule']!r}")
    files = d["files"]
    if not isinstance(files, Mapping) or not files:
        raise ConfigError(f"{path}.files: atteso un blocco non vuoto percorso -> sha256")
    out = {}
    for k, v in files.items():
        _str(k, f"{path}.files")
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ConfigError(f"{path}.files[{k!r}]: atteso uno sha256 esadecimale minuscolo, trovato {v!r}")
        out[k] = v
    return out


# ---------------------------------------------------------------- manifest
@dataclass(frozen=True)
class FileRecord:
    relpath: str        # relativo a data_dir, separatore "/"
    sha256: str


@dataclass(frozen=True)
class MachineEntry:
    name: str
    group: str
    role: Optional[str]
    files: Mapping[str, FileRecord]
    data_version: str
    labels_version: str


def _version(prefix: str, payload: Dict[str, Any]) -> str:
    return prefix + canonical_hash(payload)[:VERSION_LEN]


def expected_relpaths(spec: DatasetSpec, machine: str) -> Dict[str, str]:
    return {k: p.format(m=machine) for k, p in LAYOUTS[spec.layout].patterns.items()}


class Manifest:
    """Inventario verificato di un dataset. Hash calcolati alla prima richiesta (nessuna cache su disco)."""

    def __init__(self, spec: DatasetSpec, root: Path, roles: Optional[Dict[str, Tuple[str, ...]]],
                 reference: Optional[Dict[str, str]]):
        self.spec = spec
        self.root = root
        self._roles = roles
        self._reference = reference
        self._entries: Dict[str, MachineEntry] = {}
        self._checked: set = set()

    # -------------------------------------------------- elenchi
    def machines(self, role: Optional[str] = None) -> Tuple[str, ...]:
        if role is None:
            return tuple(self.spec.machines)
        if role not in ROLES:
            raise ValueError(f"ruolo {role!r} non ammesso, ammessi {ROLES}")
        if self._roles is None:
            raise ConfigError(f"lo spec {self.spec.name!r} non ha split_file: ruoli dev/eval non definiti")
        return self._roles[role]

    def resolve(self, selection: Union[str, Sequence[str]]) -> Tuple[str, ...]:
        """``data.machines`` della config: "dev" | "eval" | lista esplicita (nomi verificati)."""
        if isinstance(selection, str):
            return self.machines(selection)
        unknown = [m for m in selection if m not in self.spec.machines]
        if unknown:
            raise ConfigError(f"data.machines: macchine non presenti nello spec {self.spec.name!r}: {unknown}")
        return tuple(selection)

    def group_of(self, machine: str) -> str:
        return self.spec.machines[self._known(machine)]

    def role_of(self, machine: str) -> Optional[str]:
        self._known(machine)
        if self._roles is None:
            return None
        return next(r for r in ROLES if machine in self._roles[r])

    def _known(self, machine: str) -> str:
        if machine not in self.spec.machines:
            raise KeyError(f"macchina {machine!r} non presente nello spec {self.spec.name!r}")
        return machine

    # -------------------------------------------------- versioni [D-59]
    def entry(self, machine: str) -> MachineEntry:
        """Hash dei file della macchina, confrontati con il riferimento se lo spec lo prevede."""
        self._known(machine)
        if machine not in self._entries:
            files, failures = {}, []
            for kind, rel in expected_relpaths(self.spec, machine).items():
                sha = file_sha256(self.root / rel)
                if sha is None:
                    failures.append(f"{machine}: file mancante {rel}")
                    continue
                if self._reference is not None and self._reference.get(rel) != sha:
                    failures.append(f"{machine}: hash di {rel} diverso dal riferimento "
                                    f"({self._reference.get(rel)} != {sha})")
                files[kind] = FileRecord(rel, sha)
            if failures:
                raise ManifestError(failures)
            base = {"manifest_format": MANIFEST_FORMAT, "dataset": self.spec.name, "machine": machine}
            dv = _version("dv1-", {**base, "files": {k: files[k].sha256 for k in FIT_FILE_KINDS}})
            lv = _version("lv1-", {**base, "files": {LABEL_FILE_KIND: files[LABEL_FILE_KIND].sha256}})
            self._entries[machine] = MachineEntry(machine, self.group_of(machine), self.role_of(machine),
                                                  files, dv, lv)
        return self._entries[machine]

    def data_version(self, machine: str) -> str:
        """Versione dei dati usata nel ``fit_key``: solo train e test [D-59]."""
        return self.entry(machine).data_version

    def labels_version(self, machine: str) -> str:
        """Versione delle label: fuori dal ``fit_key``, controllata dallo stadio di analisi [D-59]."""
        return self.entry(machine).labels_version

    def dataset_version(self) -> str:
        """Hash di tutte le versioni (provenance e registro, non ``fit_key``)."""
        return _version("ds1-", {
            "manifest_format": MANIFEST_FORMAT, "dataset": self.spec.name,
            "machines": {m: {"data_version": self.data_version(m), "labels_version": self.labels_version(m)}
                         for m in self.machines()},
        })

    # -------------------------------------------------- contenuto [D-32, D-58]
    def check_content(self, machine: str, train: np.ndarray, test: np.ndarray, labels: np.ndarray) -> None:
        s, f = self.spec, []
        for what, x in (("train", train), ("test", test)):
            if x.ndim != 2:
                f.append(f"{machine}: {what} con {x.ndim} dimensioni, attese 2")
                continue
            if x.shape[1] != s.n_features:
                f.append(f"{machine}: {what} con {x.shape[1]} sensori, attesi {s.n_features}")
            if not np.isfinite(x).all():
                f.append(f"{machine}: {what} contiene valori non finiti")
            elif s.value_range is not None and x.size:
                lo, hi = float(x.min()), float(x.max())
                if lo < s.value_range[0] or hi > s.value_range[1]:
                    f.append(f"{machine}: {what} con valori in [{lo}, {hi}], fuori da {list(s.value_range)}")
        if s.max_len_diff is not None and abs(len(train) - len(test)) > s.max_len_diff:
            f.append(f"{machine}: |T_train - T_test| = {abs(len(train) - len(test))} > {s.max_len_diff}")
        if labels.ndim != 1:
            f.append(f"{machine}: label con {labels.ndim} dimensioni, attesa 1")
        elif len(labels) != len(test):
            f.append(f"{machine}: {len(labels)} label per {len(test)} punti di test")
        bad = np.setdiff1d(np.unique(labels), np.asarray(s.label_values))
        if bad.size:
            f.append(f"{machine}: label con valori non ammessi {bad.tolist()}, ammessi {list(s.label_values)}")
        if f:
            raise ManifestError(f)

    def load(self, machine: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hash (e riferimento) poi contenuto: un fallimento ferma tutto. Restituisce (train, test, label)."""
        entry = self.entry(machine)
        train, test, labels = LAYOUTS[self.spec.layout].loader(self.root, machine)
        # i file letti devono essere quelli di cui si è calcolato l'hash (nessun cambio a metà esecuzione)
        changed = [r.relpath for r in entry.files.values() if file_sha256(self.root / r.relpath) != r.sha256]
        if changed:
            raise ManifestError([f"{machine}: file cambiato dopo il calcolo dell'hash: {changed}"])
        self.check_content(machine, train, test, labels)
        self._checked.add(machine)
        return train, test, labels

    def check_all(self) -> None:
        """Controlli di contenuto su tutte le macchine; raccoglie tutti i fallimenti prima di fermarsi."""
        failures = []
        for m in self.machines():
            try:
                self.load(m)
            except ManifestError as exc:
                failures += exc.failures
        if failures:
            raise ManifestError(failures)

    def to_dict(self) -> Dict[str, Any]:
        """Riassunto per la provenance (nessun percorso assoluto)."""
        return {
            "manifest_format": MANIFEST_FORMAT, "dataset": self.spec.name, "layout": self.spec.layout,
            "dataset_version": self.dataset_version(),
            "machines": {m: {"group": e.group, "role": e.role, "data_version": e.data_version,
                             "labels_version": e.labels_version,
                             "files": {k: {"path": r.relpath, "sha256": r.sha256} for k, r in e.files.items()}}
                         for m, e in ((m, self.entry(m)) for m in self.machines())},
        }


def build_manifest(spec: DatasetSpec, data_dir: Union[str, Path]) -> Manifest:
    """Inventario su tutto il dataset (bloccante); hash e contenuto si controllano per macchina."""
    root = Path(data_dir).resolve()
    failures: List[str] = []
    if not root.is_dir():
        raise ManifestError([f"cartella dei dati inesistente: {root}"])

    expected = set(spec.machines)
    for kind, pattern in LAYOUTS[spec.layout].patterns.items():
        folder, _, fname = pattern.partition("/")
        suffix = fname.replace("{m}", "")
        found = {p.name[: -len(suffix)] for p in (root / folder).glob(f"*{suffix}")} if (root / folder).is_dir() else set()
        missing, extra = sorted(expected - found), sorted(found - expected)
        if missing:
            failures.append(f"{folder}/: macchine mancanti {missing}")
        if extra:
            failures.append(f"{folder}/: file non previsti dallo spec {extra}")

    roles = None
    if spec.split_file is not None:
        try:
            roles = load_role_split(spec.split_file)
        except (ConfigError, OSError) as exc:
            failures.append(f"file dev/eval non valido: {exc}")
        else:
            dev, ev = set(roles["dev"]), set(roles["eval"])
            if dev & ev:
                failures.append(f"dev ed eval non disgiunte: {sorted(dev & ev)}")
            if dev | ev != expected:
                failures.append(f"dev ∪ eval diverso dalle macchine dello spec: mancanti {sorted(expected - dev - ev)}, "
                                f"in più {sorted((dev | ev) - expected)}")

    reference = None
    if spec.hashes_file is not None:
        try:
            reference = load_reference_hashes(spec.hashes_file, spec.name)
        except (ConfigError, OSError) as exc:
            failures.append(f"riferimento degli hash non valido o mancante (D-61): {exc}")
        else:
            exp_files = {rel for m in spec.machines for rel in expected_relpaths(spec, m).values()}
            if set(reference) != exp_files:
                failures.append(f"riferimento degli hash non coerente con lo spec: mancanti "
                                f"{sorted(exp_files - set(reference))[:5]}, in più {sorted(set(reference) - exp_files)[:5]}")

    if failures:
        raise ManifestError(failures)
    return Manifest(spec, root, roles, reference)


def compute_file_hashes(spec: DatasetSpec, data_dir: Union[str, Path]) -> Dict[str, str]:
    """Hash di tutti i file previsti dallo spec (per ``scripts/make_data_hashes.py``, D-61)."""
    root = Path(data_dir).resolve()
    out, missing = {}, []
    for m in spec.machines:
        for rel in expected_relpaths(spec, m).values():
            sha = file_sha256(root / rel)
            if sha is None:
                missing.append(rel)
            else:
                out[rel] = sha
    if missing:
        raise ManifestError([f"file mancanti: {missing[:10]}"])
    return out
