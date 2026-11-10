"""Test della lista delle macchine di sviluppo (D-51): scripts/make_dev_split.py e configs/dev_machines.yaml."""

from pathlib import Path

import yaml

from scripts.make_dev_split import FIXED, OUT_PATH, group_of, make_dev_split
from src.data.loader import SMD_MACHINES


def test_six_dev_and_22_eval_disjoint_covering_all_machines():
    split = make_dev_split()
    dev, ev = split["dev"], split["eval"]
    assert len(dev) == 6 and len(ev) == 22
    assert not set(dev) & set(ev)
    assert set(dev) | set(ev) == set(SMD_MACHINES)


def test_two_dev_machines_per_group_and_fixed_ones_included():
    dev = make_dev_split()["dev"]
    assert sorted(group_of(m) for m in dev) == [1, 1, 2, 2, 3, 3]
    assert set(FIXED) <= set(dev)
    assert set(FIXED) == {"machine-1-1", "machine-3-8"}


def test_same_list_at_every_run():
    assert make_dev_split() == make_dev_split()
    assert make_dev_split()["seed"] == 20261110


def test_committed_yaml_matches_the_procedure():
    """Il file registrato coincide con quanto produce la procedura (rilevare modifiche a mano)."""
    assert OUT_PATH == Path(__file__).resolve().parents[1] / "configs" / "dev_machines.yaml"
    saved = yaml.safe_load(OUT_PATH.read_text(encoding="utf-8"))
    split = make_dev_split()
    for key in ("seed", "procedure", "dev", "eval"):
        assert saved[key] == split[key], key
