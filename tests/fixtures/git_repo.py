"""Repository git temporaneo per i test di provenance e store (F2): il repository vero non si tocca mai."""

import subprocess
from pathlib import Path

_GIT_CFG = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
            "-c", "init.defaultBranch=main", "-c", "core.autocrlf=false"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *_GIT_CFG, "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout


def make_repo(path: Path) -> Path:
    """Repository con un commit: ``tracked.txt`` tracciato, ``.gitignore`` che ignora ``ignored/`` e ``store/``."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    (path / ".gitignore").write_text("ignored/\nstore/\n", encoding="utf-8")
    git(path, "add", "tracked.txt", ".gitignore")
    git(path, "commit", "-q", "-m", "init")
    return path
