import gzip
import os
from pathlib import Path

from .models import PromptResponsePair


def write_prepared_pairs(path: Path, pairs: list[PromptResponsePair]) -> Path:
    """Atomically materialize the exact dataset extraction used by every task."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(pair.model_dump_json())
            handle.write("\n")
    os.replace(partial, path)
    return path


def load_prepared_pairs(path: Path) -> list[PromptResponsePair]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset artifact not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        pairs = [
            PromptResponsePair.model_validate_json(line)
            for line in handle
            if line.strip()
        ]
    if not pairs:
        raise ValueError(f"Prepared dataset artifact is empty: {path}")
    return pairs
