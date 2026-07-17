import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import BaseDatasetConfig, DatasetConfig, DatasetKind, DatasetSource
from .models import PromptResponsePair


def load_prompt_response_pairs(config: DatasetConfig) -> list[PromptResponsePair]:
    if config.kind == DatasetKind.OASST1:
        from .oasst1 import load_oasst1_pairs

        return load_oasst1_pairs(config)
    if config.kind == DatasetKind.WILDCHAT:
        from .wildchat import load_wildchat_pairs

        return load_wildchat_pairs(config)
    raise ValueError(f"Unsupported dataset kind: {config.kind}")


def load_dataset_rows(config: BaseDatasetConfig) -> Iterable[Mapping[str, Any]]:
    if config.source == DatasetSource.HUGGINGFACE:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "The `datasets` package is required. Install requirements-experiments.txt."
            ) from error
        splits = [config.split] if isinstance(config.split, str) else config.split
        for split in splits:
            dataset = load_dataset(
                config.dataset_name,
                split=split,
                streaming=config.streaming,
            )
            yield from dataset
        return

    assert config.local_path is not None
    if config.source == DatasetSource.JSONL:
        with Path(config.local_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    if config.source == DatasetSource.PARQUET:
        import pandas as pd

        dataframe = pd.read_parquet(config.local_path)
        yield from dataframe.to_dict(orient="records")
        return

    raise ValueError(f"Unsupported dataset source: {config.source}")
