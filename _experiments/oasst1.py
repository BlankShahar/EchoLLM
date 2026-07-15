import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import DatasetConfig, DatasetSource, ResponseSelection
from .models import PromptResponsePair


def load_oasst1_pairs(config: DatasetConfig) -> list[PromptResponsePair]:
    rows = list(_load_rows(config))
    return build_prompt_response_pairs(
        rows,
        language=config.language,
        exclude_deleted=config.exclude_deleted,
        require_positive_review=config.require_positive_review,
        selection=config.response_selection,
        max_pairs=config.max_pairs,
    )


def build_prompt_response_pairs(
    rows: list[Mapping[str, Any]],
    *,
    language: str | None,
    exclude_deleted: bool = True,
    require_positive_review: bool = True,
    selection: ResponseSelection,
    max_pairs: int | None,
) -> list[PromptResponsePair]:
    indexed_rows = [(index, dict(row)) for index, row in enumerate(rows)]
    prompts = {
        row.get("message_id"): row
        for _, row in indexed_rows
        if row.get("role") == "prompter"
        and _row_is_eligible(
            row,
            language=language,
            exclude_deleted=exclude_deleted,
            require_positive_review=require_positive_review,
        )
    }

    candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in indexed_rows:
        if row.get("role") != "assistant":
            continue
        if not _row_is_eligible(
            row,
            language=language,
            exclude_deleted=exclude_deleted,
            require_positive_review=require_positive_review,
        ):
            continue
        parent_id = row.get("parent_id")
        if parent_id not in prompts:
            continue
        candidates.setdefault(parent_id, []).append((index, row))

    selected: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for prompt_id, replies in candidates.items():
        prompt = prompts[prompt_id]
        ordered = sorted(replies, key=lambda item: _response_sort_key(item[0], item[1]))
        chosen = ordered if selection == ResponseSelection.ALL else ordered[:1]
        selected.extend((index, prompt, response) for index, response in chosen)

    selected.sort(key=lambda item: item[0])
    if max_pairs is not None:
        selected = selected[:max_pairs]

    pairs: list[PromptResponsePair] = []
    for pair_index, (source_index, prompt, response) in enumerate(selected):
        prompt_text = str(prompt.get("text", "")).strip()
        response_text = str(response.get("text", "")).strip()
        if not prompt_text or not response_text:
            continue
        pairs.append(
            PromptResponsePair(
                pair_index=len(pairs),
                prompt_id=str(prompt["message_id"]),
                response_id=str(response["message_id"]),
                message_tree_id=prompt.get("message_tree_id") or response.get("message_tree_id"),
                prompt=prompt_text,
                reference_response=response_text,
                source_index=source_index,
                created_at=prompt.get("created_date"),
            )
        )
    if not pairs:
        raise ValueError("No valid prompt-response pairs were found")
    return pairs


def _row_is_eligible(
    row: Mapping[str, Any],
    *,
    language: str | None,
    exclude_deleted: bool,
    require_positive_review: bool,
) -> bool:
    if language is not None and row.get("lang") != language:
        return False
    if exclude_deleted and row.get("deleted", False):
        return False
    if require_positive_review and row.get("review_result", True) is False:
        return False
    return True


def _response_sort_key(source_index: int, response: Mapping[str, Any]) -> tuple[float, float, int]:
    rank = response.get("rank")
    normalized_rank = float(rank) if rank is not None else float("inf")
    quality = 0.0
    labels = response.get("labels") or {}
    quality_payload = labels.get("quality") if isinstance(labels, Mapping) else None
    if isinstance(quality_payload, Mapping):
        quality = float(quality_payload.get("value") or 0.0)
    return normalized_rank, -quality, source_index


def _load_rows(config: DatasetConfig) -> Iterable[Mapping[str, Any]]:
    if config.source == DatasetSource.HUGGINGFACE:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "The `datasets` package is required. Install requirements-experiments.txt."
            ) from error
        splits = [config.split] if isinstance(config.split, str) else config.split
        for split in splits:
            dataset = load_dataset(config.dataset_name, split=split)
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
