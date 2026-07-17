from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .config import OASST1DatasetConfig, ResponseSelection
from .datasets import load_dataset_rows
from .models import PromptResponsePair


IndexedRow = tuple[int, dict[str, Any]]
SelectedPair = tuple[int, dict[str, Any], dict[str, Any]]


def load_oasst1_pairs(config: OASST1DatasetConfig) -> list[PromptResponsePair]:
    return build_prompt_response_pairs(
        list(load_dataset_rows(config)),
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
    eligible = [
        (index, row)
        for index, row in indexed_rows
        if row.get("role") in {"prompter", "assistant"}
        and _has_text(row)
        and _row_is_eligible(
            row,
            language=language,
            exclude_deleted=exclude_deleted,
            require_positive_review=require_positive_review,
        )
    ]

    if selection == ResponseSelection.SINGLE_PATH:
        selected = _select_single_paths(eligible)
    else:
        selected = _select_direct_responses(eligible, selection)

    selected.sort(key=lambda item: item[0])
    if max_pairs is not None:
        selected = selected[:max_pairs]

    pairs = [
        PromptResponsePair(
            pair_index=pair_index,
            prompt_id=str(prompt["message_id"]),
            response_id=str(response["message_id"]),
            message_tree_id=prompt.get("message_tree_id")
            or response.get("message_tree_id"),
            prompt=str(prompt["text"]).strip(),
            reference_response=str(response["text"]).strip(),
            source_index=source_index,
            created_at=prompt.get("created_date"),
        )
        for pair_index, (source_index, prompt, response) in enumerate(selected)
    ]
    if not pairs:
        raise ValueError("No valid prompt-response pairs were found")
    return pairs


def _select_single_paths(indexed_rows: list[IndexedRow]) -> list[SelectedPair]:
    children: dict[str, list[IndexedRow]] = {}
    for index, row in indexed_rows:
        parent_id = row.get("parent_id")
        if parent_id is not None:
            children.setdefault(str(parent_id), []).append((index, row))

    roots = [
        (index, row)
        for index, row in indexed_rows
        if row.get("role") == "prompter" and row.get("parent_id") is None
    ]
    roots.sort(key=lambda item: _prompt_sort_key(item[0], item[1]))

    selected: list[SelectedPair] = []
    for root_index, root in roots:
        prompt_index = root_index
        prompt = root
        visited: set[str] = set()
        while True:
            prompt_id = str(prompt["message_id"])
            if prompt_id in visited:
                break
            visited.add(prompt_id)

            replies = [
                item
                for item in children.get(prompt_id, [])
                if item[1].get("role") == "assistant"
            ]
            if not replies:
                break
            _, response = min(
                replies,
                key=lambda item: _response_sort_key(item[0], item[1]),
            )
            selected.append((prompt_index, prompt, response))

            continuations = [
                item
                for item in children.get(str(response["message_id"]), [])
                if item[1].get("role") == "prompter"
                and _has_usable_reply(item[1], children)
            ]
            if not continuations:
                break
            prompt_index, prompt = min(
                continuations,
                key=lambda item: _prompt_sort_key(item[0], item[1]),
            )

    # A malformed source may repeat a message id across roots. Never turn that
    # into repeated requests.
    unique: dict[str, SelectedPair] = {}
    for item in selected:
        unique.setdefault(str(item[1]["message_id"]), item)
    return list(unique.values())


def _select_direct_responses(
    indexed_rows: list[IndexedRow],
    selection: ResponseSelection,
) -> list[SelectedPair]:
    prompts = {
        str(row["message_id"]): row
        for _, row in indexed_rows
        if row.get("role") == "prompter" and row.get("message_id") is not None
    }
    candidates: dict[str, list[IndexedRow]] = {}
    for index, row in indexed_rows:
        parent_id = row.get("parent_id")
        if row.get("role") == "assistant" and str(parent_id) in prompts:
            candidates.setdefault(str(parent_id), []).append((index, row))

    selected: list[SelectedPair] = []
    for prompt_id, replies in candidates.items():
        ordered = sorted(replies, key=lambda item: _response_sort_key(item[0], item[1]))
        chosen = ordered if selection == ResponseSelection.ALL else ordered[:1]
        selected.extend((index, prompts[prompt_id], response) for index, response in chosen)
    return selected


def _has_usable_reply(
    prompt: Mapping[str, Any],
    children: Mapping[str, list[IndexedRow]],
) -> bool:
    return any(
        child.get("role") == "assistant"
        for _, child in children.get(str(prompt["message_id"]), [])
    )


def _has_text(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("text") or "").strip())


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


def _response_sort_key(
    source_index: int,
    response: Mapping[str, Any],
) -> tuple[float, float, float, str, int]:
    rank = _as_float(response.get("rank"), default=float("inf"))
    labels = response.get("labels") or {}
    quality_payload = labels.get("quality") if isinstance(labels, Mapping) else None
    quality = (
        _as_float(quality_payload.get("value"), default=0.0)
        if isinstance(quality_payload, Mapping)
        else 0.0
    )
    return (
        rank,
        -quality,
        _created_timestamp(response),
        str(response.get("message_id") or ""),
        source_index,
    )


def _prompt_sort_key(
    source_index: int,
    prompt: Mapping[str, Any],
) -> tuple[float, str, int]:
    return (
        _created_timestamp(prompt),
        str(prompt.get("message_id") or ""),
        source_index,
    )


def _created_timestamp(row: Mapping[str, Any]) -> float:
    value = row.get("created_date")
    if value is None:
        return float("inf")
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
