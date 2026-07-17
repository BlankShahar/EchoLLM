import heapq
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from .config import WildChatDatasetConfig
from .datasets import load_dataset_rows
from .models import PromptResponsePair


HeapEntry = tuple[float, int, PromptResponsePair]


def load_wildchat_pairs(config: WildChatDatasetConfig) -> list[PromptResponsePair]:
    rows = tqdm(
        load_dataset_rows(config),
        desc="Scanning WildChat conversations",
        unit="conversation",
    )
    return build_wildchat_pairs(
        rows,
        language=config.language,
        source_models=config.source_models,
        max_pairs=config.max_pairs,
    )


def build_wildchat_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    language: str | None,
    source_models: tuple[str, ...] | None,
    max_pairs: int | None,
) -> list[PromptResponsePair]:
    retained: list[HeapEntry] = []
    source_index = 0
    allowed_models = set(source_models) if source_models is not None else None

    for row_index, source_row in enumerate(rows):
        row = dict(source_row)
        source_model = str(row.get("model") or "").strip() or None
        if allowed_models is not None and source_model not in allowed_models:
            continue
        conversation = row.get("conversation")
        if not isinstance(conversation, list):
            continue

        conversation_id = _conversation_id(row, row_index)
        for turn_index in range(len(conversation) - 1):
            user_turn = conversation[turn_index]
            assistant_turn = conversation[turn_index + 1]
            if not isinstance(user_turn, Mapping) or not isinstance(
                assistant_turn, Mapping
            ):
                continue
            if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
                continue
            if language is not None and (
                user_turn.get("language") or row.get("language")
            ) != language:
                continue

            prompt = str(user_turn.get("content") or "").strip()
            response = str(assistant_turn.get("content") or "").strip()
            created_at = _parse_timestamp(assistant_turn.get("timestamp"))
            if not prompt or not response or created_at is None:
                continue

            pair = PromptResponsePair(
                pair_index=source_index,
                prompt_id=_turn_id(conversation_id, "user", user_turn, turn_index),
                response_id=_turn_id(
                    conversation_id,
                    "assistant",
                    assistant_turn,
                    turn_index + 1,
                ),
                message_tree_id=conversation_id,
                prompt=prompt,
                reference_response=response,
                source_index=source_index,
                created_at=created_at,
                source_model=source_model,
            )
            _retain_earliest(retained, pair, max_pairs)
            source_index += 1

    if not retained:
        raise ValueError("No valid WildChat prompt-response pairs were found")

    chronological = sorted(
        (entry[2] for entry in retained),
        key=lambda pair: (
            pair.created_at,
            pair.source_index,
            pair.prompt_id,
        ),
    )
    return [
        pair.model_copy(update={"pair_index": pair_index})
        for pair_index, pair in enumerate(chronological)
    ]


def _retain_earliest(
    retained: list[HeapEntry],
    pair: PromptResponsePair,
    max_pairs: int | None,
) -> None:
    assert pair.created_at is not None
    entry = (-pair.created_at.timestamp(), -pair.source_index, pair)
    if max_pairs is None or len(retained) < max_pairs:
        heapq.heappush(retained, entry)
        return

    latest_retained = (-retained[0][0], -retained[0][1])
    candidate = (pair.created_at.timestamp(), pair.source_index)
    if candidate < latest_retained:
        heapq.heapreplace(retained, entry)


def _conversation_id(row: Mapping[str, Any], row_index: int) -> str:
    conversation_hash = str(row.get("conversation_hash") or "conversation")
    # The dataset card explicitly notes that conversation_hash is not unique.
    return f"{conversation_hash}:{row_index}"


def _turn_id(
    conversation_id: str,
    role: str,
    turn: Mapping[str, Any],
    turn_index: int,
) -> str:
    identifier = turn.get("turn_identifier")
    suffix = identifier if identifier is not None else turn_index
    return f"{conversation_id}:{role}:{suffix}"


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
