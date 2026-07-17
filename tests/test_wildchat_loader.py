from pathlib import Path

from _experiments.config import ExperimentConfig, WildChatDatasetConfig
from _experiments.wildchat import build_wildchat_pairs


def test_loader_flattens_adjacent_user_assistant_turns_without_context() -> None:
    rows = [
        {
            "conversation_hash": "conversation",
            "model": "gpt-4-0314",
            "language": "English",
            "conversation": [
                {
                    "role": "user",
                    "content": "First question",
                    "language": "English",
                    "turn_identifier": 10,
                },
                {
                    "role": "assistant",
                    "content": "First answer",
                    "timestamp": "2023-04-09T00:01:00Z",
                    "turn_identifier": 11,
                },
                {
                    "role": "user",
                    "content": "Follow-up only",
                    "language": "English",
                    "turn_identifier": 12,
                },
                {
                    "role": "assistant",
                    "content": "Second answer",
                    "timestamp": "2023-04-09T00:02:00Z",
                    "turn_identifier": 13,
                },
            ],
        }
    ]

    pairs = build_wildchat_pairs(
        rows,
        language=None,
        source_models=None,
        max_pairs=15_000,
    )

    assert [pair.prompt for pair in pairs] == ["First question", "Follow-up only"]
    assert [pair.reference_response for pair in pairs] == [
        "First answer",
        "Second answer",
    ]
    assert all(pair.source_model == "gpt-4-0314" for pair in pairs)
    assert pairs[0].created_at is not None
    assert pairs[0].created_at.isoformat() == "2023-04-09T00:01:00+00:00"


def test_loader_keeps_globally_earliest_pairs_with_bounded_capacity() -> None:
    rows = [
        _single_turn_conversation("late", "2023-04-09T00:03:00Z"),
        _single_turn_conversation("early", "2023-04-09T00:01:00Z"),
        _single_turn_conversation("middle", "2023-04-09T00:02:00Z"),
    ]

    pairs = build_wildchat_pairs(
        rows,
        language=None,
        source_models=None,
        max_pairs=2,
    )

    assert [pair.prompt for pair in pairs] == ["early prompt", "middle prompt"]
    assert [pair.pair_index for pair in pairs] == [0, 1]


def test_loader_skips_malformed_pairs_and_filters_source_model() -> None:
    missing_timestamp = _single_turn_conversation("missing", None)
    wrong_model = _single_turn_conversation(
        "wrong-model",
        "2023-04-09T00:01:00Z",
        model="gpt-3.5-turbo-0301",
    )
    accepted = _single_turn_conversation(
        "accepted",
        "2023-04-09T00:02:00Z",
        model="gpt-4-0314",
    )

    pairs = build_wildchat_pairs(
        [missing_timestamp, wrong_model, accepted],
        language="English",
        source_models=("gpt-4-0314",),
        max_pairs=15_000,
    )

    assert [pair.prompt for pair in pairs] == ["accepted prompt"]


def test_default_wildchat_config_caps_chronological_requests_at_15k() -> None:
    config = ExperimentConfig.from_yaml(
        Path("_experiments/configs/wildchat_15k.yaml")
    )

    assert isinstance(config.dataset, WildChatDatasetConfig)
    assert config.dataset.dataset_name == "allenai/WildChat-1M"
    assert config.dataset.max_pairs == 15_000
    assert config.dataset.streaming
    assert config.trace.request_count is None


def _single_turn_conversation(
    name: str,
    timestamp: str | None,
    *,
    model: str = "gpt-4-0314",
) -> dict[str, object]:
    return {
        "conversation_hash": name,
        "model": model,
        "language": "English",
        "conversation": [
            {
                "role": "user",
                "content": f"{name} prompt",
                "language": "English",
                "turn_identifier": 1,
            },
            {
                "role": "assistant",
                "content": f"{name} response",
                "timestamp": timestamp,
                "turn_identifier": 2,
            },
        ],
    }
