from pathlib import Path

from _experiments.config import ExperimentConfig, ResponseSelection
from _experiments.oasst1 import build_prompt_response_pairs


def test_loader_selects_best_ranked_direct_assistant_reply() -> None:
    rows = [
        {
            "message_id": "p1",
            "parent_id": None,
            "role": "prompter",
            "lang": "en",
            "text": "Question",
            "created_date": "2023-02-01T12:00:00+00:00",
            "review_result": True,
            "deleted": False,
        },
        {
            "message_id": "r2",
            "parent_id": "p1",
            "role": "assistant",
            "lang": "en",
            "text": "Second",
            "rank": 1,
            "review_result": True,
            "deleted": False,
        },
        {
            "message_id": "r1",
            "parent_id": "p1",
            "role": "assistant",
            "lang": "en",
            "text": "Best",
            "rank": 0,
            "review_result": True,
            "deleted": False,
        },
    ]
    pairs = build_prompt_response_pairs(
        rows,
        language="en",
        selection=ResponseSelection.TOP_RANK,
        max_pairs=None,
    )
    assert len(pairs) == 1
    assert pairs[0].reference_response == "Best"
    assert pairs[0].created_at is not None
    assert pairs[0].created_at.isoformat() == "2023-02-01T12:00:00+00:00"


def test_loader_can_include_all_languages_reviews_and_direct_replies() -> None:
    rows = [
        {
            "message_id": "p1",
            "parent_id": None,
            "role": "prompter",
            "lang": "es",
            "text": "Pregunta",
            "created_date": "2023-02-01T12:00:00+00:00",
            "review_result": False,
            "deleted": True,
        },
        {
            "message_id": "r1",
            "parent_id": "p1",
            "role": "assistant",
            "lang": "es",
            "text": "Primera respuesta",
            "rank": 0,
            "review_result": False,
            "deleted": True,
        },
        {
            "message_id": "r2",
            "parent_id": "p1",
            "role": "assistant",
            "lang": "es",
            "text": "Segunda respuesta",
            "rank": 1,
            "review_result": True,
            "deleted": False,
        },
        {
            "message_id": "p2",
            "parent_id": "p1",
            "role": "prompter",
            "lang": "es",
            "text": "This is not an answer",
            "review_result": True,
            "deleted": False,
        },
    ]

    pairs = build_prompt_response_pairs(
        rows,
        language=None,
        exclude_deleted=False,
        require_positive_review=False,
        selection=ResponseSelection.ALL,
        max_pairs=None,
    )

    assert [pair.reference_response for pair in pairs] == [
        "Primera respuesta",
        "Segunda respuesta",
    ]


def test_default_experiment_uses_every_dataset_answer() -> None:
    config = ExperimentConfig.from_yaml(
        Path("_experiments/configs/oasst1_default.yaml")
    )

    assert config.dataset.split == ["train", "validation"]
    assert config.dataset.language is None
    assert not config.dataset.exclude_deleted
    assert not config.dataset.require_positive_review
    assert config.dataset.response_selection == ResponseSelection.ALL
    assert config.trace.request_count is None
