from pathlib import Path

from _experiments.config import (
    ExperimentConfig,
    OASST1DatasetConfig,
    ResponseSelection,
)
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


def test_loader_selects_one_deterministic_path_per_tree() -> None:
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "message_tree_id": "tree",
            "role": "prompter",
            "text": "Root question",
            "created_date": "2023-02-01T00:00:00+00:00",
        },
        {
            "message_id": "answer-other",
            "parent_id": "root",
            "message_tree_id": "tree",
            "role": "assistant",
            "text": "Other answer",
            "rank": 1,
        },
        {
            "message_id": "answer-best",
            "parent_id": "root",
            "message_tree_id": "tree",
            "role": "assistant",
            "text": "Best answer",
            "rank": 0,
        },
        {
            "message_id": "follow-late",
            "parent_id": "answer-best",
            "message_tree_id": "tree",
            "role": "prompter",
            "text": "Late branch",
            "created_date": "2023-02-03T00:00:00+00:00",
        },
        {
            "message_id": "late-answer",
            "parent_id": "follow-late",
            "message_tree_id": "tree",
            "role": "assistant",
            "text": "Late branch answer",
            "rank": 0,
        },
        {
            "message_id": "follow-early",
            "parent_id": "answer-best",
            "message_tree_id": "tree",
            "role": "prompter",
            "text": "Early branch",
            "created_date": "2023-02-02T00:00:00+00:00",
        },
        {
            "message_id": "early-answer-other",
            "parent_id": "follow-early",
            "message_tree_id": "tree",
            "role": "assistant",
            "text": "Other follow-up answer",
            "rank": 1,
        },
        {
            "message_id": "early-answer-best",
            "parent_id": "follow-early",
            "message_tree_id": "tree",
            "role": "assistant",
            "text": "Best follow-up answer",
            "rank": 0,
        },
    ]

    pairs = build_prompt_response_pairs(
        rows,
        language=None,
        exclude_deleted=False,
        require_positive_review=False,
        selection=ResponseSelection.SINGLE_PATH,
        max_pairs=None,
    )

    assert [pair.prompt_id for pair in pairs] == ["root", "follow-early"]
    assert [pair.response_id for pair in pairs] == [
        "answer-best",
        "early-answer-best",
    ]
    assert len({pair.prompt_id for pair in pairs}) == len(pairs)


def test_single_path_ignores_empty_top_ranked_response() -> None:
    rows = [
        {
            "message_id": "p1",
            "parent_id": None,
            "role": "prompter",
            "text": "Question",
        },
        {
            "message_id": "empty",
            "parent_id": "p1",
            "role": "assistant",
            "text": "  ",
            "rank": 0,
        },
        {
            "message_id": "usable",
            "parent_id": "p1",
            "role": "assistant",
            "text": "Usable",
            "rank": 1,
        },
    ]

    pairs = build_prompt_response_pairs(
        rows,
        language=None,
        exclude_deleted=False,
        require_positive_review=False,
        selection=ResponseSelection.SINGLE_PATH,
        max_pairs=None,
    )

    assert [pair.response_id for pair in pairs] == ["usable"]


def test_default_experiment_uses_one_path_per_conversation_tree() -> None:
    config = ExperimentConfig.from_yaml(
        Path("_experiments/configs/oasst1_default.yaml")
    )

    assert isinstance(config.dataset, OASST1DatasetConfig)
    assert config.dataset.split == ["train", "validation"]
    assert config.dataset.language is None
    assert not config.dataset.exclude_deleted
    assert not config.dataset.require_positive_review
    assert config.dataset.response_selection == ResponseSelection.SINGLE_PATH
    assert config.trace.request_count is None
    assert config.policy.sage_window_fraction == 0.05
    # The policy default remains soft coverage, while the hit-rate experiment
    # explicitly selects binary coverage after the trace replay ablation.
    assert config.policy.sage_soft_coverage is False
    assert config.policy.sage_ghost_capacity is None
