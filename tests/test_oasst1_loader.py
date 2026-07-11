from experiments.config import ResponseSelection
from experiments.oasst1 import build_prompt_response_pairs


def test_loader_selects_best_ranked_direct_assistant_reply() -> None:
    rows = [
        {
            "message_id": "p1",
            "parent_id": None,
            "role": "prompter",
            "lang": "en",
            "text": "Question",
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
