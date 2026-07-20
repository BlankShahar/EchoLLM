from datetime import datetime, timezone
from pathlib import Path

from _experiments.models import PromptResponsePair
from _experiments.prepared import load_prepared_pairs, write_prepared_pairs


def test_prepared_pairs_round_trip(tmp_path: Path) -> None:
    pair = PromptResponsePair(
        pair_index=0,
        prompt_id="prompt-0",
        response_id="response-0",
        message_tree_id="tree-0",
        prompt="hello",
        reference_response="world",
        source_index=7,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        source_model="model",
    )
    path = tmp_path / "pairs.jsonl.gz"

    write_prepared_pairs(path, [pair])

    assert load_prepared_pairs(path) == [pair]
    assert not path.with_name(f"{path.name}.partial").exists()
