from pathlib import Path

import numpy as np

from .config import (
    ExperimentConfig,
    LLMConfig,
    OutputConfig,
    PolicyConfig,
    QualityConfig,
    ResourceConfig,
    TraceConfig,
    TraceMode,
)
from .models import PromptResponsePair
from .runner import ExperimentRunner


def build_synthetic_workload(seed: int = 11) -> tuple[list[PromptResponsePair], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = 64
    topic_vectors = np.eye(dimension, dtype=np.float32)[:3]
    pairs: list[PromptResponsePair] = []
    prompt_vectors: list[np.ndarray] = []
    response_vectors: list[np.ndarray] = []

    def add_pair(prompt: str, response: str, prompt_vector: np.ndarray, response_vector: np.ndarray) -> None:
        index = len(pairs)
        pairs.append(
            PromptResponsePair(
                pair_index=index,
                prompt_id=f"p-{index}",
                response_id=f"r-{index}",
                message_tree_id=f"tree-{index}",
                prompt=prompt,
                reference_response=response,
                source_index=index,
            )
        )
        prompt_vectors.append(_normalized(prompt_vector))
        response_vectors.append(_normalized(response_vector))

    noise_basis = np.eye(dimension, dtype=np.float32)[3:]
    noise_cursor = 0
    for round_index in range(40):
        for topic_index, topic_name in enumerate(("password", "refund", "export")):
            prompt_noise = rng.normal(0.0, 0.015, size=dimension).astype(np.float32)
            response_noise = rng.normal(0.0, 0.01, size=dimension).astype(np.float32)
            add_pair(
                prompt=f"{topic_name} request variant {round_index}",
                response=f"canonical {topic_name} answer {round_index % 3}",
                prompt_vector=topic_vectors[topic_index] + prompt_noise,
                response_vector=topic_vectors[topic_index] + response_noise,
            )

            # One-hit noise creates cache pollution for unconditional-admission policies.
            basis = noise_basis[noise_cursor % len(noise_basis)]
            noise_cursor += 1
            add_pair(
                prompt=f"one-off unrelated request {round_index}-{topic_index}",
                response=f"one-off unrelated answer {round_index}-{topic_index}",
                prompt_vector=basis,
                response_vector=basis,
            )

    return pairs, np.vstack(prompt_vectors), np.vstack(response_vectors)


def main() -> None:
    pairs, prompt_embeddings, response_embeddings = build_synthetic_workload()
    config = ExperimentConfig(
        llm=LLMConfig(model="reference"),
        trace=TraceConfig(
            mode=TraceMode.DATASET_ORDER,
            request_count=len(pairs),
            warmup_requests=24,
            seed=11,
        ),
        policy=PolicyConfig(
            policies=["LRU", "LFU", "FIFO", "RR", "SAGE", "SPARQ"],
            cache_sizes=[3, 5, 8],
            include_unbounded_cache=False,
            hit_distance_threshold=0.08,
            sage_ghost_capacity=128,
            sage_decay_half_life_requests=96,
        ),
        quality=QualityConfig(good_hit_distance_thresholds=[0.05, 0.1, 0.2]),
        resources=ResourceConfig(enabled=False),
        output=OutputConfig(
            directory=Path("results"),
            run_name="smoke",
            write_raw_results=True,
            generate_plots=True,
        ),
    )
    output = ExperimentRunner(config, pairs, prompt_embeddings, response_embeddings).run()
    print(output)


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


if __name__ == "__main__":
    main()
