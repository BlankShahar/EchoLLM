import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from .config import ExperimentConfig, LLMProvider
from .datasets import load_prompt_response_pairs
from .embeddings import SentenceTransformerEmbeddingProvider
from .llm import build_llm
from .prepared import write_prepared_pairs
from .recorded_llm import RecordedLLMStore, RecordedResponse
from .runner import format_duration
from .trace import build_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one real LLM response per unique trace prompt"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--ollama-host")
    parser.add_argument("--prepared-pairs-output", type=Path, required=True)
    parser.add_argument("--embedding-cache-path", type=Path, required=True)
    parser.add_argument("--device")
    arguments = parser.parse_args()

    config = ExperimentConfig.from_yaml(arguments.config)
    updates: dict[str, object] = {"provider": LLMProvider.OLLAMA}
    if arguments.model:
        updates["model"] = arguments.model
    if arguments.ollama_host:
        updates["host"] = arguments.ollama_host
    llm_config = config.llm.model_copy(update=updates)

    pairs = load_prompt_response_pairs(config.dataset)
    write_prepared_pairs(arguments.prepared_pairs_output, pairs)
    trace = build_trace(
        pairs,
        np.zeros((len(pairs), 1), dtype=np.float32),
        config.trace,
    )
    prompts = list(dict.fromkeys(request.prompt for request in trace))

    store = RecordedLLMStore(arguments.output, writable=True)
    store.validate_or_initialize(model=llm_config.model, options=llm_config.options)
    missing = [prompt for prompt in prompts if store.get(prompt) is None]
    print(
        f"Recorded backend: {len(prompts)} unique prompts, "
        f"{len(prompts) - len(missing)} already complete, {len(missing)} remaining.",
        flush=True,
    )
    started = perf_counter()
    try:
        if missing:
            backend = build_llm(llm_config)
            for prompt in tqdm(missing, desc="Recording LLM responses", unit="prompt"):
                result = backend.ask(prompt)
                store.put(
                    RecordedResponse(
                        prompt=prompt,
                        response=result.response,
                        latency_ms=result.latency,
                    )
                )
        generated_responses = [
            item.response
            for prompt in prompts
            if (item := store.get(prompt)) is not None
        ]
    finally:
        store.close()
    elapsed = perf_counter() - started
    print(
        f"Recorded {len(missing)} responses in {format_duration(elapsed)}; "
        f"database: {arguments.output}",
        flush=True,
    )

    embedding_config = config.embedding.model_copy(
        update={
            "cache_path": arguments.embedding_cache_path,
            "device": arguments.device or config.embedding.device,
        }
    )
    print("Precomputing shared prompt and response embeddings...", flush=True)
    prompt_provider = SentenceTransformerEmbeddingProvider(
        embedding_config,
        embedding_config.prompt_model_name,
    )
    quality_provider = SentenceTransformerEmbeddingProvider(
        embedding_config,
        embedding_config.quality_model_name,
    )
    prompt_provider.embed_many(list(dict.fromkeys(pair.prompt for pair in pairs)))
    quality_provider.embed_many(
        list(
            dict.fromkeys(
                [pair.reference_response for pair in pairs] + generated_responses
            )
        )
    )
    print(
        f"Prepared pairs: {arguments.prepared_pairs_output}\n"
        f"Shared embeddings: {arguments.embedding_cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
