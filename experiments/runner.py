import csv
import gzip
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod
from text_similarity.vector_utils import cosine_distance

from .baselines import BaselineKind, ExactSemanticBaselineCache
from .config import ExperimentConfig
from .embeddings import PrecomputedEmbedder, SentenceTransformerEmbeddingProvider
from .latency import LatencyModel
from .metrics import MetricsAccumulator, RequestObservation, RunSummary
from .models import PromptResponsePair, TraceRequest
from .oasst1 import load_oasst1_pairs
from .plotting import generate_plots
from .trace import build_trace


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        pairs: list[PromptResponsePair],
        prompt_embeddings: np.ndarray,
        response_embeddings: np.ndarray,
    ) -> None:
        if len(pairs) != prompt_embeddings.shape[0] or len(pairs) != response_embeddings.shape[0]:
            raise ValueError("Pairs and embedding matrices must have identical row counts")
        self.config = config
        self.pairs = pairs
        self.prompt_embeddings = np.asarray(prompt_embeddings, dtype=np.float32)
        self.response_embeddings = np.asarray(response_embeddings, dtype=np.float32)
        self._prompt_vectors = {
            pair.prompt: self.prompt_embeddings[index] for index, pair in enumerate(pairs)
        }
        self._response_vectors = {
            pair.reference_response: self.response_embeddings[index]
            for index, pair in enumerate(pairs)
        }
        self._prompt_embedder = PrecomputedEmbedder(self._prompt_vectors)
        self._latency_model = LatencyModel(config.latency)
        self._trace = build_trace(pairs, self.prompt_embeddings, config.trace)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "ExperimentRunner":
        pairs = load_oasst1_pairs(config.dataset)
        prompt_provider = SentenceTransformerEmbeddingProvider(
            config.embedding,
            config.embedding.prompt_model_name,
        )
        quality_provider = SentenceTransformerEmbeddingProvider(
            config.embedding,
            config.embedding.quality_model_name,
        )
        prompt_embeddings = prompt_provider.embed_many([pair.prompt for pair in pairs])
        response_embeddings = quality_provider.embed_many(
            [pair.reference_response for pair in pairs]
        )
        return cls(config, pairs, prompt_embeddings, response_embeddings)

    def run(self) -> Path:
        run_directory = self.config.output.directory / self.config.output.run_name
        raw_directory = run_directory / "raw"
        run_directory.mkdir(parents=True, exist_ok=True)
        if self.config.output.write_raw_results:
            raw_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "config.json").write_text(
            self.config.model_dump_json(indent=2), encoding="utf-8"
        )

        summaries: list[RunSummary] = []
        for cache_size in self.config.policy.cache_sizes:
            for policy_name in self.config.policy.policies:
                summary = self._run_one(policy_name, cache_size, raw_directory)
                summaries.append(summary)

        flat_rows = [summary.flat_dict() for summary in summaries]
        pd.DataFrame(flat_rows).sort_values(["policy", "cache_size"]).to_csv(
            run_directory / "summary.csv", index=False
        )
        (run_directory / "summary.json").write_text(
            json.dumps(flat_rows, indent=2), encoding="utf-8"
        )
        if self.config.output.generate_plots:
            generate_plots(run_directory)
        return run_directory

    def _run_one(self, policy_name: str, cache_size: int, raw_directory: Path) -> RunSummary:
        cache = self._build_cache(policy_name, cache_size)
        accumulator = MetricsAccumulator(self.config.quality.good_hit_distance_thresholds)
        raw_handle = None
        writer = None
        try:
            if self.config.output.write_raw_results:
                raw_path = raw_directory / f"{policy_name.lower()}_cache_{cache_size}.csv.gz"
                raw_handle = gzip.open(raw_path, mode="wt", encoding="utf-8", newline="")
                writer = csv.DictWriter(raw_handle, fieldnames=_RAW_FIELDS)
                writer.writeheader()

            for request in self._trace:
                started = perf_counter_ns()
                lookup = cache.lookup(request.prompt)
                decision_payload: dict[str, object] = {
                    "candidate_admitted": "",
                    "admission_net_delta": "",
                }
                if lookup.hit:
                    returned_response = str(lookup.response)
                    llm_latency_ms = 0.0
                    response_distance = self._response_distance(
                        returned_response, request.reference_response
                    )
                else:
                    returned_response = request.reference_response
                    llm_latency_ms = self._latency_model.estimate_ms(
                        request.prompt, request.reference_response
                    )
                    cache.on_miss(
                        request.prompt,
                        request.reference_response,
                        llm_latency=llm_latency_ms,
                        lookup_context=lookup.context,
                    )
                    response_distance = None
                    if isinstance(cache, SAGESimilarityCache) and cache.last_decision is not None:
                        decision_payload = {
                            "candidate_admitted": cache.last_decision.admitted,
                            "admission_net_delta": cache.last_decision.net_delta,
                        }

                overhead_ms = (perf_counter_ns() - started) / 1_000_000.0
                total_latency_ms = overhead_ms + llm_latency_ms
                measured = request.request_index >= self.config.trace.warmup_requests
                observation = RequestObservation(
                    measured=measured,
                    hit=lookup.hit,
                    response_cosine_distance=response_distance,
                    total_latency_ms=total_latency_ms,
                    policy_overhead_ms=overhead_ms,
                )
                accumulator.record(observation)

                if writer is not None:
                    writer.writerow(
                        {
                            "request_index": request.request_index,
                            "measured": measured,
                            "pair_index": request.pair_index,
                            "prompt_id": request.prompt_id,
                            "response_id": request.response_id,
                            "policy": policy_name,
                            "cache_size": cache_size,
                            "hit": lookup.hit,
                            "prompt_distance": lookup.metadata.get("prompt_distance", ""),
                            "response_cosine_distance": (
                                response_distance if response_distance is not None else ""
                            ),
                            "simulated_llm_latency_ms": llm_latency_ms,
                            "policy_overhead_ms": overhead_ms,
                            "total_latency_ms": total_latency_ms,
                            **decision_payload,
                        }
                    )
        finally:
            if raw_handle is not None:
                raw_handle.close()
            close = getattr(cache, "close", None)
            if callable(close):
                close()

        return accumulator.summary(policy_name, cache_size)

    def _build_cache(self, policy_name: str, cache_size: int):
        common = {
            "max_size": cache_size,
            "hit_distance_threshold": self.config.policy.hit_distance_threshold,
            "prompt_embedder": self._prompt_embedder,
        }
        if policy_name == "SAGE":
            return SAGESimilarityCache(
                **common,
                ranking_distance_method=RankingDistanceMethod.COSINE,
                ghost_capacity=self.config.policy.sage_ghost_capacity,
                decay_half_life_requests=self.config.policy.sage_decay_half_life_requests,
                admission_margin=self.config.policy.sage_admission_margin,
            )
        return ExactSemanticBaselineCache(
            BaselineKind(policy_name),
            **common,
            distance_method=RankingDistanceMethod.COSINE,
            seed=self.config.trace.seed,
        )

    def _response_distance(self, returned: str, reference: str) -> float:
        returned_vector = self._response_vectors.get(returned)
        reference_vector = self._response_vectors.get(reference)
        if returned_vector is None or reference_vector is None:
            raise KeyError("Response embedding missing from the precomputed experiment set")
        return cosine_distance(tuple(returned_vector), tuple(reference_vector))


_RAW_FIELDS = [
    "request_index",
    "measured",
    "pair_index",
    "prompt_id",
    "response_id",
    "policy",
    "cache_size",
    "hit",
    "prompt_distance",
    "response_cosine_distance",
    "simulated_llm_latency_ms",
    "policy_overhead_ms",
    "total_latency_ms",
    "candidate_admitted",
    "admission_net_delta",
]
