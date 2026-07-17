import csv
import gzip
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from cache import ICache
from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod
from echollm import EchoLLM
from llm import ILLM
from text_similarity.vector_utils import cosine_distance

from .baselines import BaselineKind, ExactSemanticBaselineCache
from .config import ExperimentConfig
from .datasets import load_prompt_response_pairs
from .embeddings import (
    EmbeddingProvider,
    PrecomputedEmbedder,
    SentenceTransformerEmbeddingProvider,
)
from .llm import MemoizedLLM, ReferenceLLM, build_llm
from .metrics import MetricsAccumulator, RequestObservation, RunSummary
from .models import PromptResponsePair, TraceRequest
from .plotting import generate_plots
from .resources import ResourceTracker, ResourceUsage
from .trace import build_trace


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        pairs: list[PromptResponsePair],
        prompt_embeddings: np.ndarray,
        response_embeddings: np.ndarray,
        *,
        llm: ILLM | None = None,
        quality_provider: EmbeddingProvider | None = None,
    ) -> None:
        if (
            len(pairs) != prompt_embeddings.shape[0]
            or len(pairs) != response_embeddings.shape[0]
        ):
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
        backend = llm or ReferenceLLM(
            {pair.prompt: pair.reference_response for pair in pairs}
        )
        self._llm = MemoizedLLM(backend)
        self._quality_provider = quality_provider
        self._trace = build_trace(pairs, self.prompt_embeddings, config.trace)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "ExperimentRunner":
        pairs = load_prompt_response_pairs(config.dataset)
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
        return cls(
            config,
            pairs,
            prompt_embeddings,
            response_embeddings,
            llm=build_llm(config.llm),
            quality_provider=quality_provider,
        )

    def run(self) -> Path:
        run_directory = self.config.output.directory / self.config.output.run_name
        raw_directory = run_directory / "raw"
        run_directory.mkdir(parents=True, exist_ok=True)
        if self.config.output.write_raw_results:
            raw_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "config.json").write_text(
            self.config.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_directory / "dataset_stats.json").write_text(
            json.dumps(
                {
                    "prompt_response_pairs": len(self.pairs),
                    "trace_requests": len(self._trace),
                    "unique_prompt_ids": len({pair.prompt_id for pair in self.pairs}),
                    "unique_prompt_strings": len({pair.prompt for pair in self.pairs}),
                    "unique_response_ids": len({pair.response_id for pair in self.pairs}),
                    "unique_response_strings": len(
                        {pair.reference_response for pair in self.pairs}
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Dataset: {len(self.pairs)} prompt-response edges, "
            f"{len(self._trace)} trace requests, "
            f"{len({request.prompt for request in self._trace})} unique prompt strings",
            flush=True,
        )
        self._prepare_llm_responses()

        summaries: list[RunSummary] = []
        capacity_runs = self._capacity_runs()
        total_runs = len(capacity_runs) * len(self.config.policy.policies)
        run_index = 0
        for cache_size, capacity_mode in capacity_runs:
            for policy_name in self.config.policy.policies:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] Running {policy_name} "
                    f"at capacity {cache_size} ({capacity_mode})...",
                    flush=True,
                )
                experiment_started = perf_counter()
                summary = self._run_one(
                    policy_name,
                    cache_size,
                    capacity_mode,
                    raw_directory,
                )
                summaries.append(summary)
                self._write_summaries(run_directory, summaries)
                experiment_seconds = perf_counter() - experiment_started
                print(
                    f"[{run_index}/{total_runs}] {policy_name}: "
                    f"hit_rate={summary.hit_rate:.4f}, "
                    f"semantic_accuracy={_format_optional(summary.mean_hit_semantic_accuracy)}, "
                    f"mean_latency_ms={summary.mean_latency_ms:.2f}; "
                    f"completed in {format_duration(experiment_seconds)}",
                    flush=True,
                )

        if self.config.output.generate_plots:
            generate_plots(run_directory)
        return run_directory

    @staticmethod
    def _write_summaries(run_directory: Path, summaries: list[RunSummary]) -> None:
        flat_rows = [summary.flat_dict() for summary in summaries]
        pd.DataFrame(flat_rows).sort_values(["policy", "cache_size"]).to_csv(
            run_directory / "summary.csv", index=False
        )
        (run_directory / "summary.json").write_text(
            json.dumps(flat_rows, indent=2), encoding="utf-8"
        )

    def _run_one(
        self,
        policy_name: str,
        cache_size: int,
        capacity_mode: str,
        raw_directory: Path,
    ) -> RunSummary:
        cache = self._build_cache(policy_name, cache_size)
        echo_llm = EchoLLM(cache=cache, llm=self._llm)
        accumulator = MetricsAccumulator(self.config.quality.good_hit_distance_thresholds)
        resource_tracker = ResourceTracker(self.config.resources)
        resource_usage: ResourceUsage | None = None
        raw_handle = None
        writer = None
        raw_path: Path | None = None
        partial_raw_path: Path | None = None
        try:
            if self.config.output.write_raw_results:
                capacity_label = (
                    f"unbounded_{cache_size}"
                    if capacity_mode == "unbounded"
                    else str(cache_size)
                )
                raw_path = (
                    raw_directory
                    / f"{policy_name.lower()}_cache_{capacity_label}.csv.gz"
                )
                partial_raw_path = Path(f"{raw_path}.partial")
                partial_raw_path.unlink(missing_ok=True)
                raw_handle = gzip.open(
                    partial_raw_path,
                    mode="wt",
                    encoding="utf-8",
                    newline="",
                )
                writer = csv.DictWriter(raw_handle, fieldnames=_RAW_FIELDS)
                writer.writeheader()

            progress_label = f"{policy_name} cache={cache_size} ({capacity_mode})"
            with tqdm(
                self._trace,
                total=len(self._trace),
                desc=progress_label,
                unit="request",
                mininterval=5.0,
                dynamic_ncols=True,
                file=sys.stdout,
            ) as requests:
                for request in requests:
                    measured = request.request_index >= self.config.trace.warmup_requests
                    if measured:
                        resource_tracker.start()

                    result = echo_llm.ask_with_metadata(request.prompt)
                    policy_overhead_ms = result.cache_latency
                    decision_payload: dict[str, object] = {
                        "candidate_admitted": "",
                        "admission_net_delta": "",
                        "incoming_admitted": "",
                        "promoted": "",
                    }
                    response_distance = self._response_distance(
                        result.response, request.reference_response
                    )
                    if not result.cache_hit:
                        if (
                            isinstance(cache, SAGESimilarityCache)
                            and cache.last_decision is not None
                        ):
                            decision_payload = {
                                "candidate_admitted": cache.last_decision.admitted,
                                "admission_net_delta": cache.last_decision.net_delta,
                                "incoming_admitted": _optional_value(
                                    cache.last_decision.incoming_admitted
                                ),
                                "promoted": _optional_value(
                                    cache.last_decision.promoted
                                ),
                            }

                    total_latency_ms = policy_overhead_ms + result.llm_latency
                    observation = RequestObservation(
                        measured=measured,
                        hit=result.cache_hit,
                        response_cosine_distance=response_distance,
                        total_latency_ms=total_latency_ms,
                        policy_overhead_ms=policy_overhead_ms,
                    )
                    accumulator.record(observation)
                    if measured:
                        measured_index = (
                            request.request_index
                            - self.config.trace.warmup_requests
                            + 1
                        )
                        resource_tracker.sample(measured_index)

                    if writer is not None:
                        writer.writerow(
                            {
                                "request_index": request.request_index,
                                "measured": measured,
                                "created_at": (
                                    request.created_at.isoformat()
                                    if request.created_at is not None
                                    else ""
                                ),
                                "pair_index": request.pair_index,
                                "prompt_id": request.prompt_id,
                                "response_id": request.response_id,
                                "policy": policy_name,
                                "cache_size": cache_size,
                                "capacity_mode": capacity_mode,
                                "llm_model": self.config.llm.model,
                                "source_model": request.source_model or "",
                                "hit": result.cache_hit,
                                "prompt_distance": result.cache_metadata.get(
                                    "prompt_distance", ""
                                ),
                                "response_cosine_distance": response_distance,
                                "backend_latency_ms": result.llm_latency,
                                "policy_overhead_ms": policy_overhead_ms,
                                "total_latency_ms": total_latency_ms,
                                **decision_payload,
                            }
                        )
            resource_usage = resource_tracker.finish(
                len(self._trace) - self.config.trace.warmup_requests
            )
        finally:
            if raw_handle is not None:
                raw_handle.close()
            close = getattr(cache, "close", None)
            if callable(close):
                close()

        if raw_path is not None and partial_raw_path is not None:
            partial_raw_path.replace(raw_path)

        return accumulator.summary(
            policy_name,
            cache_size,
            capacity_mode=capacity_mode,
            llm_model=self.config.llm.model,
            resource_usage=resource_usage,
        )

    def _build_cache(self, policy_name: str, cache_size: int) -> ICache | None:
        if cache_size == 0:
            return None
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
                current_request_weight=self.config.policy.sage_current_request_weight,
                window_fraction=self.config.policy.sage_window_fraction,
                soft_coverage=self.config.policy.sage_soft_coverage,
                soft_coverage_power=self.config.policy.sage_soft_coverage_power,
                recent_history_multiplier=(
                    self.config.policy.sage_recent_history_multiplier
                ),
                recent_history_limit=self.config.policy.sage_recent_history_limit,
                long_history_capacity=self.config.policy.sage_long_history_capacity,
                long_history_multiplier=(
                    self.config.policy.sage_long_history_multiplier
                ),
                long_history_limit=self.config.policy.sage_long_history_limit,
                long_sample_stride=self.config.policy.sage_long_sample_stride,
                recent_evidence_weight=(
                    self.config.policy.sage_recent_evidence_weight
                ),
                long_decay_half_life_requests=(
                    self.config.policy.sage_long_decay_half_life_requests
                ),
            )
        return ExactSemanticBaselineCache(
            BaselineKind(policy_name),
            **common,
            distance_method=RankingDistanceMethod.COSINE,
            seed=self.config.trace.seed,
        )

    def _capacity_runs(self) -> list[tuple[int, str]]:
        runs: list[tuple[int, str]] = []
        seen: set[int] = set()
        for cache_size in self.config.policy.cache_sizes:
            if cache_size in seen:
                continue
            seen.add(cache_size)
            runs.append((cache_size, "no_cache" if cache_size == 0 else "bounded"))

        if self.config.policy.include_unbounded_cache:
            unbounded_size = len({request.prompt for request in self._trace})
            if unbounded_size in seen:
                runs = [
                    (size, "unbounded" if size == unbounded_size else mode)
                    for size, mode in runs
                ]
            else:
                runs.append((unbounded_size, "unbounded"))
        return runs

    def _response_distance(self, returned: str, reference: str) -> float:
        returned_vector = self._response_vectors.get(returned)
        reference_vector = self._response_vectors.get(reference)
        if returned_vector is None or reference_vector is None:
            raise KeyError("Response embedding missing from the precomputed experiment set")
        return cosine_distance(tuple(returned_vector), tuple(reference_vector))

    def _prepare_llm_responses(self) -> None:
        self._llm.prime(request.prompt for request in self._trace)
        generated = [response.response for response in self._llm.responses.values()]
        missing = list(
            dict.fromkeys(
                response
                for response in generated
                if response not in self._response_vectors
            )
        )
        if not missing:
            return
        if self._quality_provider is None:
            raise KeyError("Generated responses require a quality embedding provider")
        vectors = self._quality_provider.embed_many(missing)
        self._response_vectors.update(zip(missing, vectors, strict=True))


_RAW_FIELDS = [
    "request_index",
    "measured",
    "created_at",
    "pair_index",
    "prompt_id",
    "response_id",
    "policy",
    "cache_size",
    "capacity_mode",
    "llm_model",
    "source_model",
    "hit",
    "prompt_distance",
    "response_cosine_distance",
    "backend_latency_ms",
    "policy_overhead_ms",
    "total_latency_ms",
    "candidate_admitted",
    "admission_net_delta",
    "incoming_admitted",
    "promoted",
]


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _optional_value(value: object | None) -> object:
    return "" if value is None else value


def format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
