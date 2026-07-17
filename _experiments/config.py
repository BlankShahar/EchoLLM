from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSource(StrEnum):
    HUGGINGFACE = "huggingface"
    JSONL = "jsonl"
    PARQUET = "parquet"


class DatasetKind(StrEnum):
    OASST1 = "oasst1"
    WILDCHAT = "wildchat"


class ResponseSelection(StrEnum):
    SINGLE_PATH = "single_path"
    TOP_RANK = "top_rank"
    ALL = "all"


class TraceMode(StrEnum):
    DATASET_ORDER = "dataset_order"
    CHRONOLOGICAL = "chronological"
    SHUFFLED = "shuffled"
    ZIPF_CLUSTERED = "zipf_clustered"


class LLMProvider(StrEnum):
    OLLAMA = "ollama"


class BaseDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: DatasetSource = DatasetSource.HUGGINGFACE
    dataset_name: str
    split: str | list[str]
    local_path: Path | None = None
    language: str | None = None
    max_pairs: int | None = Field(default=None, gt=0)
    streaming: bool = False

    @model_validator(mode="after")
    def validate_local_source(self) -> Self:
        if self.source != DatasetSource.HUGGINGFACE and self.local_path is None:
            raise ValueError("local_path is required for jsonl/parquet datasets")
        splits = [self.split] if isinstance(self.split, str) else self.split
        if not splits or any(not split.strip() for split in splits):
            raise ValueError("split must contain at least one non-empty split name")
        return self


class OASST1DatasetConfig(BaseDatasetConfig):
    kind: Literal[DatasetKind.OASST1] = DatasetKind.OASST1
    dataset_name: str = "OpenAssistant/oasst1"
    split: str | list[str] = Field(default_factory=lambda: ["train", "validation"])
    exclude_deleted: bool = False
    require_positive_review: bool = False
    response_selection: ResponseSelection = ResponseSelection.SINGLE_PATH


class WildChatDatasetConfig(BaseDatasetConfig):
    kind: Literal[DatasetKind.WILDCHAT] = DatasetKind.WILDCHAT
    dataset_name: str = "allenai/WildChat-1M"
    split: str | list[str] = "train"
    max_pairs: int | None = Field(default=50_000, gt=0)
    streaming: bool = True
    source_models: tuple[str, ...] | None = None


DatasetConfig = Annotated[
    OASST1DatasetConfig | WildChatDatasetConfig,
    Field(discriminator="kind"),
]


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    quality_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    batch_size: int = Field(default=128, gt=0)
    device: str | None = None
    cache_path: Path = Path(".cache/echollm_embeddings.sqlite3")


class TraceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: TraceMode = TraceMode.CHRONOLOGICAL
    request_count: int | None = Field(default=None, gt=0)
    warmup_requests: int = Field(default=0, ge=0)
    seed: int = 7
    zipf_alpha: float = Field(default=1.1, gt=0.0)
    cluster_count: int = Field(default=1000, gt=1)

    @model_validator(mode="after")
    def validate_warmup(self) -> "TraceConfig":
        if self.request_count is not None and self.warmup_requests >= self.request_count:
            raise ValueError("warmup_requests must be smaller than request_count")
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = Field(default="qwen3:8b", min_length=1)
    host: str = Field(default="http://127.0.0.1:11434", min_length=1)
    options: dict[str, Any] = Field(default_factory=lambda: {"num_predict": 256})


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    policies: list[str] = Field(default_factory=lambda: ["LRU", "LFU", "FIFO", "RR", "SAGE"])
    cache_sizes: list[int] = Field(
        default_factory=lambda: [
            0,
            50,
            100,
            250,
            500,
            1000,
            2500,
            5000,
            10000,
            20000,
        ]
    )
    include_unbounded_cache: bool = True
    hit_distance_threshold: float = Field(default=0.25, ge=0.0)
    sage_ghost_capacity: int | None = Field(default=None, gt=0)
    sage_decay_half_life_requests: float | None = Field(default=512, gt=0.0)
    sage_admission_margin: float = Field(default=0.0, ge=0.0)
    sage_current_request_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    sage_window_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    sage_soft_coverage: bool = True
    sage_soft_coverage_power: float = Field(default=1.0, gt=0.0)
    sage_recent_history_multiplier: float = Field(default=4.0, gt=0.0)
    sage_recent_history_limit: int = Field(default=4096, gt=0)
    sage_long_history_capacity: int | None = Field(default=None, gt=0)
    sage_long_history_multiplier: float = Field(default=8.0, gt=0.0)
    sage_long_history_limit: int = Field(default=8192, gt=0)
    sage_long_sample_stride: int = Field(default=8, gt=0)
    sage_recent_evidence_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sage_long_decay_half_life_requests: float | None = Field(default=32768, gt=0.0)

    @model_validator(mode="after")
    def validate_policy_values(self) -> "PolicyConfig":
        valid = {"LRU", "LFU", "FIFO", "RR", "SAGE"}
        unknown = set(self.policies) - valid
        if unknown:
            raise ValueError(f"Unsupported policies: {sorted(unknown)}")
        if not self.cache_sizes or any(size < 0 for size in self.cache_sizes):
            raise ValueError("cache_sizes must contain non-negative integers")
        return self


class QualityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    good_hit_distance_thresholds: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: Path = Path("results")
    run_name: str = "oasst1_all_sage"
    write_raw_results: bool = True
    generate_plots: bool = True


class ResourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    sample_interval_requests: int = Field(default=100, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: DatasetConfig = Field(default_factory=OASST1DatasetConfig)
    embedding: EmbeddingConfig = EmbeddingConfig()
    trace: TraceConfig = TraceConfig()
    llm: LLMConfig = LLMConfig()
    policy: PolicyConfig = PolicyConfig()
    quality: QualityConfig = QualityConfig()
    resources: ResourceConfig = ResourceConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)
