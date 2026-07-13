from time import perf_counter, process_time

from pydantic import BaseModel, ConfigDict, Field

from .config import ResourceConfig


class ResourceUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    runner_wall_time_seconds: float = Field(ge=0.0)
    runner_cpu_time_seconds: float = Field(ge=0.0)
    baseline_process_rss_mb: float | None = Field(default=None, ge=0.0)
    peak_process_rss_mb: float | None = Field(default=None, ge=0.0)
    peak_process_rss_delta_mb: float | None = Field(default=None, ge=0.0)
    runner_throughput_qps: float | None = Field(default=None, ge=0.0)


class ResourceTracker:
    """Samples process RSS and measures the measured portion of one policy run."""

    def __init__(self, config: ResourceConfig):
        self._config = config
        self._process = None
        if config.enabled:
            try:
                import psutil
            except ImportError as error:
                raise RuntimeError(
                    "Resource measurement requires psutil; install requirements-_experiments.txt"
                ) from error
            self._process = psutil.Process()
        self._started_wall: float | None = None
        self._started_cpu: float | None = None
        self._baseline_rss_bytes: int | None = None
        self._peak_rss_bytes: int | None = None

    def start(self) -> None:
        if self._started_wall is not None:
            return
        self._started_wall = perf_counter()
        self._started_cpu = process_time()
        self._sample_rss()
        self._baseline_rss_bytes = self._peak_rss_bytes

    def sample(self, measured_request_index: int) -> None:
        if measured_request_index % self._config.sample_interval_requests == 0:
            self._sample_rss()

    def finish(self, measured_requests: int) -> ResourceUsage:
        if self._started_wall is None or self._started_cpu is None:
            raise RuntimeError("ResourceTracker.start() must be called before finish()")
        self._sample_rss()
        wall_seconds = perf_counter() - self._started_wall
        cpu_seconds = process_time() - self._started_cpu
        throughput = measured_requests / wall_seconds if wall_seconds > 0.0 else None
        return ResourceUsage(
            runner_wall_time_seconds=wall_seconds,
            runner_cpu_time_seconds=cpu_seconds,
            baseline_process_rss_mb=(
                self._baseline_rss_bytes / (1024.0 * 1024.0)
                if self._baseline_rss_bytes is not None
                else None
            ),
            peak_process_rss_mb=(
                self._peak_rss_bytes / (1024.0 * 1024.0)
                if self._peak_rss_bytes is not None
                else None
            ),
            peak_process_rss_delta_mb=(
                max(0, self._peak_rss_bytes - self._baseline_rss_bytes) / (1024.0 * 1024.0)
                if self._peak_rss_bytes is not None and self._baseline_rss_bytes is not None
                else None
            ),
            runner_throughput_qps=throughput,
        )

    def _sample_rss(self) -> None:
        if self._process is None:
            return
        rss = int(self._process.memory_info().rss)
        self._peak_rss_bytes = (
            rss if self._peak_rss_bytes is None else max(self._peak_rss_bytes, rss)
        )
