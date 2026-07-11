import numpy as np

from .config import TraceConfig, TraceMode
from .models import PromptResponsePair, TraceRequest


def build_trace(
    pairs: list[PromptResponsePair],
    prompt_embeddings: np.ndarray,
    config: TraceConfig,
) -> list[TraceRequest]:
    if prompt_embeddings.shape[0] != len(pairs):
        raise ValueError("prompt_embeddings must align with pairs")
    rng = np.random.default_rng(config.seed)

    if config.mode == TraceMode.DATASET_ORDER:
        indices = np.arange(config.request_count, dtype=np.int64) % len(pairs)
    elif config.mode == TraceMode.SHUFFLED:
        base = np.arange(len(pairs), dtype=np.int64)
        chunks: list[np.ndarray] = []
        remaining = config.request_count
        while remaining > 0:
            shuffled = rng.permutation(base)
            chunks.append(shuffled[:remaining])
            remaining -= min(remaining, len(base))
        indices = np.concatenate(chunks)
    elif config.mode == TraceMode.ZIPF_CLUSTERED:
        indices = _build_zipf_clustered_indices(prompt_embeddings, config, rng)
    else:
        raise ValueError(f"Unsupported trace mode: {config.mode}")

    trace: list[TraceRequest] = []
    for request_index, pair_index in enumerate(indices.tolist()):
        pair = pairs[int(pair_index)]
        trace.append(
            TraceRequest(
                request_index=request_index,
                pair_index=pair.pair_index,
                prompt_id=pair.prompt_id,
                response_id=pair.response_id,
                prompt=pair.prompt,
                reference_response=pair.reference_response,
            )
        )
    return trace


def _build_zipf_clustered_indices(
    embeddings: np.ndarray,
    config: TraceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for zipf_clustered traces") from error

    normalized = _normalize_rows(embeddings)
    cluster_count = min(config.cluster_count, normalized.shape[0])
    if cluster_count < 2:
        return np.zeros(config.request_count, dtype=np.int64)

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=config.seed,
        batch_size=min(4096, max(256, cluster_count * 4)),
        n_init="auto",
    )
    labels = model.fit_predict(normalized)
    members = [np.flatnonzero(labels == cluster_id) for cluster_id in range(cluster_count)]
    non_empty = [cluster for cluster in members if cluster.size]

    ranks = np.arange(1, len(non_empty) + 1, dtype=np.float64)
    probabilities = np.power(ranks, -config.zipf_alpha)
    probabilities /= probabilities.sum()
    popularity_order = rng.permutation(len(non_empty))
    probabilities = probabilities[popularity_order]
    probabilities /= probabilities.sum()

    selected_clusters = rng.choice(
        len(non_empty),
        size=config.request_count,
        replace=True,
        p=probabilities,
    )
    result = np.empty(config.request_count, dtype=np.int64)
    for index, cluster_index in enumerate(selected_clusters):
        cluster = non_empty[int(cluster_index)]
        result[index] = int(rng.choice(cluster))
    return result


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
