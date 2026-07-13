import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from .config import EmbeddingConfig


class EmbeddingProvider(Protocol):
    def embed_many(self, texts: Sequence[str]) -> np.ndarray: ...


class SQLiteEmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(model, text_hash)
                )
                """
            )

    def get(self, model: str, text: str) -> np.ndarray | None:
        key = _text_hash(text)
        with self._lock:
            row = self._connection.execute(
                "SELECT dimension, vector FROM embeddings WHERE model = ? AND text_hash = ?",
                (model, key),
            ).fetchone()
        if row is None:
            return None
        dimension, blob = row
        return np.frombuffer(blob, dtype=np.float32, count=dimension).copy()

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(vector, dtype=np.float32)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO embeddings(model, text_hash, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                (model, _text_hash(text), contiguous.shape[0], contiguous.tobytes()),
            )


class SentenceTransformerEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self._cache = SQLiteEmbeddingCache(config.cache_path)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "The `sentence-transformers` package is required. "
                "Install requirements-_experiments.txt."
            ) from error
        self._model = SentenceTransformer(model_name, device=config.device)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_positions: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(self.model_name, text)
            if cached is None:
                missing_positions.append(index)
                missing_texts.append(text)
            else:
                results[index] = cached

        if missing_texts:
            vectors = self._model.encode(
                missing_texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True,
            )
            for position, text, vector in zip(missing_positions, missing_texts, vectors, strict=True):
                prepared = np.asarray(vector, dtype=np.float32)
                self._cache.put(self.model_name, text, prepared)
                results[position] = prepared

        if any(vector is None for vector in results):
            raise RuntimeError("Embedding generation did not produce every requested vector")
        return np.vstack(results).astype(np.float32)  # type: ignore[arg-type]

    def __call__(self, text: str) -> list[float]:
        return self.embed_many([text])[0].tolist()


class PrecomputedEmbedder:
    def __init__(
        self,
        vectors_by_text: dict[str, np.ndarray],
        fallback: Callable[[str], Sequence[float]] | None = None,
    ):
        self._vectors = vectors_by_text
        self._fallback = fallback

    def __call__(self, text: str) -> list[float]:
        vector = self._vectors.get(text)
        if vector is not None:
            return vector.tolist()
        if self._fallback is None:
            raise KeyError(f"No precomputed embedding for text hash {_text_hash(text)}")
        return list(self._fallback(text))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
