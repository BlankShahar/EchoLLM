import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from .artifacts import EmbeddedRequestRecord
from .storage_client import StorageClient


class FaissClient(StorageClient):
    """
    Simple FAISS vector store with file persistence.
    - Uses cosine similarity by normalizing vectors and using IndexFlatIP.
    - Persists the FAISS index to `index_path`.
    - Persists lightweight metadata (keys <-> ids, original vectors, dim) to `index_path.with_suffix('.meta.json')`.
    """

    def __init__(self, index_path: Path = Path('resources/requests.db')):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_path.with_suffix('.meta.json')

        self.index: faiss.Index | None = None  # IndexIDMap2(IndexFlatIP)
        self.dim: int | None = None

        # key -> {"id": int, "vector": list[float]}
        self._items: dict[str, dict] = {}
        # id -> key
        self._id_to_key: dict[int, str] = {}

        self._load()

    def fetch(self, key: str) -> EmbeddedRequestRecord:
        item = self._items.get(key)
        if not item:
            raise KeyError(f"No vector with key '{key}'")
        return EmbeddedRequestRecord(vector=item["vector"])

    def fetch_k_closest(self, vector: list[float], k: int = 100) -> list[EmbeddedRequestRecord]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q = self._normalize(vector)
        if self.dim is None or len(q) != self.dim:
            raise ValueError(f"Query dim {len(q)} != index dim {self.dim}")

        xq = np.asarray([q], dtype='float32')
        k_eff = min(k, self.index.ntotal)
        _, labels = self.index.search(xq, k_eff)  # type: ignore[call-arg]

        results: list[EmbeddedRequestRecord] = []
        for lid in labels[0]:
            if lid == -1:
                continue
            key = self._id_to_key.get(int(lid))
            if not key:
                continue
            vec = self._items[key]["vector"]
            results.append(EmbeddedRequestRecord(vector=vec))
        return results

    def save(self, request: EmbeddedRequestRecord) -> str:
        vec = self._normalize(request.vector)

        if self.dim is None:
            self._new_index(len(vec))
        elif len(vec) != self.dim:
            raise ValueError(f"Vector dim {len(vec)} != index dim {self.dim}")

        key = request.key
        if key in self._items:
            # Idempotent: already stored; no re-add to avoid duplicates.
            return key

        idx = self._id_from_key(key)

        xb = np.asarray([vec], dtype='float32')
        ids = np.asarray([idx], dtype='int64')
        self.index.add_with_ids(xb, ids)  # type: ignore[arg-type]

        self._items[key] = {"id": int(idx), "vector": vec}
        self._id_to_key[int(idx)] = key

        self._persist()
        return key

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        arr = np.asarray(vector, dtype='float32')
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr.tolist()  # leave as-is; will behave like zero-similarity
        return (arr / norm).tolist()

    def _new_index(self, dim: int) -> None:
        base = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexIDMap2(base)  # attach external int64 IDs
        self.dim = dim

    @staticmethod
    def _id_from_key(key: str) -> int:
        # Stable positive 63-bit integer from key
        h = hashlib.md5(key.encode()).digest()
        return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)

    def _load(self) -> None:
        # metadata
        if self.meta_path.exists():
            with self.meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            self.dim = meta.get("dim")
            self._items = meta.get("items", {})
            self._id_to_key = {int(v["id"]): k for k, v in self._items.items()}

        # index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            # Will be created lazily on first save once we know dim
            self.index = None

    def _persist(self) -> None:
        # persist index
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        # persist metadata
        meta = {"dim": self.dim, "items": self._items}
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
