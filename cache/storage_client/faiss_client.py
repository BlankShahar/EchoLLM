import hashlib
import json
from enum import Enum
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel


class FaissVector(BaseModel):
    id: int
    vector: list[float]


class DistanceMethod(Enum):
    COSINE = "cosine"  # cosine similarity - normalize vectors, use IP
    INNER_PRODUCT = "ip"  # raw inner product
    L2 = "l2"  # squared Euclidean


class FaissClient:
    def __init__(self, distance_method: DistanceMethod, index_path: Path = Path('resources/requests.db')):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.distance_method = distance_method

        # lazy-initialized attributes
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.meta_path = self.index_path.with_suffix('.meta.json')
        self._items: dict[str, FaissVector] = {}  # key -> FaissVector
        self._id_to_key: dict[int, str] = {}  # id -> key

        self._load()

    # ---------- Public API ----------

    def fetch_k_closest(self, vector: list[float], k: int = 100) -> list[list[float]]:
        if self.index is None or self.index.ntotal == 0 or k <= 0:
            return []

        arr = np.asarray(vector, dtype=np.float32)
        if self.distance_method == DistanceMethod.COSINE:
            norm = float(np.linalg.norm(arr))
            q = (arr / norm) if norm != 0.0 else arr
        else:
            q = arr
        xq = np.ascontiguousarray([q], dtype=np.float32)

        k_eff = min(k, self.index.ntotal)
        distances, labels = self.index.search(xq, k_eff)  # distances could be similarities, depends on distance metric

        results: list[list[float]] = []
        for lid in labels[0]:
            if lid == -1:
                continue
            key = self._id_to_key.get(int(lid))
            if key:
                results.append(self._items[key].vector)
        return results

    def save(self, vector: list[float]) -> str:
        # key based on original (pre-normalization) vector for idempotency
        key = hashlib.md5(str(vector).encode()).hexdigest()
        if key in self._items:
            return key

        vec = self._prepare_vector(vector)

        # init index if needed
        if self.index is None:
            self.dim = int(len(vec))
            self.index = self._make_index(self.dim)
        elif int(len(vec)) != int(self.dim or 0):
            raise ValueError(f"Vector dim {len(vec)} != index dim {self.dim}")

        # stable int64 id from key
        id_int = int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)

        xb = np.ascontiguousarray([vec], dtype=np.float32)
        xids = np.asarray([id_int], dtype=np.int64)
        self.index.add_with_ids(xb, xids)

        self._items[key] = FaissVector(id=id_int, vector=vec)
        self._id_to_key[id_int] = key

        self._persist()
        return key

    def remove(self, key: str) -> bool:
        fv = self._items.pop(key, None)
        if fv is None:
            return False  # nothing to remove

        vid = int(fv.id)
        self._id_to_key.pop(vid, None)

        # Remove from FAISS index if present
        if self.index is not None and self.index.ntotal > 0:
            ids = np.asarray([vid], dtype=np.int64)
            self.index.remove_ids(ids)

        # Persist updated index + metadata
        self._persist()
        return True

    # ---------- Internals ----------

    def _prepare_vector(self, vector: list[float]) -> list[float]:
        """Cast to float32; L2-normalize only for cosine metric."""
        arr = np.asarray(vector, dtype=np.float32)
        if self.distance_method == DistanceMethod.COSINE:
            norm = float(np.linalg.norm(arr))
            if norm != 0.0:
                arr = (arr / norm).astype(np.float32)
        return arr.tolist()

    def _make_index(self, dim: int) -> faiss.Index:
        if self.distance_method in (DistanceMethod.COSINE, DistanceMethod.INNER_PRODUCT):
            base = faiss.IndexFlatIP(dim)  # cosine uses IP on normalized vectors
        elif self.distance_method == DistanceMethod.L2:
            base = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported distance method: {self.distance_method}")
        return faiss.IndexIDMap2(base)

    def _load(self) -> None:
        # load index if present
        if self.index_path.exists():
            self.index = faiss.read_index(self.index_path.__fspath__())
            self.dim = getattr(self.index, "d", None)

        # load metadata if present
        meta_method = None
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding='utf-8'))
            self.dim = meta.get("dim", self.dim)

            raw_items = meta.get("items", {}) or {}
            self._items = {k: FaissVector.model_validate(v) for k, v in raw_items.items()}
            self._id_to_key = {int(v.id): k for k, v in self._items.items()}

            meta_method = meta.get("distance_method")

        # If metadata has a stored method, ensure it matches the requested one
        if meta_method is not None:
            try:
                stored = DistanceMethod(meta_method) if meta_method in DistanceMethod._value2member_map_ else \
                    DistanceMethod[meta_method]
            except Exception:
                stored = None
            if stored is not None and stored != self.distance_method:
                raise ValueError(
                    f"Index at {self.index_path} was created with {stored.value} but "
                    f"this client was initialized with {self.distance_method.value}."
                )

        # if index is missing, but we have metadata, rebuild in-memory and persist
        if self.index is None and self.dim is not None:
            self.index = self._make_index(int(self.dim))
            if self._items:
                vecs = [fv.vector for fv in self._items.values()]
                ids = [fv.id for fv in self._items.values()]
                xb = np.asarray(vecs, dtype=np.float32)
                xids = np.asarray(ids, dtype=np.int64)
                if xb.size:
                    self.index.add_with_ids(xb, xids)
                    faiss.write_index(self.index, self.index_path.__fspath__())

    def _persist(self) -> None:
        # ensure dirs exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # persist index (temp then replace)
        if self.index is not None:
            tmp_index = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            faiss.write_index(self.index, tmp_index.__fspath__())
            tmp_index.replace(self.index_path)

        # persist metadata (temp then replace)
        items_dump = {k: (v.model_dump() if hasattr(v, "model_dump") else v.dict())
                      for k, v in self._items.items()}
        meta = {
            "dim": int(self.dim) if self.dim is not None else None,
            "items": items_dump,
            "distance_method": self.distance_method.value,  # persist chosen metric
        }

        tmp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp_meta.replace(self.meta_path)
