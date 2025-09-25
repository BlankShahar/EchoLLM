import hashlib
import json
from enum import Enum
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel

from text_similarity import vector_utils


class StoredVector(BaseModel):
    key: str
    vector: list[float]  # index-space vector (normalized for COSINE; raw otherwise)


class FaissVector(StoredVector):
    id: str  # Faiss ID in DB
    original_norm: float | None = None  # only set for COSINE; None for L2/IP -> used to reconstruct original vector if been normalized


class FaissDistanceMethod(Enum):
    COSINE = "cosine"  # cosine similarity - normalize vectors, use IP
    INNER_PRODUCT = "ip"  # raw inner product
    L2 = "l2"  # squared Euclidean


_CWD = Path(__file__).parent


class FaissClient:
    def __init__(self, distance_method: FaissDistanceMethod, index_path=_CWD / 'resources/requests.db'):
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

    def fetch_nearest_k(self, vector: list[float], k: int = 100) -> list[StoredVector]:
        if k <= 0:
            raise ValueError('k must be greater than 0!')
        if self.index is None or self.index.ntotal == 0:
            return []

        # query in index-space (normalize only for cosine)
        q_vec = vector_utils.normalize(vector) if self.distance_method == FaissDistanceMethod.COSINE else vector
        xq = np.ascontiguousarray([np.asarray(q_vec, dtype=np.float32)], dtype=np.float32)

        k_eff = min(k, self.index.ntotal)
        _, ids = self.index.search(xq, k_eff)  # type: ignore[call-arg]

        results: list[StoredVector] = []
        for lid in ids[0]:
            if lid == -1:
                continue
            key = self._id_to_key.get(int(lid))
            if not key:
                continue
            faiss_vector = self._items[key]
            original_vector = StoredVector(key=key, vector=self._reconstruct_original_vector(faiss_vector))
            results.append(original_vector)  # return original vector for flexible re-ranking
        return results

    def save(self, vector: list[float], key: str) -> str:
        if key in self._items:
            return key

        if self.distance_method == FaissDistanceMethod.COSINE:
            # store normalized vector in index; keep original norm to reconstruct raw later
            arr_raw = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr_raw))
            vec = (arr_raw / norm).astype(np.float32).tolist() if norm != 0.0 else arr_raw.astype(np.float32).tolist()
            original_norm = norm
        else:
            vec = list(np.asarray(vector, dtype=np.float32))
            original_norm = None

        # init index if needed
        if self.index is None:
            self.dim = int(len(vec))
            self.index = self._make_index(self.dim)
        elif int(len(vec)) != int(self.dim or 0):
            raise ValueError(f"Vector dim {len(vec)} != index dim {self.dim}")

        # stable int64 id from key
        id_int = int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)

        xb = np.ascontiguousarray([np.asarray(vec, dtype=np.float32)], dtype=np.float32)
        xids = np.asarray([id_int], dtype=np.int64)
        self.index.add_with_ids(xb, xids)  # type: ignore[call-arg]

        self._items[key] = FaissVector(key=key, id=id_int, vector=vec, original_norm=original_norm)
        self._id_to_key[id_int] = key

        self._persist()
        return key

    def remove(self, key: str) -> bool:
        fv = self._items.pop(key, None)
        if fv is None:
            return False  # nothing to remove

        vid = int(fv.id)
        self._id_to_key.pop(vid, None)

        if self.index is not None and self.index.ntotal > 0:
            ids = np.asarray([vid], dtype=np.int64)
            self.index.remove_ids(ids)

        self._persist()
        return True

    def size(self) -> int:
        return len(self._items)

    @staticmethod
    def _reconstruct_original_vector(stored_vector: FaissVector) -> list[float]:
        """Convert stored index-space vector back to raw/original space if possible (undo normalization, for Cosine case)."""
        if stored_vector.original_norm is not None and stored_vector.original_norm != 0.0:
            arr = np.asarray(stored_vector.vector, dtype=np.float32)
            return (arr * stored_vector.original_norm).astype(np.float32).tolist()
        return stored_vector.vector

    def _make_index(self, dim: int) -> faiss.Index:
        if self.distance_method in (FaissDistanceMethod.COSINE, FaissDistanceMethod.INNER_PRODUCT):
            base = faiss.IndexFlatIP(dim)  # cosine uses IP on normalized vectors
        elif self.distance_method == FaissDistanceMethod.L2:
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

        # enforce metric consistency with on-disk metadata
        if meta_method is not None:
            try:
                stored = FaissDistanceMethod(meta_method) if meta_method in FaissDistanceMethod._value2member_map_ else \
                    FaissDistanceMethod[meta_method]
            except Exception:
                stored = None
            if stored is not None and stored != self.distance_method:
                raise ValueError(
                    f"Index at {self.index_path} was created with {stored.value} but "
                    f"this client was initialized with {self.distance_method.value}."
                )

        # rebuild index from metadata if needed
        if self.index is None and self.dim is not None:
            self.index = self._make_index(int(self.dim))
            if self._items:
                vecs = [fv.vector for fv in self._items.values()]  # index-space vectors
                ids = [fv.id for fv in self._items.values()]
                xb = np.asarray(vecs, dtype=np.float32)
                xids = np.asarray(ids, dtype=np.int64)
                if xb.size:
                    self.index.add_with_ids(xb, xids)  # type: ignore[call-arg]
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
        items_dump = {k: v.model_dump() for k, v in self._items.items()}
        meta = {
            "dim": int(self.dim) if self.dim is not None else None,
            "items": items_dump,
            "distance_method": self.distance_method.value,
        }

        tmp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp_meta.replace(self.meta_path)
