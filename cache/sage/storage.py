import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock

import numpy as np

from .models import PersistedResident, SAGEStorageMetadata


class SAGEStorage(ABC):
    @abstractmethod
    def load(self) -> list[PersistedResident]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, resident: PersistedResident) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_or_initialize(self, metadata: SAGEStorageMetadata) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NullSAGEStorage(SAGEStorage):
    def load(self) -> list[PersistedResident]:
        return []

    def upsert(self, resident: PersistedResident) -> None:
        return None

    def clear(self) -> None:
        return None

    def validate_or_initialize(self, metadata: SAGEStorageMetadata) -> None:
        return None


class SQLiteSAGEStorage(SAGEStorage):
    """Small persistence layer used only on admission/eviction, not on lookup."""

    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._namespace = namespace
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sage_metadata (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sage_residents (
                    namespace TEXT NOT NULL,
                    slot INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    inserted_step INTEGER NOT NULL,
                    last_access_step INTEGER NOT NULL,
                    PRIMARY KEY(namespace, slot),
                    UNIQUE(namespace, key)
                )
                """
            )

    def validate_or_initialize(self, metadata: SAGEStorageMetadata) -> None:
        values = {
            "distance_method": metadata.distance_method,
            "hit_distance_threshold": repr(metadata.hit_distance_threshold),
            "vector_dimension": str(metadata.vector_dimension),
        }
        with self._lock, self._connection:
            existing_rows = self._connection.execute(
                "SELECT key, value FROM sage_metadata WHERE namespace = ?",
                (self._namespace,),
            ).fetchall()
            existing = dict(existing_rows)
            if existing and existing != values:
                raise ValueError(
                    "Persistent SAGE namespace was created with incompatible settings: "
                    f"stored={existing}, requested={values}"
                )
            for key, value in values.items():
                self._connection.execute(
                    "INSERT OR REPLACE INTO sage_metadata(namespace, key, value) VALUES (?, ?, ?)",
                    (self._namespace, key, value),
                )

    def load(self) -> list[PersistedResident]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT slot, key, prompt, response, vector, vector_dimension,
                       inserted_step, last_access_step
                FROM sage_residents
                WHERE namespace = ?
                ORDER BY slot
                """,
                (self._namespace,),
            ).fetchall()
        residents: list[PersistedResident] = []
        for row in rows:
            slot, key, prompt, response, blob, dimension, inserted, last_access = row
            vector = np.frombuffer(blob, dtype=np.float32, count=dimension).copy().tolist()
            residents.append(
                PersistedResident(
                    slot=slot,
                    key=key,
                    prompt=prompt,
                    response=response,
                    vector=vector,
                    inserted_step=inserted,
                    last_access_step=last_access,
                )
            )
        return residents

    def upsert(self, resident: PersistedResident) -> None:
        vector = np.asarray(resident.vector, dtype=np.float32)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM sage_residents WHERE namespace = ? AND key = ? AND slot <> ?",
                (self._namespace, resident.key, resident.slot),
            )
            self._connection.execute(
                """
                INSERT INTO sage_residents(
                    namespace, slot, key, prompt, response, vector,
                    vector_dimension, inserted_step, last_access_step
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, slot) DO UPDATE SET
                    key=excluded.key,
                    prompt=excluded.prompt,
                    response=excluded.response,
                    vector=excluded.vector,
                    vector_dimension=excluded.vector_dimension,
                    inserted_step=excluded.inserted_step,
                    last_access_step=excluded.last_access_step
                """,
                (
                    self._namespace,
                    resident.slot,
                    resident.key,
                    resident.prompt,
                    resident.response,
                    vector.tobytes(order="C"),
                    vector.shape[0],
                    resident.inserted_step,
                    resident.last_access_step,
                ),
            )

    def clear(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM sage_residents WHERE namespace = ?", (self._namespace,)
            )
            self._connection.execute(
                "DELETE FROM sage_metadata WHERE namespace = ?", (self._namespace,)
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
