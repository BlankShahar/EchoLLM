import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from sqlite3 import Connection, Cursor

from .artifacts import ResponseRecord
from .storage_client import StorageClient


class SQLiteClient(StorageClient):
    TABLE_NAME = 'responses'

    def __init__(self, db_path=Path('resources/responses.db')):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db_file = self.db_path / f"{self.TABLE_NAME}.db"
        self._connection = self.create_connection()
        self._ensure_table()

    def create_connection(self) -> Connection:
        return sqlite3.connect(self._db_file)

    def reconnect(self):
        self.disconnect()
        self._connection = self.create_connection()

    def disconnect(self):
        self._connection.close()
        self._connection = None

    def _ensure_table(self):
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                key TEXT PRIMARY KEY,
                request_key TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
            )
            """
        )

    def _execute(self, query: str, *params) -> Cursor:
        cursor = self._connection.execute(query, params)
        self._connection.commit()
        return cursor

    def save(self, response: ResponseRecord) -> str:
        """
        Inserts or updates a response record.
        Returns the record key (md5 of `response.response` via ResponseRecord.key).
        """
        self._execute(
            f"""
                        INSERT INTO {self.TABLE_NAME} (key, request_key, response, created_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            request_key=excluded.request_key,
                            response=excluded.response,
                            created_at=excluded.created_at
                        """,
            (
                response.key,
                response.request_key,
                response.response,
                datetime.now(UTC).isoformat(timespec="seconds"))
        )
        return response.key

    def fetch(self, key: str) -> ResponseRecord:
        cursor = self._execute(
            f"SELECT request_key, response FROM {self.TABLE_NAME} WHERE key = ?",
            (key,)
        )
        row: tuple[str, str] = cursor.fetchone()
        if row is None:
            raise KeyError(f"Response with key '{key}' was not found")
        request_key, response_text = row
        # Rebuild the Pydantic object; its .key property will recompute from response_text
        return ResponseRecord(request_key=request_key, response=response_text)
