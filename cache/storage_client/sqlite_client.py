import sqlite3
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any


class SQLiteClient:
    def __init__(self, db_path=Path('resources/responses.db')):
        self._db_file = db_path
        self._db_file.mkdir(parents=True, exist_ok=True)
        self._connection = self.create_connection()

    def create_connection(self) -> Connection:
        return sqlite3.connect(self._db_file)

    def disconnect(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def save(self, record: list[Any], record_key: str, table_name: str):
        # Discover table columns
        cols_cur = self._execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cols_cur.fetchall()]  # row[1] = column name

        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist.")
        if "key" not in columns:
            raise ValueError(f"Table '{table_name}' must have a 'key' column.")

        # Map values: 'key' gets record_key; the rest come from `record` in column order
        non_key_cols = [c for c in columns if c != "key"]
        if len(record) != len(non_key_cols):
            raise ValueError(
                f"Record length ({len(record)}) does not match non-key columns "
                f"({len(non_key_cols)}): {non_key_cols}"
            )

        values_by_col = {"key": record_key}
        for i, c in enumerate(non_key_cols):
            values_by_col[c] = record[i]
        ordered_values = [values_by_col[c] for c in columns]

        placeholders = ", ".join(["?"] * len(columns))
        updates = ", ".join(f"{c}=excluded.{c}" for c in non_key_cols)

        self._execute(
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(key) DO UPDATE SET {updates}",
            *ordered_values,
        )

    def fetch(self, key: str, table_name: str) -> dict[str, Any] | None:
        cur = self._execute(f"SELECT * FROM {table_name} WHERE key = ?", key)
        row = cur.fetchone()
        if row is None:
            return None

        columns = [d[0] for d in cur.description]
        return dict(zip(columns, row))

    def remove(self, key: str, table_name: str) -> bool:
        # Verify table exists and has a 'key' column
        cols_cur = self._execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cols_cur.fetchall()]  # row[1] is column name
        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist.")
        if "key" not in columns:
            raise ValueError(f"Table '{table_name}' must have a 'key' column.")

        # Delete and report whether anything was removed
        cur = self._execute(f"DELETE FROM {table_name} WHERE key = ?", key)
        return cur.rowcount > 0

    def _execute(self, query: str, *params) -> Cursor:
        cursor = self._connection.execute(query, params)
        self._connection.commit()
        return cursor

    def __del__(self):
        self.disconnect()
