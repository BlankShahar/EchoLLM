import sqlite3
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any


class SQLiteClient:
    """
    This SQL client assumes each table has a primary key column named `key`.
    """

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

    def save(self, record: dict[str, Any], table_name: str) -> str:
        if not record:
            raise ValueError("record must be a non-empty dict")

        # Introspect table columns
        cols_cur = self.execute(f"PRAGMA table_info({table_name})")
        table_columns = [row[1] for row in cols_cur.fetchall()]  # row[1] = column name
        if not table_columns:
            raise ValueError(f"Table '{table_name}' does not exist.")
        if "key" not in table_columns:
            raise ValueError(f"Table '{table_name}' must have a 'key' column for upsert.")

        # Validate provided columns match the table schema
        unknown_columns = [c for c in record.keys() if c not in table_columns]
        if unknown_columns:
            raise ValueError(f"Unknown columns for table '{table_name}': {unknown_columns}")
        missing_columns = [c for c in table_columns if c not in record]
        if missing_columns:
            raise ValueError(f"Missing columns for table '{table_name}': {missing_columns}")

        # Build UPSERT
        non_key_columns = [c for c in table_columns if c != "key"]
        placeholders = ", ".join(["?"] * len(table_columns))
        assignments = ", ".join(f"{c}=excluded.{c}" for c in non_key_columns)

        sql = (
            f"INSERT INTO {table_name} ({', '.join(table_columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(key) DO UPDATE SET {assignments}"
        )
        params = [record[c] for c in table_columns]

        self.execute(sql, *params)
        return record["key"]

    def fetch(self, key: str, table_name: str) -> dict[str, Any] | None:
        cur = self.execute(f"SELECT * FROM {table_name} WHERE key = ?", key)
        row = cur.fetchone()
        if row is None:
            return None

        columns = [d[0] for d in cur.description]
        return dict(zip(columns, row))

    def remove(self, key: str, table_name: str) -> bool:
        # Verify table exists and has a 'key' column
        cols_cur = self.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cols_cur.fetchall()]  # row[1] is column name
        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist.")
        if "key" not in columns:
            raise ValueError(f"Table '{table_name}' must have a 'key' column.")

        # Delete and report whether anything was removed
        cur = self.execute(f"DELETE FROM {table_name} WHERE key = ?", key)
        return cur.rowcount > 0

    def exists(self, key: str, table_name: str) -> bool:
        # Validate table and 'key' column
        cols_cur = self.execute(f'PRAGMA table_info({table_name})')
        columns = [row[1] for row in cols_cur.fetchall()]
        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist.")
        if "key" not in columns:
            raise ValueError(f"Table '{table_name}' must have a 'key' column.")

        cur = self.execute(f'SELECT EXISTS(SELECT 1 FROM {table_name} WHERE key = ?)', key)
        row = cur.fetchone()
        return bool(row[0]) if row else False

    def execute(self, query: str, *params) -> Cursor:
        cursor = self._connection.execute(query, params)
        self._connection.commit()
        return cursor

    def size(self, table_name: str) -> int:
        # Ensure table exists
        info_cur = self.execute(f'PRAGMA table_info({table_name})')
        if not info_cur.fetchall():
            raise ValueError(f"Table '{table_name}' does not exist.")

        cur = self.execute(f'SELECT COUNT(*) FROM {table_name}')
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def __del__(self):
        self.disconnect()
