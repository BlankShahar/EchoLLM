from ...storage_client import SQLiteClient
from ...storage_client.records import ResponseRecord


class ResponsesDB:
    _TABLE = 'responses'

    def __init__(self):
        self._sqlite_client = SQLiteClient()
        self._sqlite_client.execute(
            f'CREATE TABLE IF NOT EXISTS {self._TABLE} ('
            'key TEXT PRIMARY KEY,'
            'request_key TEXT NOT NULL,'
            'response TEXT NOT NULL'
            ');'
        )

    def fetch(self, key: str) -> ResponseRecord | None:
        record = self._sqlite_client.fetch(key, self._TABLE)
        return ResponseRecord.model_validate(record) if record is not None else None

    def save(self, response: ResponseRecord) -> str:
        key = self._sqlite_client.save(response.model_dump(), self._TABLE)
        assert response.key == key
        return key

    def remove(self, key: str) -> bool:
        return self._sqlite_client.remove(key, self._TABLE)

    def remove_by_request(self, request_key: str) -> bool:
        return self._sqlite_client.remove_by_column('request_key', request_key, self._TABLE)

    def exists(self, key: str) -> bool:
        return self._sqlite_client.exists(key, self._TABLE)

    def size(self) -> int:
        return self._sqlite_client.size(self._TABLE)
