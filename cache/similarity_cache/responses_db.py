from ..storage_client.records import ResponseRecord
from ..storage_client.sqlite_client import SQLiteClient


class ResponsesDB:
    _TABLE = 'responses'

    def __init__(self):
        self.sqlite_client = SQLiteClient()
        self.sqlite_client.execute(
            f'CREATE TABLE IF NOT EXISTS {self._TABLE} ('
            'key TEXT PRIMARY KEY,'
            'request_key TEXT NOT NULL,'
            'response TEXT NOT NULL'
            ');'
        )

    def fetch(self, key: str) -> ResponseRecord:
        record = self.sqlite_client.fetch(key, self._TABLE)
        return ResponseRecord.model_validate(record)

    def save(self, response: ResponseRecord) -> str:
        key = self.sqlite_client.save(response.model_dump(), self._TABLE)
        assert response.key == key
        return key

    def remove(self, key: str) -> bool:
        return self.sqlite_client.remove(key, self._TABLE)
