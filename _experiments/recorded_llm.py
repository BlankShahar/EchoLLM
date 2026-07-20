import hashlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field

from llm import ILLM, LLMResponse, LLMResponseChunk


class RecordedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    response: str
    latency_ms: float = Field(ge=0.0)


class RecordedLLMStore:
    """Resumable SQLite store shared by one recorder and many replay jobs."""

    def __init__(self, path: Path, *, writable: bool) -> None:
        self.path = path
        if not writable and not path.exists():
            raise FileNotFoundError(f"Recorded LLM database not found: {path}")
        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
        uri = str(path) if writable else f"file:{path.as_posix()}?mode=ro"
        self._connection = sqlite3.connect(
            uri,
            uri=not writable,
            check_same_thread=False,
        )
        self._lock = RLock()
        if writable:
            with self._connection:
                self._connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS responses (
                        prompt_hash TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        latency_ms REAL NOT NULL CHECK(latency_ms >= 0)
                    );
                    """
                )

    def validate_or_initialize(self, *, model: str, options: dict[str, object]) -> None:
        expected = {
            "model": model,
            "options": json.dumps(options, sort_keys=True, separators=(",", ":")),
        }
        with self._lock, self._connection:
            existing = dict(self._connection.execute("SELECT key, value FROM metadata"))
            if existing and existing != expected:
                raise ValueError(
                    "Recorded LLM database metadata does not match the requested model/options"
                )
            for key, value in expected.items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                    (key, value),
                )

    def get(self, prompt: str) -> RecordedResponse | None:
        digest = _prompt_hash(prompt)
        with self._lock:
            row = self._connection.execute(
                "SELECT prompt, response, latency_ms FROM responses WHERE prompt_hash = ?",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        stored_prompt, response, latency_ms = row
        if stored_prompt != prompt:
            raise RuntimeError("Recorded LLM prompt hash collision")
        return RecordedResponse(
            prompt=stored_prompt,
            response=response,
            latency_ms=float(latency_ms),
        )

    def put(self, item: RecordedResponse) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO responses(prompt_hash, prompt, response, latency_ms)
                VALUES (?, ?, ?, ?)
                """,
                (
                    _prompt_hash(item.prompt),
                    item.prompt,
                    item.response,
                    item.latency_ms,
                ),
            )

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def all(self) -> list[RecordedResponse]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT prompt, response, latency_ms FROM responses ORDER BY rowid"
            ).fetchall()
        return [
            RecordedResponse(prompt=prompt, response=response, latency_ms=float(latency_ms))
            for prompt, response, latency_ms in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class RecordedLLM(ILLM):
    def __init__(self, path: Path) -> None:
        self._store = RecordedLLMStore(path, writable=False)

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        recorded = self._store.get(prompt)
        if recorded is None:
            raise KeyError("Prompt is missing from the recorded LLM database")
        return LLMResponse(response=recorded.response, latency=recorded.latency_ms)

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        response = self.ask(prompt, **kwargs)
        yield LLMResponseChunk(
            response_chunk=response.response,
            chunk_number=1,
            delay=response.latency,
        )

    def recorded_responses(self) -> list[RecordedResponse]:
        return self._store.all()

    def close(self) -> None:
        self._store.close()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
