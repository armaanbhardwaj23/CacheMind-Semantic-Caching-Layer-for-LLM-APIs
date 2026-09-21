"""On-disk embedding cache so experiments are reproducible and re-runs are free.

Keyed by (model, text). This is a *development* aid for the offline evaluation
scripts; it is unrelated to the semantic response cache and is not used by the
proxy. Vectors are appended to a JSONL file so an interrupted run loses nothing.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from typing import Protocol


class _Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class DiskCachedEmbedder:
    def __init__(self, inner: _Embedder, *, model: str, path: Path) -> None:
        self._inner = inner
        self._model = model
        self._path = path
        self._lock = Lock()
        self._vectors: dict[str, tuple[float, ...]] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._vectors[row["key"]] = tuple(row["vector"])

    def _key(self, text: str) -> str:
        return sha256(f"{self._model}\x00{text}".encode()).hexdigest()

    def embed(self, text: str) -> tuple[float, ...]:
        key = self._key(text)
        with self._lock:
            if key in self._vectors:
                self.hits += 1
                return self._vectors[key]
        vector = self._inner.embed(text)
        with self._lock:
            self.misses += 1
            self._vectors[key] = vector
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as file:
                file.write(json.dumps({"key": key, "vector": list(vector)}) + "\n")
        return vector
