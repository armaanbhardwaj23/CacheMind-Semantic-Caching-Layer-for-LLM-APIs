"""Cache storage: the ``CacheStore`` interface and the in-memory implementation.

The store owns everything about *where* entries live: vector search, exact-text
lookup, TTL expiry, invalidation, and capacity eviction. The service owns *when*
to look up or store. Two implementations satisfy the interface:

* ``InMemorySemanticCache`` (this module): numpy-vectorised, process-local. It is
  the reference implementation used by tests and by the offline experiments.
* ``RedisVLCacheStore`` (``redis_store.py``): persistent, shared across processes.

Both only ever compare vectors *within one cache identity*. That is the safety
boundary: a similar prompt under a different model, system prompt, or sampling
configuration is never a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from threading import RLock
from time import time
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np

from semantic_cache.models import CacheIdentity, ChatCompletion

Clock = Callable[[], float]


def text_hash(text: str) -> str:
    """Stable digest of the semantic lookup text, used for exact-match lookup."""
    return sha256(text.encode("utf-8")).hexdigest()[:32]


def normalize(embedding: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Return a unit-length float32 copy so cosine similarity is a dot product."""
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embeddings must be non-empty one-dimensional vectors.")
    if norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return vector / norm


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity of two raw vectors, rejecting incompatible or zero vectors."""
    if len(left) != len(right):
        raise ValueError("Embeddings must have matching dimensions.")
    return float(np.dot(normalize(left), normalize(right)))


@dataclass
class CacheEntry:
    entry_id: str
    identity: CacheIdentity
    semantic_lookup_text: str
    embedding: np.ndarray
    response: ChatCompletion
    created_at: float
    expires_at: float | None = None
    hit_count: int = 0
    last_accessed: float = field(default=0.0)


@dataclass(frozen=True)
class CacheLookup:
    """Best identity-compatible candidate, even when it is below threshold.

    ``similarity_score`` is reported for below-threshold candidates too, because
    near-miss analysis needs to know how close a miss was.
    """

    entry: CacheEntry | None
    similarity_score: float | None


class CacheStore(Protocol):
    @property
    def entry_count(self) -> int: ...

    @property
    def evictions(self) -> int: ...

    def find_exact(
        self, *, identity: CacheIdentity, semantic_lookup_text: str
    ) -> CacheEntry | None: ...

    def lookup(
        self,
        *,
        identity: CacheIdentity,
        semantic_lookup_text: str,
        embedding: tuple[float, ...],
        similarity_threshold: float,
    ) -> CacheLookup: ...

    def store(
        self,
        *,
        identity: CacheIdentity,
        semantic_lookup_text: str,
        embedding: tuple[float, ...],
        response: ChatCompletion,
        ttl_seconds: int | None = None,
    ) -> CacheEntry: ...

    def record_hit(self, entry_id: str) -> None: ...

    def invalidate(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        system_prompt_hash: str | None = None,
    ) -> int: ...

    def clear(self) -> int: ...


class _Bucket:
    """Entries sharing one identity, with their vectors in one growable matrix."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.entries: list[CacheEntry] = []
        self._matrix = np.empty((16, dimensions), dtype=np.float32)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix[: len(self.entries)]

    def add(self, entry: CacheEntry) -> None:
        if len(self.entries) == self._matrix.shape[0]:
            grown = np.empty((self._matrix.shape[0] * 2, self.dimensions), dtype=np.float32)
            grown[: len(self.entries)] = self.matrix
            self._matrix = grown
        self._matrix[len(self.entries)] = entry.embedding
        self.entries.append(entry)

    def keep_only(self, keep: list[CacheEntry]) -> None:
        self.entries = []
        for entry in keep:
            self.add(entry)


class InMemorySemanticCache:
    """Process-local store; vector search is one matrix-vector product per lookup."""

    def __init__(self, *, max_entries: int | None = None, clock: Clock = time) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self._max_entries = max_entries
        self._clock = clock
        self._lock = RLock()
        self._buckets: dict[str, _Bucket] = {}
        self._by_id: dict[str, CacheEntry] = {}
        self._exact: dict[tuple[str, str], str] = {}
        self._next_expiry: float | None = None
        self._evictions = 0

    @property
    def entry_count(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._by_id)

    @property
    def evictions(self) -> int:
        return self._evictions

    def find_exact(
        self, *, identity: CacheIdentity, semantic_lookup_text: str
    ) -> CacheEntry | None:
        with self._lock:
            self._purge_expired()
            entry_id = self._exact.get((identity.key, text_hash(semantic_lookup_text)))
            return self._by_id.get(entry_id) if entry_id else None

    def lookup(
        self,
        *,
        identity: CacheIdentity,
        semantic_lookup_text: str,
        embedding: tuple[float, ...],
        similarity_threshold: float,
    ) -> CacheLookup:
        query = normalize(embedding)
        with self._lock:
            self._purge_expired()
            exact = self.find_exact(identity=identity, semantic_lookup_text=semantic_lookup_text)
            if exact is not None:
                return CacheLookup(entry=exact, similarity_score=1.0)

            bucket = self._buckets.get(identity.key)
            if bucket is None or not bucket.entries:
                return CacheLookup(entry=None, similarity_score=None)
            if bucket.dimensions != query.shape[0]:
                raise ValueError("Embeddings must have matching dimensions.")

            scores = bucket.matrix @ query
            best = int(np.argmax(scores))
            best_score = float(scores[best])
            if best_score >= similarity_threshold:
                return CacheLookup(entry=bucket.entries[best], similarity_score=best_score)
            return CacheLookup(entry=None, similarity_score=best_score)

    def store(
        self,
        *,
        identity: CacheIdentity,
        semantic_lookup_text: str,
        embedding: tuple[float, ...],
        response: ChatCompletion,
        ttl_seconds: int | None = None,
    ) -> CacheEntry:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        vector = normalize(embedding)
        now = self._clock()
        entry = CacheEntry(
            entry_id=str(uuid4()),
            identity=identity,
            semantic_lookup_text=semantic_lookup_text,
            embedding=vector,
            response=response,
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds is not None else None,
            last_accessed=now,
        )
        with self._lock:
            self._purge_expired()
            bucket = self._buckets.setdefault(identity.key, _Bucket(vector.shape[0]))
            if bucket.dimensions != vector.shape[0]:
                raise ValueError("Embeddings must have matching dimensions.")

            # Re-storing identical text under one identity replaces the old entry
            # instead of leaving a stale duplicate that could shadow it.
            previous_id = self._exact.get((identity.key, text_hash(semantic_lookup_text)))
            if previous_id is not None:
                self._remove_ids({previous_id})
                bucket = self._buckets.setdefault(identity.key, _Bucket(vector.shape[0]))

            bucket.add(entry)
            self._by_id[entry.entry_id] = entry
            self._exact[(identity.key, text_hash(semantic_lookup_text))] = entry.entry_id
            if entry.expires_at is not None:
                self._next_expiry = (
                    entry.expires_at
                    if self._next_expiry is None
                    else min(self._next_expiry, entry.expires_at)
                )
            self._evict_over_capacity()
        return entry

    def record_hit(self, entry_id: str) -> None:
        with self._lock:
            entry = self._by_id.get(entry_id)
            if entry is not None:
                entry.hit_count += 1
                entry.last_accessed = self._clock()

    def invalidate(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        system_prompt_hash: str | None = None,
    ) -> int:
        """Remove entries matching *all* given attributes; at least one is required."""
        if model is None and provider is None and system_prompt_hash is None:
            raise ValueError("Give at least one invalidation filter, or call clear().")
        with self._lock:
            doomed = {
                entry.entry_id
                for entry in self._by_id.values()
                if (model is None or entry.identity.model == model)
                and (provider is None or entry.identity.provider == provider)
                and (
                    system_prompt_hash is None
                    or entry.identity.system_prompt_hash == system_prompt_hash
                )
            }
            return self._remove_ids(doomed)

    def clear(self) -> int:
        with self._lock:
            removed = len(self._by_id)
            self._buckets.clear()
            self._by_id.clear()
            self._exact.clear()
            self._next_expiry = None
            return removed

    def _purge_expired(self) -> None:
        now = self._clock()
        if self._next_expiry is None or now < self._next_expiry:
            return
        expired = {
            entry.entry_id
            for entry in self._by_id.values()
            if entry.expires_at is not None and entry.expires_at <= now
        }
        self._remove_ids(expired)
        remaining = [e.expires_at for e in self._by_id.values() if e.expires_at is not None]
        self._next_expiry = min(remaining) if remaining else None

    def _evict_over_capacity(self) -> None:
        if self._max_entries is None:
            return
        while len(self._by_id) > self._max_entries:
            least_recent = min(self._by_id.values(), key=lambda e: e.last_accessed)
            self._remove_ids({least_recent.entry_id})
            self._evictions += 1

    def _remove_ids(self, entry_ids: set[str]) -> int:
        if not entry_ids:
            return 0
        touched: set[str] = set()
        removed = 0
        for entry_id in entry_ids:
            entry = self._by_id.pop(entry_id, None)
            if entry is None:
                continue
            removed += 1
            self._exact.pop((entry.identity.key, text_hash(entry.semantic_lookup_text)), None)
            touched.add(entry.identity.key)
        for key in touched:
            bucket = self._buckets[key]
            survivors = [e for e in bucket.entries if e.entry_id not in entry_ids]
            if survivors:
                bucket.keep_only(survivors)
            else:
                del self._buckets[key]
        return removed
