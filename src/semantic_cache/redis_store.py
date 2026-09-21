"""Persistent cache store on Redis 8 (built-in query engine) via RedisVL.

Requires the optional ``redis`` extra (``uv sync --extra redis``) and Redis 8 or
newer, whose core includes the query/vector-search engine (the ``redis:8`` image,
or a local ``redis-server`` >= 8, e.g. ``brew install redis``).

Layout: one Redis hash per entry under ``<prefix>:<entry_id>``.

* Vector field ``embedding`` (HNSW, cosine) holds the unit-normalised vector.
* TAG fields ``identity_key`` / ``text_hash`` / ``model`` / ``provider`` /
  ``system_prompt_hash`` are the *filters*. Every vector query is pre-filtered by
  ``identity_key`` so similarity is only computed inside one cache identity,
  exactly like the in-memory store.
* TTL is a native Redis key expiry, so expired entries leave the index without
  any application-side sweep.

Redis reports cosine *distance* (``1 - cosine``); this store converts it back to
similarity so thresholds mean the same thing across backends.

Eviction under memory pressure is delegated to Redis (``maxmemory-policy``).
"""

from __future__ import annotations

from dataclasses import asdict
import json
from time import time
from typing import Any
from uuid import uuid4

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import FilterExpression, Tag

from semantic_cache.cache import CacheEntry, CacheLookup, normalize, text_hash
from semantic_cache.models import CacheIdentity, ChatCompletion, TokenUsage

_CANDIDATES = 5  # nearest neighbours fetched so that an expired top hit cannot mask a live one

_STORED_FIELDS = [
    "entry_id",
    "semantic_lookup_text",
    "response_json",
    "identity_json",
    "created_at",
    "expires_at",
    "hit_count",
]


def _is_expired(row: dict[str, Any], now: float | None = None) -> bool:
    expires_at = float(row.get("expires_at") or 0.0)
    return expires_at > 0 and expires_at <= (time() if now is None else now)


def build_schema(*, index_name: str, prefix: str, dimensions: int) -> dict[str, Any]:
    """RedisVL schema; a pure function so it can be tested without a server."""
    return {
        "index": {"name": index_name, "prefix": prefix, "storage_type": "hash"},
        "fields": [
            {"name": "identity_key", "type": "tag"},
            {"name": "text_hash", "type": "tag"},
            {"name": "model", "type": "tag"},
            {"name": "provider", "type": "tag"},
            {"name": "system_prompt_hash", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
            {"name": "hit_count", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": dimensions,
                    "algorithm": "hnsw",
                    "datatype": "float32",
                    "distance_metric": "cosine",
                },
            },
        ],
    }


def identity_filter(identity: CacheIdentity) -> FilterExpression:
    return Tag("identity_key") == identity.key


def invalidation_filter(
    *, model: str | None, provider: str | None, system_prompt_hash: str | None
) -> FilterExpression:
    if model is None and provider is None and system_prompt_hash is None:
        raise ValueError("Give at least one invalidation filter, or call clear().")
    expression: FilterExpression | None = None
    for field, value in (
        ("model", model),
        ("provider", provider),
        ("system_prompt_hash", system_prompt_hash),
    ):
        if value is None:
            continue
        term = Tag(field) == value
        expression = term if expression is None else expression & term
    assert expression is not None
    return expression


def serialize_response(response: ChatCompletion) -> str:
    return json.dumps(
        {
            "content": response.content,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "usage": asdict(response.usage) if response.usage else None,
        }
    )


def deserialize_response(raw: str) -> ChatCompletion:
    data = json.loads(raw)
    usage = TokenUsage(**data["usage"]) if data.get("usage") else None
    return ChatCompletion(
        content=data["content"],
        model=data["model"],
        usage=usage,
        finish_reason=data.get("finish_reason", "stop"),
    )


class RedisVLCacheStore:
    def __init__(
        self,
        *,
        redis_url: str,
        embedding_dimensions: int,
        index_name: str = "cachemind",
        prefix: str = "cachemind:entry",
        index: SearchIndex | None = None,
    ) -> None:
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive.")
        self._prefix = prefix
        self._dimensions = embedding_dimensions
        schema = build_schema(
            index_name=f"{index_name}_{embedding_dimensions}", prefix=prefix, dimensions=embedding_dimensions
        )
        self._index = index or SearchIndex.from_dict(schema, redis_url=redis_url)
        # overwrite=False keeps existing entries across restarts (persistence).
        self._index.create(overwrite=False)

    @property
    def entry_count(self) -> int:
        return int(self._index.info().get("num_docs", 0))

    @property
    def evictions(self) -> int:
        try:
            return int(self._index.client.info("stats").get("evicted_keys", 0))
        except Exception:  # noqa: BLE001 - metrics must never break requests
            return 0

    def _exact_rows(self, identity: CacheIdentity, semantic_lookup_text: str) -> list[dict]:
        """Every stored row for this identity and text, live or expired (normally zero or one)."""
        query = FilterQuery(
            filter_expression=identity_filter(identity) & (Tag("text_hash") == text_hash(semantic_lookup_text)),
            return_fields=_STORED_FIELDS,
            num_results=_CANDIDATES,
        )
        return list(self._index.query(query))

    def find_exact(
        self, *, identity: CacheIdentity, semantic_lookup_text: str
    ) -> CacheEntry | None:
        # An expired row that has not been purged yet must not shadow a live duplicate.
        for row in self._exact_rows(identity, semantic_lookup_text):
            if not _is_expired(row):
                return self._entry_from_row(row)
        return None

    def lookup(
        self,
        *,
        identity: CacheIdentity,
        semantic_lookup_text: str,
        embedding: tuple[float, ...],
        similarity_threshold: float,
    ) -> CacheLookup:
        exact = self.find_exact(identity=identity, semantic_lookup_text=semantic_lookup_text)
        if exact is not None:
            return CacheLookup(entry=exact, similarity_score=1.0)

        vector = normalize(embedding)
        if vector.shape[0] != self._dimensions:
            raise ValueError("Embeddings must have matching dimensions.")
        query = VectorQuery(
            vector=vector.tobytes(),
            vector_field_name="embedding",
            return_fields=[*_STORED_FIELDS],
            filter_expression=identity_filter(identity),
            num_results=_CANDIDATES,
            dtype="float32",
        )
        # Redis deletes expired keys lazily/actively; whether an expired document can still be
        # returned by a search for a moment is not documented, so the stored expiry is re-checked
        # here and an expired candidate never counts. Rows arrive best-first.
        live = [row for row in self._index.query(query) if not _is_expired(row)]
        if not live:
            return CacheLookup(entry=None, similarity_score=None)
        score = 1.0 - float(live[0]["vector_distance"])
        if score >= similarity_threshold:
            return CacheLookup(entry=self._entry_from_row(live[0]), similarity_score=score)
        return CacheLookup(entry=None, similarity_score=score)

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
        if vector.shape[0] != self._dimensions:
            raise ValueError("Embeddings must have matching dimensions.")

        # Replace, never duplicate: drop every earlier row for this exact text, expired ones included.
        previous = self._exact_rows(identity, semantic_lookup_text)
        if previous:
            self._index.drop_keys([self._key(row["entry_id"]) for row in previous])

        entry_id = str(uuid4())
        now = time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else 0.0
        document = {
            "entry_id": entry_id,
            "identity_key": identity.key,
            "text_hash": text_hash(semantic_lookup_text),
            "model": identity.model,
            "provider": identity.provider,
            "system_prompt_hash": identity.system_prompt_hash,
            "semantic_lookup_text": semantic_lookup_text,
            "response_json": serialize_response(response),
            "identity_json": json.dumps(
                {
                    "provider": identity.provider,
                    "model": identity.model,
                    "system_prompt_hash": identity.system_prompt_hash,
                    "temperature": identity.temperature,
                    "max_tokens": identity.max_tokens,
                    "params_hash": identity.params_hash,
                }
            ),
            "created_at": now,
            "expires_at": expires_at,
            "hit_count": 0,
            "embedding": vector.tobytes(),
        }
        self._index.load([document], keys=[self._key(entry_id)], ttl=ttl_seconds)
        return CacheEntry(
            entry_id=entry_id,
            identity=identity,
            semantic_lookup_text=semantic_lookup_text,
            embedding=vector,
            response=response,
            created_at=now,
            expires_at=expires_at or None,
            last_accessed=now,
        )

    def record_hit(self, entry_id: str) -> None:
        self._index.client.hincrby(self._key(entry_id), "hit_count", 1)

    def invalidate(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        system_prompt_hash: str | None = None,
    ) -> int:
        expression = invalidation_filter(
            model=model, provider=provider, system_prompt_hash=system_prompt_hash
        )
        removed = 0
        while True:
            rows = self._index.query(
                FilterQuery(filter_expression=expression, return_fields=["entry_id"], num_results=500)
            )
            if not rows:
                return removed
            dropped = self._index.drop_keys([self._key(row["entry_id"]) for row in rows])
            if dropped == 0:
                # Nothing was actually deleted (e.g. the keys are already gone): stop instead of
                # re-reading the same rows forever.
                return removed
            removed += dropped

    def clear(self) -> int:
        return int(self._index.clear())

    def _key(self, entry_id: str) -> str:
        return f"{self._prefix}:{entry_id}"

    @staticmethod
    def _entry_from_row(row: dict[str, Any]) -> CacheEntry:
        identity_data = json.loads(row["identity_json"])
        expires_at = float(row.get("expires_at") or 0.0)
        return CacheEntry(
            entry_id=row["entry_id"],
            identity=CacheIdentity(**identity_data),
            semantic_lookup_text=row["semantic_lookup_text"],
            embedding=np.empty(0, dtype=np.float32),
            response=deserialize_response(row["response_json"]),
            created_at=float(row.get("created_at") or 0.0),
            expires_at=expires_at or None,
            hit_count=int(float(row.get("hit_count") or 0)),
        )
