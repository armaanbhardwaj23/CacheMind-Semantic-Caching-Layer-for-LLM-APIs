"""Redis store logic that can be checked without a server (a fake index stands in for Redis).

Kept in its own module so that the module-level ``importorskip`` cannot hide unrelated tests when the
optional ``redis`` extra is not installed.
"""

import pytest

redisvl = pytest.importorskip("redisvl")

import json
from time import time

from semantic_cache.models import CacheIdentity, ChatCompletion
from semantic_cache.redis_store import RedisVLCacheStore, serialize_response


class FakeIndex:
    def __init__(self, rows, *, drop_result: int = 0) -> None:
        self.rows, self.drop_result, self.queries = rows, drop_result, 0
        self.dropped: list = []
        self.loaded: list = []

    def create(self, overwrite: bool = False) -> None:
        pass

    def query(self, _query):
        self.queries += 1
        if self.queries > 5:
            raise AssertionError("the store keeps re-querying: it would loop forever")
        limit = getattr(_query, "_num_results", None)  # RedisVL keeps the limit privately; like Redis, honour it
        return list(self.rows)[:limit] if limit else list(self.rows)

    def drop_keys(self, keys) -> int:
        self.dropped.append(keys)
        return self.drop_result

    def load(self, data, keys=None, ttl=None) -> None:
        self.loaded.append((data, keys, ttl))


IDENTITY = CacheIdentity(provider="p", model="m", system_prompt_hash="h", temperature=0.0, max_tokens=10)


def _row(entry_id: str, *, expires_at: float, distance: float = 0.0) -> dict:
    return {
        "entry_id": entry_id,
        "semantic_lookup_text": "user: hi",
        "response_json": serialize_response(ChatCompletion(content="cached", model="m")),
        "identity_json": json.dumps({"provider": "p", "model": "m", "system_prompt_hash": "h",
                                     "temperature": 0.0, "max_tokens": 10, "params_hash": ""}),
        "created_at": time() - 100,
        "expires_at": expires_at,
        "hit_count": 0,
        "vector_distance": distance,
    }


def _store(rows, **kwargs) -> RedisVLCacheStore:
    return RedisVLCacheStore(redis_url="redis://unused", embedding_dimensions=4, index=FakeIndex(rows, **kwargs))


def test_redis_store_never_serves_an_entry_whose_stored_expiry_has_passed() -> None:
    expired = _row("old", expires_at=time() - 5)
    store = _store([expired])
    assert store.find_exact(identity=IDENTITY, semantic_lookup_text="user: hi") is None
    lookup = store.lookup(identity=IDENTITY, semantic_lookup_text="user: other", embedding=(1, 0, 0, 0), similarity_threshold=0.9)
    assert lookup.entry is None


def test_redis_store_skips_an_expired_top_candidate_and_uses_the_next_live_one() -> None:
    rows = [_row("old", expires_at=time() - 5, distance=0.0), _row("new", expires_at=time() + 60, distance=0.02)]
    lookup = _store(rows).lookup(identity=IDENTITY, semantic_lookup_text="user: other", embedding=(1, 0, 0, 0), similarity_threshold=0.9)
    assert lookup.entry is not None and lookup.entry.entry_id == "new"


def test_redis_entries_without_ttl_are_not_treated_as_expired() -> None:
    store = _store([_row("forever", expires_at=0.0)])
    assert store.find_exact(identity=IDENTITY, semantic_lookup_text="user: hi") is not None


def test_redis_invalidate_stops_when_nothing_could_be_deleted() -> None:
    store = _store([_row("stuck", expires_at=0.0)], drop_result=0)
    assert store.invalidate(model="m") == 0  # would raise inside FakeIndex.query if it looped


def test_an_expired_exact_row_does_not_shadow_a_live_duplicate() -> None:
    rows = [_row("old", expires_at=time() - 5), _row("new", expires_at=time() + 60)]
    found = _store(rows).find_exact(identity=IDENTITY, semantic_lookup_text="user: hi")
    assert found is not None and found.entry_id == "new"


def test_storing_the_same_text_again_drops_every_earlier_row_including_expired_ones() -> None:
    index = FakeIndex([_row("old", expires_at=time() - 5), _row("live", expires_at=time() + 60)])
    store = RedisVLCacheStore(redis_url="redis://unused", embedding_dimensions=4, index=index)
    store.store(
        identity=IDENTITY,
        semantic_lookup_text="user: hi",
        embedding=(1, 0, 0, 0),
        response=ChatCompletion(content="fresh", model="m"),
    )
    (dropped,) = index.dropped
    assert sorted(key.rsplit(":", 1)[-1] for key in dropped) == ["live", "old"]
    assert len(index.loaded) == 1
