"""Redis store: offline checks always run; the live round-trip needs a server.

Run the integration test with Redis 8 (built-in query engine) available, e.g.
``brew install redis && redis-server`` (or the ``redis:8`` image) then ``REDIS_URL=redis://localhost:6379 uv run pytest``.
"""

import os

import pytest

pytest.importorskip("redisvl")

from semantic_cache.models import CacheIdentity, ChatCompletion, TokenUsage
from semantic_cache.redis_store import (
    RedisVLCacheStore,
    build_schema,
    deserialize_response,
    identity_filter,
    invalidation_filter,
    serialize_response,
)


def _identity(**overrides) -> CacheIdentity:
    values = dict(provider="openrouter", model="openai/gpt-4.1-mini", system_prompt_hash="abc", temperature=0.0, max_tokens=50)
    values.update(overrides)
    return CacheIdentity(**values)


def test_schema_is_valid_redisvl_and_uses_cosine_hnsw_float32() -> None:
    from redisvl.schema import IndexSchema

    schema = IndexSchema.from_dict(build_schema(index_name="t", prefix="t:e", dimensions=8))
    field = schema.fields["embedding"]
    assert field.attrs.dims == 8
    assert field.attrs.distance_metric.value.lower() == "cosine"
    assert set(schema.field_names) >= {"identity_key", "text_hash", "model", "system_prompt_hash", "embedding"}


def test_vector_queries_are_prefiltered_by_the_full_identity() -> None:
    a = str(identity_filter(_identity()))
    b = str(identity_filter(_identity(temperature=0.7)))
    assert a != b and a.startswith("@identity_key:{")


def test_invalidation_filter_combines_filters_and_requires_one() -> None:
    expression = str(invalidation_filter(model="openai/gpt-4.1-mini", provider=None, system_prompt_hash="abc"))
    assert "@model:" in expression and "@system_prompt_hash:" in expression and "@provider" not in expression
    with pytest.raises(ValueError):
        invalidation_filter(model=None, provider=None, system_prompt_hash=None)


def test_response_serialization_round_trips_usage_and_finish_reason() -> None:
    original = ChatCompletion(content="héllo", model="m", usage=TokenUsage(3, 4), finish_reason="stop")
    assert deserialize_response(serialize_response(original)) == original
    assert deserialize_response(serialize_response(ChatCompletion(content="x", model="m"))).usage is None


@pytest.mark.skipif(not os.environ.get("REDIS_URL"), reason="needs a Redis 8 server (set REDIS_URL)")
def test_live_round_trip_ttl_isolation_and_invalidation() -> None:  # pragma: no cover - needs server
    store = RedisVLCacheStore(
        redis_url=os.environ["REDIS_URL"], embedding_dimensions=4, index_name="cachemind_test", prefix="cachemind_test:e"
    )
    store.clear()
    identity = _identity()
    store.store(identity=identity, semantic_lookup_text="hello", embedding=(1, 0, 0, 0), response=ChatCompletion("hi", "m"))

    assert store.find_exact(identity=identity, semantic_lookup_text="hello") is not None
    near = store.lookup(identity=identity, semantic_lookup_text="hey", embedding=(0.99, 0.1, 0, 0), similarity_threshold=0.9)
    assert near.entry is not None and near.similarity_score > 0.9
    other = store.lookup(identity=_identity(temperature=0.5), semantic_lookup_text="hey", embedding=(1, 0, 0, 0), similarity_threshold=0.0)
    assert other.entry is None
    assert store.invalidate(model=identity.model) == 1
    assert store.entry_count == 0
