"""Store-level behaviour: identity isolation, threshold, TTL, invalidation, eviction."""

import pytest

from semantic_cache.cache import InMemorySemanticCache, cosine_similarity
from semantic_cache.models import CacheIdentity, ChatCompletion


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def identity(**overrides) -> CacheIdentity:
    values = dict(
        provider="p", model="m", system_prompt_hash="s", temperature=0.0, max_tokens=100
    )
    values.update(overrides)
    return CacheIdentity(**values)


def response(text: str = "answer") -> ChatCompletion:
    return ChatCompletion(content=text, model="m")


def test_lookup_returns_the_best_compatible_candidate_above_threshold() -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="a", embedding=(1.0, 0.0), response=response("A"))
    cache.store(identity=identity(), semantic_lookup_text="b", embedding=(0.8, 0.6), response=response("B"))

    found = cache.lookup(
        identity=identity(), semantic_lookup_text="q", embedding=(0.79, 0.61), similarity_threshold=0.95
    )

    assert found.entry is not None and found.entry.response.content == "B"
    assert found.similarity_score == pytest.approx(cosine_similarity((0.79, 0.61), (0.8, 0.6)))


def test_threshold_is_inclusive_and_below_threshold_still_reports_the_score() -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="a", embedding=(1.0, 0.0), response=response())
    score = cosine_similarity((1.0, 0.0), (0.8, 0.6))

    at = cache.lookup(identity=identity(), semantic_lookup_text="q", embedding=(0.8, 0.6), similarity_threshold=score)
    above = cache.lookup(identity=identity(), semantic_lookup_text="q", embedding=(0.8, 0.6), similarity_threshold=score + 1e-4)

    assert at.entry is not None
    assert above.entry is None
    assert above.similarity_score == pytest.approx(score)  # needed for near-miss analysis


@pytest.mark.parametrize(
    "different",
    [
        {"model": "other"},
        {"provider": "other"},
        {"system_prompt_hash": "other"},
        {"temperature": 0.7},
        {"temperature": None},
        {"max_tokens": 200},
        {"max_tokens": None},
        {"params_hash": "top_p=0.5"},
    ],
)
def test_entries_are_never_shared_across_identities(different) -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="same text", embedding=(1.0, 0.0), response=response())

    found = cache.lookup(
        identity=identity(**different),
        semantic_lookup_text="same text",
        embedding=(1.0, 0.0),
        similarity_threshold=0.0,
    )

    assert found.entry is None and found.similarity_score is None
    assert cache.find_exact(identity=identity(**different), semantic_lookup_text="same text") is None


def test_exact_text_is_found_without_a_vector() -> None:
    cache = InMemorySemanticCache()
    stored = cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response())
    assert cache.find_exact(identity=identity(), semantic_lookup_text="t") is stored


def test_restoring_identical_text_replaces_instead_of_duplicating() -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response("old"))
    cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response("new"))

    assert cache.entry_count == 1
    assert cache.find_exact(identity=identity(), semantic_lookup_text="t").response.content == "new"


def test_entries_expire_after_their_ttl() -> None:
    clock = FakeClock()
    cache = InMemorySemanticCache(clock=clock)
    cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response(), ttl_seconds=60)
    cache.store(identity=identity(), semantic_lookup_text="forever", embedding=(0.0, 1.0), response=response())

    clock.now += 59
    assert cache.find_exact(identity=identity(), semantic_lookup_text="t") is not None
    clock.now += 2

    assert cache.find_exact(identity=identity(), semantic_lookup_text="t") is None
    expired_lookup = cache.lookup(identity=identity(), semantic_lookup_text="q", embedding=(1.0, 0.0), similarity_threshold=0.9)
    assert expired_lookup.entry is None
    assert cache.entry_count == 1  # only the no-TTL entry survives


def test_invalidate_by_model_and_by_system_prompt_hash() -> None:
    cache = InMemorySemanticCache()
    for model, prompt in [("m1", "s1"), ("m1", "s2"), ("m2", "s1")]:
        cache.store(
            identity=identity(model=model, system_prompt_hash=prompt),
            semantic_lookup_text=f"{model}{prompt}",
            embedding=(1.0, 0.0),
            response=response(),
        )

    assert cache.invalidate(model="m1") == 2
    assert cache.entry_count == 1
    assert cache.invalidate(system_prompt_hash="s1") == 1
    assert cache.entry_count == 0


def test_invalidate_requires_a_filter_and_clear_removes_everything() -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response())
    with pytest.raises(ValueError):
        cache.invalidate()
    assert cache.clear() == 1 and cache.entry_count == 0


def test_capacity_evicts_the_least_recently_used_entry() -> None:
    clock = FakeClock()
    cache = InMemorySemanticCache(max_entries=2, clock=clock)
    first = cache.store(identity=identity(), semantic_lookup_text="a", embedding=(1.0, 0.0), response=response("A"))
    clock.now += 1
    cache.store(identity=identity(), semantic_lookup_text="b", embedding=(0.0, 1.0), response=response("B"))
    clock.now += 1
    cache.record_hit(first.entry_id)  # 'a' becomes most recently used
    clock.now += 1
    cache.store(identity=identity(), semantic_lookup_text="c", embedding=(0.7, 0.7), response=response("C"))

    assert cache.entry_count == 2 and cache.evictions == 1
    assert cache.find_exact(identity=identity(), semantic_lookup_text="a") is not None
    assert cache.find_exact(identity=identity(), semantic_lookup_text="b") is None


def test_hit_count_is_tracked() -> None:
    cache = InMemorySemanticCache()
    entry = cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response())
    cache.record_hit(entry.entry_id)
    cache.record_hit(entry.entry_id)
    assert entry.hit_count == 2


def test_dimension_mismatch_and_zero_vectors_are_rejected() -> None:
    cache = InMemorySemanticCache()
    cache.store(identity=identity(), semantic_lookup_text="t", embedding=(1.0, 0.0), response=response())
    with pytest.raises(ValueError):
        cache.lookup(identity=identity(), semantic_lookup_text="q", embedding=(1.0, 0.0, 0.0), similarity_threshold=0.5)
    with pytest.raises(ValueError):
        cache.store(identity=identity(), semantic_lookup_text="z", embedding=(0.0, 0.0), response=response())


def test_grows_past_initial_matrix_capacity() -> None:
    cache = InMemorySemanticCache()
    for n in range(40):
        cache.store(identity=identity(), semantic_lookup_text=str(n), embedding=(1.0, float(n) + 1), response=response(str(n)))
    found = cache.lookup(identity=identity(), semantic_lookup_text="q", embedding=(1.0, 40.0), similarity_threshold=0.99)
    assert cache.entry_count == 40
    assert found.entry is not None and found.entry.response.content == "39"
