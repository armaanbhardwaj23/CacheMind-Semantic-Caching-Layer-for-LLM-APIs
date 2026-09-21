"""Streaming, TTL policy, truncation and invalidation through the service."""

import pytest

from semantic_cache.cache import InMemorySemanticCache
from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.policy import CachePolicy
from semantic_cache.service import CacheService
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider


def make_request(text: str = "What is a mutex?", **overrides) -> ChatRequest:
    values = dict(
        provider="p", model="m", messages=[ChatMessage(role="user", content=text)], temperature=0.0, max_tokens=50
    )
    values.update(overrides)
    return ChatRequest(**values)


def make_service(provider, **kwargs) -> CacheService:
    return CacheService(embedder=DeterministicEmbedder(), provider=provider, similarity_threshold=0.95, **kwargs)


def test_truncated_response_is_returned_but_never_cached() -> None:
    provider = RecordingProvider(response_text="A mutex is a", finish_reason="length")
    service = make_service(provider)

    first = service.complete(make_request())
    second = service.complete(make_request())

    assert first.response.content == "A mutex is a"
    assert (first.cache_status, second.cache_status) == ("MISS", "MISS")
    assert provider.call_count == 2 and service.entry_count == 0
    assert service.metrics.not_stored_reasons == {"not_stored_finish_reason_length": 2}


def test_empty_response_is_not_cached() -> None:
    service = make_service(RecordingProvider(response_text="   "))
    service.complete(make_request())
    assert service.entry_count == 0


def test_realtime_request_bypasses_cache_and_records_the_reason() -> None:
    provider = RecordingProvider(response_text="Sunny")
    service = make_service(provider)

    a = service.complete(make_request("What's the weather in Toronto?"))
    b = service.complete(make_request("What's the weather in Toronto?"))

    assert (a.cache_status, b.cache_status) == ("BYPASS", "BYPASS")
    assert provider.call_count == 2 and service.entry_count == 0
    assert service.metrics.bypass_reasons == {"realtime": 2}


def test_ttl_policy_expires_entries_so_the_provider_is_called_again() -> None:
    now = [1000.0]
    cache = InMemorySemanticCache(clock=lambda: now[0])
    provider = RecordingProvider(response_text="answer")
    service = make_service(provider, cache=cache, policy=CachePolicy(time_relative_ttl_seconds=60))

    service.complete(make_request("What is the latest Python version?"))
    assert service.complete(make_request("What is the latest Python version?")).cache_status == "HIT"
    now[0] += 61
    assert service.complete(make_request("What is the latest Python version?")).cache_status == "MISS"
    assert provider.call_count == 2


def test_invalidating_a_system_prompt_stops_serving_its_entries() -> None:
    from semantic_cache.identity import hash_system_prompt

    provider = RecordingProvider(response_text="answer")
    service = make_service(provider)
    messages = [ChatMessage(role="system", content="You are v1."), ChatMessage(role="user", content="Hi there")]
    request = ChatRequest(provider="p", model="m", messages=messages)

    service.complete(request)
    assert service.complete(request).cache_status == "HIT"
    assert service.cache.invalidate(system_prompt_hash=hash_system_prompt("You are v1.")) == 1
    assert service.complete(request).cache_status == "MISS"


def test_hit_increments_the_entry_hit_count() -> None:
    service = make_service(RecordingProvider(response_text="answer"))
    service.complete(make_request())
    service.complete(make_request())
    service.complete(make_request())
    entry = next(iter(service.cache._by_id.values()))
    assert entry.hit_count == 2


# -- streaming ---------------------------------------------------------------


def consume(session) -> str:
    return "".join(delta.content for delta in session.chunks)


def test_streamed_miss_relays_deltas_then_caches_the_complete_response() -> None:
    provider = RecordingProvider(response_text="A mutex is a lock")
    service = make_service(provider)

    session = service.stream(make_request())
    assert session.cache_status == "MISS"
    assert consume(session) == "A mutex is a lock"

    replay = service.stream(make_request())
    assert replay.cache_status == "HIT" and replay.similarity_score == 1.0
    assert consume(replay) == "A mutex is a lock"
    assert provider.call_count == 1


def test_provider_failure_mid_stream_is_never_cached() -> None:
    provider = RecordingProvider(response_text="one two three four", fail_stream_after=2)
    service = make_service(provider)

    session = service.stream(make_request())
    with pytest.raises(RuntimeError):
        consume(session)

    assert service.entry_count == 0
    assert service.stream(make_request()).cache_status == "MISS"


def test_client_disconnect_mid_stream_is_never_cached() -> None:
    provider = RecordingProvider(response_text="one two three four")
    service = make_service(provider)

    session = service.stream(make_request())
    iterator = session.chunks
    next(iterator)  # client reads one delta then goes away
    iterator.close()

    assert service.entry_count == 0


def test_stream_that_ends_without_a_finish_reason_is_not_cached() -> None:
    class TruncatedStream(RecordingProvider):
        def stream(self, request):
            from semantic_cache.models import StreamDelta

            self.requests.append(request)
            yield StreamDelta(content="partial answer")  # connection just ends

    service = make_service(TruncatedStream(response_text="unused"))
    consume(service.stream(make_request()))
    assert service.entry_count == 0
    assert "not_stored_finish_reason_incomplete_stream" in service.metrics.not_stored_reasons


def test_streamed_length_finish_reason_is_not_cached() -> None:
    service = make_service(RecordingProvider(response_text="cut off", finish_reason="length"))
    consume(service.stream(make_request()))
    assert service.entry_count == 0


def test_no_store_stream_bypasses() -> None:
    provider = RecordingProvider(response_text="fine")
    service = make_service(provider)
    session = service.stream(make_request(), cacheable=False)
    assert session.cache_status == "BYPASS" and session.reason == "client_no_store"
    consume(session)
    assert service.entry_count == 0


def test_streaming_requires_a_streaming_provider() -> None:
    class NoStream:
        def complete(self, request):
            raise AssertionError("unused")

    with pytest.raises(NotImplementedError):
        make_service(NoStream()).stream(make_request())
