"""Behavior tests for the framework-free, in-memory cache path."""

from dataclasses import replace

from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.service import CacheService
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider


def test_empty_cache_is_a_miss_then_stores_the_successful_response() -> None:
    provider = RecordingProvider(response_text="Paris is the capital of France.")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="What is France's capital?")],
        temperature=0.0,
        max_tokens=100,
    )

    result = service.complete(request)

    assert result.cache_status == "MISS"
    assert result.response.content == "Paris is the capital of France."
    assert provider.call_count == 1
    assert service.entry_count == 1


def test_compatible_exact_repeat_is_a_hit_and_skips_the_provider() -> None:
    provider = RecordingProvider(response_text="Paris is the capital of France.")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="What is France's capital?")],
        temperature=0.0,
        max_tokens=100,
    )

    first = service.complete(request)
    second = service.complete(request)

    assert first.cache_status == "MISS"
    assert second.cache_status == "HIT"
    assert second.response == first.response
    assert provider.call_count == 1
    assert service.entry_count == 1


def test_exact_repeat_skips_a_second_embedding_call() -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> tuple[float, ...]:
            self.calls += 1
            return (1.0, 0.0)

    embedder = CountingEmbedder()
    provider = RecordingProvider(response_text="Paris")
    service = CacheService(embedder=embedder, provider=provider, similarity_threshold=0.95)
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="What is France's capital?")],
        temperature=0.0,
        max_tokens=100,
    )

    first = service.complete(request)
    second = service.complete(request)

    assert (first.cache_status, second.cache_status) == ("MISS", "HIT")
    assert embedder.calls == 1


def test_changed_execution_identity_never_reuses_a_response() -> None:
    provider = RecordingProvider(response_text="Paris is the capital of France.")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[
            ChatMessage(role="system", content="Answer concisely."),
            ChatMessage(role="user", content="What is France's capital?"),
        ],
        temperature=0.0,
        max_tokens=100,
    )

    requests_with_changed_identity = [
        replace(request, model="another-model"),
        replace(request, temperature=0.7),
        replace(request, max_tokens=200),
        replace(
            request,
            messages=[
                ChatMessage(role="system", content="Answer with detailed context."),
                ChatMessage(role="user", content="What is France's capital?"),
            ],
        ),
    ]

    assert service.complete(request).cache_status == "MISS"
    for changed_request in requests_with_changed_identity:
        assert service.complete(changed_request).cache_status == "MISS"

    assert provider.call_count == 5
    assert service.entry_count == 5


def test_explicit_bypass_skips_embedding_and_never_stores_a_response() -> None:
    class FailingEmbedder:
        def embed(self, text: str) -> tuple[float, ...]:
            raise AssertionError("A bypassed request must not be embedded.")

    provider = RecordingProvider(response_text="Action completed")
    service = CacheService(
        embedder=FailingEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="Send an email")],
        temperature=0.0,
        max_tokens=20,
    )

    result = service.complete(request, cacheable=False)

    assert result.cache_status == "BYPASS"
    assert provider.call_count == 1
    assert service.entry_count == 0
    assert service.metrics.bypasses == 1


def test_guard_switch_from_environment(monkeypatch) -> None:
    from semantic_cache.config import CacheSettings

    monkeypatch.delenv("CACHE_CONSTRAINT_GUARD", raising=False)
    assert CacheSettings.from_env().constraint_guard is True
    monkeypatch.setenv("CACHE_CONSTRAINT_GUARD", "0")
    assert CacheSettings.from_env().constraint_guard is False
