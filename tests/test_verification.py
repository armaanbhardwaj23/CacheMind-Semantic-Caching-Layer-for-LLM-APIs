from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.service import CacheService
from semantic_cache.testing import RecordingProvider
from semantic_cache.verification import ConstraintGuard


def test_constraint_guard_rejects_explicit_answer_changing_differences() -> None:
    guard = ConstraintGuard()

    assert not guard.verify(
        cached_text="user: Summarize this in 3 bullets.",
        incoming_text="user: Summarize this in 8 bullets.",
    ).accepted
    assert not guard.verify(
        cached_text="user: Give directions from Toronto to Ottawa.",
        incoming_text="user: Give directions from Ottawa to Toronto.",
    ).accepted
    assert not guard.verify(
        cached_text="user: List foods that contain gluten.",
        incoming_text="user: List foods that do not contain gluten.",
    ).accepted
    assert not guard.verify(
        cached_text="user: Rank cities from lowest to highest cost.",
        incoming_text="user: Rank cities from highest to lowest cost.",
    ).accepted


def test_constraint_guard_allows_when_its_limited_checks_match() -> None:
    decision = ConstraintGuard().verify(
        cached_text="user: What is the capital of France?",
        incoming_text="user: Name France's capital city.",
    )

    assert decision.accepted


def test_guarded_semantic_candidate_becomes_a_measured_miss() -> None:
    class SimilarEmbedder:
        def embed(self, text: str) -> tuple[float, ...]:
            return (1.0, 0.0)

    provider = RecordingProvider(response_text="provider response")
    service = CacheService(
        embedder=SimilarEmbedder(), provider=provider, similarity_threshold=0.95
    )
    cached_request = ChatRequest(
        provider="test", model="model",
        messages=[ChatMessage(role="user", content="Summarize this in 3 bullets.")],
        temperature=0, max_tokens=20,
    )
    changed_request = ChatRequest(
        provider="test", model="model",
        messages=[ChatMessage(role="user", content="Summarize this in 8 bullets.")],
        temperature=0, max_tokens=20,
    )

    assert service.complete(cached_request).cache_status == "MISS"
    result = service.complete(changed_request)

    assert result.cache_status == "MISS"
    assert result.verification_rejected is True
    assert provider.call_count == 2
    assert service.metrics.verification_rejections == 1
