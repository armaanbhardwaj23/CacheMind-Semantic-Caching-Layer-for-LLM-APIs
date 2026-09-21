from semantic_cache.metrics import CacheMetrics, TokenPricing
from semantic_cache.models import ChatMessage, ChatRequest, TokenUsage
from semantic_cache.service import CacheService
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider


def test_service_records_hit_miss_latency_similarity_and_avoided_tokens() -> None:
    provider = RecordingProvider(
        response_text="Paris", usage=TokenUsage(input_tokens=100, output_tokens=20)
    )
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
        metrics=CacheMetrics(
            similarity_threshold=0.95,
            token_pricing=TokenPricing(
                input_per_million_tokens_usd=2.0,
                output_per_million_tokens_usd=10.0,
            ),
        ),
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.0,
        max_tokens=20,
    )

    service.complete(request)
    service.complete(request)
    snapshot = service.metrics

    assert snapshot.total_requests == 2
    assert snapshot.eligible_requests == 2
    assert snapshot.hits == 1
    assert snapshot.misses == 1
    assert snapshot.bypasses == 0
    assert snapshot.hit_rate == 0.5
    assert snapshot.average_hit_latency_ms is not None
    assert snapshot.average_miss_latency_ms is not None
    assert snapshot.average_provider_latency_ms is not None
    assert snapshot.similarity_scores == (1.0,)
    assert snapshot.input_tokens_avoided == 100
    assert snapshot.output_tokens_avoided == 20
    assert snapshot.estimated_chat_cost_avoided_usd == 0.0004


def test_metrics_summary_endpoint_exposes_aggregates_without_prompt_content() -> None:
    from fastapi.testclient import TestClient

    from semantic_cache.api.app import create_app

    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=RecordingProvider(response_text="Paris"),
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.0,
        max_tokens=20,
    )
    service.complete(request)

    response = TestClient(create_app(service)).get("/metrics/summary")

    assert response.status_code == 200
    assert response.json()["total_requests"] == 1
    assert "Capital of France?" not in response.text


def test_bypass_latency_is_not_reported_as_miss_latency() -> None:
    provider = RecordingProvider(response_text="Paris")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.0,
        max_tokens=20,
    )

    service.complete(request, cacheable=False)
    snapshot = service.metrics

    assert snapshot.misses == 0
    assert snapshot.bypasses == 1
    assert snapshot.average_miss_latency_ms is None
    assert snapshot.average_bypass_latency_ms is not None
