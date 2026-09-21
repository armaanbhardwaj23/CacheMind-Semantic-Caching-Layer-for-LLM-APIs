from fastapi.testclient import TestClient

from semantic_cache.api.app import create_app
from semantic_cache.openrouter import OpenRouterError
from semantic_cache.service import CacheService
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider


def _payload() -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is France's capital?"}],
        "temperature": 0.0,
        "max_tokens": 100,
    }


def test_api_returns_openai_style_response_and_cache_headers() -> None:
    provider = RecordingProvider(response_text="Paris is the capital of France.")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    client = TestClient(create_app(service))

    first = client.post("/v1/chat/completions", json=_payload())
    second = client.post("/v1/chat/completions", json=_payload())

    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["object"] == "chat.completion"
    assert second.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Paris is the capital of France.",
    }
    assert provider.call_count == 1


def test_api_rejects_an_invalid_chat_request_before_calling_provider() -> None:
    provider = RecordingProvider(response_text="unused")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    client = TestClient(create_app(service))

    response = client.post("/v1/chat/completions", json={"model": "test-model"})

    assert response.status_code == 422
    assert provider.call_count == 0


def test_api_bypasses_cache_when_the_client_sends_no_store() -> None:
    provider = RecordingProvider(response_text="Action completed")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/v1/chat/completions",
        json=_payload(),
        headers={"Cache-Control": "no-store"},
    )

    assert response.status_code == 200
    assert response.headers["X-Cache"] == "BYPASS"
    assert service.entry_count == 0
    assert service.metrics.bypasses == 1


def test_api_rejects_unsupported_response_affecting_fields() -> None:
    provider = RecordingProvider(response_text="unused")
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
    )
    client = TestClient(create_app(service))
    payload = _payload() | {"tools": []}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert provider.call_count == 0


class FailingProvider:
    def complete(self, request):
        raise OpenRouterError("Upstream timeout")


def test_api_maps_provider_failures_to_bad_gateway() -> None:
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=FailingProvider(),
        similarity_threshold=0.95,
    )
    client = TestClient(create_app(service))

    response = client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream provider failed."}


# -- additions: streaming, params, usage, admin, prometheus -------------------

from semantic_cache.models import TokenUsage
from semantic_cache.prometheus import PrometheusMetrics


def _app(provider=None, *, admin_token=None, with_prometheus=False):
    provider = provider or RecordingProvider(response_text="Paris is the capital", usage=TokenUsage(5, 4))
    prom = PrometheusMetrics(similarity_threshold=0.95) if with_prometheus else None
    service = CacheService(
        embedder=DeterministicEmbedder(),
        provider=provider,
        similarity_threshold=0.95,
        observers=[prom] if prom else [],
    )
    return service, provider, TestClient(create_app(service, prometheus=prom, admin_token=admin_token))


def test_response_includes_usage_and_the_real_finish_reason() -> None:
    _, _, client = _app(RecordingProvider(response_text="cut", usage=TokenUsage(5, 4), finish_reason="length"))
    body = client.post("/v1/chat/completions", json=_payload()).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}


def test_max_tokens_and_temperature_are_optional_like_the_openai_api() -> None:
    _, provider, client = _app()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert provider.requests[0].temperature is None and provider.requests[0].max_tokens is None


def test_different_sampling_parameters_do_not_share_entries() -> None:
    _, provider, client = _app()
    client.post("/v1/chat/completions", json=_payload())
    different = client.post("/v1/chat/completions", json=_payload() | {"top_p": 0.5})
    seeded = client.post("/v1/chat/completions", json=_payload() | {"seed": 1})
    repeat = client.post("/v1/chat/completions", json=_payload() | {"top_p": 0.5})
    assert (different.headers["X-Cache"], seeded.headers["X-Cache"], repeat.headers["X-Cache"]) == ("MISS", "MISS", "HIT")
    assert provider.call_count == 3


def test_similarity_header_is_present_for_hits() -> None:
    _, _, client = _app()
    client.post("/v1/chat/completions", json=_payload())
    hit = client.post("/v1/chat/completions", json=_payload())
    assert hit.headers["X-Cache-Similarity"] == "1.0000"


def test_stream_true_returns_sse_and_populates_the_cache() -> None:
    _, provider, client = _app(RecordingProvider(response_text="Paris is the capital"))
    with client.stream("POST", "/v1/chat/completions", json=_payload() | {"stream": True}) as response:
        assert response.headers["X-Cache"] == "MISS"
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    import json as _json

    events = [_json.loads(line[6:]) for line in lines[:-1]]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "Paris is the capital"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"

    again = client.post("/v1/chat/completions", json=_payload())
    assert again.headers["X-Cache"] == "HIT" and provider.call_count == 1


def test_admin_endpoints_are_disabled_without_a_token() -> None:
    _, _, client = _app()
    assert client.delete("/v1/cache").status_code == 403
    assert client.post("/v1/cache/invalidate", json={"model": "m"}).status_code == 403


def test_admin_invalidation_by_model_system_prompt_and_clear() -> None:
    service, _, client = _app(admin_token="secret")
    with_system = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are v1."},
            {"role": "user", "content": "hello"},
        ],
    }
    client.post("/v1/chat/completions", json=with_system)
    client.post("/v1/chat/completions", json=_payload())
    assert service.entry_count == 2

    assert client.delete("/v1/cache", headers={"X-Admin-Token": "wrong"}).status_code == 401
    ok = client.post(
        "/v1/cache/invalidate", json={"system_prompt": "You are v1."}, headers={"X-Admin-Token": "secret"}
    )
    assert ok.json() == {"removed": 1}
    assert client.post("/v1/cache/invalidate", json={}, headers={"X-Admin-Token": "secret"}).status_code == 422
    assert client.delete("/v1/cache", headers={"X-Admin-Token": "secret"}).json() == {"removed": 1}
    assert service.entry_count == 0


def test_prometheus_endpoint_exposes_cache_counters_without_prompt_text() -> None:
    _, _, client = _app(with_prometheus=True)
    client.post("/v1/chat/completions", json=_payload())
    client.post("/v1/chat/completions", json=_payload())
    text = client.get("/metrics").text
    assert 'cache_requests_total{model="test-model",status="HIT"} 1.0' in text
    assert 'cache_requests_total{model="test-model",status="MISS"} 1.0' in text
    assert "cache_entries 1.0" in text
    assert "France" not in text


def test_metrics_summary_and_health() -> None:
    _, _, client = _app()
    client.post("/v1/chat/completions", json=_payload())
    summary = client.get("/metrics/summary").json()
    assert summary["misses"] == 1 and summary["entry_count"] == 1
    assert client.get("/health").json() == {"status": "ok"}
