import json

import httpx
import pytest

from semantic_cache.config import OpenRouterSettings
from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.openrouter import OpenRouterClient, OpenRouterError
from semantic_cache.service import CacheService


def _client(handler: httpx.MockTransport) -> OpenRouterClient:
    settings = OpenRouterSettings(
        api_key="test-key",
        embedding_model="openai/text-embedding-3-small",
        http_referer="https://example.test",
        app_title="CacheMind tests",
    )
    return OpenRouterClient(
        settings,
        client=httpx.Client(
            transport=handler,
            base_url="https://openrouter.ai/api/v1",
        ),
    )


def test_openrouter_client_sends_embedding_and_chat_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        if request.url.path == "/api/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-4.1-mini",
                    "choices": [{"message": {"content": "Paris."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            )
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    request = ChatRequest(
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.0,
        max_tokens=50,
    )

    assert client.embed("Capital of France?") == (0.1, 0.2)
    completion = client.complete(request)

    assert completion.content == "Paris."
    assert completion.model == "openai/gpt-4.1-mini"
    assert completion.usage is not None
    assert completion.usage.input_tokens == 5
    assert completion.usage.output_tokens == 2
    assert [item.url.path for item in requests] == [
        "/api/v1/embeddings",
        "/api/v1/chat/completions",
    ]
    assert json.loads(requests[0].content) == {
        "input": "Capital of France?",
        "model": "openai/text-embedding-3-small",
        "encoding_format": "float",
    }
    assert json.loads(requests[1].content)["stream"] is False
    assert requests[0].headers["Authorization"] == "Bearer test-key"


def test_openrouter_client_turns_http_errors_into_safe_provider_errors() -> None:
    client = _client(httpx.MockTransport(lambda _: httpx.Response(401)))

    with pytest.raises(OpenRouterError, match="status 401"):
        client.embed("Capital of France?")


def test_exact_cache_hit_avoids_openrouter_chat_and_second_embedding_lookup() -> None:
    endpoint_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoint_paths.append(request.url.path)
        if request.url.path == "/api/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Paris."}, "finish_reason": "stop"}]},
        )

    openrouter = _client(httpx.MockTransport(handler))
    service = CacheService(
        embedder=openrouter,
        provider=openrouter,
        similarity_threshold=0.95,
    )
    request = ChatRequest(
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.0,
        max_tokens=50,
    )

    first = service.complete(request)
    second = service.complete(request)

    assert (first.cache_status, second.cache_status) == ("MISS", "HIT")
    assert endpoint_paths == [
        "/api/v1/embeddings",
        "/api/v1/chat/completions",
    ]


def test_settings_require_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterSettings.from_env()


def _chat_request(**overrides) -> ChatRequest:
    values = dict(
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        messages=[ChatMessage(role="user", content="Capital of France?")],
    )
    values.update(overrides)
    return ChatRequest(**values)


def test_omitted_parameters_are_not_sent_to_the_provider() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        )

    _client(httpx.MockTransport(handler)).complete(_chat_request())
    _client(httpx.MockTransport(handler)).complete(_chat_request(temperature=0.0, top_p=0.5, seed=7, stop=("END",)))

    assert "temperature" not in seen[0] and "max_tokens" not in seen[0]
    assert seen[1]["temperature"] == 0.0 and seen[1]["top_p"] == 0.5
    assert seen[1]["seed"] == 7 and seen[1]["stop"] == ["END"]


def test_finish_reason_is_captured_and_missing_reason_is_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]})

    assert _client(httpx.MockTransport(handler)).complete(_chat_request()).finish_reason == "length"

    def handler_missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    assert _client(httpx.MockTransport(handler_missing)).complete(_chat_request()).finish_reason == "unknown"


def _sse(*events: str) -> bytes:
    return ("".join(f"data: {event}\n\n" for event in events)).encode()


def test_stream_parses_sse_deltas_usage_and_ignores_keepalives() -> None:
    body = b": OPENROUTER PROCESSING\n\n" + _sse(
        json.dumps({"model": "m", "choices": [{"delta": {"content": "Hel"}}]}),
        json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        json.dumps({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}),
        "[DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    deltas = list(_client(httpx.MockTransport(handler)).stream(_chat_request()))

    assert "".join(d.content for d in deltas) == "Hello"
    assert any(d.finish_reason == "stop" for d in deltas)
    assert deltas[-1].usage is not None and deltas[-1].usage.output_tokens == 2


def test_stream_turns_http_and_midstream_errors_into_safe_errors() -> None:
    with pytest.raises(OpenRouterError, match="status 500"):
        list(_client(httpx.MockTransport(lambda r: httpx.Response(500))).stream(_chat_request()))

    error_body = _sse(json.dumps({"error": {"message": "secret prompt text"}}))
    with pytest.raises(OpenRouterError) as info:
        list(_client(httpx.MockTransport(lambda r: httpx.Response(200, content=error_body))).stream(_chat_request()))
    assert "secret prompt text" not in str(info.value)
