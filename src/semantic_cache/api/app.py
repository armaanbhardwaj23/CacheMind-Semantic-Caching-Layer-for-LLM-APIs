"""OpenAI-compatible HTTP API over the cache's framework-free core service.

A client switches to the cache by changing its base URL. Omitted parameters stay
omitted (the provider applies its own defaults) and are part of cache identity.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from secrets import compare_digest
from time import time
from typing import Iterator, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from semantic_cache.identity import hash_system_prompt
from semantic_cache.models import ChatCompletion, ChatMessage, ChatRequest, CompletionResult, StreamDelta
from semantic_cache.openrouter import OpenRouterError
from semantic_cache.prometheus import PrometheusMetrics
from semantic_cache.service import CacheService


class ChatMessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionInput(BaseModel):
    # Unknown fields are rejected: silently ignoring a response-affecting field
    # (tools, response_format, n, ...) would serve wrong cached answers.
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessageInput] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool = False


class InvalidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = Field(default=None, description="Hashed server-side.")
    system_prompt_hash: str | None = None


def _to_chat_request(payload: ChatCompletionInput) -> ChatRequest:
    stop = [payload.stop] if isinstance(payload.stop, str) else payload.stop
    return ChatRequest(
        provider="openrouter",
        model=payload.model,
        messages=[ChatMessage(role=m.role, content=m.content) for m in payload.messages],
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
        seed=payload.seed,
        stop=tuple(stop) if stop else None,
        presence_penalty=payload.presence_penalty,
        frequency_penalty=payload.frequency_penalty,
    )


def _cache_headers(status: str, similarity: float | None) -> dict[str, str]:
    headers = {"X-Cache": status}
    if similarity is not None:
        headers["X-Cache-Similarity"] = f"{similarity:.4f}"
    return headers


def _completion_body(response: ChatCompletion) -> dict[str, object]:
    body: dict[str, object] = {
        "id": f"chatcmpl-{uuid4()}",
        "object": "chat.completion",
        "created": int(time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": response.finish_reason,
            }
        ],
    }
    if response.usage is not None:
        body["usage"] = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
        }
    return body


def _sse(chunks: Iterator[StreamDelta], *, model: str) -> Iterator[str]:
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    first = True
    for delta in chunks:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": delta.model or model,
            "choices": [
                {
                    "index": 0,
                    "delta": ({"role": "assistant"} if first else {})
                    | ({"content": delta.content} if delta.content else {}),
                    "finish_reason": delta.finish_reason,
                }
            ],
        }
        first = False
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(
    service: CacheService,
    *,
    prometheus: PrometheusMetrics | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    """Create an app around an injected service, keeping HTTP easy to test."""
    app = FastAPI(title="CacheMind", version="0.2.0")

    def require_admin(supplied: str | None) -> None:
        # Admin endpoints are disabled unless a token is configured, so a default
        # deployment cannot have its cache wiped by anyone who can reach it.
        if not admin_token:
            raise HTTPException(status_code=403, detail="Admin endpoints are disabled.")
        if supplied is None or not compare_digest(supplied.encode(), admin_token.encode()):
            raise HTTPException(status_code=401, detail="Invalid admin token.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    def create_chat_completion(
        payload: ChatCompletionInput,
        cache_control: str | None = Header(default=None),
    ) -> Response:
        request = _to_chat_request(payload)
        cacheable = not (cache_control and "no-store" in cache_control.lower())
        try:
            if payload.stream:
                session = service.stream(request, cacheable=cacheable)
                return StreamingResponse(
                    _sse(session.chunks, model=session.model),
                    media_type="text/event-stream",
                    headers=_cache_headers(session.cache_status, session.similarity_score),
                )
            result = service.complete(request, cacheable=cacheable)
        except NotImplementedError as error:
            raise HTTPException(status_code=400, detail="Streaming is not supported.") from error
        except OpenRouterError as error:
            raise HTTPException(status_code=502, detail="Upstream provider failed.") from error

        return JSONResponse(
            content=_completion_body(result.response),
            headers=_cache_headers(result.cache_status, result.similarity_score),
        )

    @app.get("/metrics/summary")
    def metrics_summary() -> dict[str, object]:
        """Aggregate measurements only: no prompts and no cached responses."""
        return asdict(service.metrics) | {
            "entry_count": service.entry_count,
            "evictions": service.evictions,
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        if prometheus is None:
            raise HTTPException(status_code=404, detail="Prometheus metrics are not enabled.")
        return Response(
            content=prometheus.render(entries=service.entry_count, evictions=service.evictions),
            media_type=PrometheusMetrics.CONTENT_TYPE,
        )

    @app.post("/v1/cache/invalidate")
    def invalidate(
        payload: InvalidateInput, x_admin_token: str | None = Header(default=None)
    ) -> dict[str, int]:
        """Drop entries by model, provider and/or system prompt (all filters must match)."""
        require_admin(x_admin_token)
        prompt_hash = payload.system_prompt_hash
        if payload.system_prompt is not None:
            prompt_hash = hash_system_prompt(payload.system_prompt)
        try:
            removed = service.cache.invalidate(
                model=payload.model, provider=payload.provider, system_prompt_hash=prompt_hash
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"removed": removed}

    @app.delete("/v1/cache")
    def clear_cache(x_admin_token: str | None = Header(default=None)) -> dict[str, int]:
        require_admin(x_admin_token)
        return {"removed": service.cache.clear()}

    return app
