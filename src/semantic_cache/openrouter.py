"""Synchronous OpenRouter adapter for embeddings and chat completions."""

from __future__ import annotations

from collections.abc import Iterator
import json
from typing import Any

import httpx

from semantic_cache.config import OpenRouterSettings
from semantic_cache.models import ChatCompletion, ChatRequest, StreamDelta, TokenUsage


class OpenRouterError(RuntimeError):
    """A provider failure whose message deliberately excludes request content."""


class OpenRouterClient:
    """Implements the cache's embedder and provider protocols using OpenRouter."""

    def __init__(
        self, settings: OpenRouterSettings, *, client: httpx.Client | None = None
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, text: str) -> tuple[float, ...]:
        payload = {
            "input": text,
            "model": self._settings.embedding_model,
            "encoding_format": "float",
        }
        body = self._post("/embeddings", payload)
        try:
            vector = body["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenRouterError("OpenRouter returned a malformed embedding response.") from error
        if not isinstance(vector, list) or not vector:
            raise OpenRouterError("OpenRouter returned an empty embedding vector.")
        if not all(isinstance(value, int | float) for value in vector):
            raise OpenRouterError("OpenRouter returned a non-numeric embedding vector.")
        return tuple(float(value) for value in vector)

    def complete(self, request: ChatRequest) -> ChatCompletion:
        body = self._post("/chat/completions", self._chat_payload(request, stream=False))
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenRouterError("OpenRouter returned a malformed completion response.") from error
        if not isinstance(content, str):
            raise OpenRouterError("OpenRouter returned non-text completion content.")

        response_model = body.get("model", request.model)
        if not isinstance(response_model, str):
            raise OpenRouterError("OpenRouter returned an invalid completion model.")
        finish_reason = choice.get("finish_reason")
        return ChatCompletion(
            content=content,
            model=response_model,
            usage=_parse_usage(body.get("usage")),
            finish_reason=finish_reason if isinstance(finish_reason, str) else "unknown",
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamDelta]:
        """Yield deltas from an SSE chat stream; raise on any transport/parse failure."""
        payload = self._chat_payload(request, stream=True)
        payload["stream_options"] = {"include_usage": True}
        try:
            with self._client.stream(
                "POST", "/chat/completions", json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or line.startswith(":"):  # blank or SSE keep-alive comment
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except ValueError as error:
                        raise OpenRouterError("OpenRouter sent a malformed stream event.") from error
                    if isinstance(event, dict) and "error" in event:
                        raise OpenRouterError("OpenRouter reported an error mid-stream.")
                    yield _delta_from_event(event)
        except httpx.HTTPStatusError as error:
            raise OpenRouterError(
                f"OpenRouter request failed with status {error.response.status_code}."
            ) from error
        except httpx.HTTPError as error:
            raise OpenRouterError("OpenRouter stream could not be completed.") from error

    @staticmethod
    def _chat_payload(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": stream,
        }
        optional = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "seed": request.seed,
            "stop": list(request.stop) if request.stop is not None else None,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        payload.update({name: value for name, value in optional.items() if value is not None})
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.http_referer:
            headers["HTTP-Referer"] = self._settings.http_referer
        if self._settings.app_title:
            headers["X-OpenRouter-Title"] = self._settings.app_title
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise OpenRouterError(
                f"OpenRouter request failed with status {error.response.status_code}."
            ) from error
        except httpx.HTTPError as error:
            raise OpenRouterError("OpenRouter request could not be completed.") from error

        try:
            body = response.json()
        except ValueError as error:
            raise OpenRouterError("OpenRouter returned a non-JSON response.") from error
        if not isinstance(body, dict):
            raise OpenRouterError("OpenRouter returned an invalid JSON response.")
        return body


def _parse_usage(value: object) -> TokenUsage | None:
    if not isinstance(value, dict):
        return None
    input_tokens = value.get("prompt_tokens")
    output_tokens = value.get("completion_tokens")
    if not isinstance(input_tokens, int) and input_tokens is not None:
        return None
    if not isinstance(output_tokens, int) and output_tokens is not None:
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _delta_from_event(event: object) -> StreamDelta:
    if not isinstance(event, dict):
        raise OpenRouterError("OpenRouter sent an invalid stream event.")
    choices = event.get("choices") or []
    content = ""
    finish_reason = None
    if choices:
        choice = choices[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        content = piece if isinstance(piece, str) else ""
        reason = choice.get("finish_reason")
        finish_reason = reason if isinstance(reason, str) else None
    model = event.get("model")
    return StreamDelta(
        content=content,
        finish_reason=finish_reason,
        usage=_parse_usage(event.get("usage")),
        model=model if isinstance(model, str) else None,
    )
