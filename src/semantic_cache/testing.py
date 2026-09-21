"""Deterministic local collaborators used by tests and offline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterator

from semantic_cache.models import ChatCompletion, ChatRequest, StreamDelta, TokenUsage


@dataclass(frozen=True)
class DeterministicEmbedder:
    """Produces a stable test vector for a text string.

    This is not a semantic embedding model. It only gives tests reproducible
    values, including an exact match for identical text.
    """

    dimensions: int = 12

    def embed(self, text: str) -> tuple[float, ...]:
        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        digest = sha256(text.encode("utf-8")).digest()
        return tuple(byte / 255 + 0.01 for byte in digest[: self.dimensions])


@dataclass
class RecordingProvider:
    """Returns a fixed completion and records every request it receives."""

    response_text: str
    usage: TokenUsage | None = None
    finish_reason: str = "stop"
    fail_stream_after: int | None = None  # raise after N deltas, to test partial streams
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: ChatRequest) -> ChatCompletion:
        self.requests.append(request)
        return ChatCompletion(
            content=self.response_text,
            model=request.model,
            usage=self.usage,
            finish_reason=self.finish_reason,
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamDelta]:
        self.requests.append(request)
        words = self.response_text.split(" ")
        for position, word in enumerate(words):
            if self.fail_stream_after is not None and position >= self.fail_stream_after:
                raise RuntimeError("simulated provider failure mid-stream")
            yield StreamDelta(content=word + (" " if position < len(words) - 1 else ""), model=request.model)
        yield StreamDelta(finish_reason=self.finish_reason, usage=self.usage, model=request.model)
