"""Small immutable records shared by the cache, providers, and tests.

These are internal models, not the public FastAPI schema. Keeping them
framework-free lets cache behaviour be tested without an HTTP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal, Sequence


MessageRole = Literal["system", "user", "assistant"]
CacheStatus = Literal["HIT", "MISS", "BYPASS"]


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("Message content must be text.")


@dataclass(frozen=True)
class ChatRequest:
    """A provider-neutral chat request.

    Every optional generation parameter uses ``None`` to mean "let the provider
    choose its default". ``None`` and an explicit value are different requests,
    so they must not share a cache entry.
    """

    provider: str
    model: str
    messages: Sequence[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("Provider must not be empty.")
        if not self.model:
            raise ValueError("Model must not be empty.")
        if not self.messages:
            raise ValueError("At least one chat message is required.")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("Temperature cannot be negative.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")

        # Requests may arrive with a list, but a stored request should not be
        # mutable through that original list.
        object.__setattr__(self, "messages", tuple(self.messages))
        if self.stop is not None:
            object.__setattr__(self, "stop", tuple(self.stop))


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by a provider, when it supplies them."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ChatCompletion:
    """A successful provider response.

    ``finish_reason`` matters for correctness: a response cut off by
    ``max_tokens`` (``"length"``) is a valid answer to the client that asked,
    but it must never be stored and replayed to a different request.
    """

    content: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str = "stop"


@dataclass(frozen=True)
class StreamDelta:
    """One incremental piece of a streamed provider response."""

    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str | None = None


@dataclass(frozen=True)
class CacheIdentity:
    """Execution details that must match before response reuse is safe.

    This is the *isolation boundary*: semantic similarity is only ever computed
    between entries that share an identity. ``params_hash`` covers the remaining
    response-affecting sampling parameters (top_p, seed, stop, penalties).
    """

    provider: str
    model: str
    system_prompt_hash: str
    temperature: float | None
    max_tokens: int | None
    params_hash: str = ""

    @property
    def key(self) -> str:
        """A stable digest of the full identity, usable as an index filter."""
        canonical = json.dumps(
            [
                self.provider,
                self.model,
                self.system_prompt_hash,
                self.temperature,
                self.max_tokens,
                self.params_hash,
            ],
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CompletionResult:
    """A completion plus the cache decision made for it."""

    response: ChatCompletion
    cache_status: CacheStatus
    similarity_score: float | None = None
    verification_rejected: bool = False
    latency_ms: float = 0.0
    provider_latency_ms: float | None = None
    embedding_chars: int = 0
    reason: str | None = None
    stored: bool = False
    model: str | None = None
