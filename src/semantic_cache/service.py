"""The cache request path, independent of FastAPI and of any vendor.

Request flow::

    policy ── bypass? ──────────────────────────────► provider  (BYPASS)
      │
    identity + semantic text
      │
    exact match?  ── yes ───────────────────────────► HIT (no embedding call)
      │ no
    embed → nearest neighbour *within this identity*
      │
    score >= threshold and guard accepts? ── yes ───► HIT
      │ no
    provider ─► store only complete, successful responses ─► MISS
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Iterator, Protocol

from semantic_cache.cache import CacheEntry, CacheStore, InMemorySemanticCache
from semantic_cache.identity import build_cache_identity
from semantic_cache.metrics import CacheMetrics, MetricsSnapshot
from semantic_cache.models import (
    CacheIdentity,
    CacheStatus,
    ChatCompletion,
    ChatRequest,
    CompletionResult,
    StreamDelta,
    TokenUsage,
)
from semantic_cache.policy import CachePolicy
from semantic_cache.representation import build_semantic_lookup_text
from semantic_cache.verification import CandidateVerifier, ConstraintGuard

__all__ = ["CacheService", "Embedder", "Provider", "StreamSession", "build_cache_identity"]


class Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class Provider(Protocol):
    def complete(self, request: ChatRequest) -> ChatCompletion: ...


class MetricsObserver(Protocol):
    def record(self, result: CompletionResult) -> None: ...


@dataclass
class StreamSession:
    """Decision made up front (so headers can be sent) plus the body iterator."""

    cache_status: CacheStatus
    similarity_score: float | None
    reason: str | None
    model: str
    chunks: Iterator[StreamDelta]


@dataclass
class _Probe:
    entry: CacheEntry | None
    similarity: float | None
    rejected: bool
    embedding: tuple[float, ...] | None
    embedding_chars: int


class CacheService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        provider: Provider,
        similarity_threshold: float,
        cache: CacheStore | None = None,
        metrics: CacheMetrics | None = None,
        candidate_verifier: CandidateVerifier | None = None,
        policy: CachePolicy | None = None,
        observers: Iterable[MetricsObserver] = (),
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1.")
        self._embedder = embedder
        self._provider = provider
        self._similarity_threshold = similarity_threshold
        self._cache: CacheStore = cache if cache is not None else InMemorySemanticCache()
        self._metrics = metrics or CacheMetrics(similarity_threshold=similarity_threshold)
        self._candidate_verifier = candidate_verifier or ConstraintGuard()
        self._policy = policy or CachePolicy()
        self._observers: tuple[MetricsObserver, ...] = (self._metrics, *observers)

    # -- introspection ----------------------------------------------------

    @property
    def entry_count(self) -> int:
        return self._cache.entry_count

    @property
    def evictions(self) -> int:
        return self._cache.evictions

    @property
    def metrics(self) -> MetricsSnapshot:
        return self._metrics.snapshot()

    @property
    def cache(self) -> CacheStore:
        return self._cache

    # -- non-streaming ----------------------------------------------------

    def complete(self, request: ChatRequest, *, cacheable: bool = True) -> CompletionResult:
        started_at = perf_counter()
        bypass_reason = self._bypass_reason(request, cacheable)
        if bypass_reason is not None:
            return self._bypass(request, started_at, bypass_reason)

        decision = self._policy.decide(request)
        identity = build_cache_identity(request)
        text = build_semantic_lookup_text(request.messages)
        probe = self._probe(identity, text)
        if probe.entry is not None:
            return self._hit(request, probe, started_at)

        provider_started_at = perf_counter()
        response = self._provider.complete(request)
        provider_ms = (perf_counter() - provider_started_at) * 1000
        stored, reason = self._maybe_store(identity, text, probe, response, decision.ttl_seconds)
        result = CompletionResult(
            response=response,
            cache_status="MISS",
            similarity_score=probe.similarity,
            verification_rejected=probe.rejected,
            latency_ms=(perf_counter() - started_at) * 1000,
            provider_latency_ms=provider_ms,
            embedding_chars=probe.embedding_chars,
            reason=reason,
            stored=stored,
            model=request.model,
        )
        self._record(result)
        return result

    # -- streaming --------------------------------------------------------

    def stream(self, request: ChatRequest, *, cacheable: bool = True) -> StreamSession:
        """Start a streamed completion.

        A HIT replays the stored answer. A MISS/BYPASS streams the provider's
        deltas straight through while buffering them, and stores the buffered
        response **only after the stream ends normally with finish_reason
        "stop"**. A dropped client, a provider error, or a truncated stream never
        produces a cache entry.
        """
        if not hasattr(self._provider, "stream"):
            raise NotImplementedError("The configured provider does not support streaming.")
        started_at = perf_counter()

        bypass_reason = self._bypass_reason(request, cacheable)
        if bypass_reason is not None:
            return StreamSession(
                cache_status="BYPASS",
                similarity_score=None,
                reason=bypass_reason,
                model=request.model,
                chunks=self._relay(request, started_at, None, None, None, bypass_reason),
            )

        decision = self._policy.decide(request)
        identity = build_cache_identity(request)
        text = build_semantic_lookup_text(request.messages)
        probe = self._probe(identity, text)
        if probe.entry is not None:
            result = self._hit(request, probe, started_at)
            return StreamSession(
                cache_status="HIT",
                similarity_score=probe.similarity,
                reason=None,
                model=result.response.model,
                chunks=iter(_replay(result.response)),
            )

        return StreamSession(
            cache_status="MISS",
            similarity_score=probe.similarity,
            reason=None,
            model=request.model,
            chunks=self._relay(request, started_at, (identity, text, probe), decision.ttl_seconds, probe, None),
        )

    def _relay(
        self,
        request: ChatRequest,
        started_at: float,
        cache_context: tuple[CacheIdentity, str, _Probe] | None,
        ttl_seconds: int | None,
        probe: _Probe | None,
        bypass_reason: str | None,
    ) -> Iterator[StreamDelta]:
        provider_started_at = perf_counter()
        parts: list[str] = []
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        response_model: str | None = None
        for delta in self._provider.stream(request):  # type: ignore[attr-defined]
            parts.append(delta.content)
            finish_reason = delta.finish_reason or finish_reason
            usage = delta.usage or usage
            response_model = delta.model or response_model
            yield delta

        # Reaching here means the provider stream ended without raising and the
        # client consumed it to the end. Anything else skips the code below.
        provider_ms = (perf_counter() - provider_started_at) * 1000
        response = ChatCompletion(
            content="".join(parts),
            model=response_model or request.model,
            usage=usage,
            finish_reason=finish_reason or "incomplete_stream",
        )
        stored = False
        reason = bypass_reason
        if cache_context is not None and probe is not None:
            identity, text, _ = cache_context
            stored, reason = self._maybe_store(identity, text, probe, response, ttl_seconds)
        self._record(
            CompletionResult(
                response=response,
                cache_status="BYPASS" if bypass_reason else "MISS",
                similarity_score=probe.similarity if probe else None,
                verification_rejected=probe.rejected if probe else False,
                latency_ms=(perf_counter() - started_at) * 1000,
                provider_latency_ms=provider_ms,
                embedding_chars=probe.embedding_chars if probe else 0,
                reason=reason,
                stored=stored,
                model=request.model,
            )
        )

    # -- shared steps -----------------------------------------------------

    def _bypass_reason(self, request: ChatRequest, cacheable: bool) -> str | None:
        if not cacheable:
            return "client_no_store"
        decision = self._policy.decide(request)
        return None if decision.cacheable else decision.reason

    def _bypass(self, request: ChatRequest, started_at: float, reason: str) -> CompletionResult:
        provider_started_at = perf_counter()
        response = self._provider.complete(request)
        result = CompletionResult(
            response=response,
            cache_status="BYPASS",
            latency_ms=(perf_counter() - started_at) * 1000,
            provider_latency_ms=(perf_counter() - provider_started_at) * 1000,
            reason=reason,
            model=request.model,
        )
        self._record(result)
        return result

    def _probe(self, identity: CacheIdentity, text: str) -> _Probe:
        exact = self._cache.find_exact(identity=identity, semantic_lookup_text=text)
        if exact is not None:
            return _Probe(exact, 1.0, False, None, 0)

        embedding = self._embedder.embed(text)
        lookup = self._cache.lookup(
            identity=identity,
            semantic_lookup_text=text,
            embedding=embedding,
            similarity_threshold=self._similarity_threshold,
        )
        if lookup.entry is None:
            return _Probe(None, lookup.similarity_score, False, embedding, len(text))

        decision = self._candidate_verifier.verify(
            cached_text=lookup.entry.semantic_lookup_text, incoming_text=text
        )
        if decision.accepted:
            return _Probe(lookup.entry, lookup.similarity_score, False, embedding, len(text))
        return _Probe(None, lookup.similarity_score, True, embedding, len(text))

    def _hit(self, request: ChatRequest, probe: _Probe, started_at: float) -> CompletionResult:
        assert probe.entry is not None
        self._cache.record_hit(probe.entry.entry_id)
        result = CompletionResult(
            response=probe.entry.response,
            cache_status="HIT",
            similarity_score=probe.similarity,
            latency_ms=(perf_counter() - started_at) * 1000,
            embedding_chars=probe.embedding_chars,
            model=request.model,
        )
        self._record(result)
        return result

    def _maybe_store(
        self,
        identity: CacheIdentity,
        text: str,
        probe: _Probe,
        response: ChatCompletion,
        ttl_seconds: int | None,
    ) -> tuple[bool, str | None]:
        if response.finish_reason != "stop":
            # "length" means max_tokens truncated it; anything else is unknown.
            return False, f"not_stored_finish_reason_{response.finish_reason}"
        if not response.content.strip():
            return False, "not_stored_empty_response"
        assert probe.embedding is not None
        self._cache.store(
            identity=identity,
            semantic_lookup_text=text,
            embedding=probe.embedding,
            response=response,
            ttl_seconds=ttl_seconds,
        )
        return True, None

    def _record(self, result: CompletionResult) -> None:
        for observer in self._observers:
            observer.record(result)


def _replay(response: ChatCompletion) -> list[StreamDelta]:
    return [
        StreamDelta(content=response.content, model=response.model),
        StreamDelta(finish_reason=response.finish_reason, usage=response.usage, model=response.model),
    ]
