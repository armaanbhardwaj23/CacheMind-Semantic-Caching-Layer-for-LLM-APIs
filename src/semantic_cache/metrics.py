"""In-process measurements for cache experiments and the ``/metrics/summary`` endpoint.

Cost figures are *estimates* from explicitly configured prices. Nothing here
assumes a provider price list, and the net figure subtracts the embedding cost
that every non-exact lookup pays, because ignoring it overstates savings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from threading import Lock

from semantic_cache.models import CompletionResult, TokenUsage


@dataclass(frozen=True)
class TokenPricing:
    """Explicit configured USD prices; never an assumed provider price list."""

    input_per_million_tokens_usd: float
    output_per_million_tokens_usd: float

    def __post_init__(self) -> None:
        if self.input_per_million_tokens_usd < 0 or self.output_per_million_tokens_usd < 0:
            raise ValueError("Token prices cannot be negative.")

    def estimate_usd(self, usage: TokenUsage) -> float:
        return (
            (usage.input_tokens or 0) * self.input_per_million_tokens_usd
            + (usage.output_tokens or 0) * self.output_per_million_tokens_usd
        ) / 1_000_000


@dataclass(frozen=True)
class EmbeddingPricing:
    """Embedding price plus a chars-per-token estimate (providers do not return it here)."""

    per_million_tokens_usd: float
    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if self.per_million_tokens_usd < 0 or self.chars_per_token <= 0:
            raise ValueError("Embedding pricing must be non-negative with positive chars/token.")

    def estimate_usd(self, characters: int) -> float:
        return characters / self.chars_per_token * self.per_million_tokens_usd / 1_000_000


@dataclass(frozen=True)
class MetricsSnapshot:
    total_requests: int
    eligible_requests: int
    hits: int
    misses: int
    bypasses: int
    hit_rate: float
    near_misses: int
    verification_rejections: int
    average_hit_latency_ms: float | None
    average_miss_latency_ms: float | None
    average_bypass_latency_ms: float | None
    average_provider_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    similarity_scores: tuple[float, ...]
    input_tokens_avoided: int
    output_tokens_avoided: int
    estimated_chat_cost_avoided_usd: float | None
    stores: int = 0
    not_stored: int = 0
    embedding_requests: int = 0
    estimated_embedding_cost_usd: float | None = None
    estimated_net_cost_saved_usd: float | None = None
    bypass_reasons: dict[str, int] = field(default_factory=dict)
    not_stored_reasons: dict[str, int] = field(default_factory=dict)
    hits_by_model: dict[str, int] = field(default_factory=dict)
    latency_ms_by_status: dict[str, dict[str, float | None]] = field(default_factory=dict)


class CacheMetrics:
    """Thread-safe in-memory metrics; intentionally not a dashboard backend."""

    def __init__(
        self,
        *,
        similarity_threshold: float,
        near_miss_margin: float = 0.02,
        token_pricing: TokenPricing | None = None,
        embedding_pricing: EmbeddingPricing | None = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._near_miss_margin = near_miss_margin
        self._token_pricing = token_pricing
        self._embedding_pricing = embedding_pricing
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._near_misses = 0
        self._verification_rejections = 0
        self._stores = 0
        self._not_stored = 0
        self._embedding_requests = 0
        self._embedding_chars = 0
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []
        self._bypass_latencies: list[float] = []
        self._provider_latencies: list[float] = []
        self._similarity_scores: list[float] = []
        self._input_tokens_avoided = 0
        self._output_tokens_avoided = 0
        self._estimated_chat_cost_avoided_usd = 0.0
        self._bypass_reasons: Counter[str] = Counter()
        self._not_stored_reasons: Counter[str] = Counter()
        self._hits_by_model: Counter[str] = Counter()

    def record(self, result: CompletionResult) -> None:
        with self._lock:
            if result.similarity_score is not None:
                self._similarity_scores.append(result.similarity_score)
            if result.verification_rejected:
                self._verification_rejections += 1
            if result.embedding_chars:
                self._embedding_requests += 1
                self._embedding_chars += result.embedding_chars

            if result.cache_status == "HIT":
                self._hits += 1
                self._hit_latencies.append(result.latency_ms)
                if result.model:
                    self._hits_by_model[result.model] += 1
                usage = result.response.usage
                if usage is not None:
                    self._input_tokens_avoided += usage.input_tokens or 0
                    self._output_tokens_avoided += usage.output_tokens or 0
                    if self._token_pricing is not None:
                        self._estimated_chat_cost_avoided_usd += self._token_pricing.estimate_usd(usage)
                return

            if result.cache_status == "BYPASS":
                self._bypasses += 1
                self._bypass_latencies.append(result.latency_ms)
                self._bypass_reasons[result.reason or "unspecified"] += 1
                if result.provider_latency_ms is not None:
                    self._provider_latencies.append(result.provider_latency_ms)
                return

            self._misses += 1
            self._miss_latencies.append(result.latency_ms)
            if result.provider_latency_ms is not None:
                self._provider_latencies.append(result.provider_latency_ms)
            if result.stored:
                self._stores += 1
            else:
                self._not_stored += 1
                self._not_stored_reasons[result.reason or "unspecified"] += 1
            if (
                result.similarity_score is not None
                and self._threshold - self._near_miss_margin
                <= result.similarity_score
                < self._threshold
            ):
                self._near_misses += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            eligible = self._hits + self._misses
            total = eligible + self._bypasses
            all_latencies = self._hit_latencies + self._miss_latencies
            chat_saved = (
                self._estimated_chat_cost_avoided_usd if self._token_pricing is not None else None
            )
            embedding_cost = (
                self._embedding_pricing.estimate_usd(self._embedding_chars)
                if self._embedding_pricing is not None
                else None
            )
            net = (
                chat_saved - embedding_cost
                if chat_saved is not None and embedding_cost is not None
                else None
            )
            return MetricsSnapshot(
                total_requests=total,
                eligible_requests=eligible,
                hits=self._hits,
                misses=self._misses,
                bypasses=self._bypasses,
                hit_rate=self._hits / eligible if eligible else 0.0,
                near_misses=self._near_misses,
                verification_rejections=self._verification_rejections,
                average_hit_latency_ms=_average(self._hit_latencies),
                average_miss_latency_ms=_average(self._miss_latencies),
                average_bypass_latency_ms=_average(self._bypass_latencies),
                average_provider_latency_ms=_average(self._provider_latencies),
                p50_latency_ms=_percentile(all_latencies, 0.50),
                p95_latency_ms=_percentile(all_latencies, 0.95),
                p99_latency_ms=_percentile(all_latencies, 0.99),
                similarity_scores=tuple(self._similarity_scores),
                input_tokens_avoided=self._input_tokens_avoided,
                output_tokens_avoided=self._output_tokens_avoided,
                estimated_chat_cost_avoided_usd=chat_saved,
                stores=self._stores,
                not_stored=self._not_stored,
                embedding_requests=self._embedding_requests,
                estimated_embedding_cost_usd=embedding_cost,
                estimated_net_cost_saved_usd=net,
                bypass_reasons=dict(self._bypass_reasons),
                not_stored_reasons=dict(self._not_stored_reasons),
                hits_by_model=dict(self._hits_by_model),
                latency_ms_by_status={
                    "HIT": _summary(self._hit_latencies),
                    "MISS": _summary(self._miss_latencies),
                    "BYPASS": _summary(self._bypass_latencies),
                },
            )


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[ceil(percentile * len(ordered)) - 1]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "count": float(len(values)),
        "average_ms": _average(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
    }
