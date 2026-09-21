"""Prometheus exposition, kept separate from the in-process summary metrics.

Uses a private ``CollectorRegistry`` per instance so several services (or tests)
can coexist without duplicate-metric errors from the global default registry.
Prompts and responses are never used as label values; ``model`` is the only
request-derived label.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from semantic_cache.models import CompletionResult

_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_SIMILARITY_BUCKETS = (0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99, 1.0)


class PrometheusMetrics:
    CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

    def __init__(self, *, similarity_threshold: float) -> None:
        self.registry = CollectorRegistry()
        self._threshold = similarity_threshold
        self.requests = Counter(
            "cache_requests_total", "Requests by cache outcome.", ["status", "model"],
            registry=self.registry,
        )
        self.near_misses = Counter(
            "cache_near_misses_total", "Misses within 0.02 below the threshold.",
            registry=self.registry,
        )
        self.verification_rejections = Counter(
            "cache_verification_rejections_total",
            "Candidates above threshold rejected by the constraint guard.",
            registry=self.registry,
        )
        self.stores = Counter("cache_stores_total", "Responses written to the cache.", registry=self.registry)
        self.not_stored = Counter(
            "cache_not_stored_total", "Misses that were not stored.", ["reason"], registry=self.registry
        )
        self.bypasses = Counter(
            "cache_bypasses_total", "Requests that skipped the cache.", ["reason"], registry=self.registry
        )
        self.latency = Histogram(
            "cache_request_latency_seconds", "End-to-end proxy latency by outcome.", ["status"],
            buckets=_LATENCY_BUCKETS, registry=self.registry,
        )
        self.provider_latency = Histogram(
            "cache_provider_latency_seconds", "Upstream provider latency.",
            buckets=_LATENCY_BUCKETS, registry=self.registry,
        )
        self.similarity = Histogram(
            "cache_similarity_score", "Best-candidate similarity for lookups that produced one.",
            buckets=_SIMILARITY_BUCKETS, registry=self.registry,
        )
        self.tokens_avoided = Counter(
            "cache_tokens_avoided_total", "Provider-reported tokens avoided by hits.", ["direction"],
            registry=self.registry,
        )
        self.entries = Gauge("cache_entries", "Entries currently stored.", registry=self.registry)
        self.evictions = Gauge("cache_evictions", "Capacity evictions so far.", registry=self.registry)

    def record(self, result: CompletionResult) -> None:
        self.requests.labels(status=result.cache_status, model=result.model or "unknown").inc()
        self.latency.labels(status=result.cache_status).observe(result.latency_ms / 1000)
        if result.provider_latency_ms is not None:
            self.provider_latency.observe(result.provider_latency_ms / 1000)
        if result.similarity_score is not None:
            self.similarity.observe(result.similarity_score)
        if result.verification_rejected:
            self.verification_rejections.inc()

        if result.cache_status == "HIT" and result.response.usage is not None:
            self.tokens_avoided.labels(direction="input").inc(result.response.usage.input_tokens or 0)
            self.tokens_avoided.labels(direction="output").inc(result.response.usage.output_tokens or 0)
        elif result.cache_status == "BYPASS":
            self.bypasses.labels(reason=result.reason or "unspecified").inc()
        elif result.cache_status == "MISS":
            if result.stored:
                self.stores.inc()
            else:
                self.not_stored.labels(reason=result.reason or "unspecified").inc()
            if (
                result.similarity_score is not None
                and self._threshold - 0.02 <= result.similarity_score < self._threshold
            ):
                self.near_misses.inc()

    def render(self, *, entries: int, evictions: int) -> bytes:
        self.entries.set(entries)
        self.evictions.set(evictions)
        return generate_latest(self.registry)
