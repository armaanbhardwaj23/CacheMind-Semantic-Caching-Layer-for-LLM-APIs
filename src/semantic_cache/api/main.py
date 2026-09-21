"""Uvicorn factory for a configured OpenRouter-backed CacheMind instance."""

from semantic_cache.api.app import create_app
from semantic_cache.cache import CacheStore, InMemorySemanticCache
from semantic_cache.config import CacheSettings, OpenRouterSettings
from semantic_cache.metrics import CacheMetrics, EmbeddingPricing, TokenPricing
from semantic_cache.openrouter import OpenRouterClient
from semantic_cache.policy import CachePolicy
from semantic_cache.prometheus import PrometheusMetrics
from semantic_cache.service import CacheService
from semantic_cache.verification import AcceptAllVerifier, ConstraintGuard


def _build_store(cache_settings: CacheSettings) -> CacheStore:
    if cache_settings.store == "redis":
        # Imported lazily so the optional `redis` extra is only needed when used.
        from semantic_cache.redis_store import RedisVLCacheStore

        return RedisVLCacheStore(
            redis_url=cache_settings.redis_url,
            embedding_dimensions=cache_settings.embedding_dimensions,
        )
    return InMemorySemanticCache(max_entries=cache_settings.max_entries)


def create_configured_app():
    settings = OpenRouterSettings.from_env()
    cache_settings = CacheSettings.from_env()
    openrouter = OpenRouterClient(settings)
    token_pricing = None
    if settings.input_token_price_per_million_usd is not None:
        token_pricing = TokenPricing(
            input_per_million_tokens_usd=settings.input_token_price_per_million_usd,
            output_per_million_tokens_usd=settings.output_token_price_per_million_usd,
        )
    embedding_pricing = (
        EmbeddingPricing(per_million_tokens_usd=settings.embedding_price_per_million_usd)
        if settings.embedding_price_per_million_usd is not None
        else None
    )
    prometheus = PrometheusMetrics(similarity_threshold=settings.similarity_threshold)
    service = CacheService(
        embedder=openrouter,
        provider=openrouter,
        similarity_threshold=settings.similarity_threshold,
        cache=_build_store(cache_settings),
        candidate_verifier=ConstraintGuard() if cache_settings.constraint_guard else AcceptAllVerifier(),
        metrics=CacheMetrics(
            similarity_threshold=settings.similarity_threshold,
            token_pricing=token_pricing,
            embedding_pricing=embedding_pricing,
        ),
        policy=CachePolicy(
            stable_ttl_seconds=cache_settings.stable_ttl_seconds,
            time_relative_ttl_seconds=cache_settings.time_relative_ttl_seconds,
            max_temperature=cache_settings.max_temperature,
            classify_time_sensitivity=cache_settings.classify_time_sensitivity,
        ),
        observers=[prometheus],
    )
    return create_app(service, prometheus=prometheus, admin_token=cache_settings.admin_token)
