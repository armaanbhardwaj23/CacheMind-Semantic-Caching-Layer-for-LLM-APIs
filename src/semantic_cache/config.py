"""Runtime configuration for the first hosted provider integration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    embedding_model: str = "openai/text-embedding-3-small"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 30.0
    http_referer: str | None = None
    app_title: str | None = "CacheMind"
    similarity_threshold: float = 0.95
    input_token_price_per_million_usd: float | None = None
    output_token_price_per_million_usd: float | None = None
    embedding_price_per_million_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set.")
        if not self.embedding_model:
            raise ValueError("An OpenRouter embedding model is required.")
        if self.timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive.")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Cache similarity threshold must be between 0 and 1.")
        if (self.input_token_price_per_million_usd is None) != (
            self.output_token_price_per_million_usd is None
        ):
            raise ValueError("Configure both input and output token prices, or neither.")

    @classmethod
    def from_env(cls) -> OpenRouterSettings:
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            embedding_model=os.environ.get(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            http_referer=os.environ.get("OPENROUTER_HTTP_REFERER") or None,
            app_title=os.environ.get("OPENROUTER_APP_TITLE") or "CacheMind",
            similarity_threshold=float(
                os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.95")
            ),
            input_token_price_per_million_usd=_optional_float(
                os.environ.get("CACHE_INPUT_PRICE_PER_MILLION_TOKENS_USD")
            ),
            output_token_price_per_million_usd=_optional_float(
                os.environ.get("CACHE_OUTPUT_PRICE_PER_MILLION_TOKENS_USD")
            ),
            embedding_price_per_million_usd=_optional_float(
                os.environ.get("CACHE_EMBEDDING_PRICE_PER_MILLION_TOKENS_USD")
            ),
        )


def _optional_float(value: str | None) -> float | None:
    return float(value) if value else None


@dataclass(frozen=True)
class CacheSettings:
    """Storage, policy and admin configuration, separate from provider credentials."""

    store: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379"
    embedding_dimensions: int = 1536  # text-embedding-3-small
    max_entries: int | None = None  # memory store only; Redis uses maxmemory-policy
    stable_ttl_seconds: int = 24 * 3600
    time_relative_ttl_seconds: int = 3600
    max_temperature: float | None = None
    classify_time_sensitivity: bool = True
    constraint_guard: bool = True
    admin_token: str | None = None

    def __post_init__(self) -> None:
        if self.store not in {"memory", "redis"}:
            raise ValueError("CACHE_STORE must be 'memory' or 'redis'.")

    @classmethod
    def from_env(cls) -> CacheSettings:
        max_entries = os.environ.get("CACHE_MAX_ENTRIES")
        return cls(
            store=os.environ.get("CACHE_STORE", "memory"),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            embedding_dimensions=int(os.environ.get("CACHE_EMBEDDING_DIMENSIONS", "1536")),
            max_entries=int(max_entries) if max_entries else None,
            stable_ttl_seconds=int(os.environ.get("CACHE_STABLE_TTL_SECONDS", str(24 * 3600))),
            time_relative_ttl_seconds=int(
                os.environ.get("CACHE_TIME_RELATIVE_TTL_SECONDS", "3600")
            ),
            max_temperature=_optional_float(os.environ.get("CACHE_MAX_TEMPERATURE")),
            classify_time_sensitivity=os.environ.get("CACHE_CLASSIFY_TTL", "1") != "0",
            constraint_guard=os.environ.get("CACHE_CONSTRAINT_GUARD", "1") != "0",
            admin_token=os.environ.get("CACHE_ADMIN_TOKEN") or None,
        )
