"""Cache admission and TTL policy.

Not every request should be cached, and cached answers do not stay true forever.
This is a transparent rule-based classifier, not a learned one: every decision
carries a ``reason`` so it can be audited and measured.

Tiers
-----
* ``realtime``      -> bypass. The answer is about *now*; any stored answer is
                      wrong by construction (prices, weather, live scores).
* ``time_relative`` -> short TTL. Answers drift over days (latest, this week).
* ``stable``        -> long TTL. Definitions, how-to, static facts.

The rules only look at the last user message. They are deliberately conservative
about caching, but their false-negative/positive behaviour has NOT been evaluated:
only a few unit tests exercise them.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from semantic_cache.models import ChatRequest

_REALTIME = re.compile(
    r"\b(right now|at the moment|as of now|currently|live (score|price|feed|traffic)|"
    r"breaking news|today(?:'s)?|tonight|this (morning|afternoon|evening|hour)|"
    r"(current|latest) (price|weather|time|score|status|exchange rate)|"
    r"weather|stock price|exchange rate)\b",
    re.IGNORECASE,
)
_TIME_RELATIVE = re.compile(
    r"\b(latest|newest|recent(ly)?|upcoming|yesterday|tomorrow|this (week|month|year)|"
    r"last (week|month|year)|next (week|month|year)|new(est)? version|"
    r"20[2-9]\d)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CacheDecision:
    cacheable: bool
    ttl_seconds: int | None
    reason: str


class CachePolicy:
    def __init__(
        self,
        *,
        stable_ttl_seconds: int = 24 * 3600,
        time_relative_ttl_seconds: int = 3600,
        max_temperature: float | None = None,
        classify_time_sensitivity: bool = True,
    ) -> None:
        if stable_ttl_seconds <= 0 or time_relative_ttl_seconds <= 0:
            raise ValueError("TTLs must be positive.")
        if max_temperature is not None and max_temperature < 0:
            raise ValueError("max_temperature cannot be negative.")
        self._stable = stable_ttl_seconds
        self._time_relative = time_relative_ttl_seconds
        self._max_temperature = max_temperature
        self._classify = classify_time_sensitivity

    def decide(self, request: ChatRequest) -> CacheDecision:
        if (
            self._max_temperature is not None
            and request.temperature is not None
            and request.temperature > self._max_temperature
        ):
            return CacheDecision(False, None, "temperature_above_cacheable_limit")
        if not self._classify:
            return CacheDecision(True, self._stable, "stable")

        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        if _REALTIME.search(last_user):
            return CacheDecision(False, None, "realtime")
        if _TIME_RELATIVE.search(last_user):
            return CacheDecision(True, self._time_relative, "time_relative")
        return CacheDecision(True, self._stable, "stable")
