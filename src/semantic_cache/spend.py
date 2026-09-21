"""A hard dollar ceiling for paid experiment calls.

Every paid call in the experiment scripts goes through a wrapper here. The ledger
is persisted (``.cache/spend.json``) so the ceiling holds across runs, and a call
that *could* exceed the cap is refused before it is sent. Costs are estimated from
explicitly supplied prices; they are estimates, not an invoice.

Limits, stated plainly: a ``--max-usd`` flag can lower the ceiling for one invocation but never raise it
above ``REPO_HARD_CAP_USD``; the check-then-charge sequence is not atomic, so concurrent callers can
overshoot by roughly one call each, and two processes sharing one ledger file can lose each other's
entries; and embedding cost is a characters-per-token estimate, not the provider's reported usage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock

from semantic_cache.metrics import TokenPricing
from semantic_cache.models import ChatCompletion, ChatRequest


# The hard spend cap for building and evaluating this repository. Changing it is a deliberate code edit.
REPO_HARD_CAP_USD = 0.20


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, path: Path, *, cap_usd: float) -> None:
        if cap_usd < 0:
            raise ValueError("cap_usd cannot be negative (use 0 to forbid every paid call).")
        self._path = path
        self._cap = min(cap_usd, REPO_HARD_CAP_USD)
        self._lock = Lock()
        self._entries: list[dict[str, object]] = []
        if path.exists():
            self._entries = json.loads(path.read_text())["entries"]

    @property
    def total_usd(self) -> float:
        return sum(float(e["usd"]) for e in self._entries)

    @property
    def cap_usd(self) -> float:
        return self._cap

    def ensure_room(self, upper_bound_usd: float) -> None:
        if self.total_usd + upper_bound_usd > self._cap:
            raise BudgetExceeded(
                f"Refusing call: spent ${self.total_usd:.4f} + up to ${upper_bound_usd:.4f} "
                f"would exceed the ${self._cap:.2f} cap."
            )

    def charge(self, usd: float, label: str) -> None:
        with self._lock:
            self._entries.append({"label": label, "usd": usd})
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"cap_usd": REPO_HARD_CAP_USD, "entries": self._entries}, indent=1))

    def summary(self) -> dict[str, float]:
        by_label: dict[str, float] = {}
        for entry in self._entries:
            by_label[str(entry["label"])] = by_label.get(str(entry["label"]), 0.0) + float(entry["usd"])
        return by_label


@dataclass
class BudgetedEmbedder:
    """Wrap an embedder; place *inside* DiskCachedEmbedder so cache hits cost nothing."""

    inner: object
    ledger: SpendLedger
    price_per_million_tokens_usd: float = 0.02
    chars_per_token: float = 4.0
    label: str = "embeddings"

    def embed(self, text: str) -> tuple[float, ...]:
        estimate = len(text) / self.chars_per_token * self.price_per_million_tokens_usd / 1e6
        self.ledger.ensure_room(estimate)
        vector = self.inner.embed(text)  # type: ignore[attr-defined]
        self.ledger.charge(estimate, self.label)
        return vector


@dataclass
class BudgetedProvider:
    inner: object
    ledger: SpendLedger
    pricing: TokenPricing
    label: str = "chat"

    def complete(self, request: ChatRequest) -> ChatCompletion:
        prompt_chars = sum(len(m.content) for m in request.messages)
        worst_case = self.pricing.estimate_usd(
            _usage(prompt_chars // 3 + 1, request.max_tokens or 1024)
        )
        self.ledger.ensure_room(worst_case)
        response = self.inner.complete(request)  # type: ignore[attr-defined]
        usage = response.usage or _usage(prompt_chars // 4, len(response.content) // 4)
        self.ledger.charge(self.pricing.estimate_usd(usage), self.label)
        return response


def _usage(input_tokens: int, output_tokens: int):
    from semantic_cache.models import TokenUsage

    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
