"""Workload-level correctness simulation with a ground-truth "oracle" provider.

The pair-level threshold experiment (``evaluation.py``) asks: *given one cached
prompt and one incoming prompt, is reuse safe?* This module asks the question
that matters operationally: *over a realistic request stream, how many requests
does the cache serve, and how many of those serve the wrong answer?*

How it stays honest without paying for LLM calls: the dataset groups prompts by
**intent** (prompts a human labelled as asking the same thing). The oracle
provider answers ``answer::<intent_id>``. A cache HIT is *valid* iff the cached
answer's intent equals the incoming request's intent; otherwise it is a *false
hit* - the cache served an answer to a different question. Embeddings are real;
only the provider is simulated, so latency is deliberately not reported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence

from semantic_cache.metrics import CacheMetrics
from semantic_cache.models import ChatCompletion, ChatMessage, ChatRequest, TokenUsage
from semantic_cache.policy import CachePolicy
from semantic_cache.service import CacheService, Embedder
from semantic_cache.verification import AcceptAllVerifier, CandidateVerifier


@dataclass(frozen=True)
class Intent:
    intent_id: str
    queries: tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    threshold: float
    guard: bool
    requests: int
    hits: int
    valid_hits: int
    false_hits: int
    provider_calls: int
    hit_rate: float
    false_hit_rate: float | None
    provider_calls_avoided_rate: float
    wrongly_served_rate: float


def load_intents(path: Path) -> list[Intent]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Intent file must be a JSON list.")
    return [Intent(str(item["intent_id"]), tuple(item["queries"])) for item in raw]


def build_request_stream(
    intents: Sequence[Intent], *, n_requests: int, seed: int = 42, zipf_s: float = 1.1
) -> list[tuple[str, str]]:
    """Zipf-popular intents, each request a random phrasing of its intent.

    Real traffic is head-heavy: a few intents are asked constantly (exact repeats
    and paraphrases), most are rare. ``zipf_s`` controls the skew; the parameter
    is reported with results so the workload is never a hidden lever.
    """
    if not intents:
        raise ValueError("At least one intent is required.")
    rng = random.Random(seed)
    order = list(intents)
    rng.shuffle(order)
    weights = [1 / (rank ** zipf_s) for rank in range(1, len(order) + 1)]
    picks = rng.choices(order, weights=weights, k=n_requests)
    return [(intent.intent_id, rng.choice(intent.queries)) for intent in picks]


class OracleProvider:
    def __init__(self, intents: Sequence[Intent]) -> None:
        self._intent_by_query: dict[str, str] = {}
        ambiguous: set[str] = set()
        for intent in intents:
            for query in intent.queries:
                if self._intent_by_query.get(query, intent.intent_id) != intent.intent_id:
                    ambiguous.add(query)
                self._intent_by_query[query] = intent.intent_id
        if ambiguous:
            raise ValueError(f"{len(ambiguous)} queries appear under more than one intent.")
        self.calls = 0

    def complete(self, request: ChatRequest) -> ChatCompletion:
        self.calls += 1
        query = request.messages[-1].content
        return ChatCompletion(
            content=f"answer::{self._intent_by_query[query]}",
            model=request.model,
            usage=TokenUsage(20, 40),
        )


def simulate(
    intents: Sequence[Intent],
    stream: Sequence[tuple[str, str]],
    embedder: Embedder,
    *,
    threshold: float,
    verifier: CandidateVerifier | None = None,
) -> SimulationResult:
    provider = OracleProvider(intents)
    service = CacheService(
        embedder=embedder,
        provider=provider,
        similarity_threshold=threshold,
        candidate_verifier=verifier or AcceptAllVerifier(),
        policy=CachePolicy(classify_time_sensitivity=False),
        metrics=CacheMetrics(similarity_threshold=threshold),
    )
    valid = false = 0
    for intent_id, query in stream:
        request = ChatRequest(
            provider="oracle",
            model="oracle",
            messages=[ChatMessage(role="user", content=query)],
            temperature=0.0,
            max_tokens=64,
        )
        result = service.complete(request)
        if result.cache_status == "HIT":
            if result.response.content == f"answer::{intent_id}":
                valid += 1
            else:
                false += 1
    hits = valid + false
    total = len(stream)
    return SimulationResult(
        threshold=threshold,
        guard=verifier is not None,
        requests=total,
        hits=hits,
        valid_hits=valid,
        false_hits=false,
        provider_calls=provider.calls,
        hit_rate=hits / total if total else 0.0,
        false_hit_rate=false / hits if hits else None,
        provider_calls_avoided_rate=(total - provider.calls) / total if total else 0.0,
        wrongly_served_rate=false / total if total else 0.0,
    )
