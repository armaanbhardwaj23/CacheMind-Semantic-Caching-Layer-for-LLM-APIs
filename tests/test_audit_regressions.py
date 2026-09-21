"""Regression tests added by the self-audit.

Each test pins a defect (or a previously untested behaviour) that the audit found by reading the
code or by deliberately breaking it: a mutation that left the suite green means the test was missing.
"""

import json

import pytest
from fastapi.testclient import TestClient

from semantic_cache.api.app import create_app
from semantic_cache.metrics import TokenPricing
from semantic_cache.models import ChatMessage, ChatRequest, TokenUsage
from semantic_cache.service import CacheService
from semantic_cache.spend import BudgetedProvider, BudgetExceeded, SpendLedger
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider
from semantic_cache.verification import ConstraintGuard


def _client(admin_token: str | None = None) -> tuple[TestClient, RecordingProvider]:
    provider = RecordingProvider(response_text="answer")
    service = CacheService(embedder=DeterministicEmbedder(), provider=provider, similarity_threshold=0.95)
    return TestClient(create_app(service, admin_token=admin_token), raise_server_exceptions=False), provider


BASE = {
    "model": "m",
    "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.0,
    "max_tokens": 100,
}
SYSTEM_SWAPPED = [{"role": "system", "content": "be verbose"}, BASE["messages"][1]]


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"messages": SYSTEM_SWAPPED},
        {"temperature": 0.5},
        {"max_tokens": 200},
        {"top_p": 0.9},
        {"seed": 7},
        {"stop": ["END"]},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_response_affecting_field_isolates_cache_entries_over_http(change) -> None:
    client, provider = _client()
    assert client.post("/v1/chat/completions", json=BASE).headers["X-Cache"] == "MISS"
    variant = client.post("/v1/chat/completions", json=BASE | change)
    assert variant.headers["X-Cache"] == "MISS", f"{next(iter(change))} must not share a cache entry"
    assert provider.call_count == 2
    # ...and the original request still hits its own entry, so the variant did not overwrite it.
    assert client.post("/v1/chat/completions", json=BASE).headers["X-Cache"] == "HIT"


def test_identical_request_over_http_is_a_hit() -> None:
    client, provider = _client()
    client.post("/v1/chat/completions", json=BASE)
    assert client.post("/v1/chat/completions", json=BASE).headers["X-Cache"] == "HIT"
    assert provider.call_count == 1


def test_budgeted_provider_refuses_on_worst_case_output_tokens_before_calling() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        ledger = SpendLedger(Path(directory) / "s.json", cap_usd=0.001)
        inner = RecordingProvider(response_text="ok", usage=TokenUsage(1, 1))
        provider = BudgetedProvider(inner, ledger, TokenPricing(0.4, 1.6))
        # 1M output tokens at $1.60/M is $1.60 in the worst case, far above the $0.001 cap.
        huge = ChatRequest(provider="p", model="m", messages=[ChatMessage("user", "hi")], max_tokens=1_000_000)
        with pytest.raises(BudgetExceeded):
            provider.complete(huge)
        assert inner.call_count == 0, "the provider must not be called when the worst case exceeds the cap"


def test_a_zero_cap_forbids_every_paid_call_but_is_a_valid_setting(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "s.json", cap_usd=0)  # `--max-usd 0` for free, cached-only reruns
    with pytest.raises(BudgetExceeded):
        ledger.ensure_room(1e-9)
    with pytest.raises(ValueError):
        SpendLedger(tmp_path / "t.json", cap_usd=-1)


def test_a_flag_can_lower_the_spend_cap_but_never_raise_it_and_the_file_records_the_repo_cap(tmp_path) -> None:
    from semantic_cache.spend import REPO_HARD_CAP_USD

    path = tmp_path / "s.json"
    low = SpendLedger(path, cap_usd=0.05)
    low.charge(0.01, "x")
    assert json.loads(path.read_text())["cap_usd"] == REPO_HARD_CAP_USD  # not the 0.05 flag: reports must not inherit it
    high = SpendLedger(path, cap_usd=100.0)
    assert high.cap_usd == REPO_HARD_CAP_USD
    with pytest.raises(BudgetExceeded):
        high.ensure_room(REPO_HARD_CAP_USD)  # 0.01 already spent


def test_non_ascii_admin_token_is_rejected_with_401_not_a_server_error() -> None:
    client, _ = _client(admin_token="secret")
    response = client.delete("/v1/cache", headers={"X-Admin-Token": "sécret".encode("latin-1")})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "cached, incoming",
    [
        ("user: Who wrote Hamlet?", "user: Which author wrote Hamlet?"),
        ("user: Why is the sky blue?", "user: What makes the sky blue?"),
        ("user: How do I learn Python?", "user: What is the best way to learn Python?"),
        ('user: "What is the boiling point of water?"', "user: What is the boiling point of water?"),
        ("user: - Who wrote Hamlet?", "user: Which author wrote Hamlet?"),
    ],
)
def test_guard_does_not_reject_paraphrases_over_sentence_initial_words_or_the_pronoun_I(cached, incoming) -> None:
    assert ConstraintGuard().verify(cached_text=cached, incoming_text=incoming).accepted


@pytest.mark.parametrize(
    "cached, incoming",
    [
        ("user: What is the capital of France?", "user: What is the capital of Germany?"),
        ("user: Paris is the capital of what?", "user: London is the capital of what?"),  # sentence-initial name
        ("user: Summarize this in 3 bullets.", "user: Summarize this in 8 bullets."),
        ("user: List foods that contain gluten.", "user: List foods that do not contain gluten."),
        ("user: Give directions from Toronto to Ottawa.", "user: Give directions from Ottawa to Toronto."),
    ],
)
def test_guard_still_rejects_real_differences(cached, incoming) -> None:
    assert not ConstraintGuard().verify(cached_text=cached, incoming_text=incoming).accepted


def test_guard_time_stays_linear_on_a_long_text_of_repeated_from() -> None:
    from time import perf_counter

    cached, incoming = "user: " + "from " * 8000 + "a", "user: " + "from " * 8000 + "b"  # 40 KB, no " to "
    started = perf_counter()
    ConstraintGuard().verify(cached_text=cached, incoming_text=incoming)
    assert perf_counter() - started < 0.5  # was ~2.4 s each with the unbounded pattern
