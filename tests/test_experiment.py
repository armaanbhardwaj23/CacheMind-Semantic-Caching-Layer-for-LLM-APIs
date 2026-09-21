import pytest

from semantic_cache.experiment import build_embedder, prefetch
from semantic_cache.openrouter import OpenRouterError
from semantic_cache.spend import BudgetExceeded, SpendLedger


class FakeClient:
    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[str] = []
        self._fail_first = fail_first

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        if self._fail_first > 0:
            self._fail_first -= 1
            raise OpenRouterError("boom")
        return (1.0, 0.0)


def make(tmp_path, client, cap=1.0):
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=cap)
    embedder = build_embedder(client, model="m", cache_path=tmp_path / "emb.jsonl", ledger=ledger)
    return embedder, ledger


def test_prefetch_embeds_each_distinct_text_once_and_rerun_is_free(tmp_path) -> None:
    client = FakeClient()
    embedder, ledger = make(tmp_path, client)
    prefetch(embedder, ["a", "b", "a", "c"], workers=2)
    assert sorted(client.calls) == ["a", "b", "c"]
    spent = ledger.total_usd
    assert spent > 0

    prefetch(embedder, ["a", "b", "c"], workers=2)  # disk cache sits outside the budget wrapper
    assert len(client.calls) == 3 and ledger.total_usd == spent

    # A fresh process reads the JSONL cache, so it costs nothing either.
    client2 = FakeClient()
    embedder2, _ = make(tmp_path, client2)
    prefetch(embedder2, ["a", "b", "c"])
    assert client2.calls == []


def test_prefetch_retries_transient_provider_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("semantic_cache.experiment.time.sleep", lambda _s: None)
    client = FakeClient(fail_first=2)
    embedder, _ = make(tmp_path, client)
    prefetch(embedder, ["a"], workers=1)
    assert len(client.calls) == 3


def test_prefetch_stops_on_budget_and_does_not_retry_it(tmp_path) -> None:
    client = FakeClient()
    embedder, _ = make(tmp_path, client, cap=1e-9)
    with pytest.raises(BudgetExceeded):
        prefetch(embedder, ["x" * 10_000])
    assert client.calls == []


def test_serve_exposes_the_app_over_real_http() -> None:
    import httpx
    from fastapi import FastAPI

    from semantic_cache.experiment import serve

    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    with serve(app) as base_url:
        assert httpx.get(f"{base_url}/ping", timeout=5).json() == {"ok": "yes"}
