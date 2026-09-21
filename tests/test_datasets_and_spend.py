import pytest

from semantic_cache.datasets import build_qqp_intents, read_rows
from semantic_cache.metrics import TokenPricing
from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.spend import BudgetedEmbedder, BudgetedProvider, BudgetExceeded, SpendLedger
from semantic_cache.testing import RecordingProvider
from semantic_cache.models import TokenUsage


def rows():
    return [
        {"question1": "How do I learn Python?", "question2": "What is the best way to learn Python?", "label": "1"},
        {"question1": "How old is the Moon?", "question2": "How old is the Sun?", "label": "0"},
        {"question1": "same", "question2": "same", "label": "1"},  # identical text: ignored
        {"question1": "How old is the Sun?", "question2": "Why is the sky blue?", "label": "0"},  # reuses Sun text
    ]


def test_intents_use_paraphrase_pairs_and_drop_texts_that_appear_in_two_intents() -> None:
    intents = build_qqp_intents(rows(), max_intents=10, seed=1)
    grouped = {tuple(i["queries"]) for i in intents}
    assert ("How do I learn Python?", "What is the best way to learn Python?") in grouped
    all_texts = [q for i in intents for q in i["queries"]]
    assert "How old is the Sun?" not in all_texts  # ambiguous across intents -> dropped
    assert len(all_texts) == len(set(all_texts))
    assert "How old is the Moon?" in all_texts and "Why is the sky blue?" in all_texts


def test_read_rows_supports_jsonl_and_tsv(tmp_path) -> None:
    j = tmp_path / "a.jsonl"; j.write_text('{"a": 1}\n{"a": 2}\n')
    t = tmp_path / "a.tsv"; t.write_text("a\tb\n1\t2\n")
    assert read_rows(j) == [{"a": "1"}, {"a": "2"}] and read_rows(t) == [{"a": "1", "b": "2"}]


def test_ledger_persists_and_refuses_calls_that_could_exceed_the_cap(tmp_path) -> None:
    path = tmp_path / "spend.json"
    ledger = SpendLedger(path, cap_usd=0.01)
    ledger.charge(0.006, "chat")
    assert SpendLedger(path, cap_usd=0.01).total_usd == pytest.approx(0.006)  # survives restart
    with pytest.raises(BudgetExceeded):
        ledger.ensure_room(0.005)
    ledger.ensure_room(0.003)


def test_budgeted_embedder_charges_and_stops_at_the_cap(tmp_path) -> None:
    class Inner:
        def embed(self, text):
            return (1.0,)

    ledger = SpendLedger(tmp_path / "s.json", cap_usd=0.0000001)
    embedder = BudgetedEmbedder(Inner(), ledger, price_per_million_tokens_usd=0.02)
    with pytest.raises(BudgetExceeded):
        embedder.embed("x" * 100_000)


def test_budgeted_provider_charges_actual_usage(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "s.json", cap_usd=1.0)
    provider = BudgetedProvider(
        RecordingProvider(response_text="ok", usage=TokenUsage(1_000_000, 0)), ledger, TokenPricing(0.4, 1.6)
    )
    provider.complete(ChatRequest(provider="p", model="m", messages=[ChatMessage("user", "hi")], max_tokens=10))
    assert ledger.total_usd == pytest.approx(0.4)


def test_intent_cap_keeps_confusable_questions_instead_of_filling_up_with_duplicates() -> None:
    many = [{"question1": f"dup a {i}", "question2": f"dup b {i}", "label": "1"} for i in range(50)]
    many += [{"question1": f"non a {i}", "question2": f"non b {i}", "label": "0"} for i in range(50)]
    intents = build_qqp_intents(many, max_intents=40, seed=3)
    singles = sum(len(i["queries"]) == 1 for i in intents)
    groups = sum(len(i["queries"]) == 2 for i in intents)
    assert singles > 0 and groups > 0
