import pytest

from semantic_cache.simulation import Intent, OracleProvider, build_request_stream, simulate
from semantic_cache.verification import ConstraintGuard

INTENTS = [
    Intent("capital-fr", ("What is the capital of France?", "Name France's capital city.")),
    Intent("capital-de", ("What is the capital of Germany?",)),
]


class MappedEmbedder:
    """France questions cluster together; Germany sits close but not identical."""

    vectors = {
        "user: What is the capital of France?": (1.0, 0.0, 0.0),
        "user: Name France's capital city.": (0.99, 0.14, 0.0),
        "user: What is the capital of Germany?": (0.97, 0.05, 0.24),
    }

    def embed(self, text):
        return self.vectors[text]


STREAM = [
    ("capital-fr", "What is the capital of France?"),
    ("capital-fr", "Name France's capital city."),
    ("capital-de", "What is the capital of Germany?"),
    ("capital-de", "What is the capital of Germany?"),
]


def test_high_threshold_serves_only_exact_repeats() -> None:
    result = simulate(INTENTS, STREAM, MappedEmbedder(), threshold=0.999)
    assert (result.hits, result.valid_hits, result.false_hits, result.provider_calls) == (1, 1, 0, 3)
    assert result.false_hit_rate == 0.0


def test_low_threshold_reuses_more_and_serves_a_wrong_answer_for_germany() -> None:
    result = simulate(INTENTS, STREAM, MappedEmbedder(), threshold=0.90)
    # Germany is answered with France's cached answer, so Germany is never stored
    # and its repeat is a *second* false hit: one bad hit poisons later traffic.
    assert (result.valid_hits, result.false_hits, result.provider_calls) == (1, 2, 1)
    assert result.wrongly_served_rate == pytest.approx(0.5)
    assert result.provider_calls_avoided_rate == pytest.approx(0.75)


def test_constraint_guard_can_block_the_false_hit() -> None:
    result = simulate(INTENTS, STREAM, MappedEmbedder(), threshold=0.90, verifier=ConstraintGuard())
    assert result.false_hits == 0  # named values differ: France vs Germany


def test_request_stream_is_seeded_and_head_heavy() -> None:
    many = [Intent(f"i{n}", (f"q{n}",)) for n in range(50)]
    a = build_request_stream(many, n_requests=500, seed=1)
    assert a == build_request_stream(many, n_requests=500, seed=1)
    counts = {}
    for intent_id, _ in a:
        counts[intent_id] = counts.get(intent_id, 0) + 1
    assert max(counts.values()) > 5 * (500 / 50) * 0.5  # popular head exists


def test_oracle_rejects_queries_shared_between_intents() -> None:
    with pytest.raises(ValueError):
        OracleProvider([Intent("a", ("same",)), Intent("b", ("same",))])
