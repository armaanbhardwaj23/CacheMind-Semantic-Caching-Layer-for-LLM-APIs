from semantic_cache.evaluation import EvaluationCase, ScoredCase, evaluate_thresholds, score_cases
from semantic_cache.verification import ConstraintGuard


class StaticEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.vectors = {
            "user: same": (1.0, 0.0),
            "user: paraphrase": (0.98, 0.2),
            "user: unsafe": (0.95, 0.3),
        }

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self.vectors[text]


def test_scoring_reuses_embeddings_for_repeated_queries() -> None:
    cases = [
        EvaluationCase("one", "exact", "same", "same", True, "same meaning"),
        EvaluationCase("two", "unsafe", "same", "unsafe", False, "changed meaning"),
    ]
    embedder = StaticEmbedder()

    scored = score_cases(cases, embedder)

    assert len(embedder.calls) == 2
    assert scored[0].similarity_score == 1.0
    assert scored[1].similarity_score < 1.0


def test_evaluation_uses_the_same_single_user_representation_as_the_service() -> None:
    case = EvaluationCase("one", "exact", "same", "same", True, "same meaning")
    embedder = StaticEmbedder()

    score_cases([case], embedder)

    assert embedder.calls == ["user: same"]


def test_threshold_results_expose_false_hits_and_valid_reuse_recall() -> None:
    cases = [
        ScoredCase(EvaluationCase("one", "exact", "", "", True, ""), 1.0),
        ScoredCase(EvaluationCase("two", "unsafe", "", "", False, ""), 0.97),
        ScoredCase(EvaluationCase("three", "paraphrase", "", "", True, ""), 0.93),
    ]

    result = evaluate_thresholds(cases, [0.95])[0]

    assert result.predicted_hits == 2
    assert result.valid_hits == 1
    assert result.false_hits == 1
    assert result.hit_precision == 0.5
    assert result.false_hit_rate == 0.5
    assert result.valid_reuse_recall == 0.5


def test_evaluation_reports_guard_rejections_separately_from_threshold_misses() -> None:
    cases = [
        ScoredCase(
            EvaluationCase(
                "number", "changed_number", "Summarize in 3 bullets", "Summarize in 8 bullets", False, ""
            ),
            0.99,
        )
    ]

    result = evaluate_thresholds(cases, [0.95], verifier=ConstraintGuard())[0]

    assert result.predicted_hits == 0
    assert result.false_hits == 0
    assert result.verification_rejections == 1


from semantic_cache.embedding_cache import DiskCachedEmbedder
from semantic_cache.evaluation import render_markdown_table, render_svg_curve, select_threshold


def _scored(pairs):
    return [
        ScoredCase(EvaluationCase(f"c{i}", cat, "", "", ok, ""), score)
        for i, (cat, ok, score) in enumerate(pairs)
    ]


def test_f1_and_false_hits_by_category() -> None:
    scored = _scored([("paraphrase", True, 0.99), ("negation", False, 0.97), ("changed_number", False, 0.96), ("paraphrase", True, 0.90)])
    result = evaluate_thresholds(scored, [0.95])[0]
    assert result.false_hits_by_category == {"negation": 1, "changed_number": 1}
    assert result.hit_precision == 1 / 3 and result.valid_reuse_recall == 0.5
    assert abs(result.f1 - 0.4) < 1e-9


def test_select_threshold_prefers_the_most_permissive_within_the_false_hit_budget() -> None:
    scored = _scored([("p", True, 0.99), ("p", True, 0.96), ("n", False, 0.93)])
    results = evaluate_thresholds(scored, [0.90, 0.95, 0.98])
    assert select_threshold(results, max_false_hit_rate=0.0).threshold == 0.95
    assert select_threshold(results, max_false_hit_rate=1.0).threshold == 0.90


def test_select_threshold_returns_none_when_no_threshold_is_safe() -> None:
    results = evaluate_thresholds(_scored([("n", False, 0.99)]), [0.9, 0.95])
    assert select_threshold(results, max_false_hit_rate=0.1) is None


def test_renderers_produce_a_table_and_valid_looking_svg() -> None:
    results = evaluate_thresholds(_scored([("p", True, 0.99), ("n", False, 0.93)]), [0.90, 0.95])
    table = render_markdown_table(results)
    assert table.count("\n") == 3 and "| 0.95 |" in table
    svg = render_svg_curve(results, title="t")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>") and "polyline" in svg


def test_disk_cached_embedder_only_calls_the_provider_once_per_text(tmp_path) -> None:
    calls = []

    class Inner:
        def embed(self, text):
            calls.append(text)
            return (1.0, 2.0)

    path = tmp_path / "emb.jsonl"
    first = DiskCachedEmbedder(Inner(), model="m", path=path)
    first.embed("a"); first.embed("a"); first.embed("b")
    second = DiskCachedEmbedder(Inner(), model="m", path=path)  # new process, same file
    assert second.embed("a") == (1.0, 2.0)
    other_model = DiskCachedEmbedder(Inner(), model="other", path=path)
    other_model.embed("a")
    assert calls == ["a", "b", "a"] and (first.hits, first.misses) == (1, 2) and second.hits == 1
