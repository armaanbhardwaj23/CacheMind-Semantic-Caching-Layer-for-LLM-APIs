"""Human-label-driven threshold evaluation for semantic-cache candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol, Sequence

from semantic_cache.cache import cosine_similarity
from semantic_cache.representation import build_single_user_lookup_text
from semantic_cache.verification import CandidateVerifier


class Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    cached_query: str
    incoming_query: str
    reuse_is_acceptable: bool
    rationale: str


@dataclass(frozen=True)
class ScoredCase:
    case: EvaluationCase
    similarity_score: float


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total_cases: int
    predicted_hits: int
    valid_hits: int
    false_hits: int
    verification_rejections: int
    hit_rate: float
    hit_precision: float | None
    false_hit_rate: float | None
    valid_reuse_recall: float | None
    f1: float | None = None
    false_hits_by_category: dict[str, int] = field(default_factory=dict)


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a JSON list of human-reviewed candidate pairs."""
    raw_cases = json.loads(path.read_text())
    if not isinstance(raw_cases, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    cases: list[EvaluationCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation case must be an object.")
        try:
            case = EvaluationCase(
                case_id=item["case_id"],
                category=item["category"],
                cached_query=item["cached_query"],
                incoming_query=item["incoming_query"],
                reuse_is_acceptable=item["reuse_is_acceptable"],
                rationale=item["rationale"],
            )
        except KeyError as error:
            raise ValueError(f"Evaluation case missing {error.args[0]!r}.") from error
        if not all(
            isinstance(value, str)
            for value in (
                case.case_id,
                case.category,
                case.cached_query,
                case.incoming_query,
                case.rationale,
            )
        ) or not isinstance(case.reuse_is_acceptable, bool):
            raise ValueError(f"Evaluation case {case.case_id!r} has invalid field types.")
        cases.append(case)
    return cases


def score_cases(cases: Sequence[EvaluationCase], embedder: Embedder) -> list[ScoredCase]:
    """Embed each unique query once, then calculate each pair's cosine score."""
    embeddings: dict[str, tuple[float, ...]] = {}

    def embedding_for(text: str) -> tuple[float, ...]:
        if text not in embeddings:
            embeddings[text] = embedder.embed(build_single_user_lookup_text(text))
        return embeddings[text]

    return [
        ScoredCase(
            case=case,
            similarity_score=cosine_similarity(
                embedding_for(case.cached_query), embedding_for(case.incoming_query)
            ),
        )
        for case in cases
    ]


def evaluate_thresholds(
    scored_cases: Sequence[ScoredCase], thresholds: Sequence[float], *, verifier: CandidateVerifier | None = None
) -> list[ThresholdResult]:
    results: list[ThresholdResult] = []
    total_acceptable = sum(case.case.reuse_is_acceptable for case in scored_cases)
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        candidates = [case for case in scored_cases if case.similarity_score >= threshold]
        hits = [
            case
            for case in candidates
            if verifier is None
            or verifier.verify(
                cached_text=build_single_user_lookup_text(case.case.cached_query),
                incoming_text=build_single_user_lookup_text(case.case.incoming_query),
            ).accepted
        ]
        valid_hits = sum(case.case.reuse_is_acceptable for case in hits)
        false_hits = len(hits) - valid_hits
        precision = valid_hits / len(hits) if hits else None
        recall = valid_hits / total_acceptable if total_acceptable else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        results.append(
            ThresholdResult(
                threshold=threshold,
                total_cases=len(scored_cases),
                predicted_hits=len(hits),
                valid_hits=valid_hits,
                false_hits=false_hits,
                verification_rejections=len(candidates) - len(hits),
                hit_rate=len(hits) / len(scored_cases) if scored_cases else 0.0,
                hit_precision=precision,
                false_hit_rate=false_hits / len(hits) if hits else None,
                valid_reuse_recall=recall,
                f1=f1,
                false_hits_by_category=dict(
                    Counter(c.case.category for c in hits if not c.case.reuse_is_acceptable)
                ),
            )
        )
    return results


def select_threshold(
    results: Sequence[ThresholdResult], *, max_false_hit_rate: float
) -> ThresholdResult | None:
    """Pick the most permissive threshold whose false-hit rate stays within budget.

    "Most permissive" means the lowest threshold, i.e. the most reuse. Returns
    ``None`` when no evaluated threshold meets the budget while producing at least
    one hit - which is itself a finding: the embedding alone cannot separate this
    workload safely.
    """
    eligible = [
        r
        for r in results
        if r.predicted_hits > 0
        and r.false_hit_rate is not None
        and r.false_hit_rate <= max_false_hit_rate
    ]
    return min(eligible, key=lambda r: r.threshold) if eligible else None


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render_markdown_table(results: Sequence[ThresholdResult]) -> str:
    lines = [
        "| Threshold | Hits | Hit rate | Precision | False-hit rate | Valid-reuse recall | F1 | Guard rejections |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.threshold:.2f} | {r.predicted_hits}/{r.total_cases} | {_pct(r.hit_rate)} | "
            f"{_pct(r.hit_precision)} | {_pct(r.false_hit_rate)} | {_pct(r.valid_reuse_recall)} | "
            f"{_pct(r.f1)} | {r.verification_rejections} |"
        )
    return "\n".join(lines)


def render_svg_curve(results: Sequence[ThresholdResult], *, title: str) -> str:
    """Dependency-free line chart of hit rate, precision and recall against threshold."""
    width, height, left, right, top, bottom = 640, 380, 56, 130, 44, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    thresholds = [r.threshold for r in results]
    low, high = min(thresholds), max(thresholds)
    span = (high - low) or 1.0

    def x(value: float) -> float:
        return left + (value - low) / span * plot_w

    def y(value: float) -> float:
        return top + (1 - value) * plot_h

    series = [
        ("Hit rate", "#2a6fdb", [r.hit_rate for r in results]),
        ("Precision", "#1a9850", [r.hit_precision for r in results]),
        ("Valid-reuse recall", "#d9822b", [r.valid_reuse_recall for r in results]),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="24" font-size="14" font-weight="bold">{title}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#e3e3e3"/>')
        parts.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#555">{int(tick * 100)}%</text>')
    for value in thresholds:
        parts.append(f'<text x="{x(value):.1f}" y="{top + plot_h + 18}" text-anchor="middle" fill="#555">{value:.2f}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" fill="#555">similarity threshold</text>')
    for name, color, values in series:
        points = [(x(r.threshold), y(v)) for r, v in zip(results, values) if v is not None]
        if not points:
            continue
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')
        last_x, last_y = points[-1]
        parts.append(f'<text x="{last_x + 8:.1f}" y="{last_y + 4:.1f}" fill="{color}">{name}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
