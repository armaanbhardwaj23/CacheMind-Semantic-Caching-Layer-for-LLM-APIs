"""Pair-level threshold experiment on a labelled dataset (real embeddings; paid API calls).

Embeddings are cached on disk (.cache/embeddings.jsonl), so re-running with the
same dataset costs nothing. Outputs a JSON report, and optionally a Markdown
table and an SVG curve for the README.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from semantic_cache.config import OpenRouterSettings
from semantic_cache.evaluation import (
    evaluate_thresholds,
    load_cases,
    render_markdown_table,
    render_svg_curve,
    score_cases,
    select_threshold,
)
from semantic_cache.experiment import build_embedder, prefetch
from semantic_cache.openrouter import OpenRouterClient
from semantic_cache.representation import build_single_user_lookup_text
from semantic_cache.spend import SpendLedger
from semantic_cache.verification import ConstraintGuard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/cache_reuse_pairs.draft.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--svg-output", type=Path)
    parser.add_argument("--with-constraint-guard", action="store_true")
    parser.add_argument("--max-false-hit-rate", type=float, default=0.05,
                        help="Budget used to report the most permissive acceptable threshold.")
    parser.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings.jsonl"))
    parser.add_argument("--spend-ledger", type=Path, default=Path(".cache/spend.json"))
    parser.add_argument("--max-usd", type=float, default=0.20,
                        help="Hard cap for ALL paid calls recorded in the ledger (default: the repo cap).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98],
    )
    args = parser.parse_args()

    settings = OpenRouterSettings.from_env()
    cases = load_cases(args.dataset)
    client = OpenRouterClient(settings)
    ledger = SpendLedger(args.spend_ledger, cap_usd=args.max_usd)
    embedder = build_embedder(client, model=settings.embedding_model, cache_path=args.embedding_cache, ledger=ledger)
    try:
        prefetch(
            embedder,
            (build_single_user_lookup_text(q) for c in cases for q in (c.cached_query, c.incoming_query)),
            workers=args.workers,
        )
        scored = score_cases(cases, embedder)
    finally:
        client.close()

    verifier = ConstraintGuard() if args.with_constraint_guard else None
    results = evaluate_thresholds(scored, args.thresholds, verifier=verifier)
    chosen = select_threshold(results, max_false_hit_rate=args.max_false_hit_rate)

    report = {
        "dataset": str(args.dataset),
        "embedding_model": settings.embedding_model,
        "cases": len(cases),
        "acceptable_reuse_cases": sum(c.reuse_is_acceptable for c in cases),
        "verification": "constraint_guard" if verifier else "none",
        "embedding_calls_paid_this_run": embedder.misses,
        "ledger_total_usd_after_run": round(ledger.total_usd, 6),
        "max_false_hit_rate_budget": args.max_false_hit_rate,
        "selected_threshold": chosen.threshold if chosen else None,
        "scored_cases": [
            {"case_id": s.case.case_id, "category": s.case.category,
             "reuse_is_acceptable": s.case.reuse_is_acceptable, "similarity_score": s.similarity_score}
            for s in scored
        ],
        "threshold_results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}  (paid embedding calls this run: {embedder.misses}; "
          f"ledger ${ledger.total_usd:.5f} of ${args.max_usd:.2f})")
    table = render_markdown_table(results)
    print(table)
    print(f"Selected threshold at <= {args.max_false_hit_rate:.0%} false-hit rate:",
          chosen.threshold if chosen else "none - no evaluated threshold is safe enough")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(table + "\n")
    if args.svg_output:
        args.svg_output.parent.mkdir(parents=True, exist_ok=True)
        args.svg_output.write_text(render_svg_curve(results, title=f"Threshold trade-off: {args.dataset.stem}"))


if __name__ == "__main__":
    main()
