"""Workload-level simulation: real embeddings + oracle provider (see semantic_cache.simulation).

Input is an *intents* file: a JSON list of {"intent_id": str, "queries": [str, ...]},
where every query in a group asks the same thing. Produce one from Quora Question
Pairs with scripts/import_quora_qqp.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from semantic_cache.config import OpenRouterSettings
from semantic_cache.evaluation import _pct
from semantic_cache.experiment import build_embedder, prefetch
from semantic_cache.openrouter import OpenRouterClient
from semantic_cache.representation import build_single_user_lookup_text
from semantic_cache.simulation import build_request_stream, load_intents, simulate
from semantic_cache.spend import SpendLedger
from semantic_cache.verification import ConstraintGuard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-cache", type=Path, default=Path(".cache/embeddings.jsonl"))
    parser.add_argument("--spend-ledger", type=Path, default=Path(".cache/spend.json"))
    parser.add_argument("--max-usd", type=float, default=0.20,
                        help="Hard cap for ALL paid calls recorded in the ledger (default: the repo cap).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.85, 0.90, 0.92, 0.95, 0.98])
    args = parser.parse_args()

    settings = OpenRouterSettings.from_env()
    intents = load_intents(args.intents)
    stream = build_request_stream(intents, n_requests=args.requests, seed=args.seed, zipf_s=args.zipf)
    client = OpenRouterClient(settings)
    ledger = SpendLedger(args.spend_ledger, cap_usd=args.max_usd)
    embedder = build_embedder(client, model=settings.embedding_model, cache_path=args.embedding_cache, ledger=ledger)
    results = []
    try:
        # Only phrasings that actually appear in the request stream need embedding.
        prefetch(embedder, (build_single_user_lookup_text(q) for _, q in stream), workers=args.workers)
        for guard in (False, True):
            for threshold in args.thresholds:
                results.append(
                    simulate(intents, stream, embedder, threshold=threshold,
                             verifier=ConstraintGuard() if guard else None)
                )
    finally:
        client.close()

    report = {
        "intents_file": str(args.intents), "intents": len(intents), "requests": args.requests,
        "zipf_exponent": args.zipf, "seed": args.seed, "embedding_model": settings.embedding_model,
        "embedding_calls_made": embedder.misses,
        "ledger_total_usd_after_run": round(ledger.total_usd, 6), "results": [asdict(r) for r in results],
        "note": "Provider is a ground-truth oracle; only correctness and calls avoided are measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    rows = ["| Guard | Threshold | Hit rate | Valid hits | False hits | False-hit rate | Provider calls avoided | Wrongly served (all requests) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        rows.append(f"| {'on' if r.guard else 'off'} | {r.threshold:.2f} | {_pct(r.hit_rate)} | {r.valid_hits} | "
                    f"{r.false_hits} | {_pct(r.false_hit_rate)} | {_pct(r.provider_calls_avoided_rate)} | {_pct(r.wrongly_served_rate)} |")
    table = "\n".join(rows)
    print(table)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(table + "\n")


if __name__ == "__main__":
    main()
