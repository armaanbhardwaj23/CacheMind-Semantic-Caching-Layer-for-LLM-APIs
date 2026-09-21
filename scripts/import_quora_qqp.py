"""Build evaluation pairs and workload intents from Quora Question Pairs rows.

Input: JSONL from scripts/fetch_hf_rows.py (GLUE QQP: question1, question2, label)
or the original Quora TSV (question1, question2, is_duplicate). Duplicate labels
are *human* labels of "same question": a proxy - not proof - that one answer serves
both, so rows are marked ``external_proxy_not_human_reviewed``.

Outputs
  --pairs-output    balanced pair file for scripts/run_threshold_experiment.py
  --intents-output  intent groups for scripts/simulate_workload.py
                    (construction rules: see semantic_cache.datasets.build_qqp_intents)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from semantic_cache.datasets import build_qqp_intents, read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--intents-output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=150)
    parser.add_argument("--max-intents", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.source)
    rng = random.Random(args.seed)
    by_label: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for row in rows:
        label = row.get("label", row.get("is_duplicate"))
        if label in by_label and row["question1"].strip() and row["question2"].strip():
            by_label[label].append(row)
    if any(len(v) < args.per_label for v in by_label.values()):
        raise SystemExit(f"Need >= {args.per_label} rows per label; got { {k: len(v) for k, v in by_label.items()} }.")

    pairs = []
    for label in ("1", "0"):
        for index, row in enumerate(rng.sample(by_label[label], args.per_label), start=1):
            reusable = label == "1"
            pairs.append({
                "case_id": f"qqp-{'dup' if reusable else 'nondup'}-{index:03d}",
                "category": "external_paraphrase_proxy" if reusable else "external_similar_but_different_proxy",
                "cached_query": row["question1"], "incoming_query": row["question2"],
                "reuse_is_acceptable": reusable,
                "rationale": "Mapped from the Quora Question Pairs human duplicate label; requires CacheMind-specific review.",
                "label_source": "Quora Question Pairs (GLUE QQP)",
                "review_status": "external_proxy_not_human_reviewed",
                "source_id": row.get("idx", row.get("id", "")),
            })
    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.write_text(json.dumps(pairs, indent=2) + "\n")

    intents = build_qqp_intents(rows, max_intents=args.max_intents, seed=args.seed)
    args.intents_output.parent.mkdir(parents=True, exist_ok=True)
    args.intents_output.write_text(json.dumps(intents, indent=2) + "\n")
    multi = sum(len(i["queries"]) > 1 for i in intents)
    print(f"Wrote {len(pairs)} pairs -> {args.pairs_output}; {len(intents)} intents ({multi} paraphrase groups) -> {args.intents_output}")


if __name__ == "__main__":
    main()
