"""Create a balanced external proxy set from official PAWS-Wiki TSV data.

PAWS paraphrase labels are not automatically equivalent to cache-response
correctness. The generated rows are explicitly marked as external proxy labels
and must not be reported as human-reviewed CacheMind ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from semantic_cache.datasets import read_rows


def build_rows(source: Path, *, per_label: int, seed: int) -> list[dict[str, object]]:
    by_label: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for row in read_rows(source):
        if row.get("label") in by_label:
            by_label[row["label"]].append(row)
    if any(len(rows) < per_label for rows in by_label.values()):
        raise ValueError("PAWS source does not contain enough examples for both labels.")

    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for label in ("1", "0"):
        for index, row in enumerate(rng.sample(by_label[label], per_label), start=1):
            reusable = label == "1"
            output.append(
                {
                    "case_id": f"paws-wiki-{'hit' if reusable else 'miss'}-{index:03d}",
                    "category": "external_paraphrase_proxy" if reusable else "external_adversarial_proxy",
                    "cached_query": row["sentence1"],
                    "incoming_query": row["sentence2"],
                    "reuse_is_acceptable": reusable,
                    "rationale": (
                        "Mapped from the PAWS-Wiki human paraphrase label; requires CacheMind-specific review."
                    ),
                    "label_source": "PAWS-Wiki labeled_final",
                    "review_status": "external_proxy_not_human_reviewed",
                    "source_id": row["id"],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.source, per_label=args.per_label, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
