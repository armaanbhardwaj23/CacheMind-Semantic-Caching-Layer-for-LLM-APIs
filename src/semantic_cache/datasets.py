"""Row loading and intent construction for external paraphrase datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Iterable


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read .jsonl (Hugging Face rows) or .tsv into a list of string-valued dicts."""
    if path.suffix == ".jsonl":
        return [
            {k: str(v) for k, v in json.loads(line).items()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def build_qqp_intents(
    rows: Iterable[dict[str, str]], *, max_intents: int, seed: int = 42
) -> list[dict[str, object]]:
    """Turn labelled question pairs into intent groups for workload simulation.

    * A *duplicate* pair becomes one intent with two phrasings (a genuine paraphrase).
    * A *non-duplicate* pair contributes its two questions as two separate,
      single-phrasing intents. These are the dangerous ones: topically similar
      questions that need different answers, so a low threshold will confuse them.

    Question text is the node identity. Any text that ends up in more than one
    intent is dropped from all of them, so the ground truth stays unambiguous.
    """
    rng = random.Random(seed)
    duplicates: list[tuple[str, str]] = []
    confusable: list[tuple[str, str]] = []
    for row in rows:
        q1, q2 = row["question1"].strip(), row["question2"].strip()
        label = row.get("label", row.get("is_duplicate"))
        if not q1 or not q2 or q1 == q2:
            continue
        (duplicates if label == "1" else confusable).append((q1, q2))

    # Shuffle *units* (a paraphrase group, or a confusable pair) together so the
    # cap keeps QQP's natural mix. Taking all duplicates first would fill the cap with
    # safe paraphrases and leave out the topically-similar-but-different questions.
    units: list[list[tuple[str, ...]]] = [[tuple(sorted(pair))] for pair in duplicates]
    units += [[(q1,), (q2,)] for q1, q2 in confusable]
    rng.shuffle(units)
    candidate_groups = [group for unit in units for group in unit]
    appearances: dict[str, int] = {}
    for group in candidate_groups:
        for text in group:
            appearances[text] = appearances.get(text, 0) + 1
    intents: list[dict[str, object]] = []
    for group in candidate_groups:
        if any(appearances[text] > 1 for text in group):
            continue
        intents.append({"intent_id": f"qqp-intent-{len(intents):04d}", "queries": list(group)})
        if len(intents) >= max_intents:
            break
    return intents
