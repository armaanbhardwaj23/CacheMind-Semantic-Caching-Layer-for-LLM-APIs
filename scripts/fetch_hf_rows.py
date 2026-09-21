"""Fetch a random sample of rows from a public Hugging Face dataset (no dataset download).

Uses the datasets-server rows API in pages of <=100 rows at seeded random offsets,
so a few hundred KB is transferred instead of the full dataset. Output is JSONL in
a git-ignored folder. Only public data is fetched; no credentials are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import httpx

API = "https://datasets-server.huggingface.co/rows"


def _get(client: httpx.Client, params: dict, *, retries: int = 6) -> httpx.Response:
    """GET with backoff for 429/5xx *and* dropped connections (the free rows API does both)."""
    last_error = "no attempt made"
    for attempt in range(retries):
        wait = 2.0 * 2**attempt
        try:
            response = client.get(API, params=params)
        except httpx.TransportError as error:  # server disconnected, connect/read timeout, ...
            last_error = type(error).__name__
            print(f"  {last_error}; waiting {wait:.0f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After", "")
        wait = float(retry_after) if retry_after.isdigit() else wait
        last_error = f"HTTP {response.status_code}"
        print(f"  {last_error}; waiting {wait:.0f}s (attempt {attempt + 1}/{retries})")
        time.sleep(wait)
    raise RuntimeError(f"Hugging Face rows API kept failing ({last_error}) after {retries} attempts.")


def fetch_sample(dataset: str, config: str, split: str, rows: int, seed: int, page: int = 100) -> list[dict]:
    rng = random.Random(seed)
    with httpx.Client(timeout=30.0) as client:
        first = _get(client, dict(dataset=dataset, config=config, split=split, offset=0, length=1))
        total = first.json()["num_rows_total"]
        offsets = sorted(rng.sample(range(0, total - page), (rows + page - 1) // page))
        collected: list[dict] = []
        for offset in offsets:
            response = _get(client, dict(dataset=dataset, config=config, split=split, offset=offset, length=page))
            collected += [item["row"] for item in response.json()["rows"]]
            time.sleep(1.0)  # be polite to a free public endpoint
    return collected[:rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page", type=int, default=100, help="rows per request (<=100); part of the sampling seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = fetch_sample(args.dataset, args.config, args.split, args.rows, args.seed, args.page)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {len(rows)} rows ({size_kb:.0f} KB) to {args.output}")


if __name__ == "__main__":
    main()
