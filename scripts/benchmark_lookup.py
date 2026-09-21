"""Lookup latency versus cache size. Offline: random vectors, no API calls.

Measures only the *vector search step* of the in-memory store (1536-d, the
text-embedding-3-small dimensionality), comparing the numpy implementation with
the original pure-Python loop it replaced. Embedding-API and provider latency are
deliberately excluded: this isolates the cost the cache itself adds.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np

from semantic_cache.cache import InMemorySemanticCache
from semantic_cache.models import CacheIdentity, ChatCompletion

DIM = 1536


def pure_python_scan(entries: list[tuple[float, ...]], query: tuple[float, ...]) -> float:
    q_norm = sqrt(sum(v * v for v in query))
    best = -1.0
    for vector in entries:
        dot = sum(a * b for a, b in zip(query, vector))
        norm = sqrt(sum(v * v for v in vector))
        best = max(best, dot / (q_norm * norm))
    return best


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 5_000, 10_000, 25_000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--python-baseline-max", type=int, default=5_000,
                        help="Largest size at which the slow pure-Python scan is also timed.")
    parser.add_argument("--output", type=Path, default=Path("evals/results/lookup-benchmark.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    identity = CacheIdentity("bench", "bench", "s", 0.0, 64)
    rows = []
    for size in args.sizes:
        vectors = rng.standard_normal((size, DIM)).astype(np.float32)
        cache = InMemorySemanticCache()
        response = ChatCompletion("x", "bench")
        for i in range(size):
            cache.store(identity=identity, semantic_lookup_text=str(i), embedding=vectors[i], response=response)
        queries = rng.standard_normal((args.queries, DIM)).astype(np.float32)

        numpy_ms = []
        for q in queries:
            start = perf_counter()
            cache.lookup(identity=identity, semantic_lookup_text="q", embedding=q, similarity_threshold=0.95)
            numpy_ms.append((perf_counter() - start) * 1000)

        row = {"entries": size, "numpy_p50_ms": median(numpy_ms), "numpy_p95_ms": percentile(numpy_ms, 0.95),
               "numpy_p99_ms": percentile(numpy_ms, 0.99)}
        if size <= args.python_baseline_max:
            listed = [tuple(map(float, v)) for v in vectors]
            py_ms = []
            for q in queries[:10]:
                start = perf_counter()
                pure_python_scan(listed, tuple(map(float, q)))
                py_ms.append((perf_counter() - start) * 1000)
            row["pure_python_p50_ms"] = median(py_ms)
            row["speedup_at_p50"] = median(py_ms) / median(numpy_ms)
        rows.append(row)
        print(row)

    report = {"dimensions": DIM, "queries_per_size": args.queries, "machine": platform.platform(),
              "python": platform.python_version(), "numpy": np.__version__,
              "note": "Random unit-scale vectors; measures brute-force cosine search only.", "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
