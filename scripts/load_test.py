"""Concurrency load test with a SYNTHETIC provider ($0, no network).

What is synthetic: the provider's latency (a configured sleep + jitter) and the embedder
(hash-derived vectors + a configured sleep). Those numbers are assumptions, not measurements
of any real API - the point of this test is the *proxy's own* behaviour under load:
hit latency at concurrency, the thread-pool ceiling on misses, and the effect of a
head-heavy request mix on throughput.

Arms
  baseline  N async workers -> synthetic provider server over HTTP (the provider sleeps async,
            so it never queues: the best case for "no cache").
  cached    the same workload -> CacheMind over HTTP; on a miss the proxy calls a *blocking*
            synthetic provider, as the real sync OpenRouter client does, so misses occupy
            worker threads. This is deliberate: it exposes the real concurrency limit.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter, sleep

import httpx
from fastapi import FastAPI

from semantic_cache.api.app import create_app
from semantic_cache.experiment import serve
from semantic_cache.metrics import CacheMetrics
from semantic_cache.models import ChatCompletion, ChatRequest, TokenUsage
from semantic_cache.service import CacheService
from semantic_cache.workload import latency_summary


class SyntheticEmbedder:
    def __init__(self, latency_ms: float) -> None:
        self._latency = latency_ms / 1000

    def embed(self, text: str) -> tuple[float, ...]:
        sleep(self._latency)
        digest = sha256(text.encode()).digest()
        return tuple(byte / 255 + 0.01 for byte in digest[:16])


class SyntheticProvider:
    """Blocking provider with jittered latency, like a sync HTTP client."""

    def __init__(self, latency_ms: float, jitter: float, seed: int) -> None:
        self._latency, self._jitter, self._rng = latency_ms / 1000, jitter, random.Random(seed)
        self._lock = Lock()  # the proxy calls this from many worker threads
        self.calls = 0

    def complete(self, request: ChatRequest) -> ChatCompletion:
        with self._lock:
            self.calls += 1
            delay = self._latency * (1 + self._rng.uniform(-self._jitter, self._jitter))
        sleep(delay)
        return ChatCompletion(content="synthetic answer", model=request.model, usage=TokenUsage(30, 40))


def provider_server(latency_ms: float, jitter: float, seed: int) -> FastAPI:
    """The 'baseline' provider: async, so it can serve unlimited concurrent requests."""
    app = FastAPI()
    rng = random.Random(seed)

    @app.post("/v1/chat/completions")
    async def completions(payload: dict) -> dict:
        await asyncio.sleep(latency_ms / 1000 * (1 + rng.uniform(-jitter, jitter)))
        return {"choices": [{"message": {"content": "synthetic answer"}}]}

    return app


def with_threadpool(app: FastAPI, size: int | None) -> FastAPI:
    """Resize the AnyIO worker pool that runs the proxy's synchronous request handlers.

    Starlette runs a sync endpoint in AnyIO's default thread pool (40 tokens unless changed), so at
    most 40 blocking requests can be in flight; hits queue behind misses once it is full.
    """
    if size is None:
        return app
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = size
        async with original(application) as state:
            yield state

    app.router.lifespan_context = lifespan
    return app


def colliding_pairs_rejected_by_guard(prompts: list[str], threshold: float) -> int:
    """Count distinct prompt pairs the synthetic embedder scores >= threshold, and require the guard to reject them all.

    The synthetic embedder is 16-dimensional and non-negative, so unrelated prompts can look similar. The
    prompts differ only in a number, which the constraint guard rejects; "provider calls avoided" and
    "redundant provider calls" are only meaningful while no such pair could be served as a wrong semantic hit.
    """
    import numpy as np

    from semantic_cache.cache import normalize
    from semantic_cache.models import ChatMessage
    from semantic_cache.representation import build_semantic_lookup_text
    from semantic_cache.verification import ConstraintGuard

    unique = sorted(set(prompts))
    embedder = SyntheticEmbedder(0)
    matrix = np.stack([normalize(embedder.embed(p)) for p in unique])
    scores = np.triu(matrix @ matrix.T, k=1)
    guard, count = ConstraintGuard(), 0
    for i, j in zip(*np.where(scores >= threshold)):
        texts = [build_semantic_lookup_text([ChatMessage("user", unique[k])]) for k in (i, j)]
        if guard.verify(cached_text=texts[0], incoming_text=texts[1]).accepted:
            raise SystemExit(f"Synthetic prompts {unique[i]!r} and {unique[j]!r} would be served as a wrong semantic hit; "
                             "the load-test counts would be invalid.")
        count += 1
    return count


def build_prompts(*, requests: int, distinct: int, unique_fraction: float, zipf: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    popular = [f"question {i}: explain topic number {i} in two sentences" for i in range(distinct)]
    weights = [1 / (rank ** zipf) for rank in range(1, distinct + 1)]
    stream: list[str] = []
    for n in range(requests):
        if rng.random() < unique_fraction:
            stream.append(f"one-off question {n}: unique text {rng.random()}")
        else:
            stream.append(rng.choices(popular, weights=weights)[0])
    return stream


async def drive(base_url: str, prompts: list[str], workers: int) -> tuple[list[dict], float]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    for prompt in prompts:
        queue.put_nowait(prompt)
    results: list[dict] = []
    limits = httpx.Limits(max_connections=workers * 2, max_keepalive_connections=workers * 2)
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0, limits=limits) as client:
        async def worker() -> None:
            while True:
                try:
                    prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                started = perf_counter()
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "synthetic", "temperature": 0.0, "max_tokens": 64,
                          "messages": [{"role": "user", "content": prompt}]},
                )
                results.append({"latency_ms": (perf_counter() - started) * 1000, "http": response.status_code,
                                "status": response.headers.get("X-Cache", "DIRECT")})

        started_all = perf_counter()
        await asyncio.gather(*(worker() for _ in range(workers)))
        wall = perf_counter() - started_all
    return results, wall


def summarize(results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["http"] == 200]
    by_status: dict[str, list[float]] = {}
    for r in ok:
        by_status.setdefault(r["status"], []).append(r["latency_ms"])
    return {
        "requests": len(results), "http_errors": len(results) - len(ok),
        "throughput_rps": len(ok) / wall, "wall_seconds": wall,
        "status_counts": dict(Counter(r["status"] for r in ok)),
        "latency_ms": latency_summary([r["latency_ms"] for r in ok]),
        "latency_ms_by_status": {s: latency_summary(v) for s, v in by_status.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--distinct", type=int, default=200)
    parser.add_argument("--unique-fraction", type=float, default=0.05)
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--workers", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--provider-latency-ms", type=float, default=800.0)
    parser.add_argument("--provider-jitter", type=float, default=0.2)
    parser.add_argument("--embed-latency-ms", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threadpool-size", type=int, default=None,
                        help="AnyIO worker threads for the proxy's sync handlers (default: AnyIO's 40).")
    parser.add_argument("--output", type=Path, default=Path("evals/results/load-test-synthetic.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("evals/results/load-test-synthetic.md"))
    args = parser.parse_args()

    prompts = build_prompts(requests=args.requests, distinct=args.distinct, unique_fraction=args.unique_fraction,
                            zipf=args.zipf, seed=args.seed)
    colliding = colliding_pairs_rejected_by_guard(prompts, 0.95)
    print(f"{colliding} synthetic prompt pairs score >= 0.95 and are all rejected by the guard")
    runs = []
    for workers in args.workers:
        with serve(provider_server(args.provider_latency_ms, args.provider_jitter, args.seed)) as url:
            baseline, wall = asyncio.run(drive(url, prompts, workers))
        baseline_summary = summarize(baseline, wall)

        provider = SyntheticProvider(args.provider_latency_ms, args.provider_jitter, args.seed)
        service = CacheService(embedder=SyntheticEmbedder(args.embed_latency_ms), provider=provider,
                               similarity_threshold=0.95, metrics=CacheMetrics(similarity_threshold=0.95))
        with serve(with_threadpool(create_app(service), args.threadpool_size)) as url:
            cached, wall = asyncio.run(drive(url, prompts, workers))
        cached_summary = summarize(cached, wall)
        cached_summary["provider_calls"] = provider.calls
        cached_summary["provider_calls_avoided_pct"] = 100 * (1 - provider.calls / len(prompts))
        # Each distinct prompt must reach the provider once; anything above that is a duplicate
        # miss that ran concurrently with its twin (the proxy has no request coalescing).
        cached_summary["redundant_provider_calls"] = provider.calls - len(set(prompts))
        runs.append({"workers": workers, "baseline": baseline_summary, "cached": cached_summary})

    report = {
        "SYNTHETIC": "Provider latency and embedder are simulated with configured sleeps; not a measurement of any real API.",
        "config": {k: getattr(args, k) for k in ("requests", "distinct", "unique_fraction", "zipf", "provider_latency_ms",
                                                  "provider_jitter", "embed_latency_ms", "seed", "threadpool_size")}
        | {"unique_prompts": len(set(prompts)), "synthetic_colliding_pairs_rejected_by_guard": colliding},
        "server_note": "FastAPI sync endpoint => AnyIO threadpool (default 40 threads); blocking misses occupy a thread each.",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    table = render_markdown(report)
    args.markdown_output.write_text(table + "\n")
    print(table)


def render_markdown(report: dict) -> str:
    c = report["config"]
    lines = [f"SYNTHETIC provider: {c['provider_latency_ms']:.0f} ms ±{c['provider_jitter']:.0%}, embedder {c['embed_latency_ms']:.0f} ms; "
             f"{c['requests']} requests, {c['unique_prompts']} unique prompts (Zipf {c['zipf']}), {c['unique_fraction']:.0%} one-offs; "
             f"proxy worker threads: {c['threadpool_size'] or 40}.", "",
             "| Workers | Arm | Throughput req/s | P50 ms | P95 ms | P99 ms | Provider calls avoided | Redundant provider calls |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for arm in ("baseline", "cached"):
            s = run[arm]
            lat = s["latency_ms"]
            avoided = f"{s['provider_calls_avoided_pct']:.1f}%" if arm == "cached" else "0%"
            redundant = str(s["redundant_provider_calls"]) if arm == "cached" else "-"
            lines.append(f"| {run['workers']} | {arm} | {s['throughput_rps']:.0f} | {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | {lat['p99_ms']:.1f} | {avoided} | {redundant} |")
        by = run["cached"]["latency_ms_by_status"]
        detail = ", ".join(f"{k} P50 {v['p50_ms']:.1f} / P99 {v['p99_ms']:.1f} ms (n={run['cached']['status_counts'][k]})" for k, v in sorted(by.items()))
        lines.append(f"|  | cached by status | | {detail} | | | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
