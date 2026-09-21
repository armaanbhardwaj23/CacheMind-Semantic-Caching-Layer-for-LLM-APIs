"""Shared wiring for the paid experiment scripts.

Layering (outermost first)::

    DiskCachedEmbedder  ->  BudgetedEmbedder  ->  OpenRouterClient

The disk cache is *outside* the budget wrapper, so a cache hit is never charged and
a re-run of the same experiment costs $0. ``prefetch`` warms the cache with a small
thread pool so a few thousand short embeddings finish in minutes, not an hour.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import socket
import threading
import time

from semantic_cache.embedding_cache import DiskCachedEmbedder
from semantic_cache.openrouter import OpenRouterClient, OpenRouterError
from semantic_cache.spend import BudgetedEmbedder, SpendLedger

DEFAULT_EMBEDDING_PRICE_PER_MILLION_USD = 0.02  # openai/text-embedding-3-small on OpenRouter, 2026-09-19


def build_embedder(
    client: OpenRouterClient,
    *,
    model: str,
    cache_path: Path,
    ledger: SpendLedger,
    price_per_million_usd: float = DEFAULT_EMBEDDING_PRICE_PER_MILLION_USD,
) -> DiskCachedEmbedder:
    budgeted = BudgetedEmbedder(client, ledger, price_per_million_tokens_usd=price_per_million_usd)
    return DiskCachedEmbedder(budgeted, model=model, path=cache_path)


def prefetch(
    embedder: DiskCachedEmbedder, texts: Iterable[str], *, workers: int = 8, retries: int = 3
) -> None:
    """Embed every distinct text once, concurrently. Failures are retried with backoff.

    Budget errors are *not* retried: ``BudgetExceeded`` propagates and stops the run.
    """
    unique = list(dict.fromkeys(texts))

    def embed_with_retry(text: str) -> None:
        for attempt in range(retries):
            try:
                embedder.embed(text)
                return
            except OpenRouterError:
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(embed_with_retry, text) for text in unique]:
            future.result()


@contextmanager
def serve(app: object, *, host: str = "127.0.0.1") -> Iterator[str]:
    """Run an ASGI app on a real localhost socket in a background thread; yield its base URL.

    Benchmarks go over real HTTP (not an in-process test client) so the measured latency
    includes the server, JSON encoding and the network stack, as a real client would see.
    """
    import uvicorn  # imported lazily: only the experiment scripts need a server

    with socket.socket() as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("Benchmark server did not start.")
        time.sleep(0.02)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
