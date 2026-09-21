"""Zero-cost demo of the cache over HTTP: no network, no API key, no spend.

The real FastAPI app runs in-process with a fake provider and a deterministic embedder (hash vectors, so only
*identical* text can match; this demonstrates identity, bypass and invalidation, not semantic matching).

    uv run python scripts/demo_offline.py
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from semantic_cache.api.app import create_app
from semantic_cache.service import CacheService
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider

TOKEN = "demo-token"
BASE = {
    "model": "demo-model",
    "temperature": 0,
    "max_tokens": 100,
    "messages": [{"role": "system", "content": "Answer briefly."},
                 {"role": "user", "content": "What is the capital of France?"}],
}


def run() -> list[tuple[str, str, int]]:
    """Return (step, X-Cache header, provider calls so far) for each request."""
    provider = RecordingProvider(response_text="Paris.")
    service = CacheService(embedder=DeterministicEmbedder(), provider=provider, similarity_threshold=0.95)
    client = TestClient(create_app(service, admin_token=TOKEN))
    rows: list[tuple[str, str, int]] = []

    def ask(step: str, body: dict, headers: dict | None = None) -> None:
        response = client.post("/v1/chat/completions", json=body, headers=headers or {})
        rows.append((step, response.headers["X-Cache"], provider.call_count))

    ask("first request", BASE)
    ask("identical request", BASE)
    ask("temperature 0.7 (different identity)", BASE | {"temperature": 0.7})
    ask("different system prompt", BASE | {"messages": [{"role": "system", "content": "Answer verbosely."}, BASE["messages"][1]]})
    ask("Cache-Control: no-store", BASE, {"Cache-Control": "no-store"})
    removed = client.post("/v1/cache/invalidate", json={"model": "demo-model"}, headers={"X-Admin-Token": TOKEN}).json()["removed"]
    rows.append((f"admin invalidation by model (removed {removed} entries)", "-", provider.call_count))
    ask("identical request after invalidation", BASE)
    return rows


def main() -> None:
    print(f"{'step':<52} {'X-Cache':<8} provider calls so far")
    for step, status, calls in run():
        print(f"{step:<52} {status:<8} {calls}")


if __name__ == "__main__":
    main()
