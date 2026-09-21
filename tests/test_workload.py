import json
from pathlib import Path

from semantic_cache.workload import latency_by_status, latency_summary, load_workload


def test_latency_summary_uses_nearest_rank_percentiles() -> None:
    assert latency_summary([10.0, 20.0, 30.0, 40.0]) == {
        "average_ms": 25.0,
        "p50_ms": 20.0,
        "p95_ms": 40.0,
        "p99_ms": 40.0,
    }


def test_load_workload_parses_valid_requests(tmp_path: Path) -> None:
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            [
                {
                    "item_id": "one",
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 20,
                }
            ]
        )
    )

    workload = load_workload(path)

    assert workload[0].item_id == "one"
    assert workload[0].temperature == 0.0


def test_latency_by_status_keeps_hits_and_misses_separate() -> None:
    result = latency_by_status(
        [
            {"cache_status": "HIT", "latency_ms": 10.0},
            {"cache_status": "MISS", "latency_ms": 100.0},
            {"cache_status": "MISS", "latency_ms": 200.0},
        ]
    )

    assert result["HIT"]["p95_ms"] == 10.0
    assert result["MISS"]["p50_ms"] == 100.0
