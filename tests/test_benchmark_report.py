import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("benchmark_real", ROOT / "scripts" / "benchmark_real.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index: int, status: str, proxy_ms: float, direct_ms: float, kind: str = "first_seen") -> dict:
    return {"index": index, "kind": kind, "intent_id": "i", "status": status, "proxy_ms": proxy_ms, "direct_ms": direct_ms,
            "direct_usd": 0.001, "proxy_usd": 0.0005, "similarity": None, "valid": True}


def test_paired_overhead_uses_the_conventional_median_for_an_even_number_of_misses() -> None:
    rows = [_row(0, "MISS", 1100, 1000), _row(1, "MISS", 1300, 1000), _row(2, "HIT", 2, 1000, "exact_repeat"), _row(3, "MISS", 1200, 1000)]
    module = _load()
    assert module._median([100.0, 300.0]) == 200.0  # the upper-middle element (300) was the old, wrong answer
    report = module.build_report(rows, planned_requests=4, threshold=0.95, seed=1, stopped_early=None, proxy_metrics=None, ledger_total=None)
    assert report["paired_proxy_minus_direct_ms_median"]["MISS"] == 200.0
    assert "4 requests" in report["caveats"][0]


def test_rebuilding_the_saved_real_report_keeps_the_headline_numbers() -> None:
    saved_path = ROOT / "evals" / "results" / "e2e-real-benchmark.json"
    if not saved_path.exists():
        pytest.skip("saved report not present")
    saved = json.loads(saved_path.read_text())
    rebuilt = _load().rebuild_report(saved_path)
    for key in ("latency_direct_ms", "latency_proxy_ms_by_status", "status_counts", "hits_valid", "hits_false",
                "estimated_cost_usd", "estimated_net_savings_pct", "completed_requests"):
        assert rebuilt[key] == saved[key], key
