import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_offline.py"


def _load():
    spec = importlib.util.spec_from_file_location("demo_offline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_demo_shows_miss_hit_isolation_bypass_and_invalidation() -> None:
    rows = _load().run()
    assert [status for _, status, _ in rows] == ["MISS", "HIT", "MISS", "MISS", "BYPASS", "-", "MISS"]
    assert [calls for _, _, calls in rows] == [1, 1, 2, 3, 4, 4, 5]  # the HIT never reached the provider
    assert "removed 3 entries" in rows[5][0]
