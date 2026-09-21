"""Workload file parsing and latency summaries for cache experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class WorkloadItem:
    item_id: str
    model: str
    messages: tuple[dict[str, str], ...]
    temperature: float
    max_tokens: int
    cache_control: str | None = None


def load_workload(path: Path) -> list[WorkloadItem]:
    raw_items = json.loads(path.read_text())
    if not isinstance(raw_items, list):
        raise ValueError("Workload must be a JSON list.")
    items: list[WorkloadItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Each workload item must be an object.")
        try:
            messages = raw["messages"]
            item = WorkloadItem(
                item_id=raw["item_id"],
                model=raw["model"],
                messages=tuple(messages),
                temperature=raw.get("temperature", 0.0),
                max_tokens=raw["max_tokens"],
                cache_control=raw.get("cache_control"),
            )
        except KeyError as error:
            raise ValueError(f"Workload item missing {error.args[0]!r}.") from error
        if (
            not isinstance(item.item_id, str)
            or not isinstance(item.model, str)
            or not isinstance(item.temperature, int | float)
            or not isinstance(item.max_tokens, int)
            or not item.messages
        ):
            raise ValueError(f"Workload item {item.item_id!r} has invalid fields.")
        if not all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            for message in item.messages
        ):
            raise ValueError(f"Workload item {item.item_id!r} has invalid messages.")
        items.append(item)
    return items


def latency_summary(values_ms: list[float]) -> dict[str, float | None]:
    if not values_ms:
        return {"average_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    ordered = sorted(values_ms)
    return {
        "average_ms": sum(values_ms) / len(values_ms),
        "p50_ms": ordered[ceil(0.50 * len(ordered)) - 1],
        "p95_ms": ordered[ceil(0.95 * len(ordered)) - 1],
        "p99_ms": ordered[ceil(0.99 * len(ordered)) - 1],
    }


def latency_by_status(request_results: list[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[float]] = {}
    for result in request_results:
        status = result["cache_status"]
        latency_ms = result["latency_ms"]
        if isinstance(status, str) and isinstance(latency_ms, float):
            grouped.setdefault(status, []).append(latency_ms)
    return {status: latency_summary(values) for status, values in grouped.items()}
