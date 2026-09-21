"""Real end-to-end benchmark: direct provider calls vs. the same requests through CacheMind.

*Paid.* Every chat and embedding call goes through the spend ledger (hard cap: --max-usd).

Method
  * The workload is built from QQP intents (evals/external/qqp_intents.json): a Zipf-popular
    stream of exact repeats, paraphrases and first-time questions, plus a few unique
    ``Cache-Control: no-store`` requests. Composition is reported, not hidden.
  * For each request we time (a) a direct provider call and (b) the same request through the
    proxy over real HTTP on localhost. The order alternates per request so drift and any
    provider-side warm-up affect both arms equally.
  * The proxy uses the *un-cached* embedder: embedding latency and cost are real, not replayed.
  * HIT validity uses the intent oracle: a HIT is valid iff the served text was originally
    produced for a request of the same intent. No text is stored in the report.
  * Costs are estimates from the configured prices below, computed from the usage the
    provider returned; they are not an invoice.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
from time import perf_counter

import httpx

from semantic_cache.api.app import create_app
from semantic_cache.config import OpenRouterSettings
from semantic_cache.experiment import serve
from semantic_cache.metrics import CacheMetrics, EmbeddingPricing, TokenPricing
from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.openrouter import OpenRouterClient, OpenRouterError
from semantic_cache.service import CacheService
from semantic_cache.simulation import Intent, build_request_stream, load_intents
from semantic_cache.spend import BudgetedEmbedder, BudgetedProvider, SpendLedger
from semantic_cache.workload import latency_summary

SYSTEM_PROMPT = "Answer in at most two sentences."
MODEL = "openai/gpt-4.1-mini"
PRICING = TokenPricing(input_per_million_tokens_usd=0.40, output_per_million_tokens_usd=1.60)
EMBEDDING_PRICING = EmbeddingPricing(per_million_tokens_usd=0.02)


def build_workload(intents: list[Intent], *, cacheable: int, no_store: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = [i for i in intents if len(i.queries) > 1]
    singles = [i for i in intents if len(i.queries) == 1]
    rng.shuffle(groups)
    rng.shuffle(singles)
    n_groups = min(len(groups), max(1, cacheable // 5))
    n_singles = min(len(singles) - no_store, n_groups)
    pool = groups[:n_groups] + singles[:n_singles]
    stream = build_request_stream(pool, n_requests=cacheable, seed=seed, zipf_s=1.1)

    seen_texts: set[str] = set()
    seen_intents: set[str] = set()
    requests: list[dict] = []
    for intent_id, text in stream:
        kind = "exact_repeat" if text in seen_texts else ("paraphrase" if intent_id in seen_intents else "first_seen")
        seen_texts.add(text)
        seen_intents.add(intent_id)
        requests.append({"intent_id": intent_id, "text": text, "kind": kind, "no_store": False})
    for intent in singles[n_singles : n_singles + no_store]:
        position = rng.randrange(len(requests) + 1)
        requests.insert(position, {"intent_id": intent.intent_id, "text": intent.queries[0], "kind": "no_store", "no_store": True})
    return requests


def chat_request(text: str) -> ChatRequest:
    return ChatRequest(
        provider="openrouter", model=MODEL, temperature=0.0, max_tokens=150,
        messages=[ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", text)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intents", type=Path, default=Path("evals/external/qqp_intents.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/e2e-real-benchmark.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("evals/results/e2e-real-benchmark.md"))
    parser.add_argument("--cacheable-requests", type=int, default=110)
    parser.add_argument("--no-store-requests", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spend-ledger", type=Path, default=Path(".cache/spend.json"))
    parser.add_argument("--max-usd", type=float, default=0.20)
    parser.add_argument("--recover-from", type=Path,
                        help="Offline: rebuild a clean report for the completed prefix of an interrupted raw run. Makes no API calls.")
    parser.add_argument("--rebuild-from", type=Path,
                        help="Offline: recompute every derived figure of a saved complete report from its per-request rows. Makes no API calls.")
    args = parser.parse_args()

    if args.rebuild_from:
        write_outputs(rebuild_report(args.rebuild_from), args)
        return

    if args.recover_from:
        report = recover_report(args.recover_from, args.spend_ledger)
        write_outputs(report, args)
        return

    settings = OpenRouterSettings.from_env()
    ledger = SpendLedger(args.spend_ledger, cap_usd=args.max_usd)
    workload = build_workload(
        load_intents(args.intents), cacheable=args.cacheable_requests, no_store=args.no_store_requests, seed=args.seed
    )

    client = OpenRouterClient(settings)
    direct_provider = BudgetedProvider(client, ledger, PRICING, label="e2e-chat-direct")
    proxy_provider = BudgetedProvider(client, ledger, PRICING, label="e2e-chat-proxy")
    embedder = BudgetedEmbedder(client, ledger, price_per_million_tokens_usd=0.02, label="e2e-embeddings-proxy")
    service = CacheService(
        embedder=embedder, provider=proxy_provider, similarity_threshold=args.threshold,
        metrics=CacheMetrics(similarity_threshold=args.threshold, token_pricing=PRICING, embedding_pricing=EMBEDDING_PRICING),
    )

    rows: list[dict] = []
    stopped_early: dict | None = None
    intent_by_content: dict[str, set[str]] = {}
    with serve(create_app(service)) as base_url, httpx.Client(base_url=base_url, timeout=60.0) as http:
        http.get("/health")  # open the connection before timing anything
        for index, item in enumerate(workload):
            def call_direct() -> tuple[float | None, float]:
                usd_before = ledger.total_usd
                started = perf_counter()
                try:
                    direct_provider.complete(chat_request(item["text"]))
                except OpenRouterError:
                    return None, ledger.total_usd - usd_before
                return (perf_counter() - started) * 1000, ledger.total_usd - usd_before

            def call_proxy() -> tuple[float, httpx.Response, float]:
                usd_before = ledger.total_usd
                started = perf_counter()
                response = http.post(
                    "/v1/chat/completions",
                    json={"model": MODEL, "temperature": 0.0, "max_tokens": 150,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item["text"]}]},
                    headers={"Cache-Control": "no-store"} if item["no_store"] else {},
                )
                return (perf_counter() - started) * 1000, response, ledger.total_usd - usd_before

            if index % 2 == 0:
                direct_ms, direct_usd = call_direct()
                proxy_ms, response, proxy_usd = call_proxy()
            else:
                proxy_ms, response, proxy_usd = call_proxy()
                direct_ms, direct_usd = call_direct()
            # Any failure ends the run. A silently skipped request in one arm would make the
            # arms cover different requests and bias the cost comparison.
            if direct_ms is None or response.status_code != 200:
                stopped_early = {"at_request_index": index, "direct_arm_failed": direct_ms is None,
                                 "proxy_http_status": response.status_code}
                break
            status = response.headers["X-Cache"]
            content = response.json()["choices"][0]["message"]["content"]
            row = {
                "index": index, "kind": item["kind"], "intent_id": item["intent_id"], "status": status,
                "proxy_ms": proxy_ms, "direct_ms": direct_ms, "direct_usd": direct_usd, "proxy_usd": proxy_usd,
                "similarity": float(response.headers["X-Cache-Similarity"]) if "X-Cache-Similarity" in response.headers else None,
            }
            if status == "HIT":
                row["valid"] = item["intent_id"] in intent_by_content.get(content, set())
            else:
                intent_by_content.setdefault(content, set()).add(item["intent_id"])
            rows.append(row)
        proxy_metrics = http.get("/metrics/summary").json()
    client.close()

    report = build_report(
        rows, planned_requests=len(workload), threshold=args.threshold, seed=args.seed,
        stopped_early=stopped_early, proxy_metrics=proxy_metrics, ledger_total=ledger.total_usd,
    )
    write_outputs(report, args)
    print(f"ledger ${ledger.total_usd:.5f} of ${args.max_usd:.2f}")


def write_outputs(report: dict, args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(report) + "\n")
    print(render_markdown(report))
    print(f"\nWrote {args.output}")


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_report(
    rows: list[dict], *, planned_requests: int, threshold: float, seed: int, stopped_early: dict | None,
    proxy_metrics: dict | None, ledger_total: float | None, extra_proxy_usd: float = 0.0, recovered_from: str | None = None,
) -> dict:
    """Summarise *completed* requests only (both arms succeeded)."""
    by_status: dict[str, list[float]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r["proxy_ms"])
    hits = [r for r in rows if r["status"] == "HIT"]
    eligible = [r for r in rows if r["status"] != "BYPASS"]
    direct_total = sum(r["direct_usd"] for r in rows)
    proxy_total = sum(r["proxy_usd"] for r in rows) + extra_proxy_usd
    paired = {s: [r["proxy_ms"] - r["direct_ms"] for r in rows if r["status"] == s] for s in ("MISS", "BYPASS")}
    return {
        "what_this_is": "Real provider + real embeddings, localhost HTTP, in-memory store. Costs are estimates from configured prices.",
        "status": "interrupted" if stopped_early else "complete",
        "stopped_early": stopped_early,
        "recovered_from": recovered_from,
        "model": MODEL, "threshold": threshold, "seed": seed,
        "constraint_guard": "on (CacheService default)",
        "pricing_usd_per_million_tokens": {"input": PRICING.input_per_million_tokens_usd, "output": PRICING.output_per_million_tokens_usd,
                                           "embedding": EMBEDDING_PRICING.per_million_tokens_usd},
        "planned_requests": planned_requests, "completed_requests": len(rows),
        "composition": dict(Counter(r["kind"] for r in rows)),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "hit_rate_of_cacheable": len(hits) / len(eligible) if eligible else None,
        "provider_calls_direct": len(rows),
        "provider_calls_via_proxy": sum(r["status"] != "HIT" for r in rows),
        "hits_valid": sum(r["valid"] for r in hits), "hits_false": sum(not r["valid"] for r in hits),
        "latency_direct_ms": latency_summary([r["direct_ms"] for r in rows]),
        "latency_proxy_ms_all": latency_summary([r["proxy_ms"] for r in rows]),
        "latency_proxy_ms_by_status": {s: latency_summary(v) for s, v in by_status.items()},
        "paired_proxy_minus_direct_ms_median": {s: _median(v) for s, v in paired.items()},
        "estimated_cost_usd": {"direct_total": round(direct_total, 6), "proxy_total": round(proxy_total, 6)},
        "estimated_net_savings_pct": 100 * (1 - proxy_total / direct_total) if direct_total else None,
        "proxy_metrics_summary": proxy_metrics,
        "ledger_total_usd_after_run": ledger_total,
        "caveats": [
            f"Single run, one seed, {len(rows)} requests: percentiles at P95 and above rest on very few samples.",
            "Provider latency is shared internet latency and varies run to run; compare the arms within this run only.",
            "Proxy and direct arms run in one process on one machine; the in-memory store has no network hop.",
            "HIT validity is judged by the QQP-derived intent oracle, a proxy label, not a human review.",
            f"Only {sum(r['kind'] == 'paraphrase' for r in rows)} paraphrase requests occur, so this run mostly measures exact-repeat hits, not semantic uplift.",
        ],
        "requests_detail": rows,
    }


def rebuild_report(saved_path: Path) -> dict:
    """Recompute all derived figures of a saved report from its per-request rows (no API calls).

    Everything that cannot be derived from the rows (proxy metrics snapshot, ledger total, the embedding
    cost added on top of the per-request proxy cost) is carried over unchanged.
    """
    saved = json.loads(saved_path.read_text())
    rows = saved["requests_detail"]
    extra = saved["estimated_cost_usd"]["proxy_total"] - sum(r["proxy_usd"] for r in rows)
    report = build_report(
        rows, planned_requests=saved["planned_requests"], threshold=saved["threshold"], seed=saved["seed"],
        stopped_early=saved["stopped_early"], proxy_metrics=saved["proxy_metrics_summary"],
        ledger_total=saved["ledger_total_usd_after_run"], extra_proxy_usd=max(extra, 0.0),
        recovered_from=saved["recovered_from"],
    )
    # The saved costs were rounded to 6 decimals; re-deriving the embedding add-on from them would shift the saving slightly.
    report["estimated_cost_usd"] = saved["estimated_cost_usd"]
    report["estimated_net_savings_pct"] = saved["estimated_net_savings_pct"]
    return report


def recover_report(raw_path: Path, ledger_path: Path) -> dict:
    """Rebuild a clean report for the completed prefix of an interrupted run, from saved artefacts.

    The interrupted run predates per-request cost tracking, so costs are re-attributed from the
    ledger, whose entries are chronological per label: the last D ``e2e-chat-direct`` entries are
    the run's D successful direct calls (all inside the prefix); the last P ``e2e-chat-proxy``
    entries are its P successful proxy provider calls, one per MISS/BYPASS in request order.
    Embedding cost (~$0.00001 for the whole run) is added in full as an upper bound.
    """
    raw = json.loads(raw_path.read_text())
    detail = sorted(raw["requests_detail"], key=lambda r: r["index"])
    failed = [r["index"] for r in detail if r["status"] == "ERROR" or r.get("direct_ms") is None]
    first_failure = min(failed) if failed else None
    prefix = [r for r in detail if first_failure is None or r["index"] < first_failure]
    entries = json.loads(ledger_path.read_text())["entries"]

    def usd(label: str) -> list[float]:
        return [float(e["usd"]) for e in entries if e["label"] == label]

    completed = [r for r in detail if r["status"] != "ERROR"]
    direct_successes = sum(r.get("direct_ms") is not None for r in completed)
    proxy_successes = sum(r["status"] in ("MISS", "BYPASS") for r in completed)
    direct_usd = usd("e2e-chat-direct")[-direct_successes:]
    proxy_usd = usd("e2e-chat-proxy")[-proxy_successes:]
    if len(direct_usd) != direct_successes or len(proxy_usd) != proxy_successes:
        raise SystemExit("Ledger does not contain enough entries to attribute this run's costs.")
    if any(r.get("direct_ms") is None for r in prefix):
        raise SystemExit("Unexpected direct failure inside the prefix.")
    if direct_successes != len(prefix):
        raise SystemExit("Direct successes extend past the prefix; the attribution assumption fails.")

    proxy_iter = iter(proxy_usd)
    rows = []
    for position, r in enumerate(prefix):
        chat = next(proxy_iter) if r["status"] in ("MISS", "BYPASS") else 0.0
        rows.append({**r, "direct_usd": direct_usd[position], "proxy_usd": chat})
    embeddings_upper_bound = sum(usd("e2e-embeddings-proxy"))  # includes the 12-request smoke run
    return build_report(
        rows, planned_requests=int(raw["requests"]), threshold=float(raw["threshold"]), seed=int(raw["seed"]),
        stopped_early={"at_request_index": first_failure, "reason": "OpenRouter returned HTTP 402 (insufficient credits) from here on"},
        proxy_metrics=None, ledger_total=None, extra_proxy_usd=embeddings_upper_bound,
        recovered_from=str(raw_path),
    )


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def render_markdown(report: dict) -> str:
    rows = [f"Status: **{report['status']}** ({report['completed_requests']} of {report['planned_requests']} planned requests completed)", "",
            "| Path | n | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|"]
    direct = report["latency_direct_ms"]
    rows.append(f"| direct provider | {report['provider_calls_direct']} | {_ms(direct['p50_ms'])} | {_ms(direct['p95_ms'])} | {_ms(direct['p99_ms'])} |")
    for status, summary in sorted(report["latency_proxy_ms_by_status"].items()):
        rows.append(f"| proxy {status} | {report['status_counts'][status]} | {_ms(summary['p50_ms'])} | {_ms(summary['p95_ms'])} | {_ms(summary['p99_ms'])} |")
    cost = report["estimated_cost_usd"]
    savings = report["estimated_net_savings_pct"]
    rows += [
        "",
        f"- composition: {report['composition']}; statuses: {report['status_counts']}",
        f"- provider calls: {report['provider_calls_direct']} direct vs {report['provider_calls_via_proxy']} via proxy",
        f"- hit validity (intent oracle): {report['hits_valid']} valid, {report['hits_false']} false",
        f"- estimated cost: direct ${cost['direct_total']:.5f} vs proxy ${cost['proxy_total']:.5f} (chat + embeddings)"
        + (f"; net savings {savings:.1f}%" if savings is not None else ""),
        f"- threshold {report['threshold']}, seed {report['seed']}, model {report['model']}",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
