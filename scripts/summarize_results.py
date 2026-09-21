"""Aggregate the saved experiment reports in evals/results/ into one Markdown file.

Every number in the README's results tables is copied from evals/results/summary.md, which
this script derives *only* from JSON reports written by the experiment scripts (plus the
spend ledger, whose per-label totals are copied to evals/results/cost-to-build.json).
No API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

RESULTS = Path("evals/results")
DATASETS = [("curated", "Curated safety set (118 pairs, AI-drafted labels, cross-checked by a second model, not human-verified)"),
            ("paws", "PAWS-Wiki proxy (300 pairs: adversarial word-swaps)"),
            ("qqp", "Quora QQP proxy (300 pairs: natural questions)")]
PAIR_THRESHOLDS = (0.90, 0.92, 0.95, 0.98)
WORKLOAD_THRESHOLDS = (0.80, 0.90, 0.92, 0.95, 0.98)


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def pct2(x: float | None) -> str:
    """Two decimals for rates below 1%, where one decimal would round 0.15% down to 0.1%."""
    return "-" if x is None else f"{100 * x:.2f}%"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def pair_section() -> list[str]:
    lines = ["## Pair-level threshold experiment", "",
             "Given one cached prompt and one incoming prompt, is reuse acceptable? Positives are reusable pairs. "
             "**False-hit rate = false hits / hits**; it depends on the negative/positive mix of the dataset, so it is "
             "not comparable across datasets and is not a production rate.", ""]
    for key, title in DATASETS:
        none, guard = load(f"threshold-{key}-none.json"), load(f"threshold-{key}-guard.json")
        scores = none["scored_cases"]
        pos = [c["similarity_score"] for c in scores if c["reuse_is_acceptable"]]
        neg = [c["similarity_score"] for c in scores if not c["reuse_is_acceptable"]]
        lines += [f"### {title}", "",
                  f"Cosine similarity, reusable pairs: median {median(pos):.3f}, min {min(pos):.3f}; "
                  f"non-reusable pairs: median {median(neg):.3f}, max {max(neg):.3f}. "
                  f"Threshold chosen under a 5% false-hit budget: guard off = {none['selected_threshold']}, guard on = {guard['selected_threshold']}.", "",
                  "| Threshold | Guard | Hit rate | False-hit rate | Valid-reuse recall |", "|---:|---|---:|---:|---:|"]
        for threshold in PAIR_THRESHOLDS:
            for name, report in (("off", none), ("on", guard)):
                row = next(r for r in report["threshold_results"] if abs(r["threshold"] - threshold) < 1e-9)
                lines.append(f"| {threshold:.2f} | {name} | {pct(row['hit_rate'])} | {pct(row['false_hit_rate'])} | {pct(row['valid_reuse_recall'])} |")
        lines.append("")
    return lines


def workload_section() -> list[str]:
    reports = [load(p.name) for p in sorted(RESULTS.glob("workload-qqp-seed*.json"))]
    seeds = [r["seed"] for r in reports]
    first = reports[0]
    lines = ["## Workload-level simulation (QQP intents, real embeddings, oracle provider)", "",
             f"{first['requests']} requests per seed, Zipf {first['zipf_exponent']}, {first['intents']} intents, seeds {seeds}; "
             "the intent set is fixed (built once with seed 42), so seeds vary only the request stream. "
             "Provider is a ground-truth oracle, so latency is not reported here. Values are means over seeds "
             "(range in brackets).", ""]

    def get(report: dict, guard: bool, threshold: float) -> dict:
        return next(r for r in report["results"] if r["guard"] == guard and abs(r["threshold"] - threshold) < 1e-9)

    lines += ["| Guard | Threshold | Hit rate | Uplift over exact-only | False hits / hits | Wrongly served (of all requests) |",
              "|---|---:|---:|---:|---:|---:|"]
    for guard in (False, True):
        for threshold in WORKLOAD_THRESHOLDS:
            rows = [get(r, guard, threshold) for r in reports]
            exact_only = [get(r, guard, 1.0)["hit_rate"] for r in reports]
            uplift = [a["hit_rate"] - b for a, b in zip(rows, exact_only)]
            fhr = [r["false_hit_rate"] or 0.0 for r in rows]
            wrong = [r["wrongly_served_rate"] for r in rows]
            lines.append(
                f"| {'on' if guard else 'off'} | {threshold:.2f} | {pct(mean(r['hit_rate'] for r in rows))} | "
                f"{100 * mean(uplift):+.2f} pp | {pct2(mean(fhr))} [{pct2(min(fhr))}-{pct2(max(fhr))}] | "
                f"{pct2(mean(wrong))} [{pct2(min(wrong))}-{pct2(max(wrong))}] |")
    exact = [get(r, False, 1.0)["hit_rate"] for r in reports]
    lines += ["", f"Exact-only baseline (threshold 1.00): hit rate {pct(mean(exact))}. "
              "This workload is dominated by identical repeats, so semantic matching adds little on top of exact matching.", ""]
    return lines


def e2e_section() -> list[str]:
    report = load("e2e-real-benchmark.json")
    lines = ["## Real end-to-end benchmark (OpenRouter, gpt-4.1-mini, localhost proxy)", "",
             f"Status: **{report['status']}** - {report['completed_requests']} of {report['planned_requests']} planned requests completed"
             + ("." if report["status"] == "complete" else "; the run stopped early, so this covers the completed prefix only.")
             + f" Threshold {report['threshold']}, guard on, seed {report['seed']}.", "",
             "| Path | n | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|"]
    d = report["latency_direct_ms"]
    lines.append(f"| direct provider | {report['provider_calls_direct']} | {d['p50_ms']:.1f} | {d['p95_ms']:.1f} | {d['p99_ms']:.1f} |")
    for status, s in sorted(report["latency_proxy_ms_by_status"].items()):
        lines.append(f"| proxy {status} | {report['status_counts'][status]} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | {s['p99_ms']:.1f} |")
    cost = report["estimated_cost_usd"]
    paired = report["paired_proxy_minus_direct_ms_median"]
    lines += ["",
              f"- Composition: {report['composition']}.",
              f"- Provider calls: {report['provider_calls_direct']} direct vs {report['provider_calls_via_proxy']} via proxy.",
              f"- HIT validity (intent oracle): {report['hits_valid']} valid, {report['hits_false']} false.",
              f"- Estimated cost (configured prices, embedding cost included): direct ${cost['direct_total']:.5f} vs "
              f"proxy ${cost['proxy_total']:.5f} = **{report['estimated_net_savings_pct']:.1f}% net saving**.",
              f"- Median paired overhead of a MISS vs the same request sent directly: {paired['MISS']:+.0f} ms "
              f"(embedding + lookup + store; noisy because provider latency varies per call).", ""]
    return lines


def load_test_section() -> list[str]:
    report = load("load-test-synthetic.json")
    c = report["config"]
    lines = ["## Concurrency load test (SYNTHETIC provider and embedder)", "",
             f"Provider {c['provider_latency_ms']:.0f} ms ±{c['provider_jitter']:.0%} and embedder {c['embed_latency_ms']:.0f} ms are configured sleeps, "
             f"not measurements. {c['requests']} requests, {c['unique_prompts']} unique prompts (Zipf {c['zipf']}), {c['unique_fraction']:.0%} one-offs. "
             "\"Redundant\" provider calls are calls beyond one per unique prompt: duplicate misses that ran concurrently.", "",
             "| Proxy threads | Workers | Arm | Throughput req/s | P50 ms | P95 ms | P99 ms | Provider calls avoided | Redundant provider calls |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    reports = [report]
    tp = RESULTS / "load-test-synthetic-threadpool200.json"
    if tp.exists():
        reports.append(json.loads(tp.read_text()))
    for rep in reports:
        threads = rep["config"]["threadpool_size"] or 40
        for run in rep["runs"]:
            for arm in ("baseline", "cached"):
                st = run[arm]
                lat = st["latency_ms"]
                avoided = f"{st['provider_calls_avoided_pct']:.1f}%" if arm == "cached" else "0%"
                redundant = str(st["redundant_provider_calls"]) if arm == "cached" else "-"
                lines.append(f"| {threads if arm == 'cached' else '-'} | {run['workers']} | {arm} | {st['throughput_rps']:.0f} | {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | {lat['p99_ms']:.1f} | {avoided} | {redundant} |")
    for rep in reports:
        threads = rep["config"]["threadpool_size"] or 40
        for run in rep["runs"]:
            by = run["cached"]["latency_ms_by_status"]
            lines += ["", f"{threads} proxy threads, {run['workers']} workers, cached arm by status: " + ", ".join(
                f"{k} P50 {v['p50_ms']:.1f} / P99 {v['p99_ms']:.1f} ms (n={run['cached']['status_counts'][k]})" for k, v in sorted(by.items()))]
    lines.append("")
    return lines


NON_BILLED_LABELS = {"diagnostics"}  # flat placeholders for calls the provider rejected (never billed)


def cost_section() -> list[str]:
    ledger_path = Path(".cache/spend.json")
    cost_path = RESULTS / "cost-to-build.json"
    if ledger_path.exists():  # refresh the committed copy whenever the local ledger exists
        raw = json.loads(ledger_path.read_text())
        by_label: dict[str, float] = {}
        for entry in raw["entries"]:
            by_label[entry["label"]] = by_label.get(entry["label"], 0.0) + entry["usd"]
        billed = sum(v for k, v in by_label.items() if k not in NON_BILLED_LABELS)
        cost_path.write_text(json.dumps({
            "cap_usd": raw["cap_usd"], "total_usd": billed, "total_usd_including_placeholder": sum(by_label.values()),
            "by_label_usd": by_label,
            "note": "Estimates at configured prices, not an invoice. 'diagnostics' is a flat placeholder for six probe calls the provider "
                    "rejected (HTTP 402); it is excluded from total_usd. See cost-crosscheck.json for a comparison with the provider's own counter."},
            indent=2) + "\n")
    cost = json.loads(cost_path.read_text())
    lines = ["## Cost to build", "",
             f"Spend-ledger total: **${cost['total_usd']:.4f}** against a ${cost['cap_usd']:.2f} cap for this repository "
             f"(per label: {', '.join(f'{k} ${v:.5f}' for k, v in cost['by_label_usd'].items() if k not in NON_BILLED_LABELS)}). "
             "Estimates, not an invoice."]
    cross = RESULTS / "cost-crosscheck.json"
    if cross.exists():
        x = json.loads(cross.read_text())
        lines += ["", f"Cross-check against OpenRouter's own usage counter (manual readings, shared key): about "
                  f"${x['provider_counter_total_for_this_repo_usd']:.4f} vs ledger ${x['ledger_total_excluding_placeholder_usd']:.4f}."]
    return lines + [""]


def main() -> None:
    lines = ["# Results summary", ""]
    lines += pair_section() + workload_section() + e2e_section() + load_test_section() + cost_section()
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
