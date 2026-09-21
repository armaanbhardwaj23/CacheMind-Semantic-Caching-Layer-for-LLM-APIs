# Results summary

## Pair-level threshold experiment

Given one cached prompt and one incoming prompt, is reuse acceptable? Positives are reusable pairs. **False-hit rate = false hits / hits**; it depends on the negative/positive mix of the dataset, so it is not comparable across datasets and is not a production rate.

### Curated safety set (118 pairs, AI-drafted labels, cross-checked by a second model, not human-verified)

Cosine similarity, reusable pairs: median 0.939, min 0.781; non-reusable pairs: median 0.852, max 1.000. Threshold chosen under a 5% false-hit budget: guard off = None, guard on = None.

| Threshold | Guard | Hit rate | False-hit rate | Valid-reuse recall |
|---:|---|---:|---:|---:|
| 0.90 | off | 39.0% | 63.0% | 63.0% |
| 0.90 | on | 26.3% | 48.4% | 59.3% |
| 0.92 | off | 33.1% | 56.4% | 63.0% |
| 0.92 | on | 23.7% | 42.9% | 59.3% |
| 0.95 | off | 21.2% | 52.0% | 44.4% |
| 0.95 | on | 17.8% | 42.9% | 44.4% |
| 0.98 | off | 15.3% | 61.1% | 25.9% |
| 0.98 | on | 13.6% | 56.2% | 25.9% |

### PAWS-Wiki proxy (300 pairs: adversarial word-swaps)

Cosine similarity, reusable pairs: median 0.980, min 0.753; non-reusable pairs: median 0.956, max 0.999. Threshold chosen under a 5% false-hit budget: guard off = None, guard on = None.

| Threshold | Guard | Hit rate | False-hit rate | Valid-reuse recall |
|---:|---|---:|---:|---:|
| 0.90 | off | 92.7% | 48.2% | 96.0% |
| 0.90 | on | 59.7% | 51.4% | 58.0% |
| 0.92 | off | 87.0% | 48.3% | 90.0% |
| 0.92 | on | 57.7% | 51.4% | 56.0% |
| 0.95 | off | 67.7% | 43.8% | 76.0% |
| 0.95 | on | 48.7% | 46.6% | 52.0% |
| 0.98 | off | 34.3% | 28.2% | 49.3% |
| 0.98 | on | 26.7% | 28.7% | 38.0% |

### Quora QQP proxy (300 pairs: natural questions)

Cosine similarity, reusable pairs: median 0.871, min 0.590; non-reusable pairs: median 0.609, max 0.944. Threshold chosen under a 5% false-hit budget: guard off = 0.92, guard on = 0.92.

| Threshold | Guard | Hit rate | False-hit rate | Valid-reuse recall |
|---:|---|---:|---:|---:|
| 0.90 | off | 19.7% | 8.5% | 36.0% |
| 0.90 | on | 15.7% | 6.4% | 29.3% |
| 0.92 | off | 14.3% | 4.7% | 27.3% |
| 0.92 | on | 12.0% | 2.8% | 23.3% |
| 0.95 | off | 8.3% | 0.0% | 16.7% |
| 0.95 | on | 7.7% | 0.0% | 15.3% |
| 0.98 | off | 2.3% | 0.0% | 4.7% |
| 0.98 | on | 2.3% | 0.0% | 4.7% |

## Workload-level simulation (QQP intents, real embeddings, oracle provider)

2000 requests per seed, Zipf 1.1, 400 intents, seeds [42, 43, 44]; the intent set is fixed (built once with seed 42), so seeds vary only the request stream. Provider is a ground-truth oracle, so latency is not reported here. Values are means over seeds (range in brackets).

| Guard | Threshold | Hit rate | Uplift over exact-only | False hits / hits | Wrongly served (of all requests) |
|---|---:|---:|---:|---:|---:|
| off | 0.80 | 86.2% | +1.53 pp | 1.90% [1.74%-2.15%] | 1.63% [1.50%-1.85%] |
| off | 0.90 | 85.1% | +0.45 pp | 0.49% [0.23%-0.76%] | 0.42% [0.20%-0.65%] |
| off | 0.92 | 85.0% | +0.40 pp | 0.39% [0.06%-0.76%] | 0.33% [0.05%-0.65%] |
| off | 0.95 | 84.8% | +0.17 pp | 0.04% [0.00%-0.12%] | 0.03% [0.00%-0.10%] |
| off | 0.98 | 84.6% | +0.00 pp | 0.00% [0.00%-0.00%] | 0.00% [0.00%-0.00%] |
| on | 0.80 | 85.7% | +1.10 pp | 1.30% [0.99%-1.81%] | 1.12% [0.85%-1.55%] |
| on | 0.90 | 85.0% | +0.37 pp | 0.22% [0.18%-0.29%] | 0.18% [0.15%-0.25%] |
| on | 0.92 | 85.0% | +0.32 pp | 0.12% [0.00%-0.18%] | 0.10% [0.00%-0.15%] |
| on | 0.95 | 84.8% | +0.17 pp | 0.04% [0.00%-0.12%] | 0.03% [0.00%-0.10%] |
| on | 0.98 | 84.6% | +0.00 pp | 0.00% [0.00%-0.00%] | 0.00% [0.00%-0.00%] |

Exact-only baseline (threshold 1.00): hit rate 84.6%. This workload is dominated by identical repeats, so semantic matching adds little on top of exact matching.

## Real end-to-end benchmark (OpenRouter, gpt-4.1-mini, localhost proxy)

Status: **complete** - 165 of 165 planned requests completed. Threshold 0.95, guard on, seed 42.

| Path | n | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| direct provider | 165 | 978.2 | 1437.1 | 1602.1 |
| proxy BYPASS | 15 | 1020.7 | 1414.1 | 1414.1 |
| proxy HIT | 104 | 1.4 | 2.1 | 2.3 |
| proxy MISS | 46 | 1127.8 | 1483.6 | 1840.1 |

- Composition: {'first_seen': 38, 'no_store': 15, 'exact_repeat': 104, 'paraphrase': 8}.
- Provider calls: 165 direct vs 61 via proxy.
- HIT validity (intent oracle): 104 valid, 0 false.
- Estimated cost (configured prices, embedding cost included): direct $0.01811 vs proxy $0.00600 = **66.9% net saving**.
- Median paired overhead of a MISS vs the same request sent directly: +232 ms (embedding + lookup + store; noisy because provider latency varies per call).

## Concurrency load test (SYNTHETIC provider and embedder)

Provider 800 ms ±20% and embedder 60 ms are configured sleeps, not measurements. 3000 requests, 322 unique prompts (Zipf 1.1), 5% one-offs. "Redundant" provider calls are calls beyond one per unique prompt: duplicate misses that ran concurrently.

| Proxy threads | Workers | Arm | Throughput req/s | P50 ms | P95 ms | P99 ms | Provider calls avoided | Redundant provider calls |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| - | 20 | baseline | 25 | 806.1 | 950.1 | 961.9 | 0% | - |
| 40 | 20 | cached | 181 | 1.2 | 905.5 | 1015.4 | 87.9% | 41 |
| - | 50 | baseline | 62 | 805.6 | 949.3 | 961.0 | 0% | - |
| 40 | 50 | cached | 279 | 36.7 | 957.0 | 1083.2 | 86.8% | 74 |
| - | 50 | baseline | 62 | 804.5 | 948.6 | 960.0 | 0% | - |
| 200 | 50 | cached | 376 | 1.4 | 919.0 | 1008.7 | 86.0% | 99 |

40 proxy threads, 20 workers, cached arm by status: HIT P50 1.1 / P99 4.1 ms (n=2637), MISS P50 882.3 / P99 1036.9 ms (n=363)

40 proxy threads, 50 workers, cached arm by status: HIT P50 27.6 / P99 325.4 ms (n=2604), MISS P50 914.5 / P99 1533.9 ms (n=396)

200 proxy threads, 50 workers, cached arm by status: HIT P50 1.2 / P99 5.7 ms (n=2579), MISS P50 866.2 / P99 1029.2 ms (n=421)

## Cost to build

Spend-ledger total: **$0.0419** against a $0.20 cap for this repository (per label: embeddings $0.00071, e2e-chat-direct $0.03008, e2e-embeddings-proxy $0.00003, e2e-chat-proxy $0.01105). Estimates, not an invoice.

Cross-check against OpenRouter's own usage counter (manual readings, shared key): about $0.0419 vs ledger $0.0419.
