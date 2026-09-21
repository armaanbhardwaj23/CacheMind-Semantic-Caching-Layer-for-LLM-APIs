SYNTHETIC provider: 800 ms ±20%, embedder 60 ms; 3000 requests, 322 unique prompts (Zipf 1.1), 5% one-offs; proxy worker threads: 200.

| Workers | Arm | Throughput req/s | P50 ms | P95 ms | P99 ms | Provider calls avoided | Redundant provider calls |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | baseline | 62 | 804.5 | 948.6 | 960.0 | 0% | - |
| 50 | cached | 376 | 1.4 | 919.0 | 1008.7 | 86.0% | 99 |
|  | cached by status | | HIT P50 1.2 / P99 5.7 ms (n=2579), MISS P50 866.2 / P99 1029.2 ms (n=421) | | | | |
