SYNTHETIC provider: 800 ms ±20%, embedder 60 ms; 3000 requests, 322 unique prompts (Zipf 1.1), 5% one-offs; proxy worker threads: 40.

| Workers | Arm | Throughput req/s | P50 ms | P95 ms | P99 ms | Provider calls avoided | Redundant provider calls |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | baseline | 25 | 806.1 | 950.1 | 961.9 | 0% | - |
| 20 | cached | 181 | 1.2 | 905.5 | 1015.4 | 87.9% | 41 |
|  | cached by status | | HIT P50 1.1 / P99 4.1 ms (n=2637), MISS P50 882.3 / P99 1036.9 ms (n=363) | | | | |
| 50 | baseline | 62 | 805.6 | 949.3 | 961.0 | 0% | - |
| 50 | cached | 279 | 36.7 | 957.0 | 1083.2 | 86.8% | 74 |
|  | cached by status | | HIT P50 27.6 / P99 325.4 ms (n=2604), MISS P50 914.5 / P99 1533.9 ms (n=396) | | | | |
