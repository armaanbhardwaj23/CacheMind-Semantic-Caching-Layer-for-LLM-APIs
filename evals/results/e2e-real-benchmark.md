Status: **complete** (165 of 165 planned requests completed)

| Path | n | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| direct provider | 165 | 978.2 | 1,437.1 | 1,602.1 |
| proxy BYPASS | 15 | 1,020.7 | 1,414.1 | 1,414.1 |
| proxy HIT | 104 | 1.4 | 2.1 | 2.3 |
| proxy MISS | 46 | 1,127.8 | 1,483.6 | 1,840.1 |

- composition: {'first_seen': 38, 'no_store': 15, 'exact_repeat': 104, 'paraphrase': 8}; statuses: {'MISS': 46, 'BYPASS': 15, 'HIT': 104}
- provider calls: 165 direct vs 61 via proxy
- hit validity (intent oracle): 104 valid, 0 false
- estimated cost: direct $0.01811 vs proxy $0.00600 (chat + embeddings); net savings 66.9%
- threshold 0.95, seed 42, model openai/gpt-4.1-mini
