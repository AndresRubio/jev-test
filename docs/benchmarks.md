# Benchmarks (2026-10-20)

Machine: Apple M1 Pro, 16 GB, macOS 26.6. laya-mlx 0.4.0, local-jev 0.1.0.
Latency is end to end over HTTP from a local client. Memory is the process's physical footprint from `top`, which counts a MB as 1,048,576 bytes.

| Measure | Result | Target | Pass |
|---|---|---|---|
| One short question, median | 22.9 ms | ≤ 100 ms | yes |
| One short question, 95th percentile | 23.8 ms |  |  |
| 10 questions in one call, median | 107.5 ms | ≤ 400 ms | yes |
| First request (loads English) | 572 ms | ≤ 10,000 ms | yes |
| First request after idle unload | 267 ms | ≤ 10,000 ms | yes |
| Memory with English loaded | 1595 MB |  |  |
| Memory with both checkpoints loaded | 2406 MB | ≤ 2,500 MB | yes |
| Memory after idle unload | 284 MB | ≤ 400 MB | yes |
| Models unloaded after idle | True | True | yes |
| Batch of 200 short items, one yes/no question, median of 3 | 46 items/s |  |  |
