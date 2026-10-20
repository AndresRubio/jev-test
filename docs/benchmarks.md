# Benchmarks (2026-10-20)

Machine: Apple M1 Pro, 16 GB, macOS 26.6. laya-mlx 0.4.0, local-jev 0.1.0.
Latency is end to end over HTTP from a local client. Memory is the process's physical footprint from `top`, which counts a MB as 1,048,576 bytes.

| Measure | Result | Target | Pass |
|---|---|---|---|
| One short question, median | 24.0 ms | ≤ 100 ms | yes |
| One short question, 95th percentile | 27.9 ms |  |  |
| 10 questions in one call, median | 107.7 ms | ≤ 400 ms | yes |
| First request (loads English) | 808 ms | ≤ 10,000 ms | yes |
| First request after idle unload | 348 ms | ≤ 10,000 ms | yes |
| Memory with English loaded | 1593 MB |  |  |
| Memory with both checkpoints loaded | 2402 MB | ≤ 2,500 MB | yes |
| Memory after idle unload | 281 MB | ≤ 400 MB | yes |
| Models unloaded after idle | True | True | yes |
| Batch of 200 short items, one yes/no question, median of 3 | 47 items/s |  |  |
| Batch of 200 short items, two yes/no questions, median of 3 | 30 items/s |  |  |
| Batch of 200 items of about 430 tokens, one yes/no question, median of 3 | 8 items/s |  |  |
