# Kaggle T4x2 Shard/B_MICRO Sweep - 2026-05-28

## Scope
- Hardware: Kaggle GPU T4 x2.
- Puzzle: `puzzle_id=20`.
- Beam: `2**26 + 15_506_660 = 82615524`.
- Depth limit: `72`.
- History: `static_hybrid`.
- Solution shortcut: `BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4`.
- K2: disabled, `BEAM_STREAM2_SUFFIX_RADIUS=0`.
- Code scope: notebook/config sweep plus isolated stream benchmark; no C++ architecture changes during this sweep.

## Production Sweep

| Run | Config | Shards | B_MICRO | S3/S4 batch | S4 trigger | Slots | Cap scale | Result | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v111 | `sh64_b8192` | 64 | 8192 | 196608 | 393216 | 4 | 125% | solved, `352.100s` | reference working config |
| v118 | `sh128_b4096` | 128 | 4096 | 98304 | 196608 | 4 | 125% | solved, `350.402s` | corrected 128-shard run |
| v115 | `sh128_b8192` | 128 | 8192 | 196608 | 364544 | 2 | 125% | fail depth 6 | Stream4 A/B capacity saturation |
| v116 | `sh128_b8192` | 128 | 8192 | 196608 | 196608 | 2 | 125% | fail depth 6 | lower trigger did not fix capacity |
| v117 | `sh128_b4096` label | 128 | 8192 | 196608 | 196608 | 4 | 125% | fail init | edit bug: B_MICRO stayed 8192; CUDA graph OOM |
| v114 | `sh128_b8192` | 128 | 8192 | 196608 | 323584 | 2 | 100% | fail depth 6 | capacity too tight |
| v114 | `sh256_b6656` | 256 | 6656 | 159744 | 161792 | 2 | 100% | fail init | CUDA graph OOM |
| v114 | `sh512_b3328` | 512 | 3328 | 79872 | 80896 | 2 | 100% | fail init | CUDA graph OOM |
| v114 | `sh1024_b1664` | 1024 | 1664 | 39936 | 40960 | 2 | 100% | fail budget | required `16170552240` > budget `15176892416` |

## Working Config Comparison

| Metric | `sh64_b8192` | `sh128_b4096` | Delta |
| --- | ---: | ---: | ---: |
| solved seconds | `352.100` | `350.402` | `-0.48%` |
| depth 7..10 avg sec | `65.792` | `65.670` | `-0.19%` |
| depth 6 sec | `17.256` | `16.475` | `-4.5%` |
| full-depth Stream3 jobs | `5048` | `10112` | `2.00x` |
| full-depth Stream4 jobs | `~1020` | `~2045` | `2.00x` |
| static allocation | `14.013 GiB` | `13.806 GiB` | `-0.207 GiB` |
| scratch pool | `9.083 GiB` | `8.869 GiB` | `-0.214 GiB` |
| free after all allocations | `279.8 MiB` | `513.8 MiB` | `+234.0 MiB` |

## Stream1 B_MICRO And Parallel Model Benchmark

Isolated `stream_benchmark` results from v109, TensorOp CUTLASS on T4:

| B_MICRO | concurrency 1 | concurrency 2 | concurrency 3 | concurrency 4 | Best |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 17.46M cand/s | 20.06M cand/s | 20.37M cand/s | 20.63M cand/s | 20.63M |
| 2048 | 18.35M cand/s | 20.10M cand/s | 33.76M cand/s | 40.60M cand/s | 40.60M |
| 4096 | 36.79M cand/s | 38.49M cand/s | 38.13M cand/s | 37.77M cand/s | 38.49M |
| 8192 | 32.06M cand/s | 32.84M cand/s | 33.11M cand/s | 32.84M cand/s | 33.11M |

## Conclusions
- Shard-count plateau for this T4x2 memory layout is `64..128` shards.
- `128` shards work only after reducing `B_MICRO` and Stream3/Stream4 batch to `4096 * 24 = 98304`.
- `128` shards are not meaningfully faster than `64` shards: `350.402s` vs `352.100s`; the useful gain is memory headroom, not throughput.
- `256` and `512` shards did not instantiate CUDA graphs in tested config-only layouts.
- `1024` shards exceeded the manual GPU budget before runtime.
- Isolated Stream1 optimum was `B_MICRO=2048` with `4` parallel model launches at `40.60M cand/s`.
- Production runner currently exposes `B_MICRO` as config, but parallel Stream1 model launch count was only measured in `stream_benchmark`; production parallel model tuning would require separate code/config plumbing.

## Packaged output and parent-local provenance

The public package includes the production result
[`beam_run_results.csv`](beam_run_results.csv) and this derived sweep summary.
The original raw tuning directories below are parent-repository-local provenance
and are not included in the public package:

- `test_results/kaggle_v109_shard_bmicro_sweep/`
- `test_results/kaggle_v111_shard_sweep_k1r4/`
- `test_results/kaggle_v114_shard_sweep_cap100/`
- `test_results/kaggle_v115_sh128_slots2/`
- `test_results/kaggle_v116_sh128_trigger196/`
- `test_results/kaggle_v117_sh128_b4096_slots4/`
- `test_results/kaggle_v118_sh128_b4096_slots4_corrected/`
