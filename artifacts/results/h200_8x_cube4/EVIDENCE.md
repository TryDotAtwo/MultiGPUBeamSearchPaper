# Eight-H200 Cube4 paper evidence

This note distinguishes a completed search depth from a completed puzzle solve. It is a read-only audit of previously captured 8xH200 artifacts; no GPU job was launched for the paper update.

## Source artifacts

- Original report: `worktrees/hopper-stream1-fusion/test_results/vast_h200_frontier_2026-09-04/REPORT.md`.
- Complete rank-0 production log: `worktrees/hopper-stream1-fusion/test_results/vast_h200_frontier_2026-09-04/solves/p1000_b2900_sh64/logs/production_runner_p1000_d50_b2900000000_manual_original.log`, SHA-256 `0b1ffbbde1554c75324e7ce07c2b0ed111b7df23d745dfa9809296f01d77dee4`.
- Other seven rank logs: sibling `logs/ranks_manual_original/rank1.log` through `rank7.log`. Each records `depth_done=8` and `next_frontier_size=362545152`.
- Device telemetry: sibling `logs/tuning_manual/nvidia_smi_original.log`; 3,008 H200 samples, observed maximum device-wide used memory 141,820 MiB per GPU, device total 143,771 MiB. This is not PyTorch allocated/reserved VRAM.
- Original compressed run bundle: `worktrees/hopper-stream1-fusion/test_results/vast_h200_frontier_2026-09-04/h200_frontier_2026-09-04.tar.gz`.

In the public paper repository, this audit is `artifacts/results/h200_8x_cube4/EVIDENCE.md`. The compact raw files are in the same directory as `rank0.log` through `rank7.log`, `nvidia_smi.log`, `boundary_2940m.log`, `boundary_3000m.log`, and `ORIGINAL_REPORT.md`. The large original compressed bundle and reconstructable transient history arenas are not included in the public repository.

## Measured contract and derivation

- Vast instance 49878187, eight NVIDIA H200 GPUs; one physically sharded global Cube4 beam, puzzle 1000, 96-byte state, 24 generators, 24-output FP16 piece Transformer. Model architecture: 57 tokens, 256 hidden units, eight attention heads, four Transformer blocks, FFN width 1024; 3,383,064 parameters per prior matched-Cube4 model audit.
- Solver revision `3b756afe9e7636e53520fa6de7e60a167bbaa79a`. Native CUDA Graph Stream 1, 64 shards/GPU, outer microbatch 192, Stream 1 concurrency 12, Stream 3 ring slots 12. NCCL reported 2.28.9+cuda12.9.
- Model-export manifest SHA-256 `03f55e5b617874cbbfaace5796f7e348588eb2c0b5ef079e45c06b05d84d0cb0`. The local public Cube4 checkpoint referenced by that export has SHA-256 `58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164`; the original H200 report verified the export manifest identity, not a separate remote checkpoint file digest.
- Requested beam 2,900,000,000; aligned effective retained beam 2,900,361,216 = 8 x 362,545,152. All eight ranks completed saturated depth 8 in 931.266 s (rank4-7 show 931.265 s, a 1 ms reporting difference). The rank-0 log records `depth_start=8 frontier_size=362545152` and `depth_done=8 next_frontier_size=362545152`.
- Nominal raw parent-action pairs for a saturated depth: 2,900,361,216 x 24 = 69,608,669,184. Derived throughput: 3,114,428.333 parents/s and 74,746,279.993 nominal pairs/s. These are not distinct post-deduplication states/s or a count from a native node counter.
- Depth 9 began but was stopped on user request. Puzzle 1000 was not solved; there is no complete-run wall time or depth-9 timing.
- The 2.94B-request profile failed at its first NCCL count exchange, not a documented CUDA OOM. The 3.00B-request profile failed the static budget gate (`required=147513567824`, `budget=144821649408`). Therefore 2.900361216B is a measured completed frontier, not a proven hardware maximum.
- This Cube4/H200 run is not workload-matched to the Megaminx/2xT4 paper run. Their rates cannot be divided into a strong-scaling speedup.
