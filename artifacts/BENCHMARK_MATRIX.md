# Paper benchmark matrix — 2026-08-12

## Contract

The primary endpoint is native search-system cost, not solution quality.  No
algorithm, scorer, model, stopping rule, or data placement is modified for the
comparison.  Each released system is run in its author-provided configuration
on the same 50 deterministic Cube3 states wherever its puzzle contract permits.
We report time, processed nodes/rate, peak host RAM and VRAM, maximum completed
scale, and failure boundary.  Solution length is excluded from the primary
tables.  Because the systems use different models and search semantics, the
results compare complete implementations rather than isolate a single kernel.
DeepCubeA is weighted A* and is indexed by its actual search/NN batches, never
by a fictitious beam width.

The paper's separate scale headline is not a cell in the one-T4 comparison.
The eight-H200 Cube4 raw logs establish a requested 2,900,000,000-state beam,
aligned retained width 2,900,361,216, and one saturated depth-8 step completed
in 931.266 s across all eight ranks. With 24 generators, the nominal raw
parent-action count is 69,608,669,184 and derived end-to-end rate is
74.746M pairs/s. Puzzle 1000 was stopped during depth 9 without a solution.
The older eight-A100 capacity summary records requested
`B_req=770,883,178`, depth 8, and `39,745 MiB/GPU`. Runtime alignment makes
`B_eff >= B_req`, but exact effective width and wall time are unavailable.
Thus `24 * B_req = 18,501,196,272` is reported as the minimum nominal logical
child-candidate scale of a saturated depth, not as a timed rate. The two-T4
speed point instead records requested `82,615,524`, effective `82,837,504`,
and `65.670 s` per saturated depth, giving `30.274M` logical children/s.
Novelty scope, counterexamples, and the retained-beam/raw-candidate taxonomy
are audited in `test_results/paper_prior_art_novelty_audit_2026-08-25.md`.
The H200 raw-log audit and exact evidence paths are in
`test_results/h200_paper_evidence_2026-09-26.md`. Cube4/H200 and Megaminx/T4
are different workloads, not a strong-scaling pair.

| Platform | Puzzle/model | Requested / effective retained beam | Completed timed step | Derived nominal rate | Peak device-wide used VRAM | Boundary and outcome |
|---|---|---:|---:|---:|---:|---|
| 8xH200, 143,771 MiB each | Cube4 puzzle 1000, FP16 24-output piece Transformer, 3,383,064 parameters, 24 moves | 2,900,000,000 / 2,900,361,216 | depth 8, 931.266 s on all eight ranks | 74,746,280 parent-action pairs/s | 141,820 MiB/GPU | depth 9 interrupted by request, unsolved; 2.94B first NCCL count exchange failed, 3.00B static budget gate; neither classified as CUDA OOM |

- Local GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, driver 572.70.
- Runtime: Python 3.11.5, PyTorch 2.8.0+cu128, CUDA runtime 12.8.
- Solver-level scramble: `D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2`.
- AlphaCube uses normal 18-move HTM (`allow_wide=False`) and the official `large` model.
- DeepCubeA uses the official cube3 model, weighted A* with `weight=0.6`, search batch 10000, and neural batch 10000.
- Timed processes are isolated. Model download is excluded. Peak CUDA allocated and reserved memory are recorded.

## Native-system cost table (50 states; pending remote measurements)

| GPU | System | Native model/search parameter | States | Total wall, s | Mean/p50/p90 wall per state | Total nodes | Nodes/s | Peak host RAM | Peak VRAM | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1xT4 | AlphaCube 0.1.6 | `large`, beam-width sweep | 50 | pending | pending | pending | pending | pending | pending | not tested |
| 1xT4 | DeepCubeA | official cube3, weight 0.6, search/NN batch 10,000/10,000 | 50 | pending | pending | pending | pending | pending | pending | not tested |
| 1xP100 | AlphaCube / DeepCubeA | same native parameters | 50 | pending | pending | pending | pending | pending | pending | not tested |

The final comparison contains five distinct algorithms: Pilgrim, native
CayleyPy, MultiGPUBeamSearch, AlphaCube, and DeepCubeA.  Pilgrim and native
CayleyPy are never merged into one label.  The matched Megaminx scalar test
uses the same checkpoint, state, and 24-generator graph for Pilgrim, native
CayleyPy, and MultiGPUBeamSearch.  Native CayleyPy commit `5b2b53f` accepts one
score per state, so its native 24-output cell is `incompatible`; no Q-head
adapter is introduced.  Megaminx measurements remain separate from Cube3.

| Hardware | Algorithm | Shared Megaminx output-1 | Shared Megaminx output-24 | Current status |
|---|---|---|---|---|
| 1xT4 | Pilgrim Searcher/QSearcher | max tested complete 5,898,240; 1,462.557 s; 14,433 MiB peak; 6,291,456 CUDA OOM | max tested complete 20,971,520; 197.358 s; 23,068,672 CUDA OOM | scalar bracket 5,898,240 complete / 6,291,456 OOM; output-24 bracket 20,971,520 complete / 23,068,672 OOM |
| 1xT4 | native CayleyPy `CayleyGraph.beam_search(iterated)` | broad grid: `2^10`--`2^24` complete, `2^25` CUDA OOM; earlier targeted max complete 20,971,520 and OOM 21,037,056 | incompatible: native predictor requires one score per state | broad powers-of-two sweep COMPLETE; no trend confirmation selected |
| 1xT4 | MultiGPUBeamSearch `WORLD_SIZE=1` | max tested complete 29,360,128; confirmation 2,817.158 s; 14,215 MiB peak; 31,457,280 CUDA OOM | max tested complete 29,360,128; 116.904 s; 13,717 MiB peak; 33,554,432 CUDA OOM | scalar bracket 29,360,128 complete / 31,457,280 OOM; output-24 bracket 29,360,128 complete / 33,554,432 OOM |
| 1xT4 | AlphaCube 0.1.6 | different Cube3 model/workload | different Cube3 model/workload | measured complete through 1,572,864; first failure 1,703,936 host-RAM SIGKILL |
| 1xT4 | DeepCubeA | different Cube3 model/workload | different Cube3 model/workload | measured complete with weight 0.6 and search/NN batches 10,000/10,000 |

### Exact model architectures used by the reported tests

Parameter counts refer to the trainable source model before batch-normalization
folding. Ordered block sequences are recorded explicitly so that rows with
different scorers cannot be mistaken for an algorithm-only comparison.

| Workload / model family | Trainable parameters | Ordered architecture | Used by |
|---|---:|---|---|
| Cube3 AlphaCube 0.1.6 `large` | 118,939,666 | one-hot 324 -> 8 x [Linear 4096, ReLU, BatchNorm] -> Linear 18 | AlphaCube only |
| Cube3 DeepCubeA official `cube3` | 14,663,001 | one-hot 324 -> Linear 5000, BN, ReLU -> Linear 1000, BN, ReLU -> 4 x [Linear 1000, BN, ReLU, Linear 1000, BN, residual add, ReLU] -> Linear 1 | DeepCubeA only |
| Megaminx scalar MLP, `hd1=2048`, `hd2=512`, `nrd=8` | 34,766,849 | position-class 120x120 -> EmbeddingBag-equivalent Linear 2048, BN, ReLU -> Linear 512, BN, ReLU -> 8 x [Linear 512, BN, ReLU, Linear 512, BN, residual add, ReLU] -> Linear 1 | Pilgrim Searcher, native CayleyPy, MultiGPUBeamSearch; byte-identical checkpoint SHA-256 `7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759` |
| Megaminx QMLP2RB, `hd1=1536`, `hd2=512`, `nrd=2` | 23,978,008 | position-class 120x120 -> EmbeddingBag-equivalent Linear 1536, BN, ReLU -> Linear 512, BN, ReLU -> 2 x [Linear 512, BN, ReLU, Linear 512, BN, residual add, ReLU] -> Linear 24 | Pilgrim QSearcher, MultiGPUBeamSearch; byte-identical checkpoint SHA-256 `e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670` |
| Cube4 Q Transformer | 3,383,064 | 96 tokens / 6 classes -> projection + CLS -> 4 x [pre-LN, 8-head self-attention (`d=256`), residual, pre-LN, FFN 256-1024-256 with ReLU, residual] -> LN -> Linear 24 | separate matched Cube4 Pilgrim/MultiGPU comparison only |

### Native CayleyPy Megaminx capacity, matched scalar protocol

The private 1xT4 broad sweep used native CayleyPy commit `5b2b53f`,
`CayleyGraph.beam_search` in `iterated` mode, `history_depth=0`, no returned
path, and exactly eight layers.  It used one visible Tesla T4 (15,636,037,632
bytes), Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 9.10.2, unpacked
int8 states, and the shared FP16 scalar MLP (`hd1=2048`, `hd2=512`, `nrd=8`),
checkpoint SHA-256
`7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759`.
The graph is fixed Megaminx `p900-t000` with 24 generators and fixed-state
SHA-256 `8cbf1d728c44ca820a59990e768da78fc5ebcb6f3105a0d822c3808bfda6cc75`.

| Beam | Status | Wall, s | s/depth | Peak allocated, MiB | Peak reserved, MiB |
|---:|---|---:|---:|---:|---:|
| 1,024 | completed | 0.930 | 0.116 | 85.9 | 92.0 |
| 2,048 | completed | 1.237 | 0.155 | 96.3 | 112.0 |
| 4,096 | completed | 2.189 | 0.274 | 117.3 | 128.0 |
| 8,192 | completed | 3.828 | 0.478 | 158.6 | 230.0 |
| 16,384 | completed | 6.776 | 0.847 | 241.5 | 292.0 |
| 32,768 | completed | 13.722 | 1.715 | 407.5 | 510.0 |
| 65,536 | completed | 31.399 | 3.925 | 739.3 | 858.0 |
| 131,072 | completed | 63.664 | 7.958 | 772.2 | 1,180.0 |
| 262,144 | completed | 114.551 | 14.319 | 834.8 | 1,168.0 |
| 524,288 | completed | 219.991 | 27.499 | 962.1 | 2,156.0 |
| 1,048,576 | completed | 430.789 | 53.849 | 1,216.7 | 3,234.0 |
| 2,097,152 | completed | 745.341 | 93.168 | 1,726.0 | 2,740.0 |
| 4,194,304 | completed | 1,320.000 | 165.000 | 2,745.2 | 4,492.0 |
| 8,388,608 | completed | 2,489.356 | 311.169 | 5,155.3 | 6,462.0 |
| 16,777,216 | completed | 4,741.581 | 592.698 | 10,233.7 | 14,652.0 |
| 33,554,432 | CUDA OOM | n/a | n/a | 12,922.2 | 14,484.0 |

Thus the broad-grid maximum complete is `2^24` and its first failure is
`2^25` CUDA OOM. The memory-trend estimator returned `2^24` itself, so the
protocol correctly scheduled no redundant confirmation point. Separate older
targeted evidence with the identical runtime contract remains valid:
20,971,520 completed in 4,258.602 s and 21,037,056 CUDA-OOMed; it is retained
as a finer historical bracket, not presented as part of the broad-grid policy.
The `2^25` OOM attempted a 1.96-GiB allocation. The monkey-patched public
`get_neighbors` method did not intercept
CayleyPy's internal iterated implementation, so its frontier/candidate counters
remained zero.  Consequently nodes/s is **not measured** and no rate is derived
from those counters.  Raw JSON and per-width logs are retained under
`test_results/paper_benchmarks/broad_sweeps_2026-08-15/cayleypy_v9_complete/`;
the older targeted bracket remains under
`test_results/paper_benchmarks/cayleypy_native_t4_history0_v5_complete/`.

## Paired Cube4 protocol: Pilgrim versus ours

## Native approaches on one T4

This is the paper-facing four-row view. Cube3 and Megaminx remain different
workloads; only native counters are reported as nodes/s, and no `beam/wall`
proxy is substituted for missing instrumentation.

| Approach | Workload / native scale | Max tested complete | Wall at that point | Counted nodes/s | Status |
|---|---|---:|---:|---:|---|
| AlphaCube 0.1.6 | Cube3 HTM-18, beam | 1,572,864 | 461.488 s | 42,387 | first failure 1,703,936 host-RAM SIGKILL |
| DeepCubeA | Cube3 weighted A*, search/NN batch 10,000/10,000 | n/a | 146.474 s | 41,880 | completed, solution length 21; no beam width |
| Pilgrim Searcher | Megaminx scalar, depth 8 | 5,898,240 | 1,462.557 s | not measured | max tested-complete; 182.820 s/depth; 14,433 MiB peak; first failure 6,291,456 CUDA OOM |
| Pilgrim QSearcher | Megaminx native output-24, depth 8 | 20,971,520 | 197.358 s | not measured | max tested-complete; 24.670 s/depth; first failure 23,068,672 CUDA OOM |
| native CayleyPy `5b2b53f` | Megaminx scalar, depth 8, `history_depth=0` | 20,971,520 | 4,258.602 s | not measured | broad grid: 2^24 complete, 2^25 CUDA OOM; older targeted OOM 21,037,056 |
| MultiGPUBeamSearch `WORLD_SIZE=1` | Megaminx scalar, depth 8 | 29,360,128 | 2,817.158 s | not measured | confirmed max complete; 352.145 s/depth; 14,215 MiB peak; first failure 31,457,280 CUDA OOM |
| MultiGPUBeamSearch `WORLD_SIZE=1` | Megaminx native output-24, depth 8 | 29,360,128 | 116.904 s | not measured | max tested-complete; 14.613 s/depth; 13,717 MiB peak; first failure 33,554,432 CUDA OOM |

Derived native-step times (wall divided only by an evidenced completed step
count): AlphaCube at its widest completed point is 461.488/17 = 27.146 s per
completed beam depth; DeepCubeA is 146.474/55 = 2.663 s per weighted-A* search
iteration (not a beam depth); native CayleyPy at 20,971,520 is 4,258.602/8 =
532.325 s/depth; and MultiGPUBeamSearch at 25,165,824 is 2,416.150/8 = 302.019
s/depth. The Cube4 table reports the measured tenth-layer time directly, so its
existing wall values are already seconds for one depth.

The final Pilgrim v5 pass completed all three scalar widths: 5,636,096,
5,767,168, and 5,898,240. The widest point took 1,462.557 s
(182.820 s/depth) and peaked at 14,433 MiB. A later scalar-only pass
CUDA-OOMed at 6,291,456 after 873.781 s with a 14,011 MiB device peak; its
automatic confirmation at 5,242,880 completed in 1,307.224 s. The measured
scalar bracket is therefore 5,898,240 complete versus 6,291,456 OOM. The
notebook then ended with a duplicate-checkpoint fail-closed error after both
point records had been written; this terminal error is not the failure class
of either retained point. For output-24,
20,971,520 completed in 197.358 s (24.670 s/depth) at 12,333 MiB, while
23,068,672 CUDA-OOMed. The measured output-24 bracket is therefore 20,971,520
complete versus 23,068,672 OOM. Raw v5 artifacts are under
`test_results/paper_benchmarks/pilgrim_boundary_v5_2026-08-18/`.
The scalar upper-failure artifacts are under
`test_results/paper_benchmarks/pilgrim_latest_error2_2026-08-18/`.

The corrected collaborator-run MultiGPU pass and the final scalar-only v27
confirmation retained the following boundary points. Both native heads now
have measured complete/OOM pairs:

| Beam | Wall, s | s/depth | Peak device, MiB | Status |
|---:|---:|---:|---:|---|
| 25,165,824 | 3,027.673 | 378.459 | 12,267 | completed, scalar |
| 27,262,976 | 3,080.206 | 385.026 | 13,241 | completed, scalar |
| 29,360,128 | 2,817.158 | 352.145 | 14,215 | completed, scalar confirmation |
| 31,457,280 | n/a | n/a | 14,909 | CUDA OOM, scalar |
| 1,048,576 | 8.296 | 1.037 | 773 | completed, output-24 |
| 4,194,304 | 22.735 | 2.842 | 2,253 | completed, output-24 |
| 8,388,608 | 41.091 | 5.136 | 4,159 | completed, output-24 |
| 16,777,216 | 77.460 | 9.683 | 7,983 | completed, output-24 |
| 20,971,520 | 96.124 | 12.015 | 9,893 | completed, output-24 |
| 25,165,824 | 111.130 | 13.891 | 11,799 | completed, output-24 |
| 29,360,128 | 116.904 | 14.613 | 13,717 | completed, output-24 |
| 33,554,432 | 1.563 | n/a | 14,909 | CUDA OOM, output-24 |

The scalar OOM recurred for every row-budget profile from 49,152 through 24,
so it is a measured capacity failure rather than a single profile-selection
failure. The complete confirmation recorded per-depth times
0.613, 0.572, 0.580, 0.866, 5.075, 71.692, 1,087.45, and 1,644.13 s; the
primary common metric remains total wall / 8 = 352.145 s/depth. The failed
probe spent 164.414 s across all fallback attempts; its final attempt lasted
1.374 s and is not a completed-depth time. Raw v27 evidence is under
`test_results/paper_benchmarks/multigpu_scalar_boundary_v27_2026-08-24/`.

The output-24 OOM was reproduced while the runtime descended its row-budget
profiles from 2,048 to 1. The recorded 1.563 s is the final failed search
attempt, not a completed depth time. After persisting all point records, the
notebook failed closed because the scalar checkpoint was visible both inside
the self-contained dataset and as a separately attached model. This terminal
error does not change the retained point classifications. Raw evidence is under
`test_results/paper_benchmarks/multigpu_latest_error_2026-08-18/`.

At the matched broad-grid Megaminx width 16,777,216, the latest CayleyPy wall
is 4,741.581 s versus the historical MultiGPU wall 1,728.460 s (2.74x lower).
At the older targeted width 20,971,520, MultiGPU wall is 1.95x lower. These are
end-to-end wall ratios, not inferred nodes/s ratios. The final scalar
confirmation measured 29,360,128 complete versus 31,457,280 CUDA OOM. The
matched output-24 pass measured 29,360,128 complete versus 33,554,432 CUDA OOM.
Superseded static-profile failures and the cancelled v10/v11 runs remain
excluded.

Those ratios belong only to the shared scalar `output_dim=1` contract: the
checkpoint, architecture, output head, fixed state, generators, depth, width,
FP16 dtype, and one-T4 hardware match. Pilgrim QSearcher is not compared with
scalar CayleyPy. Output-24 is a separate matched family containing Pilgrim
QSearcher and MultiGPUBeamSearch; native CayleyPy is incompatible with it.

The paired Cube4 experiment uses the same 24-action Q Transformer in both
native search implementations.  The checkpoint SHA-256 is
`58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164`:
96 position tokens, six classes, `d_model=256`, eight heads, four encoder
layers, feed-forward width 1024, ReLU, zero dropout, CLS pooling, and 24 Q
outputs (3,383,064 parameters).  Generator and facelet orderings are taken
from the same private model bundle.

Every capacity/speed point runs to exactly depth 10.  The primary reported
measurement is the tenth expansion layer itself, not the mean over startup
layers: `depth10_wall_s`, frontier size entering depth 10, generated candidates
at depth 10, depth-10 candidates/s, host RSS at depth 10, and allocated/reserved
VRAM at depth 10.  End-to-end wall time through depth 10 is retained as a
secondary check.  One fixed deterministic sufficiently distant Cube4 state is used so the
native goal test does not terminate the run before the tenth layer.

| Hardware | Implementation | Full model card | Beam | Depth-10 frontier/candidates | Depth-10 wall | Depth-10 candidates/s | RSS at depth 10 | VRAM alloc./reserved at depth 10 | Total wall through d10 | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1xT4 | Pilgrim QSearcher | same Q Transformer/checkpoint; PyTorch 2.10.0+cu128, cuDNN 9.10.2, uint8 states, eval batch 384 | 65,536 | 65,536 / 1,572,864 raw parent-action pairs | 2.76685 s | 568,467 raw pairs/s | 1.622 GB peak RSS | 211,883,008 / 297,795,584 B | 32.77 s measured search-step sum (includes 9.19 s first-step initialization) | completed; 10/10 layers, unsolved |
| 1xT4 | MultiGPUBeamSearch (`df534eb`) | Q Transformer: 3,383,064 params, FP16, 96 positions/6 classes, 256/1024, 8 heads, 4 layers, 24 Q outputs; checkpoint SHA-256 above | 65,536 | 65,536 / 1,572,864 raw parent-action pairs | 2.60199 s | 604,484 raw pairs/s | pending RSS telemetry | 385 MiB peak device-used (`nvidia-smi`; allocated/reserved unavailable in C++ runner) | 16.8617 s solver | completed; 10/10 layers, unsolved |
| 1xP100 | Pilgrim QSearcher | same Q Transformer/checkpoint; PyTorch 2.7.1+cu126, cuDNN 9.5.1, native `sm_60`, uint8 states, eval batch 384 | 65,536 | 65,536 / 1,572,864 raw parent-action pairs | 4.87182 s | 322,849 raw pairs/s | 1.407 GB peak RSS | 210,834,432 / 295,698,432 B | 40.58 s measured search-step sum (includes 5.53 s first-step initialization) | completed; 10/10 layers, unsolved |
| 1xP100 | MultiGPUBeamSearch (`df534eb`) | same Q Transformer/checkpoint; PyTorch 2.7.1+cu126, native `sm_60` | 65,536 | 65,536 / 1,572,864 raw parent-action pairs | 5.03888 s | 312,142 raw pairs/s | pending RSS telemetry | 551 MiB peak device-used (`nvidia-smi`; allocated/reserved unavailable in C++ runner) | 31.4240 s solver | completed; 10/10 layers, unsolved |
| 1xT4 | Pilgrim QSearcher | same model/state; PyTorch 2.10.0+cu128; eval batch 384 | 1,048,576 | 1,048,576 / 25,165,824 raw pairs | 46.6109 s | 539,913 raw pairs/s | 1.680 GB | 540,691,456 / 893,386,752 B | 247.76 s measured step sum | completed; 10/10 layers, unsolved |
| 1xT4 | MultiGPUBeamSearch (`df534eb`) | same model/state; FP16; microbatch 384 | 1,048,576 | 1,048,576 / 25,165,824 raw pairs | 40.6211 s | 619,526 raw pairs/s | pending | 735 MiB peak device-used | 209.409 s solver | completed; 10/10 layers, unsolved |
| 1xP100 | Pilgrim QSearcher | same model/state; PyTorch 2.7.1+cu126; eval batch 384 | 1,048,576 | 1,048,576 / 25,165,824 raw pairs | 77.9965 s | 322,653 raw pairs/s | 1.483 GB | 539,643,392 / 893,386,752 B | 409.99 s measured step sum | completed; 10/10 layers, unsolved |
| 1xP100 | MultiGPUBeamSearch (`df534eb`) | same model/state; FP16; microbatch 384, native `sm_60` | 1,048,576 | 1,048,576 / 25,165,824 raw pairs | 75.1501 s | 334,874 raw pairs/s | pending | pending device monitor extraction | 389.028 s solver | completed; 10/10 layers, unsolved |
| 1xT4 | Pilgrim QSearcher | same model/state; PyTorch 2.10.0+cu128, cuDNN 9.10.2; uint8 states; eval batch 384 | 4,194,304 | 4,194,304 / 100,663,296 raw pairs | 198.153 s | 508,007 raw pairs/s | 1.904 GB | 2,099,602,432 / 3,110,076,416 B | 970.24 s measured search-step sum | completed; 10/10 layers, unsolved |
| 1xT4 | MultiGPUBeamSearch (`df534eb`) | same model/state; FP16; microbatch 384; native `sm_75`; CUDA 12.8; no cuDNN | 4,194,304 | 4,194,304 / 100,663,296 raw pairs | 181.282 s | 555,293 raw pairs/s | pending | 1,949 MiB peak device-used (`nvidia-smi`; allocated/reserved unavailable in C++ runner) | 845.612 s solver | completed; 10/10 layers, unsolved |
| 1xP100 | Pilgrim QSearcher | same model/state; PyTorch 2.7.1+cu126, cuDNN 9.5.1; native `sm_60`; uint8 states; eval batch 384 | 4,194,304 | 4,194,304 / 100,663,296 raw pairs | 314.487 s | 320,087 raw pairs/s | 1.709 GB | 2,098,553,856 / 3,110,076,416 B | 1,525.68 s measured search-step sum | completed; 10/10 layers, unsolved |
| 1xP100 | MultiGPUBeamSearch (`df534eb`) | same model/state; FP16; microbatch 384; native `sm_60`; CUDA 12.8; no cuDNN | 4,194,304 | 4,194,304 / 100,663,296 raw pairs | 301.955 s | 333,378 raw pairs/s | pending | 1,949 MiB peak device-used (`nvidia-smi`; allocated/reserved unavailable in C++ runner) | 1,478.38 s solver | completed; 10/10 layers, unsolved |

The paired comparison is strictly single-GPU.  Kaggle's T4 machine exposes two
physical T4 devices, but both processes set `CUDA_VISIBLE_DEVICES=0` before
importing CUDA libraries.  Our production runner uses its existing
`WORLD_SIZE=1`, `RANK=0`, `LOCAL_RANK=0` branch; the public exactly-2xT4 launcher
is not used.  The P100 row uses one visible P100 in the same way.

## Source versions

| Implementation | Commit | Search |
|---|---|---|
| Pilgrim | bundled measured Searcher/QSearcher implementation | PyTorch beam search |
| native CayleyPy | `5b2b53f` (latest working commit supplied by the upstream author) | `CayleyGraph.beam_search`, iterated mode |
| AlphaCube 0.1.6 | `d549835d62f56cf7816b0f02a3281fd6f7ff686f` | beam search |
| DeepCubeA | `919489f14ecbbc80dc1bf1539ac0a462ffaca7c5` | batched weighted A* |
| Ours | current checkout; record SHA with final run | GPU-resident global beam search |

DeepCubeA required one compatibility-only change for NumPy >=1.24: `np.float` was replaced by builtin `float`. Search and model semantics were unchanged.

## RTX 3070: AlphaCube (local GPU now paused)

| Beam | Solver time, s | Nodes | Nodes/s | Solution, HTM | Peak allocated | Peak reserved | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 0.891 | 18,720 | 21,010 | 20 | 258.5 MiB | 268.0 MiB | completed |
| 4,096 | 1.500 | 73,484 | 48,989 | 20 | 327.6 MiB | 532.0 MiB | completed |
| 16,384 | 6.110 | 282,380 | 46,216 | 20 | 603.1 MiB | 1.084 GiB | completed |
| 65,536 | 21.578 | 903,448 | 41,869 | 17 | 1.661 GiB | 5.662 GiB | completed |
| 131,072 | 40.890 | 1,755,416 | 42,931 | 17 | 1.683 GiB | 5.650 GiB | completed |
| 262,144 | 84.359 | 3,459,352 | 41,009 | 17 | 1.735 GiB | 5.632 GiB | completed |
| 524,288 | 184.813 | 6,867,224 | 37,158 | 17 | 1.838 GiB | 7.010 GiB | completed |

The later isolated Docker run used Python 3.10, PyTorch 2.7.1+cu128, CUDA 12.8,
the same AlphaCube 0.1.6 commit/checkpoint/scramble, and completed widths through
`786,432` in 356.321 s (10,124,136 nodes, 28,413 nodes/s).  Width `1,048,576`
terminated with process return code `-9` before writing CUDA telemetry.  This is
a process-failure boundary, not a demonstrated CUDA OOM.  Further local GPU work
is explicitly paused by the user.

AlphaCube internally caps DNN batches at `2^16`, so device memory does not grow
linearly with beam width.

## Kaggle single-GPU baselines

Both kernels were private and accepted only after status `COMPLETE` and local
download of `results.json`, `hardware.json`, and the kernel log.  Kaggle's T4
shape exposed two physical devices; `CUDA_VISIBLE_DEVICES=0` made exactly one
T4 visible.  The P100 required the official PyTorch 2.7.1 cu126 wheel because
Kaggle's default PyTorch 2.10 cu128 binary omitted `sm_60` kernels.

| GPU/runtime | Method and exact parameters | Max completed beam | Time, s | Nodes | Nodes/s | Peak alloc. | Peak reserved | Solution | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tesla T4 15.64 GB; Torch 2.10.0+cu128 | AlphaCube 0.1.6 `large`; MLP 324--4096, 8 Linear--ReLU--BN blocks, 18 outputs; FP32; HTM-18; DNN batch cap 65,536 | 1,572,864 | 461.488 | 19,561,320 | 42,387 | 2.325 GiB | 4.685 GiB | 17 | completed; first measured failure at 1,703,936: host-RAM `SIGKILL (-9)`, not CUDA OOM |
| Tesla P100 PCIe 16 GB; Torch 2.7.1+cu126 | same AlphaCube model/search parameters | 1,572,864 | 768.623 | 19,561,320 | 25,451 | 2.324 GiB | 4.685 GiB | 17 | completed; first measured failure at 1,703,936: host-RAM `SIGKILL (-9)`, not CUDA OOM |
| Tesla T4 15.64 GB; Torch 2.10.0+cu128 | DeepCubeA official cube3 checkpoint; ResNet: 54x6 one-hot, hidden 5000, residual width 1000, 4 blocks, scalar value; weighted A*, weight 0.6; search/NN batch 10,000/10,000 | n/a | 146.474 | 6,134,352 | 41,880 | 1.196 GiB | 1.422 GiB | 21 | completed |
| Tesla P100 PCIe 16 GB; Torch 2.7.1+cu126 | same DeepCubeA model/search parameters | n/a | 122.233 | 6,134,352 | 50,186 | 1.195 GiB | 1.420 GiB | 21 | completed |

The fixed scramble for all four rows is
`D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2`.
DeepCubeA is weighted A*, not beam search; its beam width is therefore marked
not applicable.  The surprising P100/T4 ordering for DeepCubeA is reported as
measured and is not generalized beyond this single fixed instance.

### Why AlphaCube exits with SIGKILL near 1.7M

Source inspection identifies a host-memory amplification in AlphaCube 0.1.6.
`update_candidates` materializes all `18B` children in NumPy before sorting and
cutting back to `B`.  Cube states use `int64` with 54 entries, while paths use
`int8`, scores use `float32`, and `np.argsort` returns `int64` indices.  At
`B=1,703,936`, the expanded layer contains 30,670,848 rows.  The repeated state
array alone is 12.340 GiB; path (at depth 16), score, sort-index, mask, and state
index arrays raise the directly accountable minimum to about 13.282 GiB.  This
excludes the input beam, simultaneous temporaries created by `repeat`, `hstack`,
boolean indexing and `argsort`, Python/PyTorch/model memory, and worker reserve.
The DNN is separately batched at 65,536, explaining why measured CUDA reserved
memory remains far below device capacity when the worker sends SIGKILL.

Private T4/P100 diagnostic kernels have been launched to record cgroup
`memory.events`, RSS, search phase, and VRAM every 0.2 s.  Until those artifacts
are downloaded, the direct source-level mechanism is established but the
specific Kaggle cgroup `oom_kill` counter remains pending.

## RTX 3070: DeepCubeA

| State | Weight | Search batch | NN batch | Time | Status |
|---|---:|---:|---:|---:|---|
| official test state 0 | 0.6 | 10,000 | 10,000 | >300 s | timeout |
| same AlphaCube scramble | 0.6 | 10,000 | 10,000 | >300 s | timeout |

The process was terminated by the benchmark timeout, not by CUDA OOM. DeepCubeA has no beam width; reporting a beam-limit cell for it would be invalid.

## Missing paired cells

- CayleyPy defines the standard `cube_3/3/3_18gensHTM` graph, but commit
  `fbdde24b891d956b9ec905b939f848ecab711978` exposes pretrained predictors only
  for `lrx-16` and `lrx-32`.  Its default Hamming predictor can be benchmarked
  as a different heuristic, but cannot form a matched neural-model cell against
  AlphaCube or DeepCubeA.  We therefore do not substitute LRX weights, a
  picture-cube graph, or Hamming scores into the paired neural table.
- Our current GPU-resident production configuration is Megaminx-specific and
  has no verified adapter/checkpoint for the same Cube3 HTM-18 state encoding
  and predictor used above.  Megaminx throughput is reported separately and is
  never mixed into this Cube3 solver-level table.
- Kaggle T4 capacity boundary: width 1,572,864 completed; widths 1,703,936,
  1,835,008, and 2,097,152 exited with `SIGKILL (-9)` before telemetry.  A
  further refinement at 1,638,400 is not currently running.  These failures
  are classified as host-RAM process kills, not CUDA OOM.
- Kaggle P100 capacity boundary: width 1,572,864 completed; widths 1,703,936,
  1,835,008, and 2,097,152 exited with `SIGKILL (-9)` before telemetry.  The
  measured stable/process-failure interval is therefore
  `(1,572,864, 1,703,936)`.  These failures are not labelled CUDA OOM.
- Maximum beam: use a monotone sweep followed by binary refinement. Report largest completed beam and first OOM/timeout separately.

## Raw evidence

Local AlphaCube JSON is under `test_results/paper_benchmarks/rtx3070_alphacube_docker/`.
Downloaded Kaggle evidence is under `test_results/paper_benchmarks/kaggle_t4_v2/`
and `test_results/paper_benchmarks/kaggle_p100_v3/`; T4 capacity evidence is
under `test_results/paper_benchmarks/kaggle_t4_capacity_v1/`. Earlier local DeepCubeA
timeout attempts remain documented separately; they are superseded for the
paired remote table by the completed T4/P100 runs above.
