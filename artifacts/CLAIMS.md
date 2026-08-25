# Claim evidence ledger

This ledger separates measured facts, mathematical scope, literature findings,
and wording allowed in the [English](../paper/main.tex) and
[Russian](../paper/main_ru.tex) paper sources. The detailed literature review
is packaged as [PRIOR_ART.md](PRIOR_ART.md).

## Central systems claim

| Claim component | Evidence | Status and allowed wording |
|---|---|---|
| One logical beam is physically distributed across GPUs | The five-stream contract and proof are included in the [English paper source](../paper/main.tex); the pinned public implementation revision contains the corresponding [architecture contract](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/ARCHITECTURE_NEED.md). Streams 3--5 route each 128-bit key to one owner, perform cross-rank exchange, globally reduce duplicate keys, and make one global score cut. Stream 4 has no shard-local top-k or semantic shard cap. | Verified architecture. Say **one sharded global beam**, not independent beams, replicas, batched queries, or a surrogate local top-k. |
| Candidate/frontier hot data remain on GPU | Candidate metadata, survivor shards, histograms, exchange buffers, and materialized next-frontier states remain on devices. The pinned public [`cuda/dispatcher.cu`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/cuda/dispatcher.cu) copies per-rank less/equal count vectors and compact 32-byte selected-candidate metadata used for ancestry to the host. | Say **GPU-resident candidate/frontier data plane**. Do not say the CPU receives nothing except history; it receives O(world_size) control counts and compact selected metadata, but not full state/frontier arrays. |
| Exact global top-B | Scalar scoring starts from a child-state float; output-24 starts from parent-action floats. Both paths rank the same clipped and quantized integer `score_key` used by the implementation. The production GPU conversion uses `rintf` in its default round-to-nearest, ties-to-even mode. De-duplication retains the lexicographically minimum `(score_key, payload)` for each key. As candidates stream in, key domains only grow and existing per-key minima only decrease, so the B-th reduced score is monotone. Dropping only candidate scores greater than T_j cannot remove a final selected per-key minimum. Stream 4 applies no local semantic cap. The final cut returns `min(B_eff, total_available)` keys: exactly B_eff only when the de-duplicated layer is saturated. | Exact global **reduced-key score** top-B for the same candidate multiset, GPU score-key conversion, 128-bit-key model, and fixed layout. Covers both scalar and output-24 paths. This is not approximate sharded selection and is not a claim of exact ordering under the unquantized floats. The theorem's monolithic reference consumes those same GPU-defined integer keys; it does not rely on the separate CPU `std::lround` helper. |
| Equal-score boundary | The pinned public [`cuda/dispatcher.cu`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/cuda/dispatcher.cu) assigns equal-score room by rank prefix; [`cuda/threshold.cu`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/cuda/threshold.cu) takes candidates in physical survivor-array order. Owner and shard depend on the hash and layout. | Deterministic for a fixed rank/shard layout. Changing the layout can change equal-score boundary IDs. Do **not** claim partition-invariant selected IDs or a global hash-sorted tie order. |
| State identity | The candidate path uses one logical 128-bit Zobrist key and does not compare all 120 logical state bytes after a key match. | Exact under the stated 128-bit-key identity model. Residual hash-collision risk must remain explicit. |

## Scale and performance

| Measurement | Source evidence | Exact interpretation |
|---|---|---|
| Eight-A100 capacity | Packaged summary [`a100x8_capacity.json`](a100x8_capacity.json): job 33363, three Megaminx instances, requested beam 770,883,178, depth 8 stable, peak 39,745 MiB/GPU. | Summary-backed capacity result. Runtime rounds B_req upward to B_eff, but raw rank logs/config, exact B_eff, commit, checkpoint SHA, and wall time are not available in the supporting capacity record. Write B_req=770,883,178 and B_eff >= B_req, not an exact retained runtime count or speed measurement. |
| Nominal raw layer at A100 capacity request | Exact arithmetic: 24 * 770,883,178 = 18,501,196,272. | At least 18.501B nominal logical child candidates for a saturated 24-generator depth because alignment rounds effective width upward. This is a candidate count, not 18.501B distinct neural forward invocations. |
| Two-T4 end-to-end point | Packaged [`beam_run_results.csv`](results/multigpu_2xt4/beam_run_results.csv): requested 82,615,524, effective 82,837,504, solve 350.402 s; packaged [`saturated_depth_summary.md`](results/multigpu_2xt4/saturated_depth_summary.md): saturated depths 7--10 mean 65.670 s. | 24 * 82,837,504 = 1,988,100,096 logical children and 30.274M/s when normalized by effective saturated width. The total solve uses the valid radius-4 solved-neighborhood shortcut and must be labelled accordingly. Saturated-depth time is summary-backed; the original raw rank logs are absent. |
| Scale-out extent | Production execution is measured at two GPUs; capacity is reported at eight GPUs. The pinned public [`cuda/dispatcher.cu`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/cuda/dispatcher.cu) statically permits at most 128 ranks. | Say measured speed through two GPUs and capacity through eight. The 128-rank value is an implementation bound, not evidence of scaling to 100+ GPUs. |
| Relative speed | Matched scalar and output-24 rows and the Cube4 comparison are retained in the packaged [benchmark matrix](BENCHMARK_MATRIX.md). | Claim only the explicitly matched ratios. Do not claim global fastest or superiority without a matched workload/model/hardware result. |

## Novelty audit

The primary-source audit falsifies broad phrases such as
first parallel beam search, first GPU beam search, first multi-GPU beam search,
first distributed expand-deduplicate-top-k, and first billion-scale graph
search.

The closest systems include:

- Wijs--Lisser distributed global CPU beam search;
- Frohner et al. shared-memory beams through 20M;
- CayleyPy single-GPU learned combinatorial beams through 2^24;
- multi-GPU speech/language decoding with independent tiny beams or
  model/utterance parallelism;
- GPUexplore exhaustive multi-GPU traversal of more than 3.15B states;
- QiankunNet-cuSCI distributed generation, global de-duplication, neural
  scoring, and top-K on 64 A100s, but with retained sources at 32K/256K,
  full-candidate host staging, and no presentation as a beam-search system.

The defensible priority sentence is:

> We report the first published beam-search system that physically shards one
> logically global cardinality-bounded frontier across multiple GPUs, retains
> the candidate/frontier hot-data path on device, globally de-duplicates under
> a 128-bit state-key identity model, and returns the exact monolithic
> reduced-key score top-B under the same GPU score quantization and fixed-layout
> order without shard-local semantic caps. The full conjunction is demonstrated
> at a requested retained width of B_req=770,883,178; the exact aligned B_eff
> and wall time are unavailable. With 24 generators, the request corresponds to
> at least 18,501,196,272 nominal logical children in a saturated depth.

The downstream Streams 2--5 contract is scorer-agnostic provided Stream 1
emits deterministic quantized `score_key` values.  The shipped production
Stream 1 backends and the reported benchmarks are neural; supporting a
hand-engineered heuristic requires implementing that Stream 1 contract rather
than merely changing a runtime flag.

This first-published statement applies only to the full conjunction above. Any
earlier published system satisfying every component falsifies it; no priority
is claimed for the individual components.

## Public artifacts and publication gate

- The implementation repository is public at
  https://github.com/TryDotAtwo/MultiGPUBeamSearch.
- The public Megaminx result collection, including the 51-move submission
  family, is at https://github.com/TryDotAtwo/cayleypy-beam-results.
- The paper may say that an accompanying public evidence package provides the
  architecture contract, benchmark matrices, available raw outputs and logs,
  derivations, and replay checks only after stable public URLs resolve to those
  files. The local paper/evidence files are not themselves proof of publication.
- Public-artifact wording must retain the missing-data caveat: the eight-A100
  capacity point is represented by a summary, while its exact B_eff, wall time,
  raw rank logs, runtime configuration, implementation commit, and checkpoint
  hash are unavailable.

## Megaminx superflip

| Evidence | Interpretation |
|---|---|
| Packaged [`superflip/PUBLIC_RECORD_AUDIT.md`](superflip/PUBLIC_RECORD_AUDIT.md) records the public 55 -> 54 -> 53 -> 51 chronology. | The immediately preceding located public result is 53, not 55. |
| Independent CPU replay of packaged [`superflip/submission_69749.csv`](superflip/submission_69749.csv) applies the associated generator permutations and reaches the exact centre after 51 moves; file and path SHA-256 values are recorded in the audit. | Replay-valid 51-move path in the same 24 signed 72-degree face-turn metric. |
| No shorter same-metric public path was located in the public-record review; there is no formal registry or optimality proof. | Say **shortest public same-metric solution located**, not unqualified world record or optimal solution. |
| The original run manifest is absent. | Do not infer its hardware, checkpoint, wall-time provenance, or identity with the eight-A100 capacity run. |
