# Exact high-throughput beam search with one global frontier across many GPUs

This repository is the public evidence and reproduction package for the paper
**“One Global Beam Across Many GPUs: Exact High-Throughput Beam Search at
Billion-State Frontier Scale.”** The CUDA/C++ implementation lives in the
separate [MultiGPUBeamSearch repository](https://github.com/TryDotAtwo/MultiGPUBeamSearch).

The system physically shards one logical beam across GPUs. Its candidate and
frontier hot-data arrays stay on the devices while ranks jointly perform global
128-bit-key duplicate reduction and exact monolithic reduced-key score
top-\(B\) selection under the stated score quantization, fixed-layout tie order,
and identity model. There are no shard-local semantic caps.

## Headline evidence

| Evidence | Hardware | Retained frontier | Work per saturated depth | Measured execution |
|---|---|---:|---:|---:|
| Capacity and throughput | 8× NVIDIA H200 | requested 2,900,000,000; effective 2,900,361,216 | 69,608,669,184 nominal parent-action pairs at 24 generators | completed depth 8 in 931.266 s; derived 74.746 million nominal pairs/s |
| Capacity | 8× NVIDIA A100 40GB | requested \(B_{req}=770,883,178\); \(B_{eff}\ge B_{req}\) | at least 18,501,196,272 nominal children at 24 generators | wall time and exact \(B_{eff}\) unavailable |
| Throughput | 2× NVIDIA T4 | \(B_{eff}=82,837,504\) | 1,988,100,096 nominal children | 65.670 s per saturated depth; 30.274 million logical children/s |

The eight-H200 Cube4 run measures a completed search depth, not a solved puzzle;
its nominal pair rate is derived from retained width times 24 generators, not
a native distinct-state counter. Its raw rank logs and device telemetry are in
[`artifacts/results/h200_8x_cube4`](artifacts/results/h200_8x_cube4/EVIDENCE.md).
The A100 record establishes capacity only. The two-T4 Megaminx result is a
different workload, so these points do not establish strong-scaling speedup.
The implementation permits at most 128 ranks; execution beyond eight GPUs
has not been measured.

## What is public here

- `paper/`: English and Russian LaTeX sources, bibliography, and built PDFs.
- `artifacts/`: benchmark matrix, claim ledger, prior-art audit, compact raw
  JSON/JSONL/CSV results, and the explicit capacity record.
- `reproduce/notebooks/`: self-contained Kaggle notebooks for the final one-T4
  Pilgrim and MultiGPU capacity boundaries.
- `reproduce/scripts/`: pinned baseline runners and replay/verification
  utilities.
- `reproduce/runtime/`: the minimal source-visible Pilgrim Megaminx runtime,
  with pinned upstream provenance and Apache-2.0 change notices.
- `reproduce/data/`: fixed Megaminx state, generators, model metadata, and
  benchmark manifest.
- [Release v1.0.0](https://github.com/TryDotAtwo/MultiGPUBeamSearchPaper/releases/tag/v1.0.0):
  the two matched Megaminx checkpoints, the matched Cube4 checkpoint, and the
  two paper PDFs.

The complete file and SHA-256 inventory is in
[`artifacts/checksums.sha256`](artifacts/checksums.sha256). Large checkpoints
are release assets rather than Git blobs; their download-directory checksum
file is [`artifacts/release-assets.sha256`](artifacts/release-assets.sha256).

## Reproduce

Follow [`REPRODUCE.md`](REPRODUCE.md). The paper benchmarks pin these public
implementation revisions:

- measured one-T4 path: [`a1db0e6d9bb5458c8a842b37dfa99572d3025667`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/a1db0e6d9bb5458c8a842b37dfa99572d3025667);
- matched Cube4 path: [`df534ebb1ef624abd42443211778b5588916d6d2`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/df534ebb1ef624abd42443211778b5588916d6d2);
- cluster/capacity launcher snapshot: [`2d5f978fcfc1f018813fc49e9d9aeec32ea5e22c`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/2d5f978fcfc1f018813fc49e9d9aeec32ea5e22c);
- native CayleyPy comparison: [`cayleypy@5b2b53f`](https://github.com/cayleypy/cayleypy/commit/5b2b53f).
- DeepCubeA baseline: [`919489f14ecbbc80dc1bf1539ac0a462ffaca7c5`](https://github.com/forestagostinelli/DeepCubeA/commit/919489f14ecbbc80dc1bf1539ac0a462ffaca7c5).

Public Kaggle package and importable notebook sources:

- [self-contained benchmark dataset](https://www.kaggle.com/datasets/trydotatwo/gpu-beam-search-paper-1xt4-reviewer-notebooks)
- [`01_pilgrim_boundary_1xt4.ipynb`](reproduce/notebooks/01_pilgrim_boundary_1xt4.ipynb)
- [`02_multigpu_boundary_1xt4.ipynb`](reproduce/notebooks/02_multigpu_boundary_1xt4.ipynb)
- [`03_pilgrim_broad_sweep_1xt4.ipynb`](reproduce/notebooks/03_pilgrim_broad_sweep_1xt4.ipynb)
- [`04_multigpu_broad_sweep_1xt4.ipynb`](reproduce/notebooks/04_multigpu_broad_sweep_1xt4.ipynb)

The repository does not depend on the visibility of an owner's saved Kaggle
kernel page. Import an included notebook into Kaggle, attach the public dataset,
enable **Internet**, and use the normal **Save Version -> Save & Run All**
workflow. Each runner downloads only its named release checkpoint when the
asset is not attached and rejects a SHA-256 mismatch.

## Exactness scope

“Exact” means exact equivalence to a monolithic reduced-key score top-\(B\)
under the same GPU round-to-nearest-even score quantization, fixed rank/shard
layout tie order, and 128-bit state-key identity model. Duplicate elimination
is exact with respect to that key. It is not a proof that distinct full states
can never collide in 128 bits, and it is not a claim of layout-independent tie
ordering.

## License and provenance

This repository does not grant a general license for its original contents;
see [`LICENSE.md`](LICENSE.md). The separately distributed Kaggle reproduction
dataset publishes its original contents under CC BY 4.0, while its embedded
Pilgrim runtime remains under Apache-2.0. The linked implementation repository
has its own copyright status. Component-level provenance and redistribution
terms are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
