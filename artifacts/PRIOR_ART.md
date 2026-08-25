# Prior-art and novelty audit for the multi-GPU beam-search paper

Date cutoff: **2026-08-25**.  This document is the detailed evidence companion
to the concise related-work sections in the packaged [English](../paper/main.tex)
and [Russian](../paper/main_ru.tex) paper sources.

## Bottom line

The literature does **not** support any of the following broad claims:

- first parallel beam search;
- first distributed beam search;
- first GPU beam search;
- first multi-GPU graph/search workload;
- first distributed GPU generation plus de-duplication plus top-k;
- first multi-GPU global top-k over billions of input items; or
- first GPU system to traverse billions of graph states.

The strongest direct priority statement supported by the primary sources
located in this audit is the following full conjunction:

> We report the first published beam-search system that physically shards one
> logically global cardinality-bounded frontier across multiple GPUs, keeps the
> candidate/frontier hot-data path on device, globally de-duplicates under a
> 128-bit state-key identity model, and returns the exact monolithic reduced-key
> score top-B under the same GPU score quantization and fixed-layout order,
> without shard-local semantic caps. The conjunction is demonstrated at a
> requested retained width of `B_req = 770,883,178`; the exact aligned `B_eff`
> and wall time are unavailable.

This statement applies only to the full conjunction. It does not claim priority
for any component in isolation, and one earlier published system satisfying
every component would falsify it.

## What the headline scale means

The available capacity summary reports a stable eight-A100 request at
`B_req = 770,883,178`.  Runtime alignment rounds the effective retained width
upward, but the exact `B_eff` and raw configuration are unavailable.  This is
a beam request, not a dataset size or cumulative counter.  Megaminx has 24
generators, so a saturated depth contains at least the nominal

`24 * 770,883,178 = 18,501,196,272`

logical child candidates before pruning and duplicate elimination.  The implementation
streams those candidates; it does not materialize all 18.501 billion at once.

The capacity summary records depth 8 and 39,745 MiB/GPU, but not wall
time.  It therefore proves capacity and successful operation of the distributed
frontier request, **not** speed at `B_req = 770,883,178`.  Speed is supported
separately by end-to-end measurements at smaller beams.  The saturated 2xT4
point records `B_req = 82,615,524`, `B_eff = 82,837,504`, and 65.670 s/depth;
normalization by the aligned effective width gives 30.274 million logical
children/s before accounting for duplicates.  The paper must keep those two
proof axes separate.

## Scale taxonomy used in the audit

| Quantity | Definition | May be called a beam width? |
|---|---|---|
| Retained frontier `B` | States surviving global duplicate elimination and selection for the next layer | Yes |
| Raw candidates `mB` | Logical children evaluated from a saturated frontier before de-duplication | No |
| Globally unique candidates | Distinct state keys after cross-worker de-duplication | No |
| Cumulative generated/expanded states | Sum across depths, tasks, or queries | No |
| Graph, index, or dataset size | Objects that exist in the searched structure | No |
| Batched beams/queries | Many independent small searches executed together | No |

This taxonomy excludes several superficially billion-scale systems from a
retained-beam comparison without diminishing their actual contributions.

## Search procedure and limitations

The audit searched title, abstract, full-text, bibliography, and public-code
records in ACM/IEEE-indexed proceedings, AAAI/SoCS/ICAPS, USENIX, ACL/ISCA,
OpenReview, arXiv, institutional repositories, DBLP/DOI metadata, and public
implementations.  Query families included combinations of `beam search`,
`parallel`, `distributed`, `GPU`, `multi-GPU`, `global beam`, `top-k`,
`deduplication`, `implicit graph`, `combinatorial search`, `speech decoding`,
`sequence decoding`, `ANN graph search`, `BFS`, and `selected configuration
interaction`.  Backward and forward citation trails were followed from the
closest systems.

No finite review can establish literal universal exhaustiveness.  The claim is
scoped to the full conjunction and is falsifiable: one earlier published system
satisfying every item above invalidates it.

## Chronological beam-search predecessors

| Year | Work and primary source | Hardware / parallel unit | Largest relevant live width located | Relation to the claim |
|---:|---|---|---:|---|
| 1986 | Anantharaman and Bisiani, [A Hardware Accelerator for Speech Recognition Algorithms](https://doi.org/10.1145/17356.17382) | Two simulated custom parallel architectures | Not reported as a comparable fixed `B` | Establishes hardware-parallel beam search; broad priority claims are false. |
| 1989 | Bisiani, Anantharaman, and Butcher, [BEAM: An Accelerator for Speech Recognition](https://doi.org/10.1109/ICASSP.1989.266544) | Three Weitek processors, shared 8 MB | About 4,000 active states | Built specialized parallel beam decoder, not GPUs or a large combinatorial frontier. |
| 1997 | Phillips and Rogers, [Parallel Speech Recognition](https://www.isca-archive.org/eurospeech_1997/phillips97_eurospeech.pdf) | 4/8/12 CPU processors | Not reported | Up to 6x speedup; independent evidence of mature parallel decoding. |
| 2007 | Wijs and Lisser, [Distributed Extended Beam Search for Quantitative Model Checking](https://doi.org/10.1007/978-3-540-74128-2_11) ([PDF](https://eecs.ceas.uc.edu/~wilseypa/classes/ece975/sp2011/papers/wijs-07.pdf)) | Distributed CPUs | Global beta = 50,000 | A real distributed global beam over an implicit state space.  The reported 135,964,662 is cumulative states, not width. |
| 2008 | Chong et al., [Data-Parallel LVCSR on Graphics Processors](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-69.html) | One GeForce 8800 GTX | Active-state cap 20,000 | GPU data-parallel beam/Viterbi search with intermediate data on device; single GPU and speech semantics. |
| 2011 | Langdon, Yoo, and Harman, [Non-Recursive Beam Search on GPU for Formal Concept Analysis](https://device.report/m/339b5ef9ba9e05edfeb94cac11dab0d413daa9af598c8aed86f74b7823903330.pdf) | GTX 295 / Tesla C2050 | About 1.8M simultaneous GPU operations | GPU search, but overflow remains in a host queue and multiple depths coexist; multi-GPU is explicitly future work. |
| 2022/23 | Frohner et al., [Parallel Beam Search for Combinatorial Optimization](https://doi.org/10.1145/3547276.3548633); [thesis evidence](https://repositum.tuwien.at/bitstream/20.500.12708/187422/1/Frohner%20Nikolaus%20-%202023%20-%20Advancing%20State%20Space%20Search%20for%20Static%20and%20Dynamic...pdf) | 46 CPU threads, 512 GiB; MPI runs independent randomized beams | **20,000,000** | Largest peer-reviewed retained beam located.  One beam across distributed workers is left as future work. |
| 2023 | Takano, [AlphaCube](https://openreview.net/pdf?id=bnBeNFB27b) | One T4 | `2^18 = 262,144` | Neural single-GPU cube beam search. |
| 2024 | Kuroiwa and Beck, [Parallel Beam Search Algorithms for DIDP](https://doi.org/10.1609/aaai.v38i18.30062) | Up to 32 CPU threads | Maximum not reported | SBS globally selects; HDBS retains local `b/k` and is explicitly not necessarily the global best `b`. |
| 2025 | Chervov et al., [A Machine Learning Approach That Beats Large Rubik's Cubes](https://arxiv.org/abs/2502.13266) | Single-GPU agents | `2^24 = 16,777,216` | Very close learned implicit-graph semantics, but each beam resides on one GPU and agents are independent. |
| 2025/26 | Soibelman et al., [CayleyPy RL](https://arxiv.org/abs/2502.18663) | One P100 in the published sweep | `2^20 = 1,048,576` | Direct algorithmic baseline; no multi-GPU global frontier. |
| public code | [HATETRIS public beam search](https://github.com/thecog19/HATETRIS-public-beamsearch) | Multicore CPU/Rayon | **25,000,000** in its documented record history | Largest public retained beam located outside peer-reviewed literature; not GPU or distributed. |

At requested `B_req = 770,883,178`, the reported capacity point is 38.54x the 20M
peer-reviewed CPU beam and 30.84x the 25M public CPU implementation.  These are
width ratios only, not speedups and not comparisons of solution difficulty.

## Closest system-level counterexamples

### QiankunNet-cuSCI / cuNNQS-SCI (HPDC 2026)

Sun et al., [A Fully GPU-Accelerated Framework for High-Performance
Configuration Interaction Selection with Neural Network Quantum
States](https://doi.org/10.1145/3806645.3807583)
([arXiv PDF](https://arxiv.org/pdf/2604.15768)), is the closest architecture
found and must be cited prominently.

It performs CUDA configuration generation, local and distributed sort-based
global de-duplication, neural Transformer scoring, streaming top-K reduction,
and global communication on up to 64 A100 GPUs.  Its N2 strong-scaling run falls
from 825 s on two GPUs to 29 s on 64 GPUs and reports 91.06% efficiency.
Therefore we cannot claim priority for a generic distributed
expand--de-duplicate--infer--top-k pipeline or for scaling such a pipeline to
dozens of GPUs.

The distinction is specific and measurable:

- the work is presented as a selected-configuration-interaction pipeline, not
  as a beam-search system;
- the retained SCI source set is 32,000 (Cr2) or 256,000 (N2);
- its approximately `10^9` figure is raw candidates (`32,000 * 30,000`), not a
  retained top-K frontier;
- the paper states that the full candidate set must be materialized in host
  memory at a global barrier and documents D2H/H2D offload/reload;
Thus QiankunNet invalidates a broad priority claim for the generic distributed
pipeline but not the full beam-search conjunction: one physically sharded
logical beam at a near-billion retained-width request, an on-device
candidate/frontier data plane, global keyed duplicate reduction, and exact
monolithic reduced-key score top-B under the stated semantics.

### GPUAStar (SoCS 2026)

Soltani and Agostinelli, [GPU-Accelerated A* Search with Deep Neural Network
Heuristics](https://ojs.aaai.org/index.php/SOCS/article/download/43083/50643/47188),
keeps learned A* search on one H200.  It extracts `K = 10,000` nodes per
iteration.  Its roughly `6.99e8` count is cumulative generation over 100
tasks, not a live frontier.  Multi-accelerator execution is future work.  It
invalidates `first accelerator-resident learned heuristic search`, but not the
multi-GPU global-beam claim.

### GPUexplore 3.0 (2024)

Wijs and Osama, [GPUexplore 3.0](https://doi.org/10.3389/fhpcp.2024.1285349),
performs exact hash-owned exhaustive state-space exploration across four V100s
and reports models with more than 3.15 billion reachable states.  It has no
score-ranked bounded global top-B.  It invalidates `first multi-GPU
combinatorial state exploration at billion scale`, but its billion is explored
state space, not one retained beam.

### PathWeaver (USENIX ATC 2025)

[PathWeaver](https://www.usenix.org/conference/atc25/presentation/kim) pipelines
multi-GPU approximate nearest-neighbor graph traversal and reports 3.24x
geometric-mean speedup.  It serves many approximate queries over a stored graph;
the output `k` is 10 and the internal queue width is not reported as a large
global beam.  It neither performs exact within-layer state-key de-duplication
nor preserves a single depth-synchronous combinatorial frontier.

### HeteroCPPR (ICCAD 2021)

[HeteroCPPR](https://guozz.cn/publication/heterocppriccad-21/heterocppriccad-21.pdf)
uses one to four RTX 2080 Ti GPUs for iterative path expansion, sort/prune, and
an exact global merge on GPU 0, with top-k up to 100,000.  Its explicit circuit
DAG, analytical score, absence of state de-duplication, and division into
independent depth tasks distinguish it from sharding one implicit-graph beam.

### Exact multi-GPU top-k is not itself novel

[Dr. Top-k](https://arxiv.org/abs/2109.08219) computes local top-k and an exact
global merge on up to 16 V100 GPUs for inputs of `2^30` to `2^33` elements,
with `k = 128`.  This invalidates `first exact multi-GPU top-k over billions of
candidates`; there is no graph expansion, state de-duplication, ancestry, or
large retained `B`.

## GPU and distributed graph/search systems outside beam search

| Work | Reported scale | Why it is not the same object |
|---|---:|---|
| Kishimoto et al., [HDA*](https://arxiv.org/abs/1201.3204) | Up to 2,400 CPU processes and 10.5 TB | Exact hash-distributed best-first search with local OPEN/CLOSED queues, not a deterministic global beam. |
| Edelkamp et al., [GPU BFS for combinatorial search](https://ojs.aaai.org/index.php/ICAPS/article/view/13414) | Single-GPU permutation spaces | Exhaustive BFS/perfect-hash frontiers, no score-ranked top-B. |
| Merrill et al., [Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf) | Four GPUs | BFS with duplicate handling, no ranking/top-B. |
| Zhou and Zeng, [GA*](https://doi.org/10.1609/aaai.v29i1.9367) | One K20c, up to 45x over CPU A* | Many local queues on one GPU, not one global beam. |
| He et al., [DA*](https://www.sciencedirect.com/science/article/pii/S0167739X21001321) | About 3x on four GPUs | Distributed A* with multiple queues/replacement hashing; no deterministic global top-B. |
| Pan et al., [Distributed BFS](https://arxiv.org/abs/1803.03922) | 124 GPUs, scale-33 graph, 259.8 GTEPS | Exact BFS; `2^33` is graph size and maximum frontier width is not reported. |
| Chen and Arvind, [G2Miner](https://www.usenix.org/conference/osdi22/presentation/chen) | Near-linear 1--8 V100 scaling | Graph-pattern enumeration, no score-ranked global beam. |
| [BLEST](https://arxiv.org/abs/2606.05081) | 100 H100; 65.6M vertices / 3.6B edges | Batches many independent exact BFS sources; not one coherent frontier. |

These works demonstrate that multi-GPU graph parallelism and hundred-GPU graph
execution are established.  Our current implementation has measurements only
through eight GPUs.  The current static exchange tables admit at most 128 ranks,
but that is an implementation bound, **not** measured scaling evidence.  The
paper must not claim demonstrated scaling to hundreds of GPUs.

## Sequence, speech, and generative decoding

Sequence beam search is important prior art for GPU kernels and model-parallel
execution, but normally keeps a tiny beam per independent input and performs no
state equality reduction across paths.

| Work | GPU scope / beam | Distinction |
|---|---|---|
| Chen et al., [GPU WFST lattice generation](https://arxiv.org/abs/1804.03243) | GPU speech decoding | Token/lattice recombination, not an implicit combinatorial state beam of enormous width. |
| Junczys-Dowmunt et al., [Marian deployment study](https://aclanthology.org/2016.iwslt-1.5/) | Four GTX 1080; beam 5 | One sentence per GPU: multi-GPU throughput from independent beams, not one sharded beam. |
| Seki et al., [GPU beam-search vectorization](https://arxiv.org/abs/1811.04568) | One K80; beam 20 | Parallelizes hypotheses within one utterance and batches utterances, but is single-GPU. |
| Braun et al., [GPU ASR system](https://arxiv.org/abs/1910.10032) | DGX-1, eight V100; max-active 10,000 | Multi-GPU efficiency comes from independent utterance streams/lanes, not one cross-GPU frontier. |
| Yang et al., [Streaming Parallel Beam Search](https://aclanthology.org/2020.emnlp-main.366/) | One V100; reported beam 50 | Batches sequence hypotheses and streams vocabulary top-k. |
| Kang et al., [Fast and Parallel Decoding for Transducer](https://arxiv.org/abs/2211.00484) | GPU FSA batched beam search | Parallel speech decoding; small per-utterance beam and no huge cross-device state frontier. |
| Chan et al., [Trie-Based Decoding](https://aclanthology.org/2025.emnlp-main.748/) | Four V100s; beam at most 15 in main tables and 30 in a diagnostic | Multi-GPU model execution and prefix/KV sharing, not frontier sharding. |
| [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md) | Multi-GPU/multi-node; examples use small beams such as 4 or 32 | Tensor/model parallel decoding; each request retains its own tiny beam. |
| [FlashTrie](https://arxiv.org/abs/2607.10044) | Single GPU; beam up to 1,000 | Its 800M figure is keyword-library size, not beam width. |
| [D5P4](https://arxiv.org/abs/2603.19146) | Multi-GPU-compatible diffusion selection | Generalized diffusion decoding, not one physically sharded, exact duplicate-eliminated, cardinality-bounded beam at enormous retained width. |

These sources invalidate `first multi-GPU beam search` without threatening the
scoped claim about physically sharding one enormous, globally selected state
frontier.

## Billion-scale ANNS: index size is not beam width

| Work | Billion-scale object | Actual search list / semantic difference |
|---|---|---|
| Johnson et al., [FAISS](https://arxiv.org/abs/1702.08734) | 1B-vector index | Approximate IVF/PQ; output k is generally below 1,000. |
| Manohar, [CMU-CS-25-100 thesis](https://www.csd.cs.cmu.edu/sites/default/files/phd-thesis/CMU-CS-25-100.pdf) | SSNPP-100M is a dataset name | Search beam is 100, not 100M; a concrete false positive. |
| [ParlayANN](https://arxiv.org/abs/2305.04359) | Up to 1B vectors | CPU approximate search lists around 128--400. |
| [BANG](https://ssomesh.github.io/docs/papers/tbd25.pdf) | 1B vectors | Worklist around 152; graph resides in host RAM. |
| [GustANN](https://minhui-xie.github.io/papers/sigmod26-GustANN.pdf) | 1B vectors | List `L = 300`; many independent queries; CPU/SSD participates each iteration. |
| [CAGRA](https://arxiv.org/abs/2308.15136) | Up to 100M in the paper | Single-GPU approximate graph search with a small candidate list and forgettable visited hash. |

## Mathematical and implementation qualifications

The paper proves exact equivalence of its monotone thresholding and final
selection to a monolithic reduced-key score top-B **for the same candidate
multiset, clipped-and-quantized integer score, 128-bit-key model, and fixed
layout order**.  Scalar inference emits child-state floats and output-24 emits
parent-action floats; both are converted to the implementation's integer
`score_key`, and de-duplication defines each key by the lexicographically
minimum `(score_key, payload)`. Streaming can only add keys or decrease an
existing per-key minimum, preserving the monotone-threshold argument. No
shard-local semantic quota approximates the score cut. The
implementation's equal-score boundary order is rank prefix followed
by physical survivor-array order, so changing GPU or shard layout may change
which tied IDs are retained.  Partition-invariant selected IDs are not claimed.

The implementation represents state identity by a 128-bit Zobrist key and does
not carry all 120 logical state bytes through the candidate pipeline.
Consequently, duplicate elimination is exact with respect to that key, while
the correspondence `key equality => state equality` is probabilistic.  The
paper must state this collision model rather than imply collision-resolved full
state comparison.

A stronger future reproducibility package should additionally show:

1. no shard-local semantic top-k before the global cutoff;
2. a partition-invariant complete tie order, if that stronger property is desired;
3. selected-frontier checksum or ID-set tests for 1/2/4/8-GPU partitions after
   such a tie order is implemented;
4. separate raw, globally unique, and retained counts;
5. profiler evidence that candidate/frontier hot data do not traverse host
   memory (compressed ancestry may); and
6. strong scaling at fixed global `B` and weak scaling at fixed `B/GPU`.

## Defensible paper language

### Abstract-sized claim

> We report the first published beam-search system that physically shards one
> logically global cardinality-bounded frontier across multiple GPUs, keeps the
> candidate/frontier hot-data path on device, globally de-duplicates under a
> 128-bit state-key identity model, and returns the exact monolithic reduced-key
> score top-B under the same GPU score quantization and fixed-layout order,
> without shard-local semantic caps. An eight-A100 capacity summary records a
> requested retained width `B_req = 770,883,178`, corresponding to at least
> 18.501 billion nominal logical child candidates in one saturated 24-generator
> depth; the exact aligned `B_eff` and wall time are unavailable. At a separate
> two-T4 point, `B_eff = 82,837,504` yields 30.274 million nominal logical
> children/s end to end.

### Claims that must remain separate

- **Capacity:** stable requested `B_req = 770,883,178` at depth 8 on 8xA100
  40 GB, peak 39,745 MiB/GPU; exact aligned `B_eff`, raw logs, and wall time
  are absent from the supporting summary.
- **End-to-end speed:** measured at smaller frontiers, including 2xT4
  `B_eff = 82,837,504`, 65.670 s per saturated depth, about 30.274M raw logical
  children/s.
- **Matched baseline speed:** only matched-model, matched-workload comparisons
  may be expressed as ratios.
- **Scale-out:** end-to-end speed is measured through two GPUs and capacity is
  reported through eight; the 128-rank code bound is not a scaling experiment.

## Public artifacts and publication gate

- Public implementation: <https://github.com/TryDotAtwo/MultiGPUBeamSearch>.
- Public Megaminx results and 51-move submission family:
  <https://github.com/TryDotAtwo/cayleypy-beam-results>.
- The paper may describe an accompanying public evidence package only after
  stable public URLs resolve to the architecture contract, benchmark matrices,
  available raw outputs and logs, derivations, and replay checks. Local files
  alone do not establish public availability.
- The public availability statement must not imply that missing evidence exists:
  the eight-A100 capacity point has no exact `B_eff`, wall time, raw rank logs,
  runtime configuration, implementation commit, or checkpoint hash.

## Primary-source bibliography and evidence links

- Anantharaman and Bisiani 1986: <https://doi.org/10.1145/17356.17382>
- Bisiani et al. 1989: <https://doi.org/10.1109/ICASSP.1989.266544>
- Phillips and Rogers 1997: <https://www.isca-archive.org/eurospeech_1997/phillips97_eurospeech.pdf>
- Wijs and Lisser 2007: <https://doi.org/10.1007/978-3-540-74128-2_11>
- Chong et al. 2008: <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-69.html>
- Junczys-Dowmunt et al. 2016: <https://aclanthology.org/2016.iwslt-1.5/>
- Seki et al. 2018: <https://arxiv.org/abs/1811.04568>
- Braun et al. 2020: <https://arxiv.org/abs/1910.10032>
- Langdon et al. 2011: <https://device.report/m/339b5ef9ba9e05edfeb94cac11dab0d413daa9af598c8aed86f74b7823903330.pdf>
- Frohner et al. 2022: <https://doi.org/10.1145/3547276.3548633>
- AlphaCube: <https://openreview.net/pdf?id=bnBeNFB27b>
- Kuroiwa and Beck 2024: <https://doi.org/10.1609/aaai.v38i18.30062>
- CayleyPy RL: <https://arxiv.org/abs/2502.18663>
- Large Rubik's cubes: <https://arxiv.org/abs/2502.13266>
- PathWeaver: <https://www.usenix.org/conference/atc25/presentation/kim>
- QiankunNet-cuSCI: <https://doi.org/10.1145/3806645.3807583>
- GPUAStar: <https://ojs.aaai.org/index.php/SOCS/article/download/43083/50643/47188>
- GPUexplore 3.0: <https://doi.org/10.3389/fhpcp.2024.1285349>
- HATETRIS code/history: <https://github.com/thecog19/HATETRIS-public-beamsearch>

## Packaged evidence for our system

- Capacity: [`a100x8_capacity.json`](a100x8_capacity.json).
- 2xT4 E2E: [`beam_run_results.csv`](results/multigpu_2xt4/beam_run_results.csv)
  and [`saturated_depth_summary.md`](results/multigpu_2xt4/saturated_depth_summary.md).
- 2xT4 tuning: the production-sweep and Stream1 tables in
  [`saturated_depth_summary.md`](results/multigpu_2xt4/saturated_depth_summary.md).
- Single-GPU comparison matrix: [`BENCHMARK_MATRIX.md`](BENCHMARK_MATRIX.md),
  with its packaged JSON/JSONL/CSV/log inputs under [`results/`](results/).
- Algorithm contract: the five-stream proof in the packaged
  [English paper source](../paper/main.tex) and the pinned public
  [`ARCHITECTURE_NEED.md`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/a1db0e6d9bb5458c8a842b37dfa99572d3025667/ARCHITECTURE_NEED.md).
