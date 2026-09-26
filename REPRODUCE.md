# Reproduction guide

## Fastest path: Kaggle Save Version

1. Download one of the `.ipynb` files under `reproduce/notebooks/`, then use
   **File -> Import Notebook** in Kaggle.
2. Attach the public
   [self-contained benchmark dataset](https://www.kaggle.com/datasets/trydotatwo/gpu-beam-search-paper-1xt4-reviewer-notebooks).
3. Select one NVIDIA T4 accelerator, enable **Internet**, and run
   **Save Version -> Save & Run All**. MultiGPUBeamSearch, CayleyPy, and the
   baseline runners clone pinned public revisions; all runners may download
   their exact release checkpoint when it is not attached.
4. The notebooks locate the
   manifest by content and fail closed if zero or multiple benchmark bundles
   are visible.
5. Download the compact outputs (`summary.json`, `hardware.json`,
   `results.csv`, and `results.jsonl`) from the completed version.
6. Compare them with the corresponding directory under `artifacts/results/`.

The protocol is one visible T4, one fixed Megaminx state, depth 8, 24
generators, and `history_depth=0` where the native interface exposes it.
Pilgrim and MultiGPUBeamSearch are measured with native scalar and 24-output
heads. Native CayleyPy supports only one scalar score per state, so an
output-24 CayleyPy cell is intentionally not fabricated.

## Models

The notebooks download the required named checkpoint automatically when it is
not attached. For a manual or offline run, download the three checkpoint assets from
[Release v1.0.0](https://github.com/TryDotAtwo/MultiGPUBeamSearchPaper/releases/tag/v1.0.0)
and verify:

| File | Architecture | Parameters | SHA-256 |
|---|---|---:|---|
| `weights_megaminx2048_512_8_e4000.pth` | 120×120 position-class input → 2048 → 512 → 8 residual blocks of 512→512 → scalar | 34,766,849 | `7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759` |
| `p900-t000-q-sym_1777988767_best.pth` | 120×120 position-class input → 1536 → 512 → 2 residual blocks of 512→512 → 24 outputs | 23,978,008 | `e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670` |
| `cube4_piece_transformer_24out.pth` | 96 six-class state positions → 56 piece embeddings/projections + position/type embeddings → CLS (57 tokens) → input LN → 4×[pre-LN, 8-head attention (width 256), residual, pre-LN, 256→1024→256 ReLU FFN, residual] → CLS pooling → output LN → Linear(256,24) | 3,383,064 | `58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164` |

Both members of each matched comparison use the same byte-identical
checkpoint. Scalar and output-24 rows are separate model contracts. The Cube4
checkpoint is used by the matched Cube4 Pilgrim/MultiGPU comparison. The 8×H200 run uses the same model architecture; its exported-weight identity is recorded separately in the H200 evidence, not inferred from this checkpoint-file SHA.

## Native CUDA implementation

Clone the implementation at the revision corresponding to the experiment:

```bash
git clone https://github.com/TryDotAtwo/MultiGPUBeamSearch.git
cd MultiGPUBeamSearch
git checkout a1db0e6d9bb5458c8a842b37dfa99572d3025667
```

The exact run inputs and public artifact hashes are recorded in
`reproduce/data/benchmark_manifest.json` and `artifacts/manifest.json`.
Do not silently substitute a scorer, output head, move set, state, history
policy, or stopping rule.

## Eight-A100 capacity launcher

The summary-backed capacity result itself is intentionally reported only as
`B_req=770,883,178`, `B_eff>=B_req`, depth 8 stable, and 39,745 MiB/GPU. Its
raw rank logs, exact effective width, wall time, runtime configuration,
implementation commit, and checkpoint hash were not retained; no missing
field is reconstructed.

For a current portable cluster launch, use the separately published launcher
at revision `2d5f978fcfc1f018813fc49e9d9aeec32ea5e22c`:

```bash
git checkout 2d5f978fcfc1f018813fc49e9d9aeec32ea5e22c
chmod +x hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh
CUTLASS_DIR=/path/to/cutlass \
NINJA_VENV_DIR=/path/to/ninja-venv \
PUZZLE_ID=992 \
sbatch -p YOUR_PARTITION hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh
```

That launcher targets an eight-A100 **80GB** node and defaults to a 1.4B
requested beam. It is a published reproduction launcher, not the missing raw
configuration of the earlier eight-A100 40GB capacity record.

## Verification policy

- `RUNNING` or `QUEUED` is never a result.
- Keep completed, CUDA OOM, host-memory kill, timeout, incompatible, and not
  tested as distinct statuses.
- Primary seconds/step is total clean search wall divided by the evidenced
  number of native completed steps. For the fixed depth-8 Megaminx sweeps this
  is total wall divided by 8.
- Never infer nodes/s when a native counter is absent.
- Never compare scalar and 24-output model timings as an algorithm-only speedup.
