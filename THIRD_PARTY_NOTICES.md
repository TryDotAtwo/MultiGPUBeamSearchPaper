# Third-party notices

This file records the provenance and licensing of the third-party Python code
included under `reproduce/runtime/` and in the mechanically generated
`reproduce/data/bundle_payload.zip`. It does not grant a license to original
code, checkpoints, or data elsewhere in this repository.

## Pilgrim runtime

The included minimal Megaminx runtime is based on:

- upstream repository:
  <https://github.com/AnanasClassic/cayleypy-neighbour-model-training>
- pinned upstream commit:
  [`893dbc162a597b8a80d2bcaf92bc2c399fa67dba`](https://github.com/AnanasClassic/cayleypy-neighbour-model-training/commit/893dbc162a597b8a80d2bcaf92bc2c399fa67dba)
- upstream license: Apache License 2.0
- exact license copy:
  [`third_party/AnanasClassic-Apache-2.0-LICENSE.txt`](third_party/AnanasClassic-Apache-2.0-LICENSE.txt)
  (SHA-256 `c9631fda6a6f6f665d96c3f00b7e68d5a5f530bedfe79f14640dc2af2b1d8ea8`)

`pilgrim/parallel.py` is byte-identical to that upstream commit before bundle
assembly. `test.py`, `pilgrim/__init__.py`, `factory.py`, `model.py`,
`qsearcher.py`, `searcher.py`, and `utils.py` are modified derivatives. Each
modified file carries a prominent change notice naming the upstream commit,
as required by Apache-2.0 section 4(b). The upstream license is reproduced in
full in this repository.

The public runtime is intentionally limited to the code imported by the
Megaminx scalar and native 24-output benchmark paths. Optional training,
best-first, and Cube4-reduction modules from the former internal bundle are not
redistributed and are not needed by the four published one-T4 notebooks.

## Dataset and repository license scope

The checkpoint, fixed-state, target, generator, notebook, and wrapper artifacts
are published for research reproduction with exact filenames, model contracts,
training-run identifiers, byte sizes, and SHA-256 hashes in
`reproduce/data/benchmark_manifest.json` and `artifacts/manifest.json`.

The public Kaggle dataset is a separately distributed package: its original
contents are offered under CC BY 4.0 as stated in its metadata, while the
embedded Pilgrim runtime remains under Apache-2.0. The dataset includes both a
component-level notice and the complete Apache license. Original material in
this GitHub repository is governed by [`LICENSE.md`](LICENSE.md); public source
availability alone does not grant a broader license.
