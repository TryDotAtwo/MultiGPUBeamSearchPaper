# Megaminx superflip public-result audit — 2026-08-25

## Metric

The compared paths use the same 24-generator Megaminx move set: one signed
single-face 72-degree turn per token.  This is the metric used by the local
`puzzle_info.json` and by the submitted path.

## Public chronology located

| Length | Public evidence |
|---:|---|
| 55 | [SpeedSolving post 1696242](https://www.speedsolving.com/threads/development-of-a-megaminx-solver.93202/post-1696242) |
| 54 | [SpeedSolving post 1696217](https://www.speedsolving.com/threads/development-of-a-megaminx-solver.93202/post-1696217) |
| 53 | [SpeedSolving post 1700151](https://www.speedsolving.com/threads/development-of-a-megaminx-solver.93202/post-1700151) |
| 51 | [SpeedSolving post 1705451](https://www.speedsolving.com/threads/development-of-a-megaminx-solver.93202/post-1705451) |

The immediately preceding public best located by this audit is therefore 53,
not 55.  No shorter same-metric public solution was located by the cutoff, but
there is no formal registry and no optimality proof.  The defensible wording is
`shortest public same-metric solution located through 2026-08-25`, not an
unqualified world-record or optimal-solution claim.

## Independent local replay

- Submission: `real-state-rank-parity-clone/data/submissions/submission_69749.csv`
- Submission SHA-256:
  `B26D6ADBD78D752D7A52F56990C2AF6CD711D7DE1DBCC3D9373BD623C9A5D8F9`
- Puzzle: `initial_state_id=0`
- Move count: `51`
- Path SHA-256:
  `18435e4675503bbfd92b309b905e993f205dc6162f834ddf103f83491d7f95aa`
- Replay rule: `child[p] = parent[generator[move][p]]`
- Target: `central_state` from
  `real-state-rank-parity-clone/data/puzzle_info.json`
- Final differences from target: `0`
- Result: `replay_valid=true`

The replay validates the path itself.  It does not reconstruct the missing
near-billion run wall-time/hardware manifest and does not prove optimality.
