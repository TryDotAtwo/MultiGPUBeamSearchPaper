# Scalar MultiGPU timing provenance

The paper deliberately keeps C1 (comparison) and C2 (boundary) separate. All timings below concern native output_dim=1, depth 8, the shared eight-residual-block checkpoint SHA `7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759`. No averaging or replacement across runs is performed.

## C1: historical comparison

Raw inventory and runtime metadata: `results/multigpu_scalar_comparison_c1/{results.csv,results.jsonl,summary.json,hardware.json}`. The adjacent per-point `combined.log` files contain the solver's `puzzle_solved=0 ... seconds=...` timer. At B=25,165,824 this is 2416.15 seconds, printed in the paper as 2,416.150; dividing by 8 gives 302.01875 seconds/step. Trailing zeros are formatting, not additional measurement precision. The wrapper's `orchestrator_wall_s` is not this timer. Failed historical points remain raw evidence, not the current capacity boundary.

## C2: later boundary series

The mixed-head raw inventory `results/multigpu_output24_t4/results.csv` includes the scalar B=25,165,824 record: `search_wall_s=3027.672875526`, `seconds_per_depth=378.45910944075`, device-used peak 12267 MiB. Its original per-point log is retained under `results/multigpu_scalar_boundary_c2/`; internal solver time is 3023.58 seconds. The paper retains the recorded wrapper time 3,027.673, not the inner timer. The worker in `reproduce/workers/multigpu_beam_search.py` times the executable, including initialization, outside build/export; it does not time only the last depth.

Additional upper-bound points are in `results/multigpu_scalar_t4/{results.csv,results.jsonl,summary.json,hardware.json}`. These are not C1 reruns. Complete and failure records must remain distinct; a failure duration is not successful throughput.

Thus total/completed-steps is the shared normalization, not evidence of identical timer scopes. C1 and C2 should not be pooled into one performance curve or used interchangeably. Implementation/runtime/profile metadata remain in the raw records; no unrecorded setting or counter is inferred.
