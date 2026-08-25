
import hashlib, json, sys, time
from pathlib import Path
import torch
from cayleypy import CayleyGraphDef, CayleyGraph, Predictor

width, max_steps, output = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
bundle = next(Path('/kaggle/input').rglob('benchmark_manifest.json')).parent
runtime = bundle / 'pilgrim_runtime'
sys.path.insert(0, str(runtime))
from pilgrim.factory import build_model_from_info
spec = json.loads((runtime / 'generators' / 'p900.json').read_text())
target = torch.load(runtime / 'targets' / 'p900-t000.pt', map_location='cpu', weights_only=True).numpy()
start = torch.load(runtime / 'paper_state.pt', map_location='cpu', weights_only=False).to(torch.int8).numpy()
definition = CayleyGraphDef.create(generators=spec['actions'], generator_names=spec['names'], central_state=target)
graph = CayleyGraph(definition, device='cuda', dtype=torch.int8, bit_encoding_width=None,
                    batch_size=2**16, hash_chunk_size=2**16)
info = json.loads((runtime / 'model_output1.json').read_text())
weights = next(Path('/kaggle/input/models').rglob('weights_megaminx2048_512_8_e4000.pth'))
sha = hashlib.sha256(weights.read_bytes()).hexdigest()
assert sha == '7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759'
model = build_model_from_info(info, num_classes=120, state_size=120, output_dim=1)
model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False), strict=True)
model.eval().half().cuda(); model.dtype = torch.float16
probe = model(torch.as_tensor(start, device='cuda').reshape(1, -1))
assert tuple(probe.shape) == (1,)
predictor = Predictor(graph, model)

counter = {"expanded_frontier_states": 0, "generated_candidates": 0, "calls": 0}
original_get_neighbors = graph.get_neighbors
def counted_get_neighbors(states):
    counter["expanded_frontier_states"] += int(states.shape[0])
    counter["generated_candidates"] += int(states.shape[0]) * definition.n_generators
    counter["calls"] += 1
    return original_get_neighbors(states)
graph.get_neighbors = counted_get_neighbors

torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
try:
    result = graph.beam_search(
        start_state=start, beam_mode="iterated", predictor=predictor,
        beam_width=width, max_steps=max_steps, history_depth=0,
        hashed_neigbourhood=0, memory_cleanup=False,
        return_path=False, path_device='cuda', verbose=0,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    row = {
        "beam_width": width, "status": "completed", "wall_s": wall,
        "path_found": result.path_found, "path_length": result.path_length,
        "checkpoint_sha256": sha, "output_dim": 1,
        "steps_executed": counter["calls"], **counter,
        "generated_candidates_per_s": counter["generated_candidates"] / wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
except torch.OutOfMemoryError as exc:
    row = {"beam_width": width, "status": "cuda_oom", "error": str(exc), **counter,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
except RuntimeError as exc:
    message = str(exc)
    status = "cuda_oom" if "out of memory" in message.lower() else "runtime_error"
    row = {"beam_width": width, "status": status, "error": message, **counter,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
output.write_text(json.dumps(row, indent=2))
