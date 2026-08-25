import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True, capture_output=True, check=True,
).stdout.splitlines()[0]
if "P100" in gpu_name and os.environ.get("P100_TORCH_BOOTSTRAPPED") != "1":
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126",
    ], check=True)
    os.environ["P100_TORCH_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

import torch

ROOT = Path("/kaggle/working/cayleypy_capacity")
ROOT.mkdir(parents=True, exist_ok=True)
COMMIT = "5b2b53f"
MODEL_SHA256 = "7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759"
MIN_WIDTH = 2**10
MAX_WIDTH = 2**25
BOUNDARY_ALIGNMENT = 2**20
BOUNDARY_SAFETY = 0.95
MAX_STEPS = 8


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    repo = Path("/tmp/cayleypy")
    subprocess.run(["git", "clone", "-q", "https://github.com/cayleypy/cayleypy.git", str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", COMMIT], cwd=repo, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)], check=True)

    worker = ROOT / "one_width.py"
    worker.write_text(r'''
import hashlib, json, shutil, sys, time
from pathlib import Path
from urllib.request import urlretrieve
import torch
from cayleypy import CayleyGraphDef, CayleyGraph, Predictor

width, max_steps, output = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
runtime = None
for manifest_path in Path('/kaggle/input').rglob('benchmark_manifest.json'):
    for candidate in (manifest_path.parent / 'pilgrim_runtime',
                      manifest_path.parent / 'bundle_payload' / 'pilgrim_runtime'):
        if (candidate / 'pilgrim' / 'factory.py').is_file():
            runtime = candidate
            break
    if runtime is not None:
        break
if runtime is None:
    archives = list(Path('/kaggle/input').rglob('bundle_payload.zip'))
    if len(archives) == 1:
        unpacked = Path('/tmp/cayleypy_verified_bundle')
        if unpacked.exists():
            shutil.rmtree(unpacked)
        shutil.unpack_archive(archives[0], unpacked)
        candidate = unpacked / 'pilgrim_runtime'
        if (candidate / 'pilgrim' / 'factory.py').is_file():
            runtime = candidate
if runtime is None:
    raise RuntimeError('attached benchmark bundle does not contain a valid pilgrim_runtime')
sys.path.insert(0, str(runtime))
from pilgrim.factory import build_model_from_info
spec = json.loads((runtime / 'generators' / 'p900.json').read_text())
target = torch.load(runtime / 'targets' / 'p900-t000.pt', map_location='cpu', weights_only=True).numpy()
state_path = runtime / 'paper_state.pt'
expected_state_sha = '8cbf1d728c44ca820a59990e768da78fc5ebcb6f3105a0d822c3808bfda6cc75'
state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
if state_sha != expected_state_sha:
    raise RuntimeError(f'paper_state.pt SHA256 mismatch: {state_sha} != {expected_state_sha}')
start = torch.load(state_path, map_location='cpu', weights_only=True).to(torch.int8).numpy()
definition = CayleyGraphDef.create(generators=spec['actions'], generator_names=spec['names'], central_state=target)
graph = CayleyGraph(definition, device='cuda', dtype=torch.int8, bit_encoding_width=None,
                    batch_size=2**16, hash_chunk_size=2**16)
info = json.loads((runtime / 'model_output1.json').read_text())
checkpoint_name = 'weights_megaminx2048_512_8_e4000.pth'
expected_sha = '7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759'
matches = list(Path('/kaggle/input').rglob(checkpoint_name))
if len(matches) > 1:
    raise RuntimeError(f'expected at most one attached scalar checkpoint: {matches}')
if matches:
    weights = matches[0]
else:
    cache = Path('/tmp/paper_checkpoint_assets'); cache.mkdir(parents=True, exist_ok=True)
    weights = cache / checkpoint_name
    if not weights.is_file():
        urlretrieve(f'https://github.com/TryDotAtwo/MultiGPUBeamSearchPaper/releases/download/v1.0.0/{checkpoint_name}', weights)
sha = hashlib.sha256(weights.read_bytes()).hexdigest()
if sha != expected_sha:
    raise RuntimeError(f'checkpoint SHA256 mismatch: {sha} != {expected_sha}')
model = build_model_from_info(info, num_classes=120, state_size=120, output_dim=1)
model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
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
''')

    hardware = {
        "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "total_vram_bytes": torch.cuda.mem_get_info()[1],
        "python": sys.version, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "compiled_cuda_arches": torch.cuda.get_arch_list(),
        "implementation": "native CayleyPy CayleyGraph.beam_search", "commit": COMMIT,
        "graph": "Megaminx p900-t000", "move_set": "24 generators from p900.json",
        "predictor": "shared Pilgrim MLP scalar checkpoint",
        "checkpoint_sha256": MODEL_SHA256, "output_dim": 1,
        "model_architecture": "Pilgrim MLP hd1=2048 hd2=512 nrd=8",
        "state_encoding": "unpacked int8", "dtype": "int8 states / fp16 model",
        "beam_mode": "iterated", "history_depth": 0, "return_path": False,
        "max_steps": MAX_STEPS, "fixed_state_sha256": "8cbf1d728c44ca820a59990e768da78fc5ebcb6f3105a0d822c3808bfda6cc75",
        "capacity_search": {
            "minimum_width": MIN_WIDTH, "maximum_width": MAX_WIDTH,
            "grid": [2**power for power in range(10, 26)],
            "confirmation_alignment": BOUNDARY_ALIGNMENT,
            "policy": "powers of two to first failure, then one memory-trend confirmation below the estimated boundary",
        },
        "timeout_policy": "no per-width timeout; Kaggle platform limit only",
    }
    rows = []

    def save(summary=None):
        payload = {"hardware": hardware, "rows": rows}
        if summary is not None:
            payload["capacity_summary"] = summary
        (ROOT / "results.json").write_text(json.dumps(payload, indent=2))

    def run_width(width, attempt=1, phase="search"):
        suffix = "" if attempt == 1 else f"_attempt{attempt}"
        output = ROOT / f"width_{width}{suffix}.json"
        log = ROOT / f"width_{width}{suffix}.log"
        with log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                [sys.executable, str(worker), str(width), str(MAX_STEPS), str(output)],
                text=True, stdout=log_file, stderr=subprocess.STDOUT,
            )
        if output.exists():
            row = json.loads(output.read_text())
        else:
            log_text = log.read_text(encoding="utf-8", errors="replace")
            exact_oom = "cuda out of memory" in log_text.lower() or "torch.outofmemoryerror" in log_text.lower()
            row = {"beam_width": width,
                   "status": "cuda_oom" if exact_oom else "process_exit",
                   "returncode": proc.returncode}
        row.setdefault("returncode", proc.returncode)
        row.update({"attempt": attempt, "phase": phase, "log": log.name})
        rows.append(row)
        save()
        return row

    completed = []
    first_failure = None
    for width in [2**power for power in range(10, 26)]:
        row = run_width(width, phase="power_of_two_grid")
        if row["status"] == "completed":
            completed.append(row)
            continue
        first_failure = row
        break

    confirmation = None
    estimated_boundary = None
    if first_failure is not None and completed:
        estimated_boundary = float(first_failure["beam_width"])
        usable = [r for r in completed if r.get("peak_reserved_bytes") is not None]
        if len(usable) >= 2:
            left, right = usable[-2], usable[-1]
            delta_beam = right["beam_width"] - left["beam_width"]
            delta_mem = right["peak_reserved_bytes"] - left["peak_reserved_bytes"]
            if delta_beam > 0 and delta_mem > 0:
                slope = delta_mem / delta_beam
                predicted = right["beam_width"] + (0.97 * hardware["total_vram_bytes"] - right["peak_reserved_bytes"]) / slope
                estimated_boundary = min(estimated_boundary, max(float(right["beam_width"]), predicted))
        candidate = int(estimated_boundary * BOUNDARY_SAFETY) // BOUNDARY_ALIGNMENT * BOUNDARY_ALIGNMENT
        if completed[-1]["beam_width"] < candidate < first_failure["beam_width"]:
            confirmation = run_width(candidate, phase="single_below_estimate_confirmation")

    summary = {
        "status": "first_failure_observed" if first_failure else "maximum_requested_width_completed",
        "max_power_of_two_complete": max((r["beam_width"] for r in completed), default=None),
        "first_failure_width": first_failure["beam_width"] if first_failure else None,
        "first_failure_status": first_failure["status"] if first_failure else None,
        "estimated_boundary": int(estimated_boundary) if estimated_boundary is not None else None,
        "confirmation_width": confirmation["beam_width"] if confirmation else None,
        "confirmation_status": confirmation["status"] if confirmation else None,
        "output_dim_24": "incompatible: native CayleyPy Predictor requires one scalar score per state",
    }
    save(summary)

    print(json.dumps({"hardware": hardware, "rows": rows, "capacity_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
