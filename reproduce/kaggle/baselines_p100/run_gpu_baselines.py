import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Kaggle's T4 machine shape physically exposes two GPUs.  The paper protocol is
# deliberately single-GPU, so hide all but device 0 before importing PyTorch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Kaggle's default torch 2.10+cu128 wheel excludes Pascal (sm_60).  Bootstrap
# the official cu126 wheel on P100 before importing torch, then re-exec once.
physical_gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    text=True, capture_output=True, check=True).stdout.splitlines()[0]
if "P100" in physical_gpu and os.environ.get("P100_TORCH_BOOTSTRAPPED") != "1":
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126",
    ], check=True)
    os.environ["P100_TORCH_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

import torch


ROOT = Path("/kaggle/working/paper_gpu_baselines")
ROOT.mkdir(parents=True, exist_ok=True)
SCRAMBLE = "D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2"
EXPECTED_GPU = os.environ.get("EXPECTED_GPU", "Tesla")
DEEPCUBEA_COMMIT = "919489f14ecbbc80dc1bf1539ac0a462ffaca7c5"


def run(command, cwd=None, timeout=None):
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def hardware_manifest():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    assert EXPECTED_GPU.lower() in name.lower(), (EXPECTED_GPU, name)
    free, total = torch.cuda.mem_get_info()
    manifest = {
        "gpu": name,
        "gpu_count": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "total_vram_bytes": total,
        "initial_free_vram_bytes": free,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "compiled_cuda_arches": torch.cuda.get_arch_list(),
    }
    (ROOT / "hardware.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def alpha_sweep(hardware):
    run([sys.executable, "-m", "pip", "install", "-q", "alphacube==0.1.6"])
    import alphacube

    alphacube.load("large")
    rows = []
    for width in [1024, 4096, 16384, 65536, 131072, 262144, 524288]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = alphacube.solve(SCRAMBLE, beam_width=width, allow_wide=False)
            torch.cuda.synchronize()
            row = {
                **hardware,
                "implementation": "AlphaCube",
                "version": "0.1.6",
                "commit": "d549835d62f56cf7816b0f02a3281fd6f7ff686f",
                "algorithm": "beam search",
                "model": "large",
                "network": "MLP: one-hot 324; width 4096; 8 hidden Linear-ReLU-BN blocks; 18 outputs",
                "dtype": str(next(alphacube.solver.model.parameters()).dtype),
                "move_set": "3x3x3 HTM, 18 moves, no wide moves",
                "scramble": SCRAMBLE,
                "beam_width": width,
                "internal_max_dnn_batch": 65536,
                "status": "completed" if result is not None else "no_solution",
                "process_wall_s": time.perf_counter() - started,
                "solver_wall_s": None if result is None else result["time"],
                "num_nodes": None if result is None else result["num_nodes"],
                "solution_lengths": None if result is None else [len(x.split()) for x in result["solutions"]],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        except torch.OutOfMemoryError as exc:
            row = {
                **hardware,
                "implementation": "AlphaCube",
                "version": "0.1.6",
                "beam_width": width,
                "status": "oom",
                "error": str(exc),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            rows.append(row)
            break
        rows.append(row)
        (ROOT / "alphacube.json").write_text(json.dumps(rows, indent=2))
    return rows


def deepcubea(hardware):
    repo = ROOT / "DeepCubeA"
    run([
        "git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/forestagostinelli/DeepCubeA.git", str(repo),
    ], timeout=300)
    run(["git", "fetch", "--depth", "1", "origin", DEEPCUBEA_COMMIT], cwd=repo, timeout=300)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual_commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    if actual_commit != DEEPCUBEA_COMMIT:
        raise RuntimeError(f"DeepCubeA commit mismatch: {actual_commit}")
    source = repo / "environments/environment_abstract.py"
    source.write_text(source.read_text().replace("np.float", "float"))
    harness = ROOT / "benchmark_deepcubea.py"
    harness.write_text(r'''
import json
import sys
import time
from pathlib import Path

import torch

repo, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from search_methods.astar import AStar, get_path
from utils import env_utils, nnet_utils, search_utils

scramble = "D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2"
env = env_utils.get_environment("cube3")
state = env.generate_goal_states(1)[0]
move_to_action = {move: i for i, move in enumerate(env.moves)}
for token in scramble.split():
    face = token[0]
    actions = ([move_to_action[face + "1"]] * 2 if token.endswith("2")
               else [move_to_action[face + "-1"]] if token.endswith("'")
               else [move_to_action[face + "1"]])
    for action in actions:
        state = env.next_state([state], action)[0][0]

device, devices, on_gpu = nnet_utils.get_device()
heuristic_fn = nnet_utils.load_heuristic_fn(
    str(repo / "saved_models/cube3/current"), device, on_gpu,
    env.get_nnet_model(), env, clip_zero=True, batch_size=10000)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
astar = AStar([state], env, heuristic_fn, [0.6])
iterations = 0
while not min(astar.has_found_goal()):
    astar.step(heuristic_fn, 10000, verbose=False)
    iterations += 1
torch.cuda.synchronize()
goal = astar.get_goal_node_smallest_path_cost(0)
_, solution, path_cost = get_path(goal)
assert search_utils.is_valid_soln(state, solution, env)
row = {
    "status": "completed", "wall_s": time.perf_counter() - started,
    "iterations": iterations, "solution_length": len(solution),
    "path_cost": path_cost, "num_nodes": astar.get_num_nodes_generated(0),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
}
output.write_text(json.dumps(row, indent=2))
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    command = [
        sys.executable, str(harness), str(repo), str(ROOT / "deepcubea.json"),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=300)
        if completed.returncode == 0:
            result = json.loads((ROOT / "deepcubea.json").read_text())
            shutil.rmtree(repo)
            harness.unlink()
            return result
        result = {
            **hardware,
            "implementation": "DeepCubeA",
            "commit": DEEPCUBEA_COMMIT,
            "algorithm": "batched weighted A*",
            "network": "ResNet: input 54x6 one-hot; hidden 5000; residual width 1000; 4 blocks; scalar value",
            "weight": 0.6,
            "search_batch": 10000,
            "nnet_batch": 10000,
            "scramble": SCRAMBLE,
            "status": "error",
            "wall_s": time.perf_counter() - started,
            "stderr_tail": completed.stderr[-4000:],
        }
        shutil.rmtree(repo)
        harness.unlink()
        return result
    except subprocess.TimeoutExpired:
        result = {
            **hardware,
            "implementation": "DeepCubeA",
            "commit": DEEPCUBEA_COMMIT,
            "algorithm": "batched weighted A*",
            "network": "ResNet: input 54x6 one-hot; hidden 5000; residual width 1000; 4 blocks; scalar value",
            "weight": 0.6,
            "search_batch": 10000,
            "nnet_batch": 10000,
            "scramble": SCRAMBLE,
            "status": "timeout",
            "wall_s": 300,
        }
        shutil.rmtree(repo)
        harness.unlink()
        return result


def main():
    hardware = hardware_manifest()
    results = {"hardware": hardware, "alphacube": alpha_sweep(hardware)}
    results["deepcubea"] = deepcubea(hardware)
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
