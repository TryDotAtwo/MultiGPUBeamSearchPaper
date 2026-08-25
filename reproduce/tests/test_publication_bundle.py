import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_RELEASE_ASSETS = {
    "weights_megaminx2048_512_8_e4000.pth": (
        139_196_146,
        "7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759",
    ),
    "p900-t000-q-sym_1777988767_best.pth": (
        95_961_231,
        "e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670",
    ),
    "cube4_piece_transformer_24out.pth": (
        13_555_825,
        "58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_json_and_jsonl_are_parseable() -> None:
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in ROOT.rglob("*.jsonl"):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"{path}:{line_number}: {exc}") from exc


def test_notebooks_have_code_and_no_embedded_output() -> None:
    for path in (ROOT / "reproduce" / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells, path
        for cell in code_cells:
            assert cell.get("outputs", []) == [], path
            assert cell.get("execution_count") is None, path


def test_bundle_zip_exactly_matches_its_manifest() -> None:
    manifest = json.loads(
        (ROOT / "reproduce" / "data" / "benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected = manifest["sha256"]
    with zipfile.ZipFile(ROOT / "reproduce" / "data" / "bundle_payload.zip") as archive:
        names = sorted(name.replace("\\", "/") for name in archive.namelist())
        assert names == sorted(expected)
        assert "DATASET_LICENSES.md" in names
        assert "third_party/AnanasClassic-Apache-2.0-LICENSE.txt" in names
        for name, expected_sha in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected_sha


def test_minimal_pilgrim_runtime_has_provenance_notices() -> None:
    runtime = ROOT / "reproduce" / "runtime" / "pilgrim_runtime"
    assert not (runtime / "unified_training").exists()
    assert not (runtime / "pilgrim" / "best_first.py").exists()
    assert not (runtime / "pilgrim" / "trainer.py").exists()
    modified = [
        runtime / "test.py",
        runtime / "pilgrim" / "__init__.py",
        runtime / "pilgrim" / "factory.py",
        runtime / "pilgrim" / "model.py",
        runtime / "pilgrim" / "qsearcher.py",
        runtime / "pilgrim" / "searcher.py",
        runtime / "pilgrim" / "utils.py",
    ]
    for path in modified:
        source = path.read_text(encoding="utf-8")
        assert "Modified for the MultiGPUBeamSearchPaper" in source
        assert "893dbc162a597b8a80d2bcaf92bc2c399fa67dba" in source
    assert (ROOT / "third_party" / "AnanasClassic-Apache-2.0-LICENSE.txt").is_file()


def test_capacity_arithmetic_and_missing_fields_are_explicit() -> None:
    record = json.loads(
        (ROOT / "artifacts" / "a100x8_capacity.json").read_text(encoding="utf-8")
    )
    width = record["frontier"]["requested_width"]
    generators = record["workload"]["generator_count"]
    assert generators * width == record["derived"][
        "minimum_nominal_children_per_saturated_depth"
    ]
    assert record["frontier"]["exact_effective_width"] is None
    assert record["timing"]["wall_seconds"] is None
    assert record["timing"]["logical_children_per_second"] is None


def test_release_assets_match_when_staged_locally() -> None:
    release_dir = ROOT / "release_assets"
    if not release_dir.exists():
        return
    for name, (expected_size, expected_sha) in EXPECTED_RELEASE_ASSETS.items():
        path = release_dir / name
        assert path.stat().st_size == expected_size
        assert sha256(path) == expected_sha


def test_repository_checksum_paths_are_posix() -> None:
    checksum_file = ROOT / "artifacts" / "checksums.sha256"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        assert separator == "  "
        assert len(digest) == 64
        assert "\\" not in relative
        assert (ROOT / relative).is_file()


def test_no_literature_pdfs_or_secrets_are_packaged() -> None:
    assert not (ROOT / "paper" / "sources").exists()
    forbidden_names = {
        "kaggle.json",
        ".env",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
    }
    present = {path.name.lower() for path in ROOT.rglob("*") if path.is_file()}
    assert not (forbidden_names & present)


def test_deepcubea_runner_checks_out_and_verifies_the_pinned_commit() -> None:
    scripts = [
        ROOT / "reproduce" / "scripts" / "run_gpu_baselines.py",
        ROOT / "reproduce" / "kaggle" / "baselines_t4" / "run_gpu_baselines.py",
        ROOT / "reproduce" / "kaggle" / "baselines_p100" / "run_gpu_baselines.py",
    ]
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        assert 'DEEPCUBEA_COMMIT = "919489f14ecbbc80dc1bf1539ac0a462ffaca7c5"' in source
        assert '"fetch", "--depth", "1", "origin", DEEPCUBEA_COMMIT' in source
        assert '"checkout", "--detach", "FETCH_HEAD"' in source
        assert '"rev-parse", "HEAD"' in source
        assert "actual_commit != DEEPCUBEA_COMMIT" in source


def test_workers_pin_dependencies_and_download_exact_release_checkpoints() -> None:
    for name in ("pilgrim.py", "multigpu_beam_search.py"):
        source = (ROOT / "reproduce" / "workers" / name).read_text(encoding="utf-8")
        assert "MultiGPUBeamSearchPaper/releases/download/v1.0.0" in source
        assert "7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759" in source
        assert "e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670" in source
        assert "urlretrieve" in source
        assert "checkpoint SHA256 mismatch" in source
    multigpu = (ROOT / "reproduce" / "workers" / "multigpu_beam_search.py").read_text(
        encoding="utf-8"
    )
    assert 'cutlass_commit="afa1772203677c5118fcd82537a9c8fefbcc7008"' in multigpu
    assert "actual_cutlass_commit != cutlass_commit" in multigpu
    for path in (
        ROOT / "reproduce" / "scripts" / "run_cayleypy_capacity.py",
        ROOT / "reproduce" / "kaggle" / "cayleypy_t4" / "run_cayleypy_capacity.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "MultiGPUBeamSearchPaper/releases/download/v1.0.0" in source
        assert "urlretrieve" in source
        assert "checkpoint SHA256 mismatch" in source
        assert "8cbf1d728c44ca820a59990e768da78fc5ebcb6f3105a0d822c3808bfda6cc75" in source
        assert "paper_state.pt SHA256 mismatch" in source
        assert "weights_only=False" not in source


def test_public_notebooks_and_kernel_metadata_have_no_private_inputs() -> None:
    public_dataset = "trydotatwo/gpu-beam-search-paper-1xt4-reviewer-notebooks"
    public_dataset_mount = "gpu-beam-search-paper-1xt4-reviewer-notebooks"
    for path in (ROOT / "reproduce" / "notebooks").glob("*.ipynb"):
        source = path.read_text(encoding="utf-8")
        assert "paper-multigpu-megaminx-benchmark-bundle-v3" not in source
        assert "expected one private benchmark bundle" not in source
        assert "Reviewer-facing, private" not in source
        assert public_dataset_mount in source
    for path in (ROOT / "reproduce" / "kaggle").glob("*/kernel-metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["is_private"] is False
        assert "trydotatwo/paper-multigpu-megaminx-benchmark-bundle-v3" not in metadata.get(
            "dataset_sources", []
        )
        for source in metadata.get("dataset_sources", []):
            assert source == public_dataset
