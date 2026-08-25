"""Build the deterministic self-contained Kaggle runtime bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reproduce" / "data"
RUNTIME = ROOT / "reproduce" / "runtime" / "pilgrim_runtime"
WORKERS = ROOT / "reproduce" / "workers"
MANIFEST = DATA / "benchmark_manifest.json"
ARCHIVE = DATA / "bundle_payload.zip"
BINARY_SUFFIXES = {".pt", ".pth"}


def normalized_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() not in BINARY_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def bundle_members() -> dict[str, Path]:
    members: dict[str, Path] = {
        "DATASET_LICENSES.md": DATA / "DATASET_LICENSES.md",
        "third_party/AnanasClassic-Apache-2.0-LICENSE.txt": (
            ROOT / "third_party" / "AnanasClassic-Apache-2.0-LICENSE.txt"
        ),
    }
    for path in RUNTIME.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            members[f"pilgrim_runtime/{path.relative_to(RUNTIME).as_posix()}"] = path
    for name in ("multigpu_beam_search.py", "pilgrim.py"):
        members[f"workers/{name}"] = WORKERS / name
    return dict(sorted(members.items()))


def main() -> None:
    members = bundle_members()
    missing = [str(path) for path in members.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bundle sources: {missing}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["sha256"] = {
        name: hashlib.sha256(normalized_bytes(path)).hexdigest()
        for name, path in members.items()
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    temporary = ARCHIVE.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, path in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, normalized_bytes(path), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(ARCHIVE)

    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "members": len(members),
                "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                "bytes": ARCHIVE.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
