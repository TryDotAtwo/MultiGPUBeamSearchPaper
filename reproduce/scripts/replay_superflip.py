#!/usr/bin/env python3
"""CPU-only replay of the published 51-move Megaminx superflip path."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBMISSION = ROOT / "artifacts" / "superflip" / "submission_69749.csv"
DEFAULT_TEST = ROOT / "reproduce" / "data" / "test.csv"
DEFAULT_PUZZLE_INFO = ROOT / "reproduce" / "data" / "puzzle_info.json"

EXPECTED_SUBMISSION_SHA256 = (
    "b26d6adbd78d752d7a52f56990c2af6cd711d7de1dbcc3d9373bd623c9a5d8f9"
)
EXPECTED_PATH_SHA256 = (
    "18435e4675503bbfd92b309b905e993f205dc6162f834ddf103f83491d7f95aa"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_row(path: Path, row_id: int, value_column: str) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["initial_state_id"]) == row_id:
                return row[value_column]
    raise KeyError(f"initial_state_id={row_id} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--puzzle-info", type=Path, default=DEFAULT_PUZZLE_INFO)
    args = parser.parse_args()

    submission_bytes = args.submission.read_bytes()
    submission_sha = sha256_bytes(submission_bytes)
    path_text = read_row(args.submission, 0, "path")
    path_sha = sha256_bytes(path_text.encode("utf-8"))
    state_text = read_row(args.test, 0, "initial_state")
    state = [int(value) for value in state_text.split(",")]
    puzzle = json.loads(args.puzzle_info.read_text(encoding="utf-8"))

    moves = path_text.split(".") if path_text else []
    for move in moves:
        permutation = puzzle["generators"][move]
        state = [state[source] for source in permutation]

    differences = sum(a != b for a, b in zip(state, puzzle["central_state"]))
    report = {
        "submission_sha256": submission_sha,
        "path_sha256": path_sha,
        "move_count": len(moves),
        "final_differences": differences,
        "replay_valid": differences == 0,
    }
    print(json.dumps(report, indent=2))

    assert submission_sha == EXPECTED_SUBMISSION_SHA256
    assert path_sha == EXPECTED_PATH_SHA256
    assert len(moves) == 51
    assert differences == 0


if __name__ == "__main__":
    main()
