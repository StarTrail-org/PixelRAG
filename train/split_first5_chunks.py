#!/usr/bin/env python3
"""Merge the first 5 filtered-HN chunks and split them into train/eval/test JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_CHUNK_INPUTS = [
    Path(
        "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_000000_009999/filtered_hn.jsonl"
    ),
    Path(
        "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_010000_019999/filtered_hn.jsonl"
    ),
    Path(
        "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_020000_029999/filtered_hn.jsonl"
    ),
    Path(
        "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_030000_039999/filtered_hn.jsonl"
    ),
    Path(
        "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/chunk_040000_049999/filtered_hn.jsonl"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_CHUNK_INPUTS],
        help="Filtered chunk JSONL files to merge before splitting.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered-hn-v2-chunks/split",
        help="Where to write train/eval/test JSONL files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratio_sum = args.train_ratio + args.eval_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    rows = []
    input_counts = {}
    for path in input_paths:
        chunk_rows = read_jsonl(path)
        rows.extend(chunk_rows)
        input_counts[str(path)] = len(chunk_rows)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    total = len(rows)
    train_end = int(total * args.train_ratio)
    eval_end = train_end + int(total * args.eval_ratio)

    train_rows = rows[:train_end]
    eval_rows = rows[train_end:eval_end]
    test_rows = rows[eval_end:]

    train_path = output_dir / "train_hn.jsonl"
    eval_path = output_dir / "eval_hn.jsonl"
    test_path = output_dir / "test_hn.jsonl"
    summary_path = output_dir / "split_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(test_path, test_rows)

    summary = {
        "inputs": input_counts,
        "seed": args.seed,
        "total_rows": total,
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "test_ratio": args.test_ratio,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "test_rows": len(test_rows),
        "output_files": {
            "train": str(train_path),
            "eval": str(eval_path),
            "test": str(test_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
