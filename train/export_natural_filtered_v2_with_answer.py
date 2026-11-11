#!/usr/bin/env python3
"""Attach answers to the natural_filtered_v2 HN dataset via source lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE_JSONL = Path(
    "/home/user/wiki-screenshot-training/training/data/lite-query-v2-full-filtered.jsonl"
)
DEFAULT_INPUT_JSONL = Path(
    "/home/user/wiki-screenshot-training/training/data/natrual_filtered_v2/"
    "lite-query-v2-full-filtered-hn.jsonl"
)
DEFAULT_OUTPUT_JSONL = Path(
    "/home/user/wiki-screenshot-training/training/data/natrual_filtered_v2/"
    "lite-query-v2-full-filtered-hn-with-answer.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/opt/dlami/nvme/kiwix_tiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    return parser.parse_args()


def normalize_source_chunk_path(chunk_path: str, image_root: Path) -> str:
    path = Path(chunk_path)
    if path.is_absolute():
        return path.as_posix()
    return (image_root / path).as_posix()


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    summary_json = (
        args.summary_json
        if args.summary_json is not None
        else args.output_jsonl.with_suffix(".summary.json")
    )

    answer_map: dict[tuple[str, str], str | None] = {}
    duplicates = 0
    source_rows = 0
    for row in iter_jsonl(args.source_jsonl):
        source_rows += 1
        key = (
            row["query"],
            normalize_source_chunk_path(row["chunk_path"], args.image_root),
        )
        if key in answer_map:
            duplicates += 1
        answer_map[key] = row.get("answer")

    total_rows = 0
    matched_rows = 0
    missing_rows = 0
    missing_examples = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as out:
        for row in iter_jsonl(args.input_jsonl):
            total_rows += 1
            key = (row["query"], row["chunk_path"])
            if key not in answer_map:
                missing_rows += 1
                if len(missing_examples) < 5:
                    missing_examples.append(
                        {"query": row["query"], "chunk_path": row["chunk_path"]}
                    )
                continue
            matched_rows += 1
            enriched = dict(row)
            enriched["answer"] = answer_map[key]
            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    summary = {
        "source_jsonl": str(args.source_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "source_rows": source_rows,
        "source_duplicate_keys": duplicates,
        "input_rows": total_rows,
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "match_rate_pct": round(100 * matched_rows / total_rows, 4)
        if total_rows
        else 0.0,
        "missing_examples": missing_examples,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
