#!/usr/bin/env python3
"""Prepare a Hugging Face dataset folder from a single HN JSONL file.

Output layout:

  <output_dir>/
    README.md
    dataset_summary.json
    <metadata_name>.jsonl
    images/<relative image tree>

Image paths in the metadata are rewritten to be relative to `images/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_INPUT_JSONL = Path(
    "/home/user/wiki-screenshot-training/training/data/natrual_filtered_v2/"
    "lite-query-v2-full-filtered-hn.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/opt/dlami/nvme/kiwix_tiles")
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/wiki-screenshot-training/hf_dataset_export/screenshot-training-natural-filtered-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata-name",
        default="lite-query-v2-full-filtered-hn.jsonl",
        help="Filename to use inside the HF dataset folder.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Use hardlinks to avoid duplicating local storage when possible.",
    )
    parser.add_argument(
        "--repo-id",
        default="Chrisyichuan/screenshot-training-natural-filtered-v2",
        help="Hugging Face dataset repo id, used in the generated README.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_relative_image_path(path: str, image_root: Path) -> str:
    rel = Path(path).relative_to(image_root)
    return rel.as_posix()


def materialize_image(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build_readme(repo_id: str, metadata_name: str, summary: dict) -> str:
    return f"""---
license: mit
task_categories:
- image-retrieval
- question-answering
language:
- en
pretty_name: screenshot-training-natural-filtered-v2
size_categories:
- 100K<n<1M
---

# {repo_id}

Wikipedia screenshot retrieval training dataset filtered for more natural, SimpleQA-like queries.

## Contents

- `{metadata_name}`
- `images/`

Each metadata row has the form:

```json
{{
  "query": "...",
  "chunk_path": "images/shard_123/shard_00001/123456.png.tiles/chunk_0000_00.png",
  "neg_chunk_paths": [
    "images/shard_234/shard_00002/234567.png.tiles/chunk_0000_01.png"
  ],
  "source_positive_rank": 1,
  "source_positive_score": 0.63
}}
```

## Summary

- rows: {summary["rows"]}
- unique_images_referenced: {summary["unique_images_referenced"]}
- avg_negatives_per_row: {summary["avg_negatives_per_row"]:.4f}

## Notes

- This export is derived from `natrual_filtered_v2/lite-query-v2-full-filtered-hn.jsonl`.
- Rows were filtered with `naturalness >= 4` and `simpleqa_style_fit >= 4`.
- Image paths are stored relative to the dataset root.
- Source images were deduplicated before export so repeated hard negatives upload once.
"""


def main() -> int:
    args = parse_args()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    transformed = []
    unique_images = set()
    total_negatives = 0
    for row in read_jsonl(args.input_jsonl):
        pos_rel = to_relative_image_path(row["chunk_path"], image_root)
        materialize_image(Path(row["chunk_path"]), images_dir / pos_rel, args.link_mode)
        unique_images.add(pos_rel)

        neg_rel_paths = []
        for neg_path in row.get("neg_chunk_paths", []):
            neg_rel = to_relative_image_path(neg_path, image_root)
            materialize_image(Path(neg_path), images_dir / neg_rel, args.link_mode)
            unique_images.add(neg_rel)
            neg_rel_paths.append(f"images/{neg_rel}")
        total_negatives += len(neg_rel_paths)

        transformed.append(
            {
                "query": row["query"],
                "chunk_path": f"images/{pos_rel}",
                "neg_chunk_paths": neg_rel_paths,
                "source_positive_rank": row.get("source_positive_rank"),
                "source_positive_score": row.get("source_positive_score"),
            }
        )

    write_jsonl(output_dir / args.metadata_name, transformed)
    summary = {
        "repo_id": args.repo_id,
        "input_jsonl": str(args.input_jsonl),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "metadata_name": args.metadata_name,
        "link_mode": args.link_mode,
        "rows": len(transformed),
        "unique_images_referenced": len(unique_images),
        "avg_negatives_per_row": (total_negatives / len(transformed))
        if transformed
        else 0.0,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (output_dir / "README.md").write_text(
        build_readme(args.repo_id, args.metadata_name, summary)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
