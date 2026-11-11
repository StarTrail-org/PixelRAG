#!/usr/bin/env python3
"""Prepare a Hugging Face dataset folder from local split JSONL files.

The output layout is:

  <output_dir>/
    README.md
    dataset_summary.json
    train.jsonl
    eval.jsonl
    test.jsonl
    train_hn.jsonl
    eval_hn.jsonl
    test_hn.jsonl
    images/<relative image tree>

Each metadata row keeps relative image paths under the `images/` folder so the
dataset can be moved to another machine without absolute-path assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_SPLIT_DIR = Path(
    "/home/user/wiki-screenshot-training/training/data/"
    "lite-query-v2-full-filtered-hn-v2-chunks/split"
)
DEFAULT_IMAGE_ROOT = Path("/opt/dlami/nvme/kiwix_tiles")
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/wiki-screenshot-training/hf_dataset_export/screenshot-training"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Use hardlinks to avoid duplicating local storage when possible.",
    )
    parser.add_argument(
        "--repo-id",
        default="Chrisyichuan/screenshot-training",
        help="Hugging Face dataset repo id, used in the generated README.",
    )
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


def transform_rows(
    split_name: str,
    rows: list[dict],
    image_root: Path,
    images_dir: Path,
    link_mode: str,
) -> tuple[list[dict], dict]:
    out_rows = []
    unique_images = set()
    total_negatives = 0

    for row in rows:
        pos_rel = to_relative_image_path(row["chunk_path"], image_root)
        pos_src = Path(row["chunk_path"])
        materialize_image(pos_src, images_dir / pos_rel, link_mode)
        unique_images.add(pos_rel)

        neg_rel_paths = []
        for neg_path in row.get("neg_chunk_paths", []):
            neg_rel = to_relative_image_path(neg_path, image_root)
            materialize_image(Path(neg_path), images_dir / neg_rel, link_mode)
            unique_images.add(neg_rel)
            neg_rel_paths.append(f"images/{neg_rel}")
        total_negatives += len(neg_rel_paths)

        out_rows.append(
            {
                "query": row["query"],
                "chunk_path": f"images/{pos_rel}",
                "neg_chunk_paths": neg_rel_paths,
                "split": split_name,
            }
        )

    stats = {
        "rows": len(out_rows),
        "unique_images_referenced": len(unique_images),
        "avg_negatives_per_row": (total_negatives / len(out_rows)) if out_rows else 0.0,
    }
    return out_rows, stats


def build_readme(repo_id: str, summary: dict) -> str:
    return f"""---
license: mit
task_categories:
- image-retrieval
- question-answering
language:
- en
pretty_name: screenshot-training
size_categories:
- 10K<n<100K
---

# {repo_id}

Wikipedia screenshot retrieval training dataset exported from local hard-negative mining.

## Contents

- `train.jsonl` / `train_hn.jsonl`
- `eval.jsonl` / `eval_hn.jsonl`
- `test.jsonl` / `test_hn.jsonl`
- `images/`

Each metadata row has the form:

```json
{{
  "query": "...",
  "chunk_path": "images/shard_123/shard_00001/123456.png.tiles/chunk_0000_00.png",
  "neg_chunk_paths": [
    "images/shard_234/shard_00002/234567.png.tiles/chunk_0000_01.png"
  ],
  "split": "train"
}}
```

## Split sizes

- train: {summary["splits"]["train"]["rows"]}
- eval: {summary["splits"]["eval"]["rows"]}
- test: {summary["splits"]["test"]["rows"]}

## Notes

- Image paths are stored relative to the dataset root.
- The source images were deduplicated before export so repeated hard negatives only upload once.
- This export was prepared from the first 5 filtered hard-negative chunks.
"""


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir
    image_root = args.image_root.resolve()
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "train": split_dir / "train_hn.jsonl",
        "eval": split_dir / "eval_hn.jsonl",
        "test": split_dir / "test_hn.jsonl",
    }

    split_rows = {name: read_jsonl(path) for name, path in input_paths.items()}
    summary = {
        "repo_id": args.repo_id,
        "split_dir": str(split_dir),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "link_mode": args.link_mode,
        "splits": {},
    }

    all_unique_images = set()
    for split_name, rows in split_rows.items():
        transformed, stats = transform_rows(
            split_name, rows, image_root, images_dir, args.link_mode
        )
        write_jsonl(output_dir / f"{split_name}.jsonl", transformed)
        write_jsonl(output_dir / f"{split_name}_hn.jsonl", transformed)
        summary["splits"][split_name] = stats
        for row in transformed:
            all_unique_images.add(row["chunk_path"])
            all_unique_images.update(row["neg_chunk_paths"])

    summary["total_rows"] = sum(info["rows"] for info in summary["splits"].values())
    summary["total_unique_images"] = len(all_unique_images)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (output_dir / "README.md").write_text(build_readme(args.repo_id, summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
