#!/usr/bin/env python3
"""Pack a prepared HF dataset export into tar shards for faster Hub uploads."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

DEFAULT_SOURCE_DIR = Path(
    "/home/user/wiki-screenshot-training/hf_dataset_export/screenshot-training"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/wiki-screenshot-training/hf_dataset_export_sharded/screenshot-training"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing shard tar if it already exists.",
    )
    return parser.parse_args()


def copy_metadata(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.iterdir()):
        if src.name == "images":
            continue
        if src.is_file():
            shutil.copy2(src, output_dir / src.name)


def pack_one_shard(shard_dir: Path, tar_path: Path, output_root: Path) -> int:
    file_count = 0
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="w") as tar:
        for path in sorted(shard_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(output_root).as_posix()
            tar.add(path, arcname=arcname, recursive=False)
            file_count += 1
    return file_count


def build_readme(output_dir: Path, shard_count: int) -> None:
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        return
    original = readme_path.read_text()
    extra = f"""

## Image Storage

The images are stored as `{shard_count}` tar shards under `image_shards/` to keep
the repository file count low and make uploads/downloads more reliable.

To materialize the images locally after download:

```bash
python extract_hf_image_shards.py --dataset-dir .
```
"""
    if "## Image Storage" not in original:
        readme_path.write_text(original.rstrip() + "\n" + extra)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    source_images = source_dir / "images"
    shard_output_dir = output_dir / "image_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_output_dir.mkdir(parents=True, exist_ok=True)

    copy_metadata(source_dir, output_dir)

    shard_dirs = sorted(p for p in source_images.iterdir() if p.is_dir())
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "shard_count": 0,
        "shards": {},
    }

    for shard_dir in shard_dirs:
        tar_path = shard_output_dir / f"{shard_dir.name}.tar"
        if tar_path.exists() and not args.overwrite:
            continue
        if tar_path.exists():
            tar_path.unlink()
        file_count = pack_one_shard(shard_dir, tar_path, source_images)
        summary["shards"][shard_dir.name] = {
            "tar_path": str(tar_path),
            "file_count": file_count,
            "size_bytes": tar_path.stat().st_size,
        }

    summary["shard_count"] = len(list(shard_output_dir.glob("*.tar")))
    summary["total_size_bytes"] = sum(
        info["size_bytes"] for info in summary["shards"].values()
    )
    (output_dir / "image_shards_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    extract_script_src = Path(
        "/home/user/wiki-screenshot-training/extract_hf_image_shards.py"
    )
    if extract_script_src.exists():
        shutil.copy2(extract_script_src, output_dir / "extract_hf_image_shards.py")

    build_readme(output_dir, summary["shard_count"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
