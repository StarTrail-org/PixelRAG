#!/usr/bin/env python3
"""Package and upload the hard-mini-v8 test set to HF dataset repo."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_MINIV8_JSON = Path("training/data/test_miniv8.json")
DEFAULT_TILES_DIR = Path("/home/user/Vis-RAG/agent/tiles-hard-mini-v8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miniv8-json", type=Path, default=DEFAULT_MINIV8_JSON)
    parser.add_argument("--tiles-dir", type=Path, default=DEFAULT_TILES_DIR)
    parser.add_argument("--repo-id", default="Chrisyichuan/screenshot-training")
    parser.add_argument("--repo-type", default="dataset")
    return parser.parse_args()


def retry_on_429(fn, max_retries=5, initial_wait=60):
    for attempt in range(max_retries):
        try:
            return fn()
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = initial_wait * (2**attempt)
                print(
                    f"Rate limited (429). Waiting {wait}s before retry {attempt + 2}/{max_retries}..."
                )
                time.sleep(wait)
            else:
                raise


def main() -> int:
    args = parse_args()
    tiles_dir = args.tiles_dir.resolve()
    miniv8_json = args.miniv8_json.resolve()

    # Rewrite test_miniv8.json with relative tiles_dir
    with open(miniv8_json) as f:
        data = json.load(f)
    data["tiles_dir"] = "test_miniv8/tiles"

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir) / "test_miniv8"
        staging.mkdir()

        # Write updated JSON
        out_json = staging / "test_miniv8.json"
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Tar the tiles
        tar_path = staging / "tiles.tar"
        print(f"Packing {tiles_dir} into {tar_path} ...")
        tile_files = sorted(tiles_dir.glob("*.png"))
        print(f"  {len(tile_files)} tiles")
        with tarfile.open(tar_path, "w") as tar:
            for p in tile_files:
                tar.add(p, arcname=p.name)
        print(f"  tar size: {tar_path.stat().st_size / 1e6:.1f} MB")

        # Upload
        api = HfApi()
        print("Uploading test_miniv8/ to HF...")
        retry_on_429(
            lambda: api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=str(staging),
                path_in_repo="test_miniv8",
            )
        )
        print(
            f"Done: https://huggingface.co/datasets/{args.repo_id}/tree/main/test_miniv8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
