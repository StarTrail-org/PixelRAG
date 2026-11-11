#!/usr/bin/env python3
"""Upload a prepared local dataset folder to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_LOCAL_DIR = Path(
    "/home/user/wiki-screenshot-training/hf_dataset_export_sharded/screenshot-training"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Chrisyichuan/screenshot-training")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip create_repo (use if repo already exists)",
    )
    return parser.parse_args()


def retry_on_429(fn, max_retries=5, initial_wait=60):
    """Retry a function on 429 rate limit errors with exponential backoff."""
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
    api = HfApi()

    if not args.skip_create:
        print("Creating repo (with retry)...")
        retry_on_429(
            lambda: api.create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                exist_ok=True,
                private=args.private,
            )
        )
        print("Repo ready.")

    print(f"Uploading {args.local_dir} ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.local_dir),
    )
    print(
        f"Uploaded {args.local_dir} -> https://huggingface.co/datasets/{args.repo_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
