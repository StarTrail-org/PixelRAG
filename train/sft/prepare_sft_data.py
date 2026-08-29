#!/usr/bin/env python3
"""Prepare SFT training data for LlamaFactory from with_answer JSONL.

Reads train/eval/test_hn_with_answer.jsonl, compresses positive images by
a given ratio, and outputs LlamaFactory-compatible ShareGPT JSON files.

Output format (per example):
{
  "messages": [
    {"role": "user", "content": "<image>\n{query}"},
    {"role": "assistant", "content": "{answer}"}
  ],
  "images": ["/path/to/compressed_image.png"]
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from PIL import Image
from tqdm import tqdm


class ShareGPTMessage(TypedDict):
    role: str
    content: str


class ShareGPTExample(TypedDict):
    messages: list[ShareGPTMessage]
    images: list[str]


class ImageInfo(TypedDict):
    src: str
    dst: str
    dst_rel: str


class DatasetColumnMap(TypedDict):
    messages: str
    images: str


class DatasetTagMap(TypedDict):
    role_tag: str
    content_tag: str
    user_tag: str
    assistant_tag: str


class DatasetInfoEntry(TypedDict):
    file_name: str
    formatting: str
    columns: DatasetColumnMap
    tags: DatasetTagMap


def compress_image(src: str, dst: str, scale_factor: float) -> bool:
    """Compress image by scale_factor per dimension. Returns True on success."""
    try:
        Image.MAX_IMAGE_PIXELS = 300_000_000
        with Image.open(src) as img:
            new_w = max(1, int(img.width * scale_factor))
            new_h = max(1, int(img.height * scale_factor))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_resized.save(dst, format="PNG")
        return True
    except Exception as e:
        print(f"  WARN: compress failed {src}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/mnt/data/hf_datasets/screenshot-training-natural-filtered-v2",
        help="Root of the HF dataset (contains images/ and *_with_answer.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/sft_data/compressed_3x",
        help="Output directory for compressed images and JSON",
    )
    parser.add_argument(
        "--compress-ratio",
        type=int,
        default=3,
        help="Pixel compression ratio (3 = each dim scaled by 1/sqrt(3))",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Parallel workers for image compression"
    )
    parser.add_argument(
        "--max-examples", type=int, default=0, help="Limit examples per split (0 = all)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    compressed_dir = output_dir / "images"
    compressed_dir.mkdir(parents=True, exist_ok=True)

    scale_factor = 1.0 / math.sqrt(args.compress_ratio)
    print(f"Compression: ratio={args.compress_ratio}, scale={scale_factor:.4f}/dim")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")

    splits = {
        "train": "train_hn_with_answer.jsonl",
        "eval": "eval_hn_with_answer.jsonl",
        "test": "test_hn_with_answer.jsonl",
    }

    for split_name, jsonl_name in splits.items():
        jsonl_path = dataset_dir / jsonl_name
        if not jsonl_path.exists():
            print(f"  SKIP {split_name}: {jsonl_path} not found")
            continue

        print(f"\n=== {split_name} ===")

        # Load JSONL
        examples = []
        with open(jsonl_path) as f:
            for line in f:
                examples.append(json.loads(line))
        if args.max_examples > 0:
            examples = examples[: args.max_examples]
        print(f"  Loaded {len(examples)} examples")

        # Collect unique positive image paths
        unique_images: dict[str, ImageInfo] = {}
        for ex in examples:
            src_rel = ex["chunk_path"]  # e.g. images/shard_760/...
            src_abs = str(dataset_dir / src_rel)
            if src_rel not in unique_images:
                # Create compressed path preserving shard structure
                compressed_rel = src_rel  # keep same relative structure
                compressed_abs = str(compressed_dir / compressed_rel)
                unique_images[src_rel] = {
                    "src": src_abs,
                    "dst": compressed_abs,
                    "dst_rel": str(compressed_dir / compressed_rel),
                }

        print(f"  Unique images to compress: {len(unique_images)}")

        # Compress images in parallel
        to_compress = []
        for info in unique_images.values():
            if not os.path.exists(info["dst"]):
                os.makedirs(os.path.dirname(info["dst"]), exist_ok=True)
                to_compress.append(info)

        if to_compress:
            print(
                f"  Compressing {len(to_compress)} new images ({len(unique_images) - len(to_compress)} cached)..."
            )
            ok = 0
            fail = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        compress_image, info["src"], info["dst"], scale_factor
                    ): info
                    for info in to_compress
                }
                for fut in tqdm(
                    as_completed(futures), total=len(futures), desc=f"  {split_name}"
                ):
                    if fut.result():
                        ok += 1
                    else:
                        fail += 1
            print(f"  Compressed: {ok} ok, {fail} failed")
        else:
            print("  All images already cached")

        # Build ShareGPT format
        sharegpt_data: list[ShareGPTExample] = []
        skipped = 0
        for ex in examples:
            src_rel = ex["chunk_path"]
            info = unique_images[src_rel]
            compressed_path = info["dst"]

            if not os.path.exists(compressed_path):
                skipped += 1
                continue

            sharegpt_data.append(
                {
                    "messages": [
                        {"role": "user", "content": "<image>\n" + ex["query"]},
                        {"role": "assistant", "content": ex["answer"]},
                    ],
                    "images": [compressed_path],
                }
            )

        out_json = output_dir / f"{split_name}.json"
        with open(out_json, "w") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=None)
        print(
            f"  Output: {out_json} ({len(sharegpt_data)} examples, {skipped} skipped)"
        )

    # Write dataset_info.json for LlamaFactory
    dataset_info: dict[str, DatasetInfoEntry] = {}
    for split_name in splits:
        out_json = output_dir / f"{split_name}.json"
        if out_json.exists():
            dataset_info[f"compressed_qa_{split_name}"] = {
                "file_name": str(out_json),
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\nDataset info: {info_path}")
    print("Done!")


if __name__ == "__main__":
    main()
