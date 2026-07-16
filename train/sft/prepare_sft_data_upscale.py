#!/usr/bin/env python3
"""Variant of prepare_sft_data.py: compress by ratio then UPSCALE BACK to original size.

The goal: feed Qwen3-VL full visual-token budget (like 0x) but with blurry content
(like Nx compression). Tests if extra tokens help extract more from lossy pixels.
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


class ImageInfo(TypedDict):
    src: str
    dst: str


def compress_then_upscale(src: str, dst: str, scale_factor: float) -> bool:
    """Downscale by scale_factor/dim, then upscale back to original size."""
    try:
        Image.MAX_IMAGE_PIXELS = 300_000_000
        with Image.open(src) as img:
            orig_w, orig_h = img.width, img.height
            new_w = max(1, int(orig_w * scale_factor))
            new_h = max(1, int(orig_h * scale_factor))
            if img.mode != "RGB":
                img = img.convert("RGB")
            small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Upscale back to original
            upscaled = small.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            upscaled.save(dst, format="PNG")
        return True
    except Exception as e:
        print(f"  WARN: compress+upscale failed {src}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compress-ratio", type=int, default=9)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    compressed_dir = output_dir / "images"
    compressed_dir.mkdir(parents=True, exist_ok=True)

    scale_factor = 1.0 / math.sqrt(args.compress_ratio)
    print(
        f"Compress ratio={args.compress_ratio}, "
        f"downscale={scale_factor:.4f}/dim THEN upscale back to original."
    )
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {output_dir}")

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

        examples = [json.loads(line) for line in open(jsonl_path)]
        if args.max_examples > 0:
            examples = examples[: args.max_examples]
        print(f"  Loaded {len(examples)} examples")

        unique_images: dict[str, ImageInfo] = {}
        for ex in examples:
            src_rel = ex["chunk_path"]
            src_abs = str(dataset_dir / src_rel)
            if src_rel not in unique_images:
                compressed_abs = str(compressed_dir / src_rel)
                unique_images[src_rel] = {"src": src_abs, "dst": compressed_abs}

        print(f"  Unique images: {len(unique_images)}")

        to_compress = [
            info for info in unique_images.values() if not os.path.exists(info["dst"])
        ]
        for info in to_compress:
            os.makedirs(os.path.dirname(info["dst"]), exist_ok=True)

        if to_compress:
            print(
                f"  Processing {len(to_compress)} new images "
                f"({len(unique_images) - len(to_compress)} cached)..."
            )
            ok = fail = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        compress_then_upscale, info["src"], info["dst"], scale_factor
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
            print(f"  Done: {ok} ok, {fail} failed")
        else:
            print("  All images cached")

        sharegpt = []
        skipped = 0
        for ex in examples:
            info = unique_images[ex["chunk_path"]]
            if not os.path.exists(info["dst"]):
                skipped += 1
                continue
            sharegpt.append(
                {
                    "messages": [
                        {"role": "user", "content": "<image>\n" + ex["query"]},
                        {"role": "assistant", "content": ex["answer"]},
                    ],
                    "images": [info["dst"]],
                }
            )
        out_json = output_dir / f"{split_name}.json"
        with open(out_json, "w") as f:
            json.dump(sharegpt, f, ensure_ascii=False)
        print(f"  Output: {out_json} ({len(sharegpt)} examples, {skipped} skipped)")

    dataset_info = {}
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


if __name__ == "__main__":
    main()
