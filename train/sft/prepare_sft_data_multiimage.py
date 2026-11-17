#!/usr/bin/env python3
"""Prepare multi-image SFT data from the top-6 retrieval outputs.

Input:
  <retrieval-dir>/{train,eval,test}.jsonl   (from fetch_top6_retrieval.py)
  <retrieval-dir>/tiles/...                 (raw mirror from download_tiles.py)

Output (per compression ratio N):
  <out-root>/<split>.json                   (LlamaFactory ShareGPT format, 6 images)
  <out-root>/images/shard_.../chunk.png     (compressed to 1/sqrt(N) per dim)
  <out-root>/dataset_info.json

Each example has exactly 6 images:
  = { gold } ∪ { top-6 hits }                            (dedup by shard-suffix)
  which collapses to:
    - gold in hits:   gold + 5 non-gold hits
    - gold not hits:  gold + top-5 hits
The gold's position among the 6 is randomized (per-example) so the model
does not learn a positional shortcut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from PIL import Image
from tqdm import tqdm


class RetrievalHit(TypedDict):
    path: str
    score: float | None
    article_id: int | None
    url: str | None


class RetrievalRow(TypedDict):
    query: str
    answer: str
    gold_path_rel: str
    gold_suffix: str
    hits: list[RetrievalHit]
    gold_in_top6_pos: int


def shard_suffix(p: str) -> str:
    parts = p.split("/")
    for i, x in enumerate(parts):
        if x.startswith("shard_"):
            return "/".join(parts[i:])
    return p


def compress_image(src: str, dst: str, scale_factor: float) -> bool:
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


def build_image_set(
    row: RetrievalRow, seed_base: int, n_images: int
) -> tuple[list[str], int]:
    """Return (list of n_images shard-suffixes, gold index) with gold position shuffled.
    Composition: gold + (n_images-1) non-gold hits from top-6."""
    gold = row["gold_suffix"]
    hit_sufs = [shard_suffix(h["path"]) for h in row["hits"]]
    non_gold = [s for s in hit_sufs if s != gold]
    chosen = [gold] + non_gold[: n_images - 1]
    while len(chosen) < n_images:
        chosen.append(gold)
    rng = random.Random(seed_base)
    idx = list(range(n_images))
    rng.shuffle(idx)
    shuffled = [chosen[i] for i in idx]
    gold_pos = idx.index(0)
    return shuffled, gold_pos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retrieval-dir",
        default="/scratch/users/zwcolin/cxr_embeds/sft_data/retrieval_raw",
    )
    p.add_argument(
        "--tiles-mirror", default=None, help="Default: <retrieval-dir>/tiles"
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Where to write compressed images + split JSONs",
    )
    p.add_argument(
        "--compress-ratio",
        type=int,
        required=True,
        help="Pixel-area ratio (2 / 3 / 4 / 5 / 9 …)",
    )
    p.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument(
        "--n-images",
        type=int,
        default=3,
        help="Images per sample. top-3 = gold + 2 non-gold hits; top-6 = gold + 5 hits",
    )
    p.add_argument(
        "--json-suffix",
        default="",
        help="Append to output JSON filenames (e.g. '_top3' → train_top3.json). Empty = overwrite train.json",
    )
    args = p.parse_args()

    retrieval_dir = Path(args.retrieval_dir)
    mirror = Path(args.tiles_mirror) if args.tiles_mirror else retrieval_dir / "tiles"
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    scale = 1.0 / math.sqrt(args.compress_ratio)
    no_compression = args.compress_ratio == 1
    print(
        f"Compress ratio: {args.compress_ratio}  (scale={scale:.4f}/dim)"
        + (" [NO-OP hardlink from mirror]" if no_compression else "")
    )
    print(f"Mirror:  {mirror}")
    print(f"Out:     {out_root}")
    print(f"Splits:  {args.splits}")

    # Pass 1: gather unique suffixes that need compression across all splits
    all_split_rows: dict[str, list[RetrievalRow]] = {}
    needed: set[str] = set()
    for split in args.splits:
        p_in = retrieval_dir / f"{split}.jsonl"
        if not p_in.exists():
            print(f"  SKIP {split}: {p_in} missing")
            continue
        rows = []
        with open(p_in) as f:
            for line in f:
                rows.append(json.loads(line))
        all_split_rows[split] = rows
        for r in rows:
            shuffled, _ = build_image_set(r, args.shuffle_seed, args.n_images)
            needed.update(shuffled)
    print(f"\nUnique suffixes needed: {len(needed):,}")

    # Filter to those not already compressed
    to_compress = []
    missing_src = 0
    for s in needed:
        dst = out_images / s
        if dst.exists():
            continue
        src = mirror / s
        if not src.exists():
            missing_src += 1
            continue
        to_compress.append((str(src), str(dst)))
    print(f"  Already compressed: {len(needed) - len(to_compress) - missing_src:,}")
    print(f"  To compress:        {len(to_compress):,}")
    print(f"  Missing src (skip): {missing_src:,}")

    # Compress in parallel (or hardlink when ratio=1)
    if to_compress:

        def _work(args_):
            src, dst = args_
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if no_compression:
                try:
                    os.link(src, dst)
                    return True
                except OSError:
                    import shutil

                    shutil.copy2(src, dst)
                    return True
            return compress_image(src, dst, scale)

        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_work, item) for item in to_compress]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="compress"):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
        print(f"  compressed: ok={ok} fail={fail}")

    # Pass 2: build ShareGPT JSON per split
    dataset_info = {}
    for split, rows in all_split_rows.items():
        sg = []
        skipped = 0
        for r in rows:
            shuffled, gold_pos = build_image_set(r, args.shuffle_seed, args.n_images)
            img_paths = [str(out_images / s) for s in shuffled]
            if any(not os.path.exists(pp) for pp in img_paths):
                skipped += 1
                continue
            # ShareGPT multi-image: N <image> tokens in user content, N paths in images
            user_content = ("<image>" * args.n_images) + "\n" + r["query"]
            sg.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": r["answer"]},
                    ],
                    "images": img_paths,
                    # meta (not used by LlamaFactory, retained for debugging)
                    "_gold_pos": gold_pos,
                    "_gold_in_top6_pos": r.get("gold_in_top6_pos", -1),
                }
            )
        out_json = out_root / f"{split}{args.json_suffix}.json"
        with open(out_json, "w") as f:
            json.dump(sg, f, ensure_ascii=False)
        print(f"  {split}: {len(sg)} examples, skipped {skipped}")
        ds_key = f"multimage_top{args.n_images}_{split}"
        dataset_info[ds_key] = {
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

    info_path = out_root / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\ndataset_info: {info_path}")


if __name__ == "__main__":
    main()
