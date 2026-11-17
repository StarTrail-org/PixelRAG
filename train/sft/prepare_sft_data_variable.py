#!/usr/bin/env python3
"""Variable-k multi-image SFT data prep.

Like prepare_sft_data_multiimage.py but each sample picks a random k in
[k_min, k_max]. Gold always included, position shuffled.

Reuses already-compressed tiles under <out-root>/images/... (does not
re-compress). Intended to run after prepare_sft_data_multiimage.py has
already populated the compressed tile cache for the same compression ratio.

For k=1..6 with gold-always-in-top-6-hits, we can always assemble the set
from top-6 hits alone (non-gold hits ≥ 5).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import TypedDict


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


def build_variable_image_set(
    row: RetrievalRow, rng: random.Random, k_min: int, k_max: int
) -> tuple[list[str], int, int]:
    """Sample k ~ Uniform[k_min, k_max], return (shard_suffixes, gold_pos, k)."""
    gold = row["gold_suffix"]
    hit_sufs = [shard_suffix(h["path"]) for h in row["hits"]]
    non_gold = [s for s in hit_sufs if s != gold]
    k = rng.randint(k_min, k_max)
    chosen = [gold] + non_gold[: k - 1]
    while len(chosen) < k:
        chosen.append(gold)
    idx = list(range(k))
    rng.shuffle(idx)
    shuffled = [chosen[i] for i in idx]
    gold_pos = idx.index(0)
    return shuffled, gold_pos, k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retrieval-dir",
        default="/scratch/users/zwcolin/cxr_embeds/sft_data/retrieval_raw",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Existing output root containing images/... already populated",
    )
    p.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=1337,
        help="Different from the top-3 prep seed so we get fresh compositions",
    )
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument(
        "--json-suffix",
        default="_vark",
        help="Suffix for output JSONs (default _vark → train_vark.json)",
    )
    p.add_argument(
        "--ds-key-prefix",
        default="multimage_vark",
        help="dataset_info key prefix: <prefix>_<split>",
    )
    args = p.parse_args()

    retrieval_dir = Path(args.retrieval_dir)
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    assert out_images.exists(), (
        f"{out_images} missing — run prepare_sft_data_multiimage.py first"
    )

    print(f"Retrieval dir: {retrieval_dir}")
    print(f"Out root:      {out_root}")
    print(f"k range:       [{args.k_min}, {args.k_max}]")
    print(f"Splits:        {args.splits}")
    print()

    # Load existing dataset_info.json if present (merge, don't clobber)
    info_path = out_root / "dataset_info.json"
    if info_path.exists():
        dataset_info = json.loads(info_path.read_text())
    else:
        dataset_info = {}

    for split in args.splits:
        p_in = retrieval_dir / f"{split}.jsonl"
        if not p_in.exists():
            print(f"  SKIP {split}: {p_in} missing")
            continue

        # Per-split seed so train/eval/test get deterministic-but-different compositions
        rng = random.Random(args.shuffle_seed + hash(split) % 10000)

        rows = []
        with open(p_in) as f:
            for line in f:
                rows.append(json.loads(line))

        sg = []
        skipped = 0
        per_k = {k: 0 for k in range(args.k_min, args.k_max + 1)}
        for r in rows:
            shuffled, gold_pos, k = build_variable_image_set(
                r, rng, args.k_min, args.k_max
            )
            img_paths = [str(out_images / s) for s in shuffled]
            if any(not os.path.exists(pp) for pp in img_paths):
                skipped += 1
                continue
            user_content = ("<image>" * k) + "\n" + r["query"]
            sg.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": r["answer"]},
                    ],
                    "images": img_paths,
                    "_gold_pos": gold_pos,
                    "_k": k,
                    "_gold_in_top6_pos": r.get("gold_in_top6_pos", -1),
                }
            )
            per_k[k] += 1

        out_json = out_root / f"{split}{args.json_suffix}.json"
        with open(out_json, "w") as f:
            json.dump(sg, f, ensure_ascii=False)

        dist = ", ".join(f"k={k}:{per_k[k]}" for k in sorted(per_k))
        print(f"  {split}: {len(sg)} examples, skipped {skipped} ({dist})")

        ds_key = f"{args.ds_key_prefix}_{split}"
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

    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\ndataset_info: {info_path}")


if __name__ == "__main__":
    main()
