#!/usr/bin/env python3
"""Build a mixed-compression training dataset by concatenating ShareGPT JSON
files from 2x / 3x / 5x / 9x compression dirs.

Each original chunk appears 4 times, once per compression level, with the SAME
query+answer but different blur. This trains a single adapter that's robust
across compression levels.

Eval: use per-compression eval.json (separate eval datasets in dataset_info).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COMPRESSIONS = ["2x", "3x", "5x", "9x"]
BASE = "/scratch/users/zwcolin/cxr_embeds/sft_data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=f"{BASE}/compressed_mixed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Train: concat all 4 compressions, shuffle ===
    train_all = []
    for c in COMPRESSIONS:
        src = Path(f"{BASE}/compressed_{c}/train.json")
        data = json.loads(src.read_text())
        print(f"  {c}/train.json: {len(data)} examples")
        # Tag with compression for provenance (optional)
        for ex in data:
            ex["_compression"] = c
        train_all.extend(data)
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    print(f"Total train (mixed): {len(train_all)}")
    # Strip _compression tag before save (LF doesn't use it)
    for ex in train_all:
        ex.pop("_compression", None)
    (out / "train.json").write_text(json.dumps(train_all, ensure_ascii=False))

    # === Eval: keep per-compression eval sets ===
    for c in COMPRESSIONS:
        src = Path(f"{BASE}/compressed_{c}/eval.json")
        dst = out / f"eval_{c}.json"
        dst.write_text(src.read_text())
        print(f"  eval_{c}.json: copied {len(json.loads(dst.read_text()))} examples")

    # === dataset_info.json ===
    info = {
        "mixed_train": {
            "file_name": str(out / "train.json"),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    for c in COMPRESSIONS:
        info[f"mixed_eval_{c}"] = {
            "file_name": str(out / f"eval_{c}.json"),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDataset info: {out / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
