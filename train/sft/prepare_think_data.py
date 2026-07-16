#!/usr/bin/env python3
"""Build think-SFT datasets using pre-generated reasoning traces.

For each compression level (2x/3x/5x/9x), produce ShareGPT-format training
data where the assistant target is `<think>{reasoning}</think>{answer}`.

Also produces a mixed-compression dataset that concatenates all 4.

Input: think_traces_Nk.jsonl with {query, chunk_path, answer, reasoning}
Output: per-compression and mixed ShareGPT JSON + dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COMPRESSIONS = ["2x", "3x", "5x", "9x"]
BASE_DATA = "/scratch/users/zwcolin/cxr_embeds/sft_data"


def format_assistant(reasoning: str, answer: str) -> str:
    # Qwen3 thinking format: <think>...</think>answer
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--traces",
        required=True,
        help="JSONL with {query, chunk_path, answer, reasoning}",
    )
    p.add_argument("--output-dir", default=f"{BASE_DATA}/think_sft")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load traces
    with open(args.traces) as f:
        traces = [json.loads(line) for line in f]
    traces = [t for t in traces if t.get("reasoning")]
    print(f"Loaded {len(traces)} traces")

    # For each compression, build think-SFT examples pointing at the compressed image
    info = {}
    rng = random.Random(args.seed)

    per_comp = {}
    for c in COMPRESSIONS:
        c_dir = Path(f"{BASE_DATA}/compressed_{c}/images")
        data = []
        skipped = 0
        for t in traces:
            # Image path: compressed_Nx/images/<src_rel>
            src_rel = t["chunk_path"]
            img_path = str(c_dir / src_rel)
            if not Path(img_path).exists():
                skipped += 1
                continue
            data.append(
                {
                    "messages": [
                        {"role": "user", "content": "<image>\n" + t["query"]},
                        {
                            "role": "assistant",
                            "content": format_assistant(t["reasoning"], t["answer"]),
                        },
                    ],
                    "images": [img_path],
                }
            )
        print(f"  {c}: {len(data)} ({skipped} skipped)")
        out_json = out / f"train_{c}.json"
        out_json.write_text(json.dumps(data, ensure_ascii=False))
        per_comp[c] = data
        info[f"think_train_{c}"] = {
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

    # Mixed dataset: concat all 4
    mixed = []
    for c in COMPRESSIONS:
        mixed.extend(per_comp[c])
    rng.shuffle(mixed)
    print(f"  mixed: {len(mixed)}")
    mixed_json = out / "train_mixed.json"
    mixed_json.write_text(json.dumps(mixed, ensure_ascii=False))
    info["think_train_mixed"] = {
        "file_name": str(mixed_json),
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    # Reuse existing eval sets (no reasoning needed, we still eval on plain Q→A)
    # ... actually to monitor think loss we should have think eval too. Let's skip for now —
    # just point eval_dataset at existing compressed_qa_eval per compression if needed.
    # For simplicity, write a small think_eval (5x only) from first 500 eval examples with reasoning.
    # Actually: eval on plain answer is fine, training loss monitors itself.

    # For LF, point eval_dataset to the existing non-think eval of the matching compression.
    # We reference those via symlink or just list them here as mixed_eval_Nx
    for c in COMPRESSIONS:
        info[f"think_eval_{c}"] = {
            "file_name": f"{BASE_DATA}/compressed_{c}/eval.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }

    info_path = out / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"\nDataset info: {info_path}")


if __name__ == "__main__":
    main()
