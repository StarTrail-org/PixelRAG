#!/usr/bin/env python3
"""Fetch top-6 retrieval hits for each query in train/eval/test splits.

Sends text queries in batches to the wiki-screenshot search API (defaults
to :30895 which serves dora-ls005-ckpt150). Saves hits + gold info as
JSONL per split. Does NOT fetch tile images — that's download_tiles.py.

Output per line:
  {
    "query": "...",
    "answer": "...",
    "gold_path_rel": "images/shard_583/.../chunk_0000_00.png",
    "gold_suffix": "shard_583/.../chunk_0000_00.png",
    "hits": [
      {"path": "/opt/dlami/nvme/kiwix_tiles/shard_.../chunk_.png",
       "score": 0.70, "url": "...", "article_id": 123}
    ],
    "gold_in_top6_pos": 0  # or -1 if miss
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict


class SplitStats(TypedDict):
    split: str
    total: int
    gold_in_top1: int
    gold_in_top3: int
    gold_in_top6: int
    gold_miss: int


def shard_suffix(p: str) -> str:
    parts = p.split("/")
    for i, x in enumerate(parts):
        if x.startswith("shard_"):
            return "/".join(parts[i:])
    return p


def search_batch(
    api_url: str, queries: list[str], n_docs: int, timeout: int = 300, retries: int = 5
) -> list[dict[str, object]]:
    payload = {"queries": [{"text": q} for q in queries], "n_docs": n_docs}
    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                api_url.rstrip("/") + "/search",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["results"]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            wait = 2**attempt
            print(
                f"  search_batch attempt {attempt + 1}/{retries} failed: {e}; retry in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"search_batch failed after {retries}: {last_err}")


def process_split(
    split_name: str,
    jsonl_path: Path,
    out_path: Path,
    api_url: str,
    batch_size: int,
    n_docs: int,
) -> SplitStats:
    # Resume: count existing lines
    existing = 0
    if out_path.exists():
        with open(out_path) as f:
            for _ in f:
                existing += 1
        print(f"  [{split_name}] resume: {existing} rows already saved")

    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))
    total = len(examples)
    print(f"  [{split_name}] total={total}, skipping first {existing}")

    examples = examples[existing:]
    if not examples:
        print(f"  [{split_name}] already complete, skipping")
        return _collect_stats(out_path)

    t0 = time.time()
    n_done = existing
    gold_in_topk = {1: 0, 3: 0, 6: 0}
    gold_miss = 0

    # Re-scan existing for stats
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                pos = r.get("gold_in_top6_pos", -1)
                if pos < 0:
                    gold_miss += 1
                else:
                    for k in (1, 3, 6):
                        if pos < k:
                            gold_in_topk[k] += 1

    with open(out_path, "a") as out_f:
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            queries = [ex["query"] for ex in batch]
            try:
                results = search_batch(api_url, queries, n_docs=n_docs)
            except Exception as e:
                print(f"  [{split_name}] FATAL at batch {i}: {e}", file=sys.stderr)
                raise

            for ex, res in zip(batch, results):
                gold_rel = ex["chunk_path"]
                gs = shard_suffix(gold_rel)
                hits = res.get("hits", [])
                hit_sufs = [shard_suffix(h["path"]) for h in hits]
                try:
                    pos = hit_sufs.index(gs)
                except ValueError:
                    pos = -1

                if pos < 0:
                    gold_miss += 1
                else:
                    for k in (1, 3, 6):
                        if pos < k:
                            gold_in_topk[k] += 1

                # Keep only fields we need per hit
                trimmed = [
                    {
                        "path": h["path"],
                        "score": h.get("score"),
                        "article_id": h.get("article_id"),
                        "url": h.get("url"),
                    }
                    for h in hits
                ]

                row = {
                    "query": ex["query"],
                    "answer": ex["answer"],
                    "gold_path_rel": gold_rel,
                    "gold_suffix": gs,
                    "hits": trimmed,
                    "gold_in_top6_pos": pos,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            n_done += len(batch)
            batch_idx = i // batch_size
            if batch_idx % 5 == 0 or n_done == total:
                el = time.time() - t0
                rate = (n_done - existing) / max(el, 1e-9)
                eta = (total - n_done) / max(rate, 1e-9) / 60
                print(
                    f"  [{split_name}] {n_done}/{total} "
                    f"({rate:.1f} q/s, eta {eta:.1f} min) "
                    f"gold@1={gold_in_topk[1] / max(1, n_done) * 100:.1f}% "
                    f"gold@3={gold_in_topk[3] / max(1, n_done) * 100:.1f}% "
                    f"gold@6={gold_in_topk[6] / max(1, n_done) * 100:.1f}% "
                    f"miss={gold_miss / max(1, n_done) * 100:.1f}%"
                )

    return {
        "split": split_name,
        "total": n_done,
        "gold_in_top1": gold_in_topk[1],
        "gold_in_top3": gold_in_topk[3],
        "gold_in_top6": gold_in_topk[6],
        "gold_miss": gold_miss,
    }


def _collect_stats(path: Path) -> SplitStats:
    gold_in_topk = {1: 0, 3: 0, 6: 0}
    gold_miss = 0
    n = 0
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n += 1
                pos = r.get("gold_in_top6_pos", -1)
                if pos < 0:
                    gold_miss += 1
                else:
                    for k in (1, 3, 6):
                        if pos < k:
                            gold_in_topk[k] += 1
    return {
        "split": path.stem,
        "total": n,
        "gold_in_top1": gold_in_topk[1],
        "gold_in_top3": gold_in_topk[3],
        "gold_in_top6": gold_in_topk[6],
        "gold_miss": gold_miss,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        default="/scratch/users/zwcolin/cxr_embeds/external_data/screenshot-training-natural-filtered-v2",
    )
    p.add_argument(
        "--output-dir",
        default="/scratch/users/zwcolin/cxr_embeds/sft_data/retrieval_raw",
    )
    p.add_argument("--api-url", default="http://localhost:30895")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-docs", type=int, default=6)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test", "eval", "train"],
        help="Splits to run; order = processing order",
    )
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": "train_hn_with_answer.jsonl",
        "eval": "eval_hn_with_answer.jsonl",
        "test": "test_hn_with_answer.jsonl",
    }

    print(f"API:     {args.api_url}")
    print(f"Output:  {output_dir}")
    print(f"Splits:  {args.splits}")
    print(f"Batch:   {args.batch_size}, n_docs={args.n_docs}")
    print()

    all_stats = []
    for split in args.splits:
        jsonl = dataset_dir / split_files[split]
        out = output_dir / f"{split}.jsonl"
        if not jsonl.exists():
            print(f"  SKIP {split}: {jsonl} missing")
            continue
        print(f"=== {split} ===")
        stats = process_split(
            split, jsonl, out, args.api_url, args.batch_size, args.n_docs
        )
        all_stats.append(stats)

    summary = {
        "api_url": args.api_url,
        "n_docs": args.n_docs,
        "batch_size": args.batch_size,
        "splits": all_stats,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    for s in all_stats:
        n = max(1, s["total"])
        print(
            f"  {s['split']:5s} n={s['total']} "
            f"gold@1={s['gold_in_top1'] / n * 100:5.1f}% "
            f"gold@3={s['gold_in_top3'] / n * 100:5.1f}% "
            f"gold@6={s['gold_in_top6'] / n * 100:5.1f}% "
            f"miss={s['gold_miss'] / n * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
