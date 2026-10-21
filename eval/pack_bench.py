"""Pack dump_bench.sh output into Parquet configs for a Hugging Face dataset repo.

    python pack_bench.py <dump_root> <out_dir> [--commit SHA] [--allow-missing FILE]

Reads <dump_root>/<bench>/records.jsonl plus the tiles and query images beside it and
writes <out_dir>/<config>/test-*.parquet, one config per bench, images embedded.
A missing or empty image stops the pack, unless its tile is listed (one "<article_id>/<chunk
file>" per line) in --allow-missing: those tiles exist nowhere any more and are packed as null.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from datasets import Dataset, Features, Image, Sequence, Value

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.grader import build_ground_truth

# dump_bench.sh bench name -> (config name, grader task)
BENCHES = {
    "nq": ("nq", "nq"),
    "nqt": ("nq_tables", "nq_tables"),
    "sqa": ("simpleqa", "simpleqa"),
    "mms": ("mmsearch", "mmsearch"),
    "evqa": ("encyclopedic_vqa_landmarks", "encyclopedic_vqa"),
}

FEATURES = Features(
    {
        "id": Value("string"),
        "question": Value("string"),
        "instructions": Value("string"),
        "answer": Value("string"),
        "query_image": Image(),
        "retrieved_images": Sequence(Image()),
        "retrieved_scores": Sequence(Value("float32")),
        "retrieved_urls": Sequence(Value("string")),
        "original_data": Value("string"),
    }
)


def _img(
    bench_dir: str, rel: str, allow_missing: frozenset = frozenset()
) -> dict | None:
    path = os.path.join(bench_dir, rel)
    if not os.path.exists(path) and rel.removeprefix("tiles/") in allow_missing:
        return None
    # Bytes, not paths: to_parquet stores a path-only Image as a dangling reference.
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        raise SystemExit(f"empty image {rel} in {bench_dir}")
    return {"bytes": data, "path": rel}


def rows(
    bench_dir: str, grader_task: str, commit: str | None, allow_missing: frozenset
):
    with open(os.path.join(bench_dir, "records.jsonl")) as f:
        recs = sorted((json.loads(line) for line in f), key=lambda r: r["load_index"])
    for r in recs:
        if commit:
            r["run_metadata"]["git_commit"] = commit
        od = r["original_data"]
        hits = r["retrieved_images"]
        yield {
            "id": r["example_id"],
            "question": r["problem"],
            "instructions": od.get("additional_instructions") or "",
            "answer": build_ground_truth(grader_task, od),
            "query_image": _img(bench_dir, r["query_image"])
            if r["query_image"]
            else None,
            "retrieved_images": [
                _img(bench_dir, h["path"], allow_missing) for h in hits
            ],
            "retrieved_scores": [h["score"] for h in hits],
            "retrieved_urls": [h["url"] for h in hits],
            "original_data": json.dumps(
                {"original_data": od, "run_metadata": r["run_metadata"]},
                ensure_ascii=False,
            ),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_root")
    ap.add_argument("out_dir")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--allow-missing", default=None)
    args = ap.parse_args()
    allow_missing = frozenset()
    if args.allow_missing:
        allow_missing = frozenset(
            line.strip() for line in open(args.allow_missing) if line.strip()
        )

    for bench, (config, grader_task) in BENCHES.items():
        bench_dir = os.path.join(args.dump_root, bench)
        if not os.path.exists(os.path.join(bench_dir, "records.jsonl")):
            print(f"[pack] {bench}: no records, skipped")
            continue
        ds = Dataset.from_list(
            list(rows(bench_dir, grader_task, args.commit, allow_missing)),
            features=FEATURES,
        )
        out = os.path.join(args.out_dir, config)
        os.makedirs(out, exist_ok=True)
        n_shards = max(1, round(ds.data.nbytes / 500e6) or 1)
        for i in range(n_shards):
            ds.shard(n_shards, i, contiguous=True).to_parquet(
                os.path.join(out, f"test-{i:05d}-of-{n_shards:05d}.parquet")
            )
        print(f"[pack] {bench} -> {config}: {len(ds)} rows, {n_shards} shard(s)")


if __name__ == "__main__":
    main()
