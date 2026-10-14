"""Export a frozen-retrieval benchmark: questions + the top-k tiles a pixel serve returned.

`run_bench.py --dump-retrieval DIR` loads examples and builds the retriever exactly as a
normal run does, then writes what retrieval produced instead of calling a reader. The
result lets a reader model be evaluated without hosting the index or the tile corpus.

Layout under DIR:
  records.jsonl            one line per example (see _record); resumable by example_id
  tiles/<article_id>/<chunk file>   each retrieved tile, decoded from the serve's image_base64
  missing_tiles.jsonl      hits the serve returned without an image (its tile is absent on
                           the serve's disk); these are filled from the tile corpus later
  query_images/<id>.<ext>  the query image, for multimodal tasks
  dump.log                 per-example start/end lines
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Examples sent to the serve per prefetch call; also the checkpoint granularity.
CHUNK = 32


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tile_rel(hit: dict) -> str:
    # Keyed by article, not the serve's path: the serve prints an unresolved sub-shard as "?".
    return f"tiles/{hit['article_id']}/{os.path.basename(hit['path'])}"


def _record(
    example: dict, hits: list[dict], query_image: str | None, run_metadata: dict
) -> dict:
    return {
        "example_id": example["id"],
        "load_index": example.get("_load_index"),
        "problem": example["problem"],
        # Same filter run_bench applies to original_data, so lib.grader reads these
        # records unchanged once a "final_response" field is added.
        "original_data": {
            k: v
            for k, v in example.items()
            if not hasattr(v, "save") and not k.startswith("_")
        },
        "query_image": query_image,
        "retrieved_images": [
            {
                "rank": rank,
                "path": _tile_rel(h),
                "serve_path": h["path"],
                "score": h.get("score"),
                "url": h.get("url"),
                "article_id": h.get("article_id"),
                "chunk_index": h.get("chunk_index"),
                "y_offset": h.get("y_offset"),
            }
            for rank, h in enumerate(hits)
        ],
        "run_metadata": run_metadata,
    }


async def dump_retrieval(
    args, examples: list[dict], retriever, run_metadata: dict
) -> None:
    out_dir = args.dump_retrieval
    os.makedirs(os.path.join(out_dir, "tiles"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "query_images"), exist_ok=True)
    records_path = os.path.join(out_dir, "records.jsonl")

    done: set[str] = set()
    if os.path.exists(records_path):
        with open(records_path) as f:
            done = {json.loads(line)["example_id"] for line in f if line.strip()}

    log = open(os.path.join(out_dir, "dump.log"), "a", buffering=1)
    log.write(
        f"[{_now()}] run_start task={args.task} n={len(examples)} already_done={len(done)} "
        f"api={args.local_api_url} top_k={args.retrieval_top_k} nprobe={args.nprobe} "
        f"commit={run_metadata.get('git_commit')}\n"
    )

    todo = [ex for ex in examples if ex["id"] not in done]
    for ex in examples:
        if ex["id"] in done:
            log.write(
                f"[{_now()}] item={ex['id']} status=skip reason=already_in_records\n"
            )

    with (
        open(records_path, "a", buffering=1) as rec_f,
        open(os.path.join(out_dir, "missing_tiles.jsonl"), "a", buffering=1) as miss_f,
    ):
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            for ex in chunk:
                log.write(f"[{_now()}] item={ex['id']} status=start\n")
            t0 = time.time()
            await retriever.prefetch(chunk)
            elapsed = time.time() - t0

            for ex in chunk:
                # A task can repeat an example id (NQ-Tables does); the retriever cache then
                # holds both copies' hits back to back, so keep only the first top_k.
                hits = retriever._cache.get(ex["id"], [])[: args.retrieval_top_k]
                if len(hits) < args.retrieval_top_k:
                    # Not written, so a rerun retries it.
                    log.write(
                        f"[{_now()}] item={ex['id']} status=fail hits={len(hits)} "
                        f"chunk_elapsed={elapsed:.1f}s\n"
                    )
                    continue

                missing = []
                for h in hits:
                    dst = os.path.join(out_dir, _tile_rel(h))
                    if not h.get("image_base64"):
                        missing.append(h)
                        continue
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst + ".part", "wb") as f:
                            f.write(base64.b64decode(h["image_base64"]))
                        os.replace(dst + ".part", dst)

                query_image = None
                src = retriever.query_image_fn(ex) if retriever.query_image_fn else None
                if src and os.path.exists(src):
                    ext = os.path.splitext(src)[1] or ".png"
                    query_image = f"query_images/{ex['id']}{ext}"
                    shutil.copyfile(src, os.path.join(out_dir, query_image))

                for h in missing:
                    miss_f.write(
                        json.dumps(
                            {
                                "example_id": ex["id"],
                                "article_id": h["article_id"],
                                "serve_path": h["path"],
                                "tile": _tile_rel(h),
                            }
                        )
                        + "\n"
                    )
                rec_f.write(
                    json.dumps(
                        _record(ex, hits, query_image, run_metadata), ensure_ascii=False
                    )
                    + "\n"
                )
                log.write(
                    f"[{_now()}] item={ex['id']} status=ok hits={len(hits)} "
                    f"missing_tiles={len(missing)}"
                    f"{' ' + ','.join(h['path'] for h in missing) if missing else ''} "
                    f"query_image={'yes' if query_image else 'no'} chunk_elapsed={elapsed:.1f}s\n"
                )
            print(
                f"[dump] {min(start + CHUNK, len(todo))}/{len(todo)} examples ({elapsed:.1f}s last chunk)"
            )

    log.write(f"[{_now()}] run_end task={args.task}\n")
    log.close()
