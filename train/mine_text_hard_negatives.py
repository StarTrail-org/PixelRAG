#!/usr/bin/env python3
"""Mine text hard negatives using a text search API.

Input rows are expected to contain at least:
- query
- article_id
- chunk_index

The script queries the text search endpoint with the query text, keeps the
retrieved top-K hits, and selects the first N non-positive hits as hard
negatives. No false-negative filtering is applied.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def search_batch(search_url: str, queries: list[str], n_docs: int) -> list[list[dict]]:
    payload = {
        "queries": [{"text": q} for q in queries],
        "n_docs": n_docs,
    }
    resp = requests.post(search_url, json=payload, timeout=120)
    resp.raise_for_status()
    results = resp.json()["results"]
    return [r.get("hits", []) for r in results]


def positive_key(row: dict) -> tuple[int | None, int | None]:
    article_id = row.get("article_id")
    chunk_index = row.get("chunk_index")
    return article_id, chunk_index


def hit_key(hit: dict) -> tuple[int | None, int | None]:
    return hit.get("article_id"), hit.get("chunk_index")


def normalize_hit(hit: dict, rank: int) -> dict:
    return {
        "rank": rank + 1,
        "score": hit.get("score", 0.0),
        "article_id": hit.get("article_id"),
        "chunk_index": hit.get("chunk_index"),
        "char_offset": hit.get("char_offset"),
        "n_tokens": hit.get("n_tokens"),
        "title": hit.get("title"),
        "url": hit.get("url"),
        "text": hit.get("text", ""),
    }


def mine_from_search(
    pairs: list[dict],
    search_url: str,
    num_negatives: int = 7,
    n_docs: int = 20,
    batch_size: int = 64,
    search_workers: int = 1,
) -> tuple[list[dict], dict]:
    unique_queries = list(dict.fromkeys(p["query"] for p in pairs))
    query_to_idx = {q: i for i, q in enumerate(unique_queries)}
    logger.info("%d unique queries (from %d pairs)", len(unique_queries), len(pairs))

    query_positives: dict[str, set[tuple[int | None, int | None]]] = {}
    for pair in pairs:
        query_positives.setdefault(pair["query"], set()).add(positive_key(pair))

    all_hits: list[list[dict] | None] = [None] * len(unique_queries)
    batches = []
    for i in range(0, len(unique_queries), batch_size):
        batch_queries = unique_queries[i : i + batch_size]
        batches.append((i, batch_queries, i // batch_size + 1))
    n_batches = len(batches)
    completed = 0

    def run_batch(batch_start: int, batch_queries: list[str], batch_idx: int):
        return (
            batch_start,
            batch_queries,
            batch_idx,
            search_batch(search_url, batch_queries, n_docs=n_docs),
        )

    with ThreadPoolExecutor(max_workers=max(1, search_workers)) as executor:
        futures = {
            executor.submit(run_batch, batch_start, batch_queries, batch_idx): (
                batch_start,
                batch_queries,
                batch_idx,
            )
            for batch_start, batch_queries, batch_idx in batches
        }
        for future in as_completed(futures):
            batch_start, batch_queries, batch_idx = futures[future]
            try:
                _, _, _, batch_hits = future.result()
                for j, hits in enumerate(batch_hits):
                    all_hits[batch_start + j] = hits
            except Exception as exc:
                logger.warning("Batch %d/%d failed: %s", batch_idx, n_batches, exc)
                for j in range(len(batch_queries)):
                    all_hits[batch_start + j] = []
            completed += len(batch_queries)
            if batch_idx % 10 == 0 or completed == len(unique_queries):
                logger.info("  Searched: %d/%d", completed, len(unique_queries))

    query_negatives: dict[str, list[dict]] = {}
    query_metadata: dict[str, dict] = {}
    stats = {
        "total": 0,
        "with_negs": 0,
        "avg_negs": 0.0,
        "avg_pos_rank": 0.0,
        "pos_found_rate": 0.0,
        "pos_recall@1": 0.0,
        "pos_recall@10": 0.0,
        "pos_recall@20": 0.0,
    }
    pos_ranks: list[int] = []

    for q in unique_queries:
        positives = query_positives[q]
        hits = all_hits[query_to_idx[q]] or []
        neg_hits: list[dict] = []
        pos_rank = None
        pos_score = None

        for rank, hit in enumerate(hits):
            hk = hit_key(hit)
            if hk in positives:
                if pos_rank is None:
                    pos_rank = rank
                    pos_score = hit.get("score", 0.0)
                continue
            if len(neg_hits) < num_negatives:
                neg_hits.append(normalize_hit(hit, rank))

        query_negatives[q] = neg_hits
        query_metadata[q] = {
            "retrieve_top20": [
                normalize_hit(hit, rank) for rank, hit in enumerate(hits)
            ],
            "positive_rank": pos_rank + 1 if pos_rank is not None else 0,
            "positive_score": pos_score if pos_score is not None else 0.0,
        }
        stats["total"] += 1
        if neg_hits:
            stats["with_negs"] += 1
        if pos_rank is not None:
            pos_ranks.append(pos_rank)

    output_pairs = []
    for pair in pairs:
        neg_hits = query_negatives.get(pair["query"], [])
        meta = query_metadata.get(pair["query"], {})
        output_pairs.append(
            {
                **pair,
                "neg_hits": neg_hits,
                "neg_passages": [hit.get("text", "") for hit in neg_hits],
                "retrieve_top20": meta.get("retrieve_top20", []),
                "positive_score": meta.get("positive_score", 0.0),
                "positive_rank": meta.get("positive_rank", 0),
            }
        )

    total_queries = len(unique_queries)
    if total_queries > 0:
        stats["avg_negs"] = (
            sum(len(query_negatives[q]) for q in unique_queries) / total_queries
        )
        stats["pos_found_rate"] = len(pos_ranks) / total_queries
        stats["pos_recall@1"] = sum(1 for r in pos_ranks if r == 0) / total_queries
        stats["pos_recall@10"] = sum(1 for r in pos_ranks if r < 10) / total_queries
        stats["pos_recall@20"] = sum(1 for r in pos_ranks if r < 20) / total_queries
    if pos_ranks:
        stats["avg_pos_rank"] = sum(pos_ranks) / len(pos_ranks)

    return output_pairs, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL with query/article_id/chunk_index"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL with added neg_hits/neg_passages"
    )
    parser.add_argument("--search-url", default="http://localhost:30889/search")
    parser.add_argument("--health-url", default="http://localhost:30889/health")
    parser.add_argument("--health-timeout", type=int, default=30)
    parser.add_argument("--num-negatives", type=int, default=7)
    parser.add_argument("--n-docs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process the first N rows (0=all)"
    )
    parser.add_argument("--stats-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        resp = requests.get(args.health_url, timeout=args.health_timeout)
        resp.raise_for_status()
        logger.info("Search API is healthy: %s", args.health_url)
    except Exception as exc:
        logger.warning("Search API health check failed, continuing anyway: %s", exc)

    pairs = []
    with open(args.input) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                "query" not in row
                or "article_id" not in row
                or "chunk_index" not in row
            ):
                raise ValueError(f"Missing required fields at line {line_no}")
            pairs.append(row)
            if args.limit > 0 and len(pairs) >= args.limit:
                break
    logger.info("Loaded %d pairs", len(pairs))

    t0 = time.time()
    output_pairs, stats = mine_from_search(
        pairs,
        search_url=args.search_url,
        num_negatives=args.num_negatives,
        n_docs=args.n_docs,
        batch_size=args.batch_size,
        search_workers=args.search_workers,
    )
    elapsed = time.time() - t0

    with open(args.output, "w") as f:
        f.writelines(
            json.dumps(pair, ensure_ascii=False) + "\n" for pair in output_pairs
        )

    n_with_negs = sum(1 for p in output_pairs if p["neg_hits"])
    logger.info("Wrote %d pairs to %s", len(output_pairs), args.output)
    logger.info(
        "  %d with negatives (%.1f%%)",
        n_with_negs,
        100.0 * n_with_negs / max(len(output_pairs), 1),
    )
    logger.info("  Avg negatives per query: %.2f", stats["avg_negs"])
    logger.info("  Avg positive rank: %.2f", stats["avg_pos_rank"])
    logger.info("  Search API recall@1: %.3f", stats["pos_recall@1"])
    logger.info("  Search API recall@10: %.3f", stats["pos_recall@10"])
    logger.info("  Search API recall@20: %.3f", stats["pos_recall@20"])
    logger.info("  Time: %.0fs", elapsed)

    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        logger.info("Wrote stats to %s", args.stats_output)


if __name__ == "__main__":
    main()
