#!/usr/bin/env python3
"""Generate fake (query, chunk_path) pairs for training pipeline validation.

Randomly samples chunks from kiwix_tiles, uses article titles to create
synthetic queries, and splits into train/eval JSONL files.

Usage:
    uv run python training/fake_data.py \
        --tiles-dir /opt/dlami/nvme/kiwix_tiles \
        --articles-json /opt/dlami/nvme/kiwix/wikipedia_en_all_maxi_2025-08.zim.articles.json \
        --output-dir training/data \
        --num-articles 1000
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUERY_TEMPLATES = [
    "What is {title}?",
    "Tell me about {title}",
    "{title} overview",
]


def find_chunk_paths(tiles_dir: Path, article_id: int) -> list[str]:
    """Find all chunk PNG paths for a given article ID."""
    shard_size = 8284
    top_shard = article_id // shard_size
    top_dir = tiles_dir / f"shard_{top_shard:03d}"
    if not top_dir.exists():
        return []
    tile_dir_name = f"{article_id}.png.tiles"
    # Search sub-shards
    for sub in top_dir.iterdir():
        if not sub.is_dir() or not sub.name.startswith("shard_"):
            continue
        candidate = sub / tile_dir_name
        if candidate.exists():
            chunks_json = candidate / "chunks.json"
            if chunks_json.exists():
                try:
                    chunks = json.loads(chunks_json.read_text())
                    return [
                        str(candidate / c["file"])
                        for c in chunks.get("chunks", [])
                        if (candidate / c["file"]).exists()
                    ]
                except (json.JSONDecodeError, KeyError):
                    pass
    return []


def title_from_slug(slug: str) -> str:
    """Convert URL slug to readable title."""
    title = slug.replace("_", " ")
    # Remove URL encoding
    title = re.sub(r"%[0-9A-Fa-f]{2}", " ", title)
    return title.strip()


def generate_queries(title: str) -> list[str]:
    """Generate fake queries from article title."""
    return [t.format(title=title) for t in QUERY_TEMPLATES]


def main():
    parser = argparse.ArgumentParser(description="Generate fake training data")
    parser.add_argument(
        "--tiles-dir", type=Path, default=Path("/opt/dlami/nvme/kiwix_tiles")
    )
    parser.add_argument(
        "--articles-json",
        type=Path,
        default=Path(
            "/opt/dlami/nvme/kiwix/wikipedia_en_all_maxi_2025-08.zim.articles.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/data"))
    parser.add_argument("--num-articles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading articles.json...")
    articles = json.loads(args.articles_json.read_text())
    num_articles = len(articles)
    logger.info(f"Loaded {num_articles} article slugs")

    # Sample random article IDs and find ones with chunks
    pairs = []
    sampled = 0
    indices = list(range(num_articles))
    random.shuffle(indices)

    for aid in indices:
        if len(pairs) >= args.num_articles * 3:  # 3 queries per article
            break
        sampled += 1
        slug = articles[aid]
        if not slug or slug.startswith("_"):
            continue
        chunk_paths = find_chunk_paths(args.tiles_dir, aid)
        if not chunk_paths:
            continue
        title = title_from_slug(slug)
        queries = generate_queries(title)
        for query in queries:
            # Pick a random chunk for this query
            chunk_path = random.choice(chunk_paths)
            pairs.append({"query": query, "chunk_path": chunk_path})

        if len(pairs) % 300 == 0:
            logger.info(
                f"Generated {len(pairs)} pairs from {sampled} sampled articles..."
            )

    random.shuffle(pairs)
    logger.info(f"Total pairs: {len(pairs)} from {sampled} sampled articles")

    # Split 80/20
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"

    for path, data in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with open(path, "w") as f:
            f.writelines(json.dumps(item) + "\n" for item in data)
        logger.info(f"Wrote {len(data)} pairs to {path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
