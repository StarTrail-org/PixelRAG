#!/usr/bin/env python3
"""Pre-chunk tile images into model-sized pieces on disk.

For each article directory (*.png.tiles/), reads every tile_XXXX.png and splits
it into a grid of <=1024px-tall x <=viewport_width-wide chunks (writing
chunk_XXXX_YY.png files) plus a chunks.json manifest recording each chunk's
x_offset/y_offset/width/height. Narrow web tiles (<= viewport_width) keep their
old single-column height-strip layout; wider sources (PDFs, landscape pages) are
also split along the width so the embedder never has to drop an oversized chunk.

Usage:
    # Single shard
    python chunk_tiles.py --shard-dir /opt/dlami/nvme/kiwix_tiles/shard_100

    # All shards (parallel)
    python chunk_tiles.py --tiles-dir /opt/dlami/nvme/kiwix_tiles --workers 96

    # Force rechunk (overwrite existing chunks, compare tile hashes)
    python chunk_tiles.py --tiles-dir /opt/dlami/nvme/kiwix_tiles --workers 96 --force

    # Force rechunk + delete tiles after each shard
    python chunk_tiles.py --tiles-dir /opt/dlami/nvme/kiwix_tiles --workers 96 --force --delete-tiles

    # Dry run (count chunks without writing)
    python chunk_tiles.py --tiles-dir /opt/dlami/nvme/kiwix_tiles --dry-run
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some tiles exceed default 178M pixel limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chunk_tiles")

CHUNK_HEIGHT = 1024
MIN_CHUNK_HEIGHT = 28  # one Qwen3-VL patch; merge tiny tails into previous


def _compute_tile_hashes(article_dir: str, tile_names: list[str]) -> dict[str, str]:
    """Compute MD5 hashes for all tile files."""
    hashes = {}
    for tn in tile_names:
        tp = os.path.join(article_dir, tn)
        if os.path.exists(tp):
            h = hashlib.md5()
            with open(tp, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            hashes[tn] = h.hexdigest()
    return hashes


def chunk_article(article_dir: str, dry_run: bool = False, force: bool = False) -> dict:
    """Chunk all tiles in one article directory.

    Args:
        article_dir: Path to *.png.tiles/ directory.
        dry_run: If True, compute chunks but don't write files.
        force: If True, rechunk even if chunks.json exists (compare tile hashes).

    Returns:
        dict with chunking results, or None if up-to-date / skipped.
    """
    tiles_json = os.path.join(article_dir, "tiles.json")
    chunks_json = os.path.join(article_dir, "chunks.json")

    if not os.path.exists(tiles_json):
        return None

    with open(tiles_json) as f:
        raw = f.read().strip()
    if not raw:
        return None
    meta = json.loads(raw)

    tile_names = meta.get("tiles", [])
    if not tile_names:
        return None

    # Compute tile hashes (stored in manifest for future change detection)
    tile_hashes = _compute_tile_hashes(article_dir, tile_names)

    # If no tiles exist on disk, skip — never delete existing chunks without tiles to rechunk
    if not tile_hashes:
        return None

    if os.path.exists(chunks_json):
        try:
            with open(chunks_json) as f:
                old_manifest = json.load(f)
        except (json.JSONDecodeError, KeyError):
            old_manifest = None

        # Always verify chunk files actually exist on disk
        chunks_ok = old_manifest is not None and all(
            os.path.exists(os.path.join(article_dir, c["file"]))
            for c in old_manifest.get("chunks", [])
        )

        if chunks_ok:
            if not force:
                return None  # chunks exist, not forced, skip
            # Force: also check tile hashes
            old_hashes = old_manifest.get("tile_hashes", {})
            if old_hashes and old_hashes == tile_hashes:
                return None  # tiles unchanged and chunks exist, skip

        # Hashes differ or missing — delete old chunk files before rechunking
        if not dry_run:
            for f in os.listdir(article_dir):
                if f.startswith("chunk_") and f.endswith((".png", ".jpg", ".jpeg")):
                    os.unlink(os.path.join(article_dir, f))

    page_height = meta.get("page_height", 0)
    viewport_width = meta.get("viewport_width", 875)
    tile_height = meta.get("tile_height", 8192)

    chunks_info = []  # list of {tile, chunk_index, file, y_offset, height}
    files_written = 0

    for tile_name in tile_names:
        tile_path = os.path.join(article_dir, tile_name)
        if not os.path.exists(tile_path):
            continue

        try:
            img = Image.open(tile_path)
            w, h = img.size
        except Exception as e:
            logger.warning("Skipping corrupt tile %s: %s", tile_path, e)
            continue
        # Handle both .png and .jpg tile files
        tile_base = tile_name.replace("tile_", "")
        for ext in (".png", ".jpg", ".jpeg"):
            tile_base = tile_base.replace(ext, "")
        tile_idx = int(tile_base)

        # Fast path: web tiles (<= viewport_width) that fit one strip are copied
        # verbatim — byte-identical to the pre-2D-tiling behavior.
        if w <= viewport_width and h <= CHUNK_HEIGHT:
            chunk_name = f"chunk_{tile_idx:04d}_00.png"
            chunk_path = os.path.join(article_dir, chunk_name)
            if not dry_run:
                shutil.copy2(tile_path, chunk_path)
                files_written += 1
            chunks_info.append(
                {
                    "tile": tile_name,
                    "tile_index": tile_idx,
                    "chunk_index": 0,
                    "file": chunk_name,
                    "x_offset": 0,
                    "y_offset": 0,
                    "height": h,
                    "width": w,
                }
            )
            continue

        # 2D grid: CHUNK_HEIGHT-tall row strips x viewport_width-wide columns.
        # Columns are a full viewport_width each (the model's native width) with
        # the remainder in the last column — not evened out — so most content
        # lands at the in-distribution width the index was built on. chunk_index
        # is a flat row-major counter, so single-column tiles keep the same
        # 0, 1, 2, ... order (and identical crops) as before.
        chunk_idx = 0
        y = 0
        while y < h:
            ch = min(CHUNK_HEIGHT, h - y)
            # Discard tiny height tail (< 28px = one Qwen3-VL patch)
            if ch < MIN_CHUNK_HEIGHT:
                break

            x = 0
            while x < w:
                cw = min(viewport_width, w - x)
                if cw < MIN_CHUNK_HEIGHT:  # discard tiny right-edge sliver
                    break

                chunk_name = f"chunk_{tile_idx:04d}_{chunk_idx:02d}.png"
                chunk_path = os.path.join(article_dir, chunk_name)
                if not dry_run:
                    img.crop((x, y, x + cw, y + ch)).save(chunk_path, format="PNG")
                    files_written += 1

                chunks_info.append(
                    {
                        "tile": tile_name,
                        "tile_index": tile_idx,
                        "chunk_index": chunk_idx,
                        "file": chunk_name,
                        "x_offset": x,
                        "y_offset": y,
                        "height": ch,
                        "width": cw,
                    }
                )
                chunk_idx += 1
                x += cw

            y += ch

        img.close()

    if not chunks_info:
        return None

    # Write chunks.json
    manifest = {
        "page_height": page_height,
        "viewport_width": viewport_width,
        "tile_height": tile_height,
        "chunk_height": CHUNK_HEIGHT,
        "num_tiles": len(tile_names),
        "num_chunks": len(chunks_info),
        "tile_hashes": tile_hashes,
        "chunks": chunks_info,
    }

    if not dry_run:
        with open(chunks_json, "w") as f:
            json.dump(manifest, f)

    return {
        "article_dir": article_dir,
        "num_tiles": len(tile_names),
        "num_chunks": len(chunks_info),
        "files_written": files_written,
    }


def _delete_tiles_in_shard(shard_dir: str) -> int:
    """Delete tile_*.png for all articles with chunks.json in a shard.

    Tiles referenced directly as chunk files (h <= 1024) are preserved.
    """
    deleted = 0
    for sub in Path(shard_dir).iterdir():
        if not sub.is_dir() or not sub.name.startswith("shard_"):
            continue
        for article_dir in sub.iterdir():
            if not article_dir.is_dir() or not article_dir.name.endswith(".png.tiles"):
                continue
            cj_path = article_dir / "chunks.json"
            if not cj_path.exists():
                continue
            # Collect tile files referenced as chunks (small tiles, not split)
            try:
                with open(cj_path) as f:
                    manifest = json.load(f)
                keep = {
                    c["file"]
                    for c in manifest.get("chunks", [])
                    if c["file"].startswith("tile_")
                }
            except (json.JSONDecodeError, KeyError):
                keep = set()
            for f in article_dir.iterdir():
                if f.name.startswith("tile_") and f.name.endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    if f.name not in keep:
                        f.unlink()
                        deleted += 1
    return deleted


def process_shard(
    shard_dir: str,
    dry_run: bool = False,
    force: bool = False,
    delete_tiles: bool = False,
) -> dict:
    """Chunk all articles in a shard directory."""
    t0 = time.time()
    total_articles = 0
    chunked_articles = 0
    skipped_articles = 0
    total_tiles = 0
    total_chunks = 0
    total_files = 0

    # Walk sub-shard directories (shard_00000, shard_00001, ...)
    sub_dirs = sorted(
        p
        for p in Path(shard_dir).iterdir()
        if p.is_dir() and p.name.startswith("shard_")
    )

    if not sub_dirs:
        # Flat structure — article dirs directly in shard_dir
        sub_dirs = [Path(shard_dir)]

    for sub_dir in sub_dirs:
        for article_dir in sorted(sub_dir.iterdir()):
            if not article_dir.is_dir() or not article_dir.name.endswith(".png.tiles"):
                continue
            total_articles += 1

            result = chunk_article(str(article_dir), dry_run=dry_run, force=force)
            if result is None:
                skipped_articles += 1
                continue

            chunked_articles += 1
            total_tiles += result["num_tiles"]
            total_chunks += result["num_chunks"]
            total_files += result["files_written"]

    # Delete tiles after chunking the whole shard
    tiles_deleted = 0
    if delete_tiles and not dry_run:
        tiles_deleted = _delete_tiles_in_shard(shard_dir)

    elapsed = time.time() - t0
    shard_name = os.path.basename(shard_dir.rstrip("/"))
    return {
        "shard": shard_name,
        "articles": total_articles,
        "chunked": chunked_articles,
        "skipped": skipped_articles,
        "tiles": total_tiles,
        "chunks": total_chunks,
        "files_written": total_files,
        "tiles_deleted": tiles_deleted,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-chunk tile images into 1024px strips"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard-dir", help="Process a single shard directory")
    group.add_argument("--tiles-dir", help="Process all shards under this directory")
    parser.add_argument(
        "--workers", type=int, default=96, help="Parallel shard workers (default: 96)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count chunks without writing files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rechunk even if chunks.json exists (compare tile hashes, skip if unchanged)",
    )
    parser.add_argument(
        "--delete-tiles",
        action="store_true",
        help="Delete tile_*.png after chunking each shard",
    )
    args = parser.parse_args()

    if args.shard_dir:
        logger.info(
            "Processing single shard: %s (force=%s, delete_tiles=%s)",
            args.shard_dir,
            args.force,
            args.delete_tiles,
        )
        result = process_shard(
            args.shard_dir,
            dry_run=args.dry_run,
            force=args.force,
            delete_tiles=args.delete_tiles,
        )
        logger.info("Result: %s", result)
        return

    # Process all shards
    tiles_dir = args.tiles_dir
    shard_dirs = sorted(
        str(p)
        for p in Path(tiles_dir).iterdir()
        if p.is_dir() and p.name.startswith("shard_")
    )
    logger.info(
        "Found %d shards in %s (workers=%d, force=%s, delete_tiles=%s, dry_run=%s)",
        len(shard_dirs),
        tiles_dir,
        args.workers,
        args.force,
        args.delete_tiles,
        args.dry_run,
    )

    t0 = time.time()
    total = {
        "shards": 0,
        "articles": 0,
        "chunked": 0,
        "skipped": 0,
        "tiles": 0,
        "chunks": 0,
        "files_written": 0,
        "tiles_deleted": 0,
    }

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_shard, sd, args.dry_run, args.force, args.delete_tiles
            ): sd
            for sd in shard_dirs
        }
        for fut in as_completed(futures):
            sd = futures[fut]
            try:
                r = fut.result()
                total["shards"] += 1
                total["articles"] += r["articles"]
                total["chunked"] += r["chunked"]
                total["skipped"] += r["skipped"]
                total["tiles"] += r["tiles"]
                total["chunks"] += r["chunks"]
                total["files_written"] += r["files_written"]
                total["tiles_deleted"] += r["tiles_deleted"]
                if r["chunked"] > 0 or r["tiles_deleted"] > 0:
                    logger.info(
                        "%s: %d chunked, %d tiles → %d chunks, %d written, %d tiles deleted (%.1fs)",
                        r["shard"],
                        r["chunked"],
                        r["tiles"],
                        r["chunks"],
                        r["files_written"],
                        r["tiles_deleted"],
                        r["elapsed_s"],
                    )
            except Exception as e:
                logger.error("Failed %s: %s", sd, e)

    elapsed = time.time() - t0
    logger.info(
        "Done: %d shards, %d articles chunked (%d skipped), "
        "%d tiles → %d chunks, %d files written, %d tiles deleted in %.0fs",
        total["shards"],
        total["chunked"],
        total["skipped"],
        total["tiles"],
        total["chunks"],
        total["files_written"],
        total["tiles_deleted"],
        elapsed,
    )


if __name__ == "__main__":
    main()
