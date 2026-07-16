#!/usr/bin/env python3
"""LiveVQA evaluation — retrieval-augmented visual question answering on news articles.

LiveVQA is a 5-option multiple-choice VQA benchmark over CNN/BBC/AP News articles.
Each query consists of a question text and an editorial photo. Grading is exact
letter match (A-E), no LLM judge required.

Supports the same modes as the PixelRAG paper:

  naive            No retrieval. Reader sees only editorial photo + question.
  pixel            Pixel retrieval via news tile search API (default :30890).
  text             Text retrieval via news text search API (default :30892).
  ocr              Pixel retrieve -> OCR tiles -> text reader.
  rendered         Text retrieve -> render chunks as images -> pixel reader.
  html             Text retrieve -> HTML DOM lookup -> text reader.
  hybrid           Union of pixel + text retrieval, both modalities to VL reader.

Two-stage pipeline:
  1. Retrieve: batch search via HTTP API (for modes that use retrieval)
  2. Read: concurrent VLM/LLM calls for MCQ answering

Usage examples:

  # Naive (no retrieval)
  python run_livevqa.py --mode naive \\
      --v4 /path/to/livevqa_v4.json \\
      --api-base http://localhost:8211/v1 --model Qwen/Qwen3-VL-4B-Instruct \\
      --output eval_output/livevqa_naive.jsonl

  # Pixel retrieval
  python run_livevqa.py --mode pixel \\
      --v4 /path/to/livevqa_v4.json \\
      --pixel-api http://localhost:30890/search \\
      --api-base http://localhost:8211/v1 --model Qwen/Qwen3-VL-4B-Instruct \\
      --output eval_output/livevqa_pixel.jsonl

  # Text retrieval
  python run_livevqa.py --mode text \\
      --v4 /path/to/livevqa_v4.json \\
      --text-api http://localhost:30892/search \\
      --api-base http://localhost:8200/v1 --model Qwen/Qwen3.5-4B \\
      --output eval_output/livevqa_text.jsonl

  # Hybrid (pixel + text)
  python run_livevqa.py --mode hybrid \\
      --v4 /path/to/livevqa_v4.json \\
      --pixel-api http://localhost:30890/search \\
      --text-api http://localhost:30892/search \\
      --api-base http://localhost:8211/v1 --model Qwen/Qwen3-VL-4B-Instruct \\
      --output eval_output/livevqa_hybrid.jsonl
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

# Add eval root to path so simpleqa imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.llm import LLMClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_livevqa")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LETTERS = "ABCDE"
NEWS_TILES_DIR = "/opt/dlami/nvme/news_tiles"
LIVEVQA_IMAGES_DIR = "/opt/dlami/nvme/livevqa"

# Default v4 JSON (canonical LiveVQA dataset with question/options/GT/img_path)
# LiveVQA dataset (question/options/GT/img_path). External data input — see README.md.
# Override with --v4-path. Retrieval is re-done live; only the QA fields are read from here.
DEFAULT_V4_PATH = os.environ.get(
    "LIVEVQA_V4_PATH", "/mnt/data/yichuan/livevqa_v4_multimodal.json"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_livevqa_dataset(v4_path: str, max_samples: int | None = None) -> list[dict]:
    """Load LiveVQA QA pairs from the v4 JSON.

    Each row has: question, img_path, corpus_url, source, level, options,
    ground_truth (letter A-E), gt_hex_id, retrieved_hex_ids.
    """
    logger.info("Loading LiveVQA from %s", v4_path)
    with open(v4_path) as f:
        data = json.load(f)
    rows = data["per_query"]
    logger.info("Loaded %d QA pairs", len(rows))
    if max_samples:
        rows = rows[:max_samples]
        logger.info("Limited to %d samples", max_samples)
    return rows


# ---------------------------------------------------------------------------
# Option shuffling (deterministic, same as old scripts)
# ---------------------------------------------------------------------------


def shuffle_options(
    options: list[str], ground_truth: str, seed: int
) -> tuple[list[str], str]:
    """Shuffle option order with a deterministic seed; return (new_options, new_gt_letter)."""
    rng = random.Random(seed)
    texts = [opt.split(". ", 1)[1] if ". " in opt else opt for opt in options]
    gt_idx = LETTERS.index(ground_truth)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    new_options = [f"{LETTERS[i]}. {texts[indices[i]]}" for i in range(len(indices))]
    new_gt_idx = indices.index(gt_idx)
    return new_options, LETTERS[new_gt_idx]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_letter(response: str) -> str:
    """Extract A-E letter from model response."""
    response = response.strip()
    if response and response[0] in LETTERS:
        return response[0]
    for ch in LETTERS:
        if ch in response:
            return ch
    return response[:1] if response else ""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def image_to_base64_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg"
    )
    return f"data:{mime};base64,{b64}"


def encode_image_base64(path: str) -> str:
    """Read and base64-encode an image. Returns the raw base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_naive_prompt(question: str, options: list[str], has_photo: bool) -> str:
    opts_str = "\n".join(options)
    if has_photo:
        return (
            "Based on the editorial photo above, answer the following question.\n\n"
            f"{question}\n\n{opts_str}\n\n"
            "Answer with ONLY the option letter (e.g. A, B, C, D, or E). Do not explain."
        )
    return (
        f"{question}\n\n{opts_str}\n\n"
        "Answer with ONLY the option letter (e.g. A, B, C, D, or E). Do not explain."
    )


def build_pixel_prompt(question: str, options: list[str], has_photo: bool) -> str:
    opts_str = "\n".join(options)
    if has_photo:
        ctx = "Based on the editorial photo and the article screenshot(s) above"
    else:
        ctx = "Based on the article screenshot(s) above"
    return (
        f"{ctx}, answer the following question.\n\n"
        f"{question}\n\n{opts_str}\n\n"
        "Answer with ONLY the option letter (e.g. A, B, C, D, or E). Do not explain."
    )


def build_text_prompt(
    question: str, options: list[str], passages: list[str], has_photo: bool
) -> str:
    ctx = "\n\n---\n\n".join(passages) if passages else "(no context retrieved)"
    opts_str = "\n".join(options)
    if has_photo:
        intro = "Use the editorial photo above and the following article excerpts to answer the question."
    else:
        intro = "Use the following article excerpts to answer the question."
    return (
        f"{intro}\n\n"
        f"{ctx}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"{opts_str}\n\n"
        "Answer with ONLY the option letter (A, B, C, D, or E). Do not explain."
    )


def build_hybrid_prompt(
    question: str,
    options: list[str],
    n_tiles: int,
    n_chunks: int,
    has_photo: bool,
) -> str:
    opts_str = "\n".join(options)
    parts = []
    if has_photo:
        parts.append("the editorial photo")
    if n_tiles > 0:
        parts.append(f"the {n_tiles} article screenshot(s)")
    if n_chunks > 0:
        parts.append(f"the {n_chunks} article excerpt(s)")
    intro = "Use " + " and ".join(parts) + " above to answer the following question."
    return (
        f"{intro}\n\n"
        f"{question}\n\n{opts_str}\n\n"
        "Answer with ONLY the option letter (e.g. A, B, C, D, or E). Do not explain."
    )


# ---------------------------------------------------------------------------
# Message building (OpenAI-compatible chat format for LLMClient)
# ---------------------------------------------------------------------------


def build_messages_for_livevqa(
    prompt: str,
    image_paths: list[str] | None = None,
    text_chunks: list[str] | None = None,
) -> list[dict]:
    """Build OpenAI-compatible messages for a LiveVQA MCQ call.

    Supports mixed content: images first, then text chunks, then the MCQ prompt.
    """
    content: list[dict] = []
    if image_paths:
        for p in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(p)},
                }
            )
    if text_chunks:
        ctx = "\n\n---\n\n".join(text_chunks)
        content.append({"type": "text", "text": ctx})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def batch_retrieve_pixel(
    queries: list[dict],
    api_url: str,
    search_k: int = 50,
    top_k: int = 10,
    batch_size: int = 16,
    nprobe: int | None = None,
    timeout: int = 180,
    db_path: str = "/opt/dlami/nvme/news_pages/state.db",
) -> list[list[dict]]:
    """Batch pixel retrieval via news tile search API.

    Returns a list (one per query) of lists of hit dicts with keys:
    hex, file, tile, chunk, score.
    """
    # Build url-to-hex mapping from news DB
    url_to_hex: dict[str, str] = {}
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, url FROM articles WHERE status = 'downloaded'")
        url_to_hex = {row[1]: row[0] for row in cur}
        conn.close()
        logger.info("Loaded url->hex map: %d entries from %s", len(url_to_hex), db_path)

    all_results: list[list[dict]] = [[] for _ in queries]
    n_batches = (len(queries) + batch_size - 1) // batch_size
    t_all = time.time()

    for bi in range(0, len(queries), batch_size):
        batch_q = queries[bi : bi + batch_size]
        payload: dict = {"queries": batch_q, "n_docs": search_k}
        if nprobe is not None:
            payload["nprobe"] = nprobe
        t0 = time.time()
        r = requests.post(api_url, json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        dt = time.time() - t0

        for qi, res in enumerate(body["results"]):
            items: list[dict] = []
            for hit in res["hits"]:
                url = hit.get("url", "")
                hex_id = url_to_hex.get(url)
                if not hex_id:
                    continue
                if len(items) >= top_k:
                    break
                items.append(
                    {
                        "hex": hex_id,
                        "file": os.path.basename(hit.get("path", "")),
                        "tile": int(hit.get("tile_index", 0)),
                        "chunk": int(hit.get("chunk_index", 0)),
                        "score": float(hit.get("score", 0.0)),
                    }
                )
            all_results[bi + qi] = items

        batch_num = bi // batch_size + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == n_batches:
            done = bi + len(batch_q)
            el = time.time() - t_all
            qps = done / el if el > 0 else 0
            eta = (len(queries) - done) / qps if qps > 0 else 0
            logger.info(
                "Pixel retrieval batch %d/%d  done=%d  %.1f q/s  last=%.2fs  ETA=%.0fs",
                batch_num,
                n_batches,
                done,
                qps,
                dt,
                eta,
            )

    return all_results


def batch_retrieve_text(
    queries: list[dict],
    api_url: str,
    search_k: int = 50,
    top_k: int = 10,
    batch_size: int = 16,
    nprobe: int | None = None,
    timeout: int = 180,
) -> list[list[dict]]:
    """Batch text retrieval via news text search API.

    Returns a list (one per query) of lists of hit dicts with keys:
    url, chunk_index, text, score.
    """
    all_results: list[list[dict]] = [[] for _ in queries]
    n_batches = (len(queries) + batch_size - 1) // batch_size
    t_all = time.time()

    for bi in range(0, len(queries), batch_size):
        batch_q = queries[bi : bi + batch_size]
        payload: dict = {"queries": batch_q, "n_docs": search_k}
        if nprobe is not None:
            payload["nprobe"] = nprobe
        t0 = time.time()
        r = requests.post(api_url, json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        dt = time.time() - t0

        for qi, res in enumerate(body["results"]):
            items: list[dict] = []
            for hit in res["hits"]:
                url = hit.get("url", "")
                if not url:
                    continue
                if len(items) >= top_k:
                    break
                items.append(
                    {
                        "url": url,
                        "chunk_index": int(hit.get("chunk_index", 0)),
                        "text": hit.get("text", ""),
                        "score": float(hit.get("score", 0.0)),
                    }
                )
            all_results[bi + qi] = items

        batch_num = bi // batch_size + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == n_batches:
            done = bi + len(batch_q)
            el = time.time() - t_all
            qps = done / el if el > 0 else 0
            eta = (len(queries) - done) / qps if qps > 0 else 0
            logger.info(
                "Text retrieval batch %d/%d  done=%d  %.1f q/s  last=%.2fs  ETA=%.0fs",
                batch_num,
                n_batches,
                done,
                qps,
                dt,
                eta,
            )

    return all_results


# ---------------------------------------------------------------------------
# Tile / chunk resolution helpers
# ---------------------------------------------------------------------------


def resolve_strip_path(
    hex_id: str, strip_file: str, tiles_dir: str = NEWS_TILES_DIR
) -> str | None:
    """Resolve a pixel tile to an absolute path on disk."""
    tile_dir = Path(tiles_dir) / f"{hex_id}.tiles"
    path = tile_dir / strip_file
    return str(path) if path.exists() else None


def resolve_editorial_photo(
    row: dict, livevqa_images: str = LIVEVQA_IMAGES_DIR
) -> str | None:
    """Return the path to the editorial photo for a QA row, or None."""
    ip = row.get("img_path")
    if not ip:
        return None
    full = os.path.join(livevqa_images, ip)
    return full if os.path.exists(full) else None


# ---------------------------------------------------------------------------
# Reader helpers (per-row context assembly for each mode)
# ---------------------------------------------------------------------------


def resolve_pixel_context(
    row: dict,
    retrieved_items: list[dict],
    top_k: int,
    include_photo: bool,
    livevqa_images: str,
    tiles_dir: str,
) -> tuple[list[str], str]:
    """Build image paths and prompt for pixel reader mode.

    Returns (image_paths, prompt_text).
    """
    images: list[str] = []
    photo = resolve_editorial_photo(row, livevqa_images) if include_photo else None
    if photo:
        images.append(photo)
    for it in retrieved_items[:top_k]:
        p = resolve_strip_path(it["hex"], it["file"], tiles_dir)
        if p:
            images.append(p)
    prompt = build_pixel_prompt(row["question"], row["options"], has_photo=bool(photo))
    return images, prompt


def resolve_text_context(
    row: dict,
    retrieved_items: list[dict],
    top_k: int,
    include_photo: bool,
    livevqa_images: str,
    chunks_db: str | None = None,
    hex_to_int: dict | None = None,
    url_to_hex: dict | None = None,
) -> tuple[list[str] | None, list[str], str]:
    """Build text passages and prompt for text reader mode.

    For text retrieval items (with 'text' key), uses text directly.
    For pixel retrieval items (cross-format), looks up text from DB.

    Returns (image_paths_or_None, passages, prompt_text).
    """
    photo = resolve_editorial_photo(row, livevqa_images) if include_photo else None
    passages: list[str] = []

    for it in retrieved_items[:top_k]:
        if "text" in it and it["text"]:
            passages.append(it["text"])
        elif chunks_db and hex_to_int:
            # Cross-format: pixel retrieval item, fetch text from DB
            hex_id = it.get("hex", "")
            if not hex_id and url_to_hex:
                hex_id = url_to_hex.get(it.get("url", ""), "")
            if hex_id and hex_id in hex_to_int:
                aid = hex_to_int[hex_id]
                ci = int(it.get("chunk", it.get("chunk_index", 0)))
                # Thread-local connection handled by caller
                text = _fetch_chunk_text(chunks_db, aid, ci)
                if text:
                    passages.append(text)

    imgs = [photo] if photo else None
    prompt = build_text_prompt(
        row["question"], row["options"], passages, has_photo=bool(photo)
    )
    return imgs, passages, prompt


# Thread-local SQLite connections for text chunk lookups
_tls = threading.local()


def _get_chunks_conn(db_path: str) -> sqlite3.Connection:
    if not hasattr(_tls, "conn") or _tls.conn_path != db_path:
        _tls.conn = sqlite3.connect(db_path)
        _tls.conn_path = db_path
    return _tls.conn


def _fetch_chunk_text(db_path: str, article_int: int, chunk_index: int) -> str | None:
    conn = _get_chunks_conn(db_path)
    cur = conn.execute(
        "SELECT text FROM chunks WHERE article_id = ? AND chunk_index = ?",
        (article_int, int(chunk_index)),
    )
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# DB / map loading helpers
# ---------------------------------------------------------------------------


def load_url_to_hex(db_path: str) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, url FROM articles WHERE status = 'downloaded'")
    m = {row[1]: row[0] for row in cur}
    conn.close()
    return m


def load_hex_to_int(path: str) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_one(
    idx: int,
    row: dict,
    llm_client: LLMClient,
    mode: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    # Pre-computed retrieval results (populated during batch retrieval phase)
    pixel_items: list[dict] | None = None,
    text_items: list[dict] | None = None,
    # Shared resources
    hex_to_int: dict | None = None,
    url_to_hex: dict | None = None,
) -> dict:
    """Evaluate a single LiveVQA question. Returns a result dict."""
    async with semaphore:
        t0 = time.time()
        error = None
        raw_response = ""
        n_images = 0
        n_chunks = 0

        try:
            if mode == "naive":
                # No retrieval: editorial photo + question only
                photo = (
                    resolve_editorial_photo(row, args.livevqa_images)
                    if args.include_editorial_photo
                    else None
                )
                images = [photo] if photo else []
                prompt = build_naive_prompt(
                    row["question"], row["options"], has_photo=bool(photo)
                )
                messages = build_messages_for_livevqa(
                    prompt, image_paths=images if images else None
                )
                n_images = len(images)

            elif mode == "pixel":
                items = pixel_items or []
                images, prompt = resolve_pixel_context(
                    row,
                    items,
                    args.top_k,
                    args.include_editorial_photo,
                    args.livevqa_images,
                    args.tiles_dir,
                )
                # Check if we have any retrieved tiles (beyond just the editorial photo)
                has_photo = args.include_editorial_photo and resolve_editorial_photo(
                    row, args.livevqa_images
                )
                n_tile_images = len(images) - (1 if has_photo else 0)
                if n_tile_images <= 0:
                    error = "no_tiles"
                messages = (
                    build_messages_for_livevqa(prompt, image_paths=images)
                    if not error
                    else []
                )
                n_images = len(images)

            elif mode == "text":
                items = text_items or []
                imgs, passages, prompt = resolve_text_context(
                    row,
                    items,
                    args.top_k,
                    args.include_editorial_photo,
                    args.livevqa_images,
                    args.chunks_db,
                    hex_to_int,
                    url_to_hex,
                )
                if not passages:
                    error = "no_chunks"
                messages = (
                    build_messages_for_livevqa(
                        prompt, image_paths=imgs if imgs else None
                    )
                    if not error
                    else []
                )
                n_chunks = len(passages)
                n_images = len(imgs) if imgs else 0

            elif mode == "hybrid":
                # Pixel items -> tile paths; text items -> text passages
                p_items = pixel_items or []
                t_items = text_items or []

                photo = (
                    resolve_editorial_photo(row, args.livevqa_images)
                    if args.include_editorial_photo
                    else None
                )
                tile_paths: list[str] = []
                for it in p_items[: args.top_k]:
                    p = resolve_strip_path(it["hex"], it["file"], args.tiles_dir)
                    if p:
                        tile_paths.append(p)

                chunks: list[str] = []
                for it in t_items[: args.top_k]:
                    if "text" in it and it["text"]:
                        chunks.append(it["text"])

                if not tile_paths and not chunks:
                    error = "no_retrieval"
                else:
                    image_paths = ([photo] if photo else []) + tile_paths
                    prompt = build_hybrid_prompt(
                        row["question"],
                        row["options"],
                        len(tile_paths),
                        len(chunks),
                        has_photo=bool(photo),
                    )
                    messages = build_messages_for_livevqa(
                        prompt,
                        image_paths=image_paths if image_paths else None,
                        text_chunks=chunks if chunks else None,
                    )
                    n_images = len(image_paths)
                    n_chunks = len(chunks)
                if error:
                    messages = []

            else:
                raise ValueError(f"Unknown mode: {mode}")

            if not error:
                raw_response, _usage = await llm_client.generate(messages)

        except Exception as e:
            error = type(e).__name__
            logger.debug("Error on idx=%d: %s", idx, e)

        latency = time.time() - t0
        gt = row.get("ground_truth", "")
        predicted = extract_letter(raw_response) if not error else ""
        is_correct = predicted == gt

        return {
            "idx": idx,
            "question": row["question"],
            "img_path": row.get("img_path", ""),
            "corpus_url": row.get("corpus_url", ""),
            "source": row.get("source", ""),
            "level": row.get("level", ""),
            "ground_truth": gt,
            "predicted": predicted,
            "raw_response": raw_response,
            "correct": is_correct,
            "error": error,
            "n_images": n_images,
            "n_chunks": n_chunks,
            "latency": round(latency, 2),
            "mode": mode,
        }


async def run_evaluation(args: argparse.Namespace):
    """Main evaluation loop."""
    # Load dataset
    rows = load_livevqa_dataset(args.v4, args.max_samples)

    # Apply option shuffling if requested
    if args.shuffle_seed is not None:
        for i, q in enumerate(rows):
            q["options"], q["ground_truth"] = shuffle_options(
                q["options"],
                q["ground_truth"],
                args.shuffle_seed + i,
            )
        logger.info("Shuffled options with seed=%d", args.shuffle_seed)

    mode = args.mode

    # ---- Retrieval phase (batch, synchronous) ----
    # Build queries for retrieval modes
    pixel_results_map: dict[int, list[dict]] = {}
    text_results_map: dict[int, list[dict]] = {}

    if mode in ("pixel", "hybrid"):
        logger.info("=== Pixel retrieval phase ===")
        queries = []
        for row in rows:
            q: dict = {"text": row["question"]}
            if args.query_instruction:
                q["text"] = f"{args.query_instruction} {q['text']}"
            if args.multimodal_query:
                photo = resolve_editorial_photo(row, args.livevqa_images)
                if photo:
                    with open(photo, "rb") as f:
                        q["image"] = base64.b64encode(f.read()).decode()
            queries.append(q)
        pixel_results = batch_retrieve_pixel(
            queries,
            args.pixel_api,
            search_k=args.search_k,
            top_k=args.retrieval_top_k,
            batch_size=args.retrieval_batch_size,
            nprobe=args.nprobe,
            timeout=args.retrieval_timeout,
            db_path=args.pages_db,
        )
        for i, items in enumerate(pixel_results):
            pixel_results_map[i] = items
        logger.info("Pixel retrieval complete: %d queries", len(pixel_results))

    if mode in ("text", "hybrid"):
        logger.info("=== Text retrieval phase ===")
        queries = []
        for row in rows:
            q = {"text": row["question"]}
            if args.multimodal_query:
                photo = resolve_editorial_photo(row, args.livevqa_images)
                if photo:
                    with open(photo, "rb") as f:
                        q["image"] = base64.b64encode(f.read()).decode()
            queries.append(q)
        text_results = batch_retrieve_text(
            queries,
            args.text_api,
            search_k=args.search_k,
            top_k=args.retrieval_top_k,
            batch_size=args.retrieval_batch_size,
            nprobe=args.nprobe,
            timeout=args.retrieval_timeout,
        )
        for i, items in enumerate(text_results):
            text_results_map[i] = items
        logger.info("Text retrieval complete: %d queries", len(text_results))

    # ---- Load shared resources for cross-format lookups ----
    hex_to_int = None
    url_to_hex = None
    if mode in ("text",) and args.chunks_db and os.path.exists(args.chunks_db):
        if args.hex_to_int_map and os.path.exists(args.hex_to_int_map):
            hex_to_int = load_hex_to_int(args.hex_to_int_map)
            logger.info("Loaded hex->int map: %d entries", len(hex_to_int))
        if args.pages_db and os.path.exists(args.pages_db):
            url_to_hex = load_url_to_hex(args.pages_db)
            logger.info("Loaded url->hex map: %d entries", len(url_to_hex))

    # ---- Reader phase (async, concurrent) ----
    logger.info("=== Reader phase: mode=%s, model=%s ===", mode, args.model)
    llm_client = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.reader_timeout,
        enable_thinking=False if args.no_think else None,
    )

    # Smoke test
    logger.info("Smoke test on first example...")
    sem = asyncio.Semaphore(args.workers)
    smoke_result = await evaluate_one(
        0,
        rows[0],
        llm_client,
        mode,
        args,
        sem,
        pixel_items=pixel_results_map.get(0),
        text_items=text_results_map.get(0),
        hex_to_int=hex_to_int,
        url_to_hex=url_to_hex,
    )
    if smoke_result["error"]:
        logger.warning("Smoke test had error: %s", smoke_result["error"])
    else:
        logger.info(
            "Smoke test OK: pred=%s gt=%s correct=%s latency=%.1fs",
            smoke_result["predicted"],
            smoke_result["ground_truth"],
            smoke_result["correct"],
            smoke_result["latency"],
        )

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint handling
    out_s = str(output_path)
    if out_s.endswith(".jsonl"):
        checkpoint_path = out_s[:-6] + "_checkpoint.jsonl"
    elif out_s.endswith(".json"):
        checkpoint_path = out_s[:-5] + "_checkpoint.json"
    else:
        checkpoint_path = out_s + "_checkpoint.jsonl"
    completed: dict[int, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                completed[r["idx"]] = r
        logger.info("Loaded checkpoint: %d completed rows", len(completed))

    # Run all examples
    tasks = []
    for i, row in enumerate(rows):
        if i in completed:
            continue
        tasks.append(
            evaluate_one(
                i,
                row,
                llm_client,
                mode,
                args,
                sem,
                pixel_items=pixel_results_map.get(i),
                text_items=text_results_map.get(i),
                hex_to_int=hex_to_int,
                url_to_hex=url_to_hex,
            )
        )

    # Process with progress tracking
    results: list[dict] = list(completed.values())
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    err_types: dict[str, int] = defaultdict(int)
    for r in results:
        if r["error"]:
            err_types[r["error"]] += 1
    level_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        lvl = r.get("level", "?")
        level_stats[lvl]["total"] += 1
        if r["correct"]:
            level_stats[lvl]["correct"] += 1

    latencies: list[float] = [r["latency"] for r in results if not r["error"]]
    t0 = time.time()
    last_log = t0
    fi = 0
    total_tasks = len(tasks) + len(completed)

    # Save checkpoint incrementally
    ckpt_fh = open(checkpoint_path, "a")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        total += 1
        fi += 1

        if result["error"]:
            errors += 1
            err_types[result["error"]] += 1
        else:
            latencies.append(result["latency"])
        if result["correct"]:
            correct += 1
        lvl = result.get("level", "?")
        level_stats[lvl]["total"] += 1
        if result["correct"]:
            level_stats[lvl]["correct"] += 1

        # Write checkpoint
        ckpt_fh.write(json.dumps(result) + "\n")
        if fi % 500 == 0:
            ckpt_fh.flush()

        # Log progress
        now = time.time()
        if fi % 200 == 0 or now - last_log > 30 or fi == len(tasks):
            el = now - t0
            qps = fi / el if el > 0 else 0
            acc = correct / total * 100 if total else 0
            p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
            eta = (len(tasks) - fi) / qps if qps > 0 else 0
            logger.info(
                "[%d/%d] acc=%.2f%% | %.1f q/s ETA %dm%ds | "
                "lat p50=%.1fs | err=%d (%s)",
                total,
                total_tasks,
                acc,
                qps,
                int(eta) // 60,
                int(eta) % 60,
                p50,
                errors,
                dict(err_types),
            )
            last_log = now

    ckpt_fh.close()

    # ---- Sort results by idx and write final output ----
    results.sort(key=lambda r: r["idx"])

    acc = correct / total * 100 if total else 0
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 64}")
    print(f"LiveVQA Evaluation — mode={mode}")
    print(f"{'=' * 64}")
    print(f"Model: {args.model}")
    print(f"Total: {total}  Correct: {correct}  Accuracy: {acc:.2f}%")
    print(f"Errors: {errors} ({dict(err_types)})")
    print(f"Time: {elapsed:.0f}s ({total / elapsed:.1f} q/s)" if elapsed > 0 else "")
    if latencies:
        ls = sorted(latencies)
        print(
            f"Latency: p50={ls[len(ls) // 2]:.2f}s "
            f"p90={ls[int(len(ls) * 0.9)]:.2f}s "
            f"p99={ls[int(len(ls) * 0.99)]:.2f}s"
        )
    print()
    print("By difficulty level:")
    for lvl in sorted(level_stats.keys()):
        s = level_stats[lvl]
        la = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  Level {lvl}: {s['correct']}/{s['total']} = {la:.1f}%")

    # By news source (outlet)
    source_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        src = r.get("source", "?")
        outlet = src.split()[0] if src else "?"  # "CNN Politics" -> "CNN"
        source_stats[outlet]["total"] += 1
        if r["correct"]:
            source_stats[outlet]["correct"] += 1
    print("\nBy outlet:")
    for outlet in sorted(source_stats.keys()):
        s = source_stats[outlet]
        la = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {outlet}: {s['correct']}/{s['total']} = {la:.1f}%")

    # Write final JSONL output
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved %d results to %s", len(results), args.output)

    # Write summary JSON (alongside the JSONL)
    out_str = str(args.output)
    if out_str.endswith(".jsonl"):
        summary_path = out_str[:-6] + "_summary.json"
    elif out_str.endswith(".json"):
        summary_path = out_str[:-5] + "_summary.json"
    else:
        summary_path = out_str + "_summary.json"
    summary = {
        "mode": mode,
        "model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "errors": errors,
        "error_types": dict(err_types),
        "elapsed_s": elapsed,
        "top_k": args.top_k,
        "include_editorial_photo": args.include_editorial_photo,
        "multimodal_query": args.multimodal_query,
        "shuffle_seed": args.shuffle_seed,
        "v4_source": args.v4,
        "level_stats": {str(k): v for k, v in level_stats.items()},
        "outlet_stats": {k: v for k, v in source_stats.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Removed checkpoint %s", checkpoint_path)


def main():
    parser = argparse.ArgumentParser(
        description="LiveVQA evaluation with multiple retrieval modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        required=True,
        choices=["naive", "pixel", "text", "hybrid"],
        help="Evaluation mode: naive (no retrieval), pixel (screenshot tiles), "
        "text (text chunks), hybrid (pixel + text combined).",
    )

    # Dataset
    parser.add_argument(
        "--v4",
        default=DEFAULT_V4_PATH,
        help="Path to LiveVQA v4 JSON with per_query data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to first N QA pairs (for debugging)",
    )
    parser.add_argument(
        "--livevqa-images",
        default=LIVEVQA_IMAGES_DIR,
        help="Base directory for LiveVQA editorial photos",
    )

    # Reader (LLM/VLM)
    parser.add_argument(
        "--api-base",
        default="http://localhost:8211/v1",
        help="OpenAI-compatible API base URL for reader model",
    )
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Reader model name"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens for reader response (just a letter)",
    )
    parser.add_argument(
        "--reader-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout for reader calls (seconds)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking via chat_template_kwargs.enable_thinking=False",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of concurrent reader requests"
    )

    # Retrieval (shared)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved items (chunk-level) to feed the reader",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
        help="Number of items to fetch from search API (before reader top-k slicing)",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=50,
        help="Number of raw candidates to fetch from FAISS (n_docs)",
    )
    parser.add_argument(
        "--retrieval-batch-size",
        type=int,
        default=16,
        help="Batch size for retrieval API calls",
    )
    parser.add_argument(
        "--nprobe", type=int, default=None, help="Override FAISS nprobe for retrieval"
    )
    parser.add_argument(
        "--retrieval-timeout",
        type=int,
        default=180,
        help="Per-batch timeout for retrieval API calls (seconds)",
    )
    parser.add_argument(
        "--multimodal-query",
        action="store_true",
        default=True,
        help="Include editorial photo in retrieval query (default: True)",
    )
    parser.add_argument(
        "--no-multimodal-query",
        dest="multimodal_query",
        action="store_false",
        help="Text-only retrieval query (no editorial photo)",
    )
    parser.add_argument(
        "--query-instruction",
        default=None,
        help="Instruction prefix for pixel retrieval queries",
    )

    # Pixel retrieval API
    parser.add_argument(
        "--pixel-api",
        default="http://localhost:30890/search",
        help="News pixel search API endpoint",
    )
    parser.add_argument(
        "--tiles-dir",
        default=NEWS_TILES_DIR,
        help="Directory containing news article tiles ({hex}.tiles/)",
    )

    # Text retrieval API
    parser.add_argument(
        "--text-api",
        default="http://localhost:30892/search",
        help="News text search API endpoint",
    )

    # DB paths for cross-format lookups
    parser.add_argument(
        "--pages-db",
        default="/opt/dlami/nvme/news_pages/state.db",
        help="News pages SQLite DB (url<->hex mapping)",
    )
    parser.add_argument(
        "--chunks-db",
        default="/opt/dlami/nvme/news_text_embeddings/text_baseline.db",
        help="Text chunks SQLite DB (for cross-format text lookups)",
    )
    parser.add_argument(
        "--hex-to-int-map",
        default="/opt/dlami/nvme/news_text_embeddings/article_id_map.json",
        help="JSON mapping hex article IDs to integer IDs in chunks DB",
    )

    # Editorial photo handling
    parser.add_argument(
        "--include-editorial-photo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include editorial photo in reader input (default: True)",
    )

    # Option shuffling
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle option order per question with this seed (seed + idx)",
    )

    # Output
    parser.add_argument("--output", required=True, help="Path for output JSONL file")

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.mode in ("pixel", "hybrid") and not args.pixel_api:
        parser.error("--pixel-api required for pixel/hybrid mode")
    if args.mode in ("text", "hybrid") and not args.text_api:
        parser.error("--text-api required for text/hybrid mode")

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
