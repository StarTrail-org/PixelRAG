#!/usr/bin/env python3
"""FAISS-based visual search API.

Serves a FAISS index over visual embeddings (Wikipedia screenshots, news images, or any pre-built index).
Supports text and image queries (single or batch) via Qwen3-VL-Embedding-2B.

Embedding backend: direct transformers-based inference (SDPA attention).
Produces embeddings aligned with indexes built via the direct_gpu pipeline (cosine = 1.0).

Usage:
    # Start server (CPU, default)
    pixelrag-serve \
        --index-dir ./index \
        --tiles-dir ./tiles \
        --articles-json ./articles.json \
        --model Qwen/Qwen3-VL-Embedding-2B \
        --port 30001

    # Start server (CUDA)
    pixelrag-serve \
        --index-dir ./index \
        --tiles-dir ./tiles \
        --articles-json ./articles.json \
        --model Qwen/Qwen3-VL-Embedding-2B \
        --device cuda \
        --port 30001

    # Search
    curl -X POST http://localhost:30001/search \
        -H "Content-Type: application/json" \
        -d '{"queries": [{"text": "Nikola Tesla"}], "n_docs": 10}'
"""

import argparse
import base64
import functools
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger("search_api")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app = FastAPI(title="PixelRAG Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state loaded at startup
_state = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Query(BaseModel):
    text: str | None = None
    image: str | None = None  # base64-encoded image
    embedding: list[float] | None = None  # pre-computed query embedding


class SearchRequest(BaseModel):
    queries: list[Query]
    n_docs: int = 10
    nprobe: int | None = None  # override default nprobe
    min_tile_height: int | None = None  # filter out small/blank chunks
    instruction: str | None = None  # override query embedding instruction
    include_images: bool = False  # return base64-encoded tile images
    articles_only: bool = False  # drop Wikipedia meta pages (Portal:, List_of_, …)


# Wikipedia meta/aggregator pages that pollute "find the article" results.
# Matches both bare titles ("Portal:The_arts/Featured_picture",
# "List_of_German_physicists") and full /wiki/ URLs. Opt-in via articles_only.
_META_RE = re.compile(
    r"(?:^|/wiki/)(?:Portal|Wikipedia|WP|Template|Category|Draft|Help|File|Module|Book|Special|Talk|MediaWiki|Topic)[ _]*:"
    r"|(?:^|/wiki/)(?:List|Outline|Index|Glossary|Timeline)[ _]of[ _]"
    r"|\(disambiguation\)"
    r"|[ _]\((?:surname|given[ _]name)\)",
    re.IGNORECASE,
)


def _is_meta(url: str) -> bool:
    return bool(_META_RE.search(url))


class Hit(BaseModel):
    score: float
    vector_id: int
    article_id: int
    tile_index: int
    chunk_index: int
    y_offset: int
    tile_height: int
    path: str
    url: str
    # Which (tile, chunk) coordinates actually exist on disk for this article,
    # e.g. "0:0-8,1:0-4" — lets agents page through an article without
    # guessing coordinates past its end.
    article_pages: str | None = None
    image_base64: str | None = None


class QueryResult(BaseModel):
    hits: list[Hit]


class SearchResponse(BaseModel):
    results: list[QueryResult]


class StatusResponse(BaseModel):
    total_vectors: int
    dimension: int
    nlist: int
    nprobe: int
    model: str
    index_dir: str = ""
    tiles_dir: str = ""
    index_built_at: str
    index_size_bytes: int
    metadata_size_bytes: int


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."


def _parse_queries(
    queries: list[Query], instruction: str | None = None
) -> tuple[list[dict], list[Image.Image | None]]:
    """Parse queries into chat messages and optional images."""
    instr = DEFAULT_INSTRUCTION if instruction is None else instruction
    messages_list = []
    images = []
    for q in queries:
        # Build chat messages for apply_chat_template
        sys_content = [{"type": "text", "text": instr}]
        user_content = []
        img = None
        if q.image:
            # Accept both a raw base64 string and a data URL
            # ("data:image/png;base64,...") — strip the prefix if present,
            # otherwise b64decode chokes on it ("Incorrect padding").
            img_data = q.image
            if img_data.startswith("data:"):
                img_data = img_data.split(",", 1)[-1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            user_content.append({"type": "image", "image": img})
        if q.text:
            user_content.append({"type": "text", "text": q.text})
        if not user_content:
            raise HTTPException(
                status_code=400, detail="Query must have text, image, or both"
            )
        messages_list.append(
            [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]
        )
        images.append(img)
    return messages_list, images


def _encode_queries(queries: list[Query], instruction: str | None = None) -> np.ndarray:
    """Encode queries via HF transformers + SDPA (~42ms/query on GPU, exact index alignment).

    Uses the same model + attention backend (SDPA) as the index-building pipeline
    (embed_tiles.py direct_gpu backend), so embeddings are identical (cosine = 1.0).
    """
    import torch

    model = _state["model"]
    processor = _state["processor"]
    device = _state["device"]
    messages_list, images = _parse_queries(queries, instruction)

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    # Separate image and non-image inputs for the processor
    img_list = [img for img in images if img is not None]
    if img_list:
        inputs = processor(
            text=texts, images=img_list, return_tensors="pt", padding=True
        )
    else:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model(**inputs)

    # Last-token pooling + L2 normalize — use last_hidden_state (post-RMSNorm)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    last_token_indices = attention_mask.sum(dim=1) - 1
    pooled = last_hidden[
        torch.arange(last_hidden.size(0), device=last_hidden.device),
        last_token_indices,
    ]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()


def _normalize_query_embeddings(queries: list[Query]) -> np.ndarray:
    """Normalize externally supplied query embeddings to match service behavior."""
    if not queries:
        return np.zeros((0, _state["dimension"]), dtype=np.float32)
    if any(q.embedding is None for q in queries):
        raise HTTPException(
            status_code=400,
            detail="When using pre-computed embeddings, every query must provide `embedding`.",
        )
    if any(q.text is not None or q.image is not None for q in queries):
        raise HTTPException(
            status_code=400,
            detail="Pre-computed embeddings cannot be mixed with text/image fields in the same request.",
        )

    embeddings = np.asarray([q.embedding for q in queries], dtype=np.float32)
    if embeddings.ndim != 2:
        raise HTTPException(
            status_code=400, detail="Embeddings must have shape [batch, dim]."
        )
    expected_dim = _state["dimension"]
    if embeddings.shape[1] != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dim mismatch: got {embeddings.shape[1]}, expected {expected_dim}.",
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


# ---------------------------------------------------------------------------
# Path / URL resolution
# ---------------------------------------------------------------------------


def _resolve_path(article_id: int, tile_index: int, chunk_index: int) -> str:
    """Resolve chunk file path from article_id, tile_index, chunk_index."""
    tiles_dir = _state["tiles_dir"]
    shard_size = _state.get("shard_size", 8284)
    tiles_dirname = f"{article_id}.png.tiles"
    chunk_name = f"chunk_{tile_index:04d}_{chunk_index:02d}.png"

    # Flat layout: tiles_dir/{article_id}.png.tiles/chunk_XXXX_YY.png
    flat_path = os.path.join(tiles_dir, tiles_dirname, chunk_name)
    if os.path.exists(flat_path):
        return flat_path

    # Sharded layout: tiles_dir/shard_XXX/sub/{article_id}.png.tiles/chunk_XXXX_YY.png
    top_shard = article_id // shard_size
    top_shard_dir = os.path.join(tiles_dir, f"shard_{top_shard:03d}")
    if os.path.isdir(top_shard_dir):
        for sub in sorted(os.listdir(top_shard_dir)):
            sub_path = os.path.join(top_shard_dir, sub, tiles_dirname)
            if os.path.isdir(sub_path):
                return os.path.join(sub_path, chunk_name)

    # Fallback: shard path without checking existence (serve may run without tiles)
    top_shard = article_id // shard_size
    return os.path.join(
        tiles_dir, f"shard_{top_shard:03d}", "?", tiles_dirname, chunk_name
    )


@functools.lru_cache(maxsize=8192)
def _article_pages(article_id: int) -> str | None:
    """Map of chunk files that exist on disk for an article: "0:0-8,1:0-4".

    Lets clients (notably the chat agent) page through an article without
    probing nonexistent coordinates. Returns None when tiles are rendered
    on demand (no files on disk) or the article dir can't be found.
    """
    if _state.get("ondemand") is not None:
        return None
    probe = _resolve_path(article_id, 0, 0)
    d = os.path.dirname(probe)
    if not os.path.isdir(d):
        return None
    tiles: dict[int, list[int]] = {}
    for name in os.listdir(d):
        m = re.match(r"chunk_(\d{4})_(\d{2})\.(?:png|jpg|jpeg)$", name)
        if m:
            tiles.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    if not tiles:
        return None
    return ",".join(f"{t}:{min(cs)}-{max(cs)}" for t, cs in sorted(tiles.items()))


def _resolve_url(article_id: int) -> str:
    """Resolve URL or title from article_id."""
    articles = _state["articles"]
    if article_id < len(articles):
        entry = articles[article_id]
        if isinstance(entry, dict):
            return entry.get("url") or entry.get("title", "")
        return str(entry)
    return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.time()

    # Encode queries
    if req.queries and all(q.embedding is not None for q in req.queries):
        query_vectors = _normalize_query_embeddings(req.queries)
    else:
        if any(q.embedding is not None for q in req.queries):
            raise HTTPException(
                status_code=400,
                detail="Do not mix pre-computed embeddings with text/image queries in one request.",
            )
        query_vectors = _encode_queries(req.queries, req.instruction)
    t_encode = time.time() - t0

    # FAISS search
    index = _state["index"]
    default_nprobe = index.nprobe
    if req.nprobe is not None:
        index.nprobe = req.nprobe

    # Over-fetch when filtering to ensure enough results after filtering.
    # Meta pages can be the majority of raw hits, so articles_only needs more.
    if req.articles_only:
        fetch_k = req.n_docs * 10
    elif req.min_tile_height:
        fetch_k = req.n_docs * 5
    else:
        fetch_k = req.n_docs
    distances, indices = index.search(query_vectors, fetch_k)

    if req.nprobe is not None:
        index.nprobe = default_nprobe
    t_search = time.time() - t0 - t_encode

    # Build results
    meta = _state["metadata"]
    article_ids = meta["article_ids"]
    tile_indices = meta["tile_indices"]
    chunk_indices = meta["chunk_indices"]
    y_offsets = meta["y_offsets"]
    tile_heights = meta["tile_heights"]
    tiles_dir = _state.get("tiles_dir", "")

    results = []
    for qi in range(len(req.queries)):
        hits = []
        for j in range(fetch_k):
            vid = int(indices[qi, j])
            if vid == -1:
                continue
            th = int(tile_heights[vid])
            if req.min_tile_height and th < req.min_tile_height:
                continue
            aid = int(article_ids[vid])
            url = _resolve_url(aid)
            if req.articles_only and _is_meta(url):
                continue
            ti = int(tile_indices[vid])
            ci = int(chunk_indices[vid])
            tile_path = _resolve_path(aid, ti, ci)
            img_b64 = None
            if req.include_images and tile_path and os.path.exists(tile_path):
                with open(tile_path, "rb") as fp:
                    img_b64 = base64.b64encode(fp.read()).decode()
            elif req.include_images and _state.get("ondemand") is not None:
                img_b64 = _ondemand_chunk_b64(aid, ti, ci, th)
            # Expose a relative tile path, not the absolute server filesystem
            # path (avoids leaking the host's directory layout; clients fetch
            # tiles via /tile/{article_id}/{tile_index}/{chunk_index}).
            rel_path = tile_path
            if tiles_dir:
                candidate = os.path.relpath(tile_path, tiles_dir)
                if not candidate.startswith(".."):
                    rel_path = candidate
            hits.append(
                Hit(
                    score=float(distances[qi, j]),
                    vector_id=vid,
                    article_id=aid,
                    tile_index=ti,
                    chunk_index=ci,
                    y_offset=int(y_offsets[vid]),
                    tile_height=th,
                    path=rel_path,
                    url=url,
                    article_pages=_article_pages(aid),
                    image_base64=img_b64,
                )
            )
            if len(hits) >= req.n_docs:
                break
        results.append(QueryResult(hits=hits))

    logger.info(
        "Search: %d queries, n_docs=%d, encode=%.3fs, search=%.3fs, total=%.3fs",
        len(req.queries),
        req.n_docs,
        t_encode,
        t_search,
        time.time() - t0,
    )

    return SearchResponse(results=results)


@app.get("/status", response_model=StatusResponse)
async def status():
    index = _state["index"]
    return StatusResponse(
        total_vectors=index.ntotal,
        dimension=_state["dimension"],
        nlist=_state["nlist"],
        nprobe=index.nprobe,
        model=_state["model_name"],
        index_dir=_state.get("index_dir", ""),
        tiles_dir=_state.get("tiles_dir", ""),
        index_built_at=_state["index_built_at"],
        index_size_bytes=_state["index_size_bytes"],
        metadata_size_bytes=_state["metadata_size_bytes"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


class ReconstructRequest(BaseModel):
    vector_ids: list[int]


@app.post("/reconstruct")
async def reconstruct(req: ReconstructRequest):
    """Reconstruct stored embeddings by vector_id (for alignment debugging)."""
    index = _state["index"]
    # Ensure direct map exists for reconstruct
    if not hasattr(index, "_direct_map_built"):
        index.make_direct_map()
        index._direct_map_built = True
    vecs = []
    for vid in req.vector_ids:
        vecs.append(index.reconstruct(vid).tolist())
    return {"embeddings": vecs}


@app.get("/tile")
async def tile(path: str):
    """Serve a tile image by its local path (legacy, use /tile/{article_id}/{tile_index}/{chunk_index} instead)."""
    tiles_dir = _state.get("tiles_dir", "./tiles")
    resolved = os.path.realpath(path)
    if not resolved.startswith(os.path.realpath(tiles_dir)):
        raise HTTPException(status_code=403, detail="Path not under tiles directory")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(resolved, media_type="image/png")


@app.get("/tile/{article_id}/{tile_index}/{chunk_index}")
async def tile_by_id(article_id: int, tile_index: int, chunk_index: int):
    """Serve a tile image by article_id, tile_index, chunk_index."""
    path = _resolve_path(article_id, tile_index, chunk_index)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Tile not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def load(args):
    """Load index, metadata, model, and articles.json."""
    import torch

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    # Load FAISS index
    index_path = os.path.join(args.index_dir, "index.faiss")
    logger.info("Loading FAISS index from %s...", index_path)
    t0 = time.time()
    index = faiss.read_index(index_path)
    logger.info("Loaded index: %d vectors in %.1fs", index.ntotal, time.time() - t0)

    # Load metadata
    metadata_path = os.path.join(args.index_dir, "metadata.npz")
    logger.info("Loading metadata from %s...", metadata_path)
    meta = np.load(metadata_path)

    # Load summary
    summary_path = os.path.join(args.index_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # Load articles.json
    logger.info("Loading articles.json from %s...", args.articles_json)
    with open(args.articles_json) as f:
        articles = json.load(f)
    logger.info("Loaded %d article slugs", len(articles))

    # Load embedding model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info("Loading model %s on device=%s dtype=%s...", args.model, device, dtype)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    )
    adapter_path = getattr(args, "peft_adapter", None)
    if adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s...", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged")
    model = model.to(device).eval()
    logger.info("Model loaded")

    # File sizes
    index_size = os.path.getsize(index_path)
    meta_size = os.path.getsize(metadata_path)

    # Index build time from file mtime
    index_mtime = os.path.getmtime(index_path)
    index_built_at = datetime.fromtimestamp(index_mtime, tz=timezone.utc).isoformat()

    _state.update(
        {
            "index": index,
            "metadata": meta,
            "articles": articles,
            "processor": processor,
            "model": model,
            "device": device,
            "model_name": args.model,
            "index_dir": args.index_dir,
            "tiles_dir": args.tiles_dir,
            "dimension": summary.get("dimension", index.d),
            "nlist": summary.get("nlist", 4096),
            "index_built_at": index_built_at,
            "index_size_bytes": index_size,
            "metadata_size_bytes": meta_size,
            "ondemand": None,
        }
    )

    # Optional: render tile images on demand from a kiwix ZIM instead of reading a
    # materialized (multi-TB) tiles/ dir. Only retrieved pages get rendered + cached.
    if getattr(args, "render_on_demand", False):
        from .render_ondemand import OnDemandTiles

        book = args.zim_book or _derive_kiwix_book(args.kiwix_url)
        if not book:
            logger.warning(
                "render-on-demand: could not derive kiwix book from %s "
                "(pass --zim-book)",
                args.kiwix_url,
            )
        cache = os.path.join(args.tiles_dir or "./tiles_cache", "_ondemand")
        _state["ondemand"] = OnDemandTiles(args.kiwix_url, book, cache)
        logger.info(
            "On-demand tile rendering enabled (kiwix=%s book=%s cache=%s)",
            args.kiwix_url,
            book,
            cache,
        )


def _derive_kiwix_book(kiwix_url: str) -> str:
    """Read the kiwix-serve catalog and return the /content/<book> id."""
    import re
    import urllib.request

    try:
        with urllib.request.urlopen(
            kiwix_url.rstrip("/") + "/catalog/v2/entries", timeout=10
        ) as r:
            xml = r.read().decode()
        m = re.search(r'href="/content/([^"/]+)"', xml)
        return m.group(1) if m else ""
    except Exception:
        return ""


def _ondemand_chunk_b64(
    article_id: int, tile_index: int, chunk_index: int, tile_height: int
):
    """Render+chunk the page on demand and return the chunk as base64 PNG."""
    od = _state.get("ondemand")
    if od is None:
        return None
    p = od.chunk_path(article_id, _resolve_url(article_id), tile_index, chunk_index)
    if not p or not os.path.exists(p):
        return None
    import io

    from PIL import Image

    im = Image.open(p)
    # The on-demand render captures a full tile_height; trim the (padded) last
    # chunk back to the height the index recorded so it matches the built tile.
    if tile_height and im.height > tile_height:
        im = im.crop((0, 0, im.width, tile_height))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description="PixelRAG FAISS Search API")
    parser.add_argument(
        "--index-dir", default=os.environ.get("PIXELRAG_INDEX_DIR", "./index")
    )
    parser.add_argument(
        "--tiles-dir", default=os.environ.get("PIXELRAG_TILES_DIR", "./tiles")
    )
    parser.add_argument(
        "--articles-json",
        default=os.environ.get("PIXELRAG_ARTICLES_JSON", "./articles.json"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on: cpu (default) or cuda",
    )
    parser.add_argument(
        "--peft-adapter",
        default=None,
        help="Path to PEFT/LoRA adapter directory (merged at load time)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument(
        "--render-on-demand",
        action="store_true",
        help="Render tile images on demand from a kiwix ZIM (no materialized tiles/ dir). "
        "Requires a running kiwix-serve (see --kiwix-url).",
    )
    parser.add_argument(
        "--kiwix-url",
        default=os.environ.get("PIXELRAG_KIWIX_URL", "http://localhost:30900"),
        help="Base URL of a running kiwix-serve, for --render-on-demand",
    )
    parser.add_argument(
        "--zim-book",
        default=os.environ.get("PIXELRAG_ZIM_BOOK"),
        help="kiwix book id for /content/<book>/ (auto-derived from --kiwix-url if omitted)",
    )
    args = parser.parse_args()

    load(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
