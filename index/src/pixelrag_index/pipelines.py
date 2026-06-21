"""End-to-end pipeline: source -> ingest -> chunk -> embed -> build."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import load_config, make_source

logger = logging.getLogger("pixelrag-index")


def build(config: dict, limit: int | None = None, force: bool = False) -> Path:
    """Build a searchable FAISS index from a document source.

    Stages: source → ingest (render) → chunk → embed → build index
    """
    import itertools

    source = make_source(config)
    try:
        docs = list(itertools.islice(source, limit)) if limit else list(source)
    finally:
        if hasattr(source, "close"):
            source.close()
    output = Path(config.get("output", "./index"))
    tiles_dir = output / "tiles"
    embeddings_dir = output / "embeddings"
    ingest_cfg = config.get("ingest", {})
    embed_cfg = config.get("embed", {})
    device = embed_cfg.get("device", "cpu")

    if force:
        import shutil

        for d in (tiles_dir, embeddings_dir):
            if d.exists():
                shutil.rmtree(d)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Render documents to tiles
    # Use sequential integer IDs as tile directory names so embed/serve can map them
    import json
    from pixelrag_render.render import render_urls, render_pdf

    logger.info("Stage 1/4: Rendering %d documents to tiles...", len(docs))

    # Collect documents into batches by type
    url_docs = []
    pdf_docs = []
    image_docs = []
    articles = []  # id → metadata mapping for serve

    for doc in docs:
        idx = len(articles)
        articles.append(
            {
                "id": str(doc.id),
                "url": doc.url,
                "path": doc.path,
                "metadata": doc.metadata or {},
            }
        )
        if doc.url:
            url_docs.append((idx, doc))
        elif doc.path and doc.path.lower().endswith(".pdf"):
            pdf_docs.append((idx, doc))
        elif doc.path:
            image_docs.append((idx, doc))

    # Render URL batch — skip already-captured articles
    if url_docs:
        new_url_docs = [
            (idx, d)
            for idx, d in url_docs
            if not (tiles_dir / f"{idx}.png.tiles" / "tiles.json").exists()
        ]
        if new_url_docs:
            urls = [d.url for _, d in new_url_docs]
            stems = [str(idx) for idx, _ in new_url_docs]
            backend = ingest_cfg.pop("backend", "cdp")
            render_urls(
                urls, str(tiles_dir), backend=backend, stems=stems, **ingest_cfg
            )
        skipped = len(url_docs) - len(new_url_docs)
        logger.info(
            "  Rendered %d URLs (%d skipped, already exist)", len(new_url_docs), skipped
        )

    # Render PDFs
    for idx, doc in pdf_docs:
        try:
            render_pdf(doc.path, str(tiles_dir))
        except Exception as e:
            logger.warning("  FAILED PDF %s: %s", doc.id, e)
    if pdf_docs:
        logger.info("  Rendered %d PDFs", len(pdf_docs))

    # Save articles.json for serve API — title + URL per article
    articles_path = output / "articles.json"
    max_idx = max(int(a["id"]) for a in articles) + 1 if articles else 0
    article_entries = [{"title": "", "url": ""}] * max_idx
    for a in articles:
        idx = int(a["id"])
        title = a.get("metadata", {}).get("title", "")
        if not title and a.get("url"):
            title = a["url"].split("/")[-1].replace("_", " ").replace("%20", " ")
        url = a.get("url", "")
        article_entries[idx] = {"title": title or str(idx), "url": url}
    with open(articles_path, "w") as f:
        json.dump(article_entries, f)
    logger.info(
        "  Saved %d article mappings to %s", len(article_entries), articles_path
    )

    # Stage 2: Chunk tiles (split large tiles into 1024px strips)
    logger.info("Stage 2/4: Chunking tiles...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pixelrag_embed.chunk",
            "--shard-dir",
            str(tiles_dir),
            "--workers",
            "8",
        ],
        check=True,
    )

    # Stage 3: Embed chunks to vectors
    logger.info("Stage 3/4: Embedding chunks (device=%s)...", device)
    if device in ("cpu", "mps", "auto"):
        # Local embedder: CPU, Apple MPS, or auto-detect
        cmd = [
            sys.executable,
            "-m",
            "pixelrag_embed.embed_cpu",
            "--shard-dir",
            str(tiles_dir),
            "--output-dir",
            str(embeddings_dir),
            "--device",
            device,
        ]
        if "model" in embed_cfg:
            cmd += ["--model", embed_cfg["model"]]
    else:
        # Use GPU embedder (vLLM/sglang)
        cmd = [
            sys.executable,
            "-m",
            "pixelrag_embed.embed",
            "--shard-dir",
            str(tiles_dir),
            "--output-dir",
            str(embeddings_dir),
        ]
        if "gpu_ids" in embed_cfg:
            cmd += ["--gpu-ids", ",".join(str(g) for g in embed_cfg["gpu_ids"])]
        if "model" in embed_cfg:
            cmd += ["--model", embed_cfg["model"]]
        if "backend" in embed_cfg:
            cmd += ["--backend", embed_cfg["backend"]]
    subprocess.run(cmd, check=True)

    # Stage 4: Build FAISS index
    # Auto-adjust nlist based on vector count (IVF needs nlist <= n_vectors)
    import numpy as np

    npz_files = sorted(embeddings_dir.glob("shard_*.npz"))
    total_vectors = sum(
        np.load(f, mmap_mode="r")["embeddings"].shape[0] for f in npz_files
    )
    nlist = min(4096, max(1, total_vectors // 40))
    logger.info(
        "Stage 4/4: Building FAISS index (%d vectors, nlist=%d)...",
        total_vectors,
        nlist,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pixelrag_embed.index",
            "build",
            "--embeddings-dir",
            str(embeddings_dir),
            "--output-dir",
            str(output),
            "--nlist",
            str(nlist),
        ],
        check=True,
    )

    logger.info("Index built at %s", output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Build a visual search index")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--config", "-c", default=None, help="Path to pixelrag.yaml")
    parser.add_argument(
        "--source", "-s", default=None, help="Source path (overrides config)"
    )
    parser.add_argument(
        "--source-type", default=None, help="Source type (kiwix/web/pdf/local)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"], help="Embedding device"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Max documents to process"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Clean output and rebuild from scratch",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config)

    if args.source:
        config.setdefault("source", {})["path"] = args.source
    if args.source_type:
        config.setdefault("source", {})["type"] = args.source_type
    if args.output:
        config["output"] = args.output
    if args.device:
        config.setdefault("embed", {})["device"] = args.device

    if args.command == "build":
        build(config, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
