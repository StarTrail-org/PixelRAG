"""End-to-end pipeline: source -> ingest -> chunk -> embed -> build."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load_config, make_source

logger = logging.getLogger("pixelrag-index")


def _needs_render(tiles_dir: Path, idx: int, doc) -> bool:
    """True if {idx}.png.tiles must be (re)rendered for doc.

    An existing tile directory is reusable only if its manifest parses and
    records the same source document (``source``, falling back to the legacy
    ``url`` field) as the doc that now occupies position ``idx``. Position
    indices shift whenever the source set changes between runs (a file added,
    removed, or renamed) — reusing the directory then would silently pair one
    document's pixels with another document's metadata. On mismatch the stale
    directory is removed and re-rendered; corrupt manifests (e.g. a crash
    mid-write) are likewise re-rendered instead of poisoning later stages.
    """
    tile_dir = tiles_dir / f"{idx}.png.tiles"
    manifest_path = tile_dir / "tiles.json"
    if not manifest_path.exists():
        return True
    expected = doc.url or doc.path
    try:
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest.get("source") or manifest.get("url")
    except (json.JSONDecodeError, OSError):
        logger.warning("  Corrupt manifest in %s — re-rendering", tile_dir.name)
        recorded = None
    if recorded == expected:
        return False
    if recorded is not None:
        logger.warning(
            "  %s was rendered from %r but position %d now holds %r — "
            "re-rendering (source set changed between runs; use --force "
            "for a clean rebuild)",
            tile_dir.name,
            recorded,
            idx,
            expected,
        )
    shutil.rmtree(tile_dir, ignore_errors=True)
    return True


def _department_of(article: dict, source_root: str) -> str:
    """Department = first sub-directory under the source root holding the file.

    Files directly under the root, non-local documents (web URLs), or paths
    outside the root have no department ("").
    """
    raw = article.get("path") or ""
    if not raw:
        url = article.get("url") or ""
        if url.startswith("file://"):
            from urllib.parse import unquote, urlparse

            raw = unquote(urlparse(url).path)
    if not raw or not source_root:
        return ""
    try:
        rel = Path(raw).resolve().relative_to(Path(source_root).expanduser().resolve())
    except ValueError:
        return ""
    return rel.parts[0] if len(rel.parts) > 1 else ""


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
    # Copy so the pop/setdefault below never mutate the caller's dict (or the
    # module-level DEFAULT_CONFIG when the yaml has no `ingest:` section).
    ingest_cfg = dict(config.get("ingest", {}))
    # Wait for network idle by default only for the `web` source (arbitrary
    # external URLs), where JS-rendered SPAs fetch content after `load` and
    # would otherwise produce blank/incomplete tiles. Everything else — kiwix
    # (localhost), local text docs (file://) — has its assets ready before
    # `load` fires (see docs/screenshot-throughput-optimization.md), and the
    # idle wait would cost >=500ms per page AND disqualify the turbo capture
    # path (cdp.py forces the standard path when wait_network_idle is set).
    # An explicit `ingest: {wait_network_idle: ...}` in pixelrag.yaml wins.
    if config.get("source", {}).get("type") == "web":
        ingest_cfg.setdefault("wait_network_idle", True)
    embed_cfg = config.get("embed", {})
    device = embed_cfg.get("device", "cpu")

    if force:
        for d in (tiles_dir, embeddings_dir):
            if d.exists():
                shutil.rmtree(d)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Render documents to tiles
    # Use sequential integer IDs as tile directory names so embed/serve can map them
    from pixelrag_render.render import render_urls, render_pdf

    logger.info("Stage 1/4: Rendering %d documents to tiles...", len(docs))

    # Collect documents into batches by type
    url_docs = []
    pdf_docs = []
    image_docs = []
    text_docs = []
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
        elif doc.path and (doc.metadata or {}).get("type") == "text":
            text_docs.append((idx, doc))
        elif doc.path:
            image_docs.append((idx, doc))

    # Render URL batch — skip already-captured articles
    if url_docs:
        new_url_docs = [
            (idx, d) for idx, d in url_docs if _needs_render(tiles_dir, idx, d)
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

    # Render text files (.md, .txt) — convert to styled HTML, then render via CDP
    if text_docs:
        import html as html_mod
        import re
        import tempfile

        import markdown as md_lib

        _HTML_TEMPLATE = (
            '<html><head><meta charset="utf-8"><style>'
            "body { font-family: -apple-system, system-ui, 'Segoe UI', Helvetica, Arial, sans-serif; "
            "max-width: 860px; margin: 40px auto; padding: 20px; line-height: 1.6; "
            "color: #1f2328; font-size: 16px; } "
            "h1, h2, h3 { border-bottom: 1px solid #d1d9e0; padding-bottom: 0.3em; } "
            "h1 { font-size: 2em; } h2 { font-size: 1.5em; } h3 { font-size: 1.25em; } "
            "code { background: #eff1f3; padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%%; "
            "font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; } "
            "pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; } "
            "pre code { background: none; padding: 0; } "
            "table { border-collapse: collapse; width: 100%%; margin: 16px 0; } "
            "th, td { border: 1px solid #d1d9e0; padding: 6px 13px; } "
            "th { background: #f6f8fa; font-weight: 600; } "
            "a { color: #0969da; text-decoration: none; } "
            "blockquote { border-left: 4px solid #d1d9e0; margin: 0; padding: 0 16px; color: #59636e; } "
            "img { max-width: 100%%; } "
            "hr { border: none; border-top: 1px solid #d1d9e0; margin: 24px 0; } "
            "</style></head><body>"
        )

        def _resolve_relative_paths(html_str: str, base_dir: str) -> str:
            """Rewrite relative src/href paths to file:// so the browser can load them."""

            def _repl(m: re.Match) -> str:
                attr, quote, url = m.group(1), m.group(2), m.group(3)
                if url.startswith(("http://", "https://", "file://", "data:", "#")):
                    return m.group(0)
                resolved = (Path(base_dir) / url).resolve()
                return f"{attr}={quote}{resolved.as_uri()}{quote}"

            return re.sub(r'(src|href)=(["\'])([^"\']+)\2', _repl, html_str)

        text_urls = []
        text_stems = []

        with tempfile.TemporaryDirectory(prefix="pixelrag_text_") as tmp_dir:
            for idx, doc in text_docs:
                if not _needs_render(tiles_dir, idx, doc):
                    continue
                src_path = Path(doc.path)
                content = src_path.read_text(errors="replace")
                ext = (doc.metadata or {}).get("extension", src_path.suffix.lower())
                if ext == ".md":
                    body = md_lib.markdown(
                        content, extensions=["tables", "fenced_code"]
                    )
                else:
                    body = f"<pre>{html_mod.escape(content)}</pre>"
                body = _resolve_relative_paths(body, str(src_path.parent))
                html_content = f"{_HTML_TEMPLATE}{body}</body></html>"
                html_path = Path(tmp_dir) / f"{idx}.html"
                html_path.write_text(html_content)
                text_urls.append(f"file://{html_path.resolve()}")
                text_stems.append(str(idx))

            if text_urls:
                text_ingest = {k: v for k, v in ingest_cfg.items() if k != "backend"}
                render_urls(
                    text_urls,
                    str(tiles_dir),
                    backend="cdp",
                    stems=text_stems,
                    **text_ingest,
                )
        logger.info("  Rendered %d text files (.md/.txt)", len(text_docs))

    # Render PDFs — use idx as tile directory name (like URLs) so directory
    # names are always the numeric article_id.
    for idx, doc in pdf_docs:
        if not _needs_render(tiles_dir, idx, doc):
            continue  # already rendered for this same document
        try:
            render_pdf(doc.path, str(tiles_dir), stem=str(idx))
        except Exception as e:
            logger.warning("  FAILED PDF %s: %s", doc.id, e)
    if pdf_docs:
        logger.info("  Rendered %d PDFs", len(pdf_docs))

    # Render local images (PNG/JPG) — copy/resize into the tile directory structure
    if image_docs:
        from PIL import Image as PILImage

        _MAX_WIDTH = 4000  # cap large images to avoid VRAM pressure during embedding

        for idx, doc in image_docs:
            if not _needs_render(tiles_dir, idx, doc):
                continue
            tile_dir = tiles_dir / f"{idx}.png.tiles"
            tile_dir.mkdir(parents=True, exist_ok=True)
            try:
                img = PILImage.open(doc.path)
                # JPEG has no alpha: composite transparent images onto white
                # before dropping the channel. A bare convert("RGB") maps
                # fully-transparent pixels to their underlying RGB — black for
                # typical chart/logo exports — turning them into dark garbage.
                if img.mode in ("RGBA", "LA", "PA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    rgba = img.convert("RGBA")
                    background = PILImage.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    img = background
                else:
                    img = img.convert("RGB")
                # Resize if too wide
                if img.width > _MAX_WIDTH:
                    ratio = _MAX_WIDTH / img.width
                    img = img.resize(
                        (int(img.width * ratio), int(img.height * ratio)),
                        PILImage.LANCZOS,
                    )
                tile_path = tile_dir / "tile_0000.jpg"
                img.save(tile_path, "JPEG", quality=90)
                manifest = {
                    "url": doc.path,
                    "page_height": img.height,
                    "tiles": ["tile_0000.jpg"],
                    "complete": True,
                }
                with open(tile_dir / "tiles.json", "w") as f:
                    json.dump(manifest, f)
            except Exception as e:
                logger.warning("  FAILED image %s: %s", doc.id, e)
        logger.info("  Rendered %d local images", len(image_docs))

    # Write article_id and the source identity into each tile directory's
    # manifests, so the embed pipeline reads the id explicitly instead of
    # guessing from the directory name, and so the next run's _needs_render
    # can detect stale directories after the source set changes. tiles.json
    # always exists here; chunks.json exists only for PDFs (pdf.py writes it
    # at render time, and chunk.py then skips those dirs). For every other
    # source chunks.json is created by Stage 2's chunk.py, which propagates
    # article_id from tiles.json. So write whichever exist now.
    for idx, doc in url_docs + text_docs + pdf_docs + image_docs:
        identity = doc.url or doc.path
        for manifest_name in ("tiles.json", "chunks.json"):
            manifest_path = tiles_dir / f"{idx}.png.tiles" / manifest_name
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("  Could not stamp unreadable %s", manifest_path)
                continue
            if manifest.get("article_id") == idx and manifest.get("source") == identity:
                continue  # already stamped — skip the rewrite
            manifest["article_id"] = idx
            manifest["source"] = identity
            # Atomic replace: a crash mid-write must not truncate the manifest
            # (a corrupt tiles.json would otherwise poison every later stage).
            tmp_path = manifest_path.with_name(manifest_name + ".tmp")
            tmp_path.write_text(json.dumps(manifest))
            os.replace(tmp_path, manifest_path)

    # Save articles.json for serve API — title + URL per article.
    # Use the pipeline's sequential *position index* (0, 1, 2, …) rather than
    # int(a["id"]), because local sources use filename stems (e.g. "art_alice")
    # as doc IDs, which are not numeric. int() on a filename stem raises ValueError
    # and crashes the entire index build step.
    articles_path = output / "articles.json"
    source_root = str(config.get("source", {}).get("path", "") or "")
    article_entries = []
    for enum_idx, a in enumerate(articles):
        title = a.get("metadata", {}).get("title", "")
        if not title and a.get("url"):
            title = a["url"].split("/")[-1].replace("_", " ").replace("%20", " ")
        if not title:
            # Fall back to original doc id (e.g. filename stem) as display title
            title = a.get("id", str(enum_idx))
        url = a.get("url", "") or a.get("path", "")
        # Department (from the directory layout of local sources) enables
        # server-side filtered search; empty string means "no department".
        article_entries.append(
            {"title": title, "url": url, "department": _department_of(a, source_root)}
        )
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
        "--device",
        default=None,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Embedding device (default: from config; auto-detects cuda/mps/cpu)",
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
