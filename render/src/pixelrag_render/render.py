"""Public API for pixelshot.

Renders documents (URLs, PDFs, local HTML/image files) to image tiles.

Entry point:
    pixelshot <inputs> --output ./tiles --backend cdp --workers 4
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pixelrag_render.render")


def render_url(
    url: str,
    output_dir: str | Path,
    backend: str = "cdp",
    *,
    tile_height: int = 8192,
    quality: int = 85,
    viewport_width: int = 875,
    workers: int = 1,
    **kwargs,
) -> list[Path]:
    """Render a URL to tiled JPEG images.

    Args:
        url: URL to capture (http:// or https:// or file://).
        output_dir: Directory to write tile subdirectories into.
        backend: Rendering backend: ``"cdp"`` (default, fastest) or
                 ``"playwright"`` (full-featured).
        tile_height: Maximum tile height in pixels (default 8192).
        quality: JPEG quality 1-100 (default 85).
        viewport_width: Browser viewport width in pixels (default 875).
        workers: Number of parallel browser processes (default 1).
        **kwargs: Additional keyword arguments forwarded to the backend.

    Returns:
        List of Path objects pointing to created tile directories.
    """
    return render_urls(
        [url],
        output_dir,
        backend=backend,
        tile_height=tile_height,
        quality=quality,
        viewport_width=viewport_width,
        workers=workers,
        **kwargs,
    )


def render_urls(
    urls: list[str],
    output_dir: str | Path,
    backend: str = "cdp",
    *,
    stems: list[str] | None = None,
    tile_height: int = 8192,
    quality: int = 85,
    viewport_width: int = 875,
    workers: int = 4,
    **kwargs,
) -> list[Path]:
    """Render a list of URLs to tiled JPEG images.

    Args:
        urls: URLs to capture.
        output_dir: Directory to write tile subdirectories into.
        backend: ``"cdp"`` (default) or ``"playwright"``.
        stems: Optional list of output directory stems (one per URL).
               If provided, tiles are written to ``{output_dir}/{stem}.png.tiles/``
               instead of deriving names from URLs. Useful for assigning
               sequential IDs (e.g. ``["0", "1", "2"]``).
        tile_height: Maximum tile height in pixels (default 8192).
        quality: JPEG quality 1-100 (default 85).
        viewport_width: Browser viewport width in pixels (default 875).
        workers: Number of parallel browser processes (default 4).
        **kwargs: Additional keyword arguments forwarded to the backend.

    Returns:
        List of Path objects pointing to created tile directories.
    """
    if backend in ("cdp", "websocket"):  # "websocket" kept as a back-compat alias
        from .backends.cdp import render_urls as _render_urls
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Choose 'cdp'."
            " The cdp backend auto-selects a turbo capture path when a turbo-capable"
            " Chrome is present."
        )

    return _render_urls(
        urls,
        output_dir,
        stems=stems,
        tile_height=tile_height,
        quality=quality,
        viewport_width=viewport_width,
        workers=workers,
        **kwargs,
    )


def render_pdf(
    path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 200,
    pages: Optional[list[int]] = None,
    quality: int = 85,
    stem: str | None = None,
) -> list[Path]:
    """Render a PDF file to tiled JPEG images.

    Args:
        path: Path to the PDF file.
        output_dir: Directory to write the tile subdirectory into.
        dpi: Rendering resolution (default 200 ≈ 1650×2200 for A4).
        pages: 1-based list of page numbers to render. ``None`` renders all.
        quality: JPEG quality 1-100 (default 85).
        stem: Override for the tile directory name (default: PDF filename stem).

    Returns:
        List containing the tile directory Path on success.
    """
    from .backends.pdf import render_pdf as _render_pdf

    return _render_pdf(
        path, output_dir, dpi=dpi, pages=pages, quality=quality, stem=stem
    )


def render_file(
    path: str | Path,
    output_dir: str | Path,
    backend: str = "cdp",
    **kwargs,
) -> list[Path]:
    """Auto-detect file type and render to tiled JPEG images.

    Dispatch rules:
    - ``.pdf`` → ``render_pdf()``
    - ``.html`` / ``.htm`` → ``render_url(file://...)``
    - ``.png`` / ``.jpg`` / ``.jpeg`` / ``.webp`` → copy into output_dir as-is
    - ``http://`` or ``https://`` prefix → ``render_url()``

    Args:
        path: Path to a local file, or a URL string.
        output_dir: Directory to write tile subdirectories into.
        backend: Browser backend for HTML/URL rendering (default ``"cdp"``).
        **kwargs: Forwarded to the underlying render function.

    Returns:
        List of Path objects pointing to created tile directories or copied files.
    """
    path_str = str(path)
    output_dir = Path(output_dir)

    # URL strings
    if path_str.startswith("http://") or path_str.startswith("https://"):
        return render_url(path_str, output_dir, backend=backend, **kwargs)

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return render_pdf(p, output_dir, **kwargs)

    if suffix in {".html", ".htm"}:
        file_url = p.resolve().as_uri()
        return render_url(file_url, output_dir, backend=backend, **kwargs)

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / p.name
        shutil.copy2(str(p), str(dest))
        logger.info("Copied image: %s → %s", p, dest)
        return [dest]

    raise ValueError(
        f"Cannot auto-detect render method for {path!r}. "
        "Supported: .pdf, .html, .htm, .png, .jpg, .jpeg, .webp, http://, https://"
    )


def main() -> None:
    """CLI entry point: pixelshot.

    Usage examples::

        # Single URL, default CDP backend
        pixelshot https://example.com --output ./tiles

        # Multiple inputs with 4 workers
        pixelshot https://a.com https://b.com --output ./tiles --workers 4

        # PDF
        pixelshot report.pdf --output ./tiles

        # Local HTML
        pixelshot index.html --output ./tiles --backend playwright

        # Pipe URLs from a file
        cat urls.txt | xargs pixelshot --output ./tiles --workers 8

        # Chrome management (folded from the former `pixelrag-chrome`)
        pixelshot install-chrome   # download the patched headless Chrome
        pixelshot which-chrome     # print the active Chrome binary path
    """
    # Chrome management subcommands — dispatch before building the render parser.
    if len(sys.argv) > 1 and sys.argv[1] in ("install-chrome", "which-chrome"):
        from pixelrag_render import chrome

        if sys.argv[1] == "install-chrome":
            chrome.install_chrome()
        else:
            try:
                print(chrome.find_chrome(auto_install=False))
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                sys.exit(1)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="pixelshot",
        description="Render documents (URLs, PDFs, HTML files) to tiled JPEG images.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="URLs or file paths to render.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./tiles",
        metavar="DIR",
        help="Output directory for tile subdirectories (default: ./tiles).",
    )
    parser.add_argument(
        "--backend",
        choices=["cdp", "playwright"],
        default="cdp",
        help="Browser backend for URL/HTML rendering (default: cdp).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel browser processes (default: 4).",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=8192,
        help="Maximum tile height in pixels (default: 8192).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85).",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=875,
        help="Browser viewport width in pixels (default: 875).",
    )
    parser.add_argument(
        "--wait-network-idle",
        action="store_true",
        help="After the page's load event, also wait until the network is quiet "
        "(~500ms) before capturing. Helps JS/SPA pages that fetch content after "
        "load; adds a quiet window per page, so off by default. Recommended for "
        "single-page renders (e.g. the pixelbrowse skill).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF rendering (default: 200).",
    )
    parser.add_argument(
        "--cdp-url",
        default=os.environ.get("PIXELSHOT_CDP_URL"),
        metavar="URL",
        help="Attach to an already-running Chrome/Brave DevTools endpoint "
        "(e.g. http://127.0.0.1:9222) instead of launching a throwaway headless "
        "browser. Renders each input in a fresh tab using that browser's existing "
        "session (cookies/logins) — so authenticated pages work — then closes only "
        "that tab. Needs no local Chrome binary. Env: PIXELSHOT_CDP_URL.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Partition inputs into URLs and files for batch processing
    urls = []
    files = []
    for inp in args.inputs:
        if inp.startswith("http://") or inp.startswith("https://"):
            urls.append(inp)
        else:
            files.append(Path(inp))

    results: list[Path] = []

    # Batch-render URLs together for efficiency
    if urls:
        logger.info(
            "Rendering %d URL(s) with backend=%s workers=%d",
            len(urls),
            args.backend,
            args.workers,
        )
        tile_dirs = render_urls(
            urls,
            output_dir,
            backend=args.backend,
            tile_height=args.tile_height,
            quality=args.quality,
            viewport_width=args.viewport_width,
            workers=args.workers,
            wait_network_idle=args.wait_network_idle,
            cdp_url=args.cdp_url,
        )
        results.extend(tile_dirs)

    # Handle files individually (they may need different backends)
    for fpath in files:
        suffix = fpath.suffix.lower()
        try:
            if suffix == ".pdf":
                tile_dirs = render_pdf(
                    fpath, output_dir, dpi=args.dpi, quality=args.quality
                )
            elif suffix in {".html", ".htm"}:
                file_url = fpath.resolve().as_uri()
                tile_dirs = render_url(
                    file_url,
                    output_dir,
                    backend=args.backend,
                    tile_height=args.tile_height,
                    quality=args.quality,
                    viewport_width=args.viewport_width,
                    workers=1,
                    wait_network_idle=args.wait_network_idle,
                    cdp_url=args.cdp_url,
                )
            elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                tile_dirs = render_file(fpath, output_dir)
            else:
                logger.warning("Unsupported file type: %s — skipping", fpath)
                continue
            results.extend(tile_dirs)
        except Exception as e:
            logger.error("Failed to render %s: %s", fpath, e)

    if results:
        logger.info("Done. %d output(s):", len(results))
        for r in results:
            print(r)
    else:
        logger.warning("No outputs produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
