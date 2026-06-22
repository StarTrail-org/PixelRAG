"""The CDP backend for pixelshot — the single rendering backend.

No Playwright dependency — launches Chrome via subprocess and talks CDP over a
raw websocket. Two capture paths, selected by the Chrome binary:

- STANDARD (default, portable): standard ``Page.captureScreenshot`` (JPEG over CDP).
  Works on any stock Chrome, any OS. Used unless a turbo-capable Chrome is present.
- TURBO: delegates to ``fast_cdp`` (rawFilePath + /dev/shm + parallel JPEG), ~2x at
  batch scale. Used automatically when the pixelrag-installed patched ``headless_shell``
  is selected (``chrome.is_turbo_capable``) and the request matches its capabilities.

Selection is deterministic (by Chrome provenance), with no runtime probe — so a stock
Chrome is never sent the patched-only CDP params (which would hang).

Requirements: websockets, pillow (no playwright needed)

Usage:
    from pixelrag_render.backends.cdp import render_urls
    tile_dirs = render_urls(["https://example.com"], "./tiles", workers=4)
"""

import asyncio
import base64
import io
import json
import logging
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

logger = logging.getLogger("pixelrag_render.backends.cdp")

VIEWPORT_W = 875
VIEWPORT_H = 1080

BROWSER_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-networking",
    "--disable-features=Translate,MediaRouter,OptimizationHints",
    "--enable-gpu-rasterization",
    "--force-gpu-rasterization",
]


def _find_chrome() -> str:
    from ..chrome import find_chrome

    return find_chrome()


async def _connect_cdp(port: int, retries: int = 5, delay: float = 1.0):
    """Connect to Chrome's CDP websocket endpoint."""
    import websockets

    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(
                f"http://localhost:{port}/json", timeout=3
            ).read()
            targets = json.loads(data)
            # Pick a real page target — Chrome's built-in component extensions
            # (Cast/Media Router) expose background_page targets that show up
            # first in /json but never render navigations, hanging CDP capture.
            pages = [t for t in targets if t.get("type") == "page"] or targets
            ws = await websockets.connect(
                pages[0]["webSocketDebuggerUrl"],
                open_timeout=10,
                max_size=50 * 1024 * 1024,
            )
            return ws
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise ConnectionError(f"Failed to connect to Chrome on port {port}")


async def _cdp_send(ws, msg_id_ref: list, method: str, params: dict | None = None):
    """Send a CDP command and wait for its response."""
    msg_id_ref[0] += 1
    mid = msg_id_ref[0]
    msg = {"id": mid, "method": method}
    if params:
        msg["params"] = params
    await ws.send(json.dumps(msg))
    while True:
        r = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
        if r.get("id") == mid:
            if "error" in r:
                raise RuntimeError(f"CDP error: {r['error']}")
            return r.get("result", {})


# Max time to wait for the `load` event (and for the optional network-idle wait)
# before giving up and capturing whatever is there. Keeps a hanging page from
# stalling a worker.
LOAD_TIMEOUT_MS = 12_000
# Network is considered idle once no new resource has been fetched for this long.
NET_QUIET_MS = 500


def _readiness_expr(wait_network_idle: bool) -> str:
    """Build the in-page readiness probe.

    Always waits for the `load` event before measuring (with a
    ``readyState === 'complete'`` shortcut so an already-loaded page returns
    immediately, and a hard timeout so a hanging page can't block). Without this,
    a client-rendered (SPA) page is measured/captured mid-hydration at a transient
    layout — often much taller than the settled page — producing blank tiles. SSR
    pages (e.g. Wikipedia) fire `load` almost immediately, so this adds ~no cost.

    When ``wait_network_idle`` is set, also waits (after load) until no new
    resource has been fetched for ``NET_QUIET_MS`` — for SPAs that fetch their
    content *after* load. This costs a quiet window per page, so it is opt-in
    (the pixelbrowse skill / single-page renders), not the batch default.

    Returns an async-IIFE expression resolving to the page height to tile.
    """
    idle_step = ""
    if wait_network_idle:
        idle_step = f"""
        await new Promise(res => {{
            let timer;
            let obs;
            const finish = () => {{ try {{ obs && obs.disconnect(); }} catch (e) {{}}
                                    clearTimeout(timer); clearTimeout(hard); res(); }};
            const bump = () => {{ clearTimeout(timer); timer = setTimeout(finish, {NET_QUIET_MS}); }};
            try {{
                obs = new PerformanceObserver(bump);
                obs.observe({{ type: 'resource', buffered: true }});
            }} catch (e) {{}}
            const hard = setTimeout(finish, {LOAD_TIMEOUT_MS});
            bump();
        }});"""
    return f"""(async () => {{
        await new Promise(res => {{
            if (document.readyState === 'complete') return res();
            const t = setTimeout(res, {LOAD_TIMEOUT_MS});
            window.addEventListener('load', () => {{ clearTimeout(t); res(); }}, {{ once: true }});
        }});{idle_step}
        await document.fonts.ready;
        // Let layout settle over two frames — but cap it: requestAnimationFrame
        // never ticks in some headless modes (e.g. google-chrome --headless=new
        // with no compositor frames scheduled), where awaiting rAF would hang.
        await Promise.race([
            new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))),
            new Promise(r => setTimeout(r, 1000)),
        ]);
        document.documentElement.style.scrollBehavior = 'auto';
        const sh = document.documentElement.scrollHeight;
        const body = document.body;
        if (body) {{
            const bottom = Math.ceil(body.getBoundingClientRect().bottom);
            return Math.min(sh, Math.max(bottom, 1));
        }}
        return sh;
    }})()"""


# Before capturing a tile below the first one, scroll it into view and wait for
# its now-visible images to load. The capture clip uses absolute page coordinates,
# but Chrome only rasterizes content near the viewport — without scrolling, tiles
# past the first (e.g. on a small tile_height) come back blank. Mirrors fast_cdp.
_SCROLL_WAIT = """new Promise(resolve => {{
    window.scrollTo(0, {y});
    // Safety net: requestAnimationFrame may never tick in headless modes that
    // don't schedule frames, which would leave this promise unresolved.
    setTimeout(resolve, 1500);
    requestAnimationFrame(() => requestAnimationFrame(() => {{
        const imgs = Array.from(document.images).filter(i => {{
            if (i.complete) return false;
            const r = i.getBoundingClientRect();
            return r.bottom > 0 && r.top < window.innerHeight;
        }});
        if (imgs.length === 0) return resolve();
        const timeout = new Promise(r => setTimeout(r, 500));
        const loaded = Promise.all(imgs.map(i => new Promise(r => {{
            i.addEventListener('load', r, {{once: true}});
            i.addEventListener('error', r, {{once: true}});
        }})));
        Promise.race([loaded, timeout]).then(resolve);
    }}));
}})"""


async def capture_url(
    ws,
    msg_id_ref: list,
    url: str,
    tile_dir: Path,
    *,
    tile_h: int = 8192,
    quality: int = 85,
    viewport_w: int = VIEWPORT_W,
    image_format: str = "jpeg",
    from_surface: bool = True,
    wait_network_idle: bool = False,
) -> int:
    """Capture a URL as tiled images via direct CDP websocket.

    Returns the number of tiles written.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)

    await _cdp_send(ws, msg_id_ref, "Page.navigate", {"url": url})

    # Wait for load (+ optional network-idle) + fonts + layout to stabilize,
    # return the page height to tile in one call. See _readiness_expr.
    result = await _cdp_send(
        ws,
        msg_id_ref,
        "Runtime.evaluate",
        {
            "expression": _readiness_expr(wait_network_idle),
            "awaitPromise": True,
            "returnByValue": True,
        },
    )
    try:
        page_height = result["result"]["value"]
    except (KeyError, TypeError):
        page_height = tile_h

    tiles = []
    y = 0
    idx = 0

    while y < page_height:
        clip_h = min(tile_h, page_height - y)
        if clip_h <= 0:
            break

        # Scroll the tile into view so Chrome rasterizes it (tiles past the first
        # are otherwise blank). The top tile is already in view after load.
        if idx > 0:
            try:
                await _cdp_send(
                    ws,
                    msg_id_ref,
                    "Runtime.evaluate",
                    {"expression": _SCROLL_WAIT.format(y=y), "awaitPromise": True},
                )
            except Exception:
                pass

        params = {
            "format": image_format,
            "fromSurface": from_surface,
            "optimizeForSpeed": True,
            "clip": {
                "x": 0,
                "y": y,
                "width": viewport_w,
                "height": clip_h,
                "scale": 1,
            },
        }
        if image_format == "jpeg":
            params["quality"] = quality

        result = await _cdp_send(ws, msg_id_ref, "Page.captureScreenshot", params)

        img_bytes = base64.b64decode(result["data"])
        tile_path = (
            tile_dir / f"tile_{idx:04d}.{'jpg' if image_format == 'jpeg' else 'png'}"
        )

        if clip_h < tile_h:
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            if h > clip_h:
                img = img.crop((0, 0, w, clip_h))
            img.save(
                tile_path, "JPEG" if image_format == "jpeg" else "PNG", quality=quality
            )
        else:
            tile_path.write_bytes(img_bytes)

        tiles.append(tile_path.name)
        idx += 1
        y += tile_h

    manifest = {
        "url": url,
        "page_height": page_height,
        "tiles": tiles,
        "complete": True,
    }
    with open(tile_dir / "tiles.json", "w") as f:
        json.dump(manifest, f)

    return len(tiles)


async def _worker(
    chrome_path: str,
    port: int,
    work_queue: asyncio.Queue,
    output_dir: Path,
    tile_height: int,
    quality: int,
    viewport_w: int,
    image_format: str,
    from_surface: bool,
    wait_network_idle: bool,
    worker_id: int,
    stats: dict,
    results: list,
):
    """Async worker: owns a Chrome process, pulls URLs from queue."""
    # Isolated profile per worker. Without --user-data-dir, a launch on a machine that
    # already has Chrome open forwards to the running instance (default profile) instead
    # of starting this headless renderer — navigation/screenshot then hang forever. A
    # unique dir also stops parallel workers from colliding on one profile. See issue #54.
    user_data_dir = tempfile.mkdtemp(prefix=f"pixelshot_chrome_{port}_")
    proc = subprocess.Popen(
        # `--headless=new`: the bare `--headless` is deprecated and hangs on modern
        # Chrome (e.g. google-chrome 149); `=new` works on both stock Chrome and the
        # patched headless_shell.
        [
            chrome_path,
            f"--remote-debugging-port={port}",
            "--headless=new",
            f"--user-data-dir={user_data_dir}",
        ]
        + BROWSER_ARGS
        + ["about:blank"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        await asyncio.sleep(3)
        ws = await _connect_cdp(port)
        msg_id_ref = [0]

        await _cdp_send(ws, msg_id_ref, "Page.enable")
        if wait_network_idle:
            # PerformanceObserver (used by the idle wait) needs no CDP domain, but
            # enabling Network keeps resource timing reliable across navigations.
            await _cdp_send(ws, msg_id_ref, "Network.enable")
        await _cdp_send(
            ws,
            msg_id_ref,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": viewport_w,
                "height": tile_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

        while True:
            try:
                item = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url = item["url"]
            stem = item["stem"]
            tile_dir = output_dir / f"{stem}.png.tiles"

            t0 = time.monotonic()
            try:
                n_tiles = await capture_url(
                    ws,
                    msg_id_ref,
                    url,
                    tile_dir,
                    tile_h=tile_height,
                    quality=quality,
                    viewport_w=viewport_w,
                    image_format=image_format,
                    from_surface=from_surface,
                    wait_network_idle=wait_network_idle,
                )
                stats["done"] += 1
                elapsed = time.monotonic() - t0
                logger.info(
                    "[w%d] %s → %d tiles (%.1fs)", worker_id, url, n_tiles, elapsed
                )
                results.append(tile_dir)
            except Exception as e:
                stats["failed"] += 1
                logger.warning("[w%d] FAIL %s: %s", worker_id, url, str(e)[:200])

        await ws.close()
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(user_data_dir, ignore_errors=True)


def _derive_stems(urls: list[str], stems: list[str] | None) -> list[str]:
    """Output-dir stem per URL (explicit stems win; else sanitize the URL).

    Shared by the standard and turbo paths so both emit identical
    ``{stem}.png.tiles`` directory names for the same inputs.
    """
    from urllib.parse import urlparse

    out: list[str] = []
    seen: dict[str, int] = {}
    for i, url in enumerate(urls):
        if stems and i < len(stems):
            out.append(str(stems[i]))
            continue
        parsed = urlparse(url)
        raw = (parsed.netloc + parsed.path).rstrip("/")
        stem = (
            raw.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
        )
        stem = stem[:200] or "page"
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        if count > 0:
            stem = f"{stem}_{count}"
        out.append(stem)
    return out


async def _run_batch(
    urls: list[str],
    output_dir: Path,
    num_workers: int,
    tile_height: int,
    quality: int,
    viewport_w: int,
    image_format: str,
    from_surface: bool,
    wait_network_idle: bool,
    stems: list[str] | None,
    chrome_path: str,
) -> list[Path]:
    work_queue: asyncio.Queue = asyncio.Queue()
    stem_list = _derive_stems(urls, stems)
    for url, stem in zip(urls, stem_list):
        work_queue.put_nowait({"url": url, "stem": stem})

    stats = {"done": 0, "failed": 0}
    results: list[Path] = []
    base_port = 9400

    actual_workers = min(num_workers, len(urls))
    workers = [
        _worker(
            chrome_path,
            base_port + wid,
            work_queue,
            output_dir,
            tile_height,
            quality,
            viewport_w,
            image_format,
            from_surface,
            wait_network_idle,
            wid,
            stats,
            results,
        )
        for wid in range(actual_workers)
    ]
    await asyncio.gather(*workers, return_exceptions=True)

    logger.info("Batch complete: done=%d failed=%d", stats["done"], stats["failed"])
    return results


def render_urls(
    urls: list[str],
    output_dir: str | Path,
    *,
    stems: list[str] | None = None,
    tile_height: int = 8192,
    quality: int = 85,
    viewport_width: int = VIEWPORT_W,
    workers: int = 4,
    image_format: str = "jpeg",
    from_surface: bool = True,
    wait_network_idle: bool = False,
    turbo: bool | None = None,
    chrome_path: str | None = None,
) -> list[Path]:
    """Render URLs to tiled images via CDP.

    Uses the TURBO path (fast_cdp: rawFilePath + parallel JPEG) when a turbo-capable
    patched Chrome is present and the request matches its capture profile; otherwise
    the portable STANDARD path. Both emit ``{stem}.png.tiles/`` with a tiles.json.

    Args:
        urls: URLs to capture.
        output_dir: Output directory for tile subdirectories.
        stems: Optional output directory name per URL.
        tile_height: Max tile height in pixels (default 8192).
        quality: JPEG quality 1-100 (default 85).
        viewport_width: Browser viewport width (default 875).
        workers: Number of parallel Chrome processes (default 4).
        image_format: 'jpeg' or 'png' (default 'jpeg').
        from_surface: CDP fromSurface param. True for batch (throughput),
                      False for serve (low latency). Default True.
        wait_network_idle: After the load event, also wait until the network has
                      been quiet (~500ms) before capturing (SPAs that fetch after
                      load). Standard path only; off by default.
        turbo: None = auto (turbo when the Chrome is turbo-capable), True/False to
                      force. Turbo only applies to the default capture profile
                      (jpeg, default viewport, fromSurface, no network-idle wait);
                      other options always use the standard path.
        chrome_path: Path to Chrome binary. Auto-detected if None.

    Returns:
        List of Path objects for created tile directories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not urls:
        return []

    chrome = chrome_path or _find_chrome()

    # Turbo only covers fast_cdp's capture profile; anything else → standard path.
    from ..chrome import is_turbo_capable

    use_turbo = is_turbo_capable(chrome) if turbo is None else turbo
    if use_turbo and (
        image_format != "jpeg"
        or viewport_width != VIEWPORT_W
        or wait_network_idle
        or not from_surface
    ):
        use_turbo = False

    if use_turbo:
        from .fast_cdp import render_articles

        stem_list = _derive_stems(urls, stems)

        # path "{stem}.png" makes fast_cdp emit "{stem}.png.tiles" — the same
        # layout the standard path / CLI / index pipeline expect. fast_cdp prepends
        # file:// to non-http inputs, so hand it a plain path for file:// URIs.
        def _navtarget(u: str) -> str:
            if u.startswith("http"):
                return u
            return u[len("file://") :] if u.startswith("file://") else u

        articles = [
            {"path": f"{stem}.png", "file": _navtarget(url)}
            for stem, url in zip(stem_list, urls)
        ]
        logger.info("Using turbo (fast_cdp) path for %d URL(s)", len(urls))
        asyncio.run(
            render_articles(
                articles,
                str(output_dir),
                chrome_path=chrome,
                n_workers=workers,
                tile_height=tile_height,
                jpeg_quality=quality,
            )
        )
        return [output_dir / f"{stem}.png.tiles" for stem in stem_list]

    return asyncio.run(
        _run_batch(
            urls,
            output_dir,
            workers,
            tile_height,
            quality,
            viewport_width,
            image_format,
            from_surface,
            wait_network_idle,
            stems,
            chrome,
        )
    )
