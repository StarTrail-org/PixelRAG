"""Fast CDP backend: raw BGRA capture → async JPEG compression.

Architecture:
  Chrome workers (n_workers)     Compression pool (n_compressors procs)
      ↓ rawFilePath                   ↓ read /dev/shm
    /dev/shm/pixelrag_render/raw/   → JPEG compress
    (28MB × n_workers slots)      → output/tiles/

Capture and compression are fully decoupled.  Chrome writes raw BGRA to
/dev/shm via Page.captureScreenshot rawFilePath.  A background asyncio task
drains the compression queue and submits work to a ProcessPoolExecutor.
Capture never waits for compression.

Requirements: pillow, websockets (no playwright needed)

Usage:
    from pixelrag_render.backends.fast_cdp import render_articles
    result = render_articles(articles, "./tiles")
    # result: {"total_tiles": N, "wall_s": T, "tiles_per_s": tps}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("pixelrag_render.backends.fast_cdp")

VIEWPORT_WIDTH = 875
TILE_HEIGHT = 8192

CHROME_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-gpu-rasterization",
    "--force-gpu-rasterization",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-networking",
    "--disable-features=Translate,MediaRouter,OptimizationHints",
]

# JS: wait for fonts + eager images, then return scrollHeight
_WAIT_FONTS_IMGS = """new Promise(resolve => {
    const waitEagerImgs = Promise.all(
        Array.from(document.images)
            .filter(i => !i.complete && i.loading !== 'lazy')
            .map(i => new Promise(r => {
                i.addEventListener('load', r, {once: true});
                i.addEventListener('error', r, {once: true});
            }))
    );
    const timeout = new Promise(r => setTimeout(r, 2000));
    Promise.race([
        Promise.all([document.fonts.ready, waitEagerImgs]),
        timeout
    ]).then(() => {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                document.documentElement.style.scrollBehavior = 'auto';
                const sh = document.documentElement.scrollHeight;
                const body = document.body;
                resolve(body
                    ? Math.min(sh, Math.max(Math.ceil(body.getBoundingClientRect().bottom), 1))
                    : sh);
            });
        });
    });
})"""


# ---------------------------------------------------------------------------
# Subprocess: JPEG compression (runs in ProcessPoolExecutor worker)
# ---------------------------------------------------------------------------


def compress_tile(raw_path: str, out_path: str, quality: int = 85) -> None:
    """Read raw BGRA file, compress to JPEG, delete raw file.

    Raw file layout (written by Chrome rawFilePath):
        bytes 0-3:  width  (uint32 LE)
        bytes 4-7:  height (uint32 LE)
        bytes 8-11: rowBytes (uint32 LE)
        bytes 12+:  BGRA pixels
    """
    from PIL import Image

    data = open(raw_path, "rb").read()
    w, h, rb = struct.unpack_from("<III", data, 0)
    img = Image.frombuffer("RGBA", (w, h), data[12:], "raw", "BGRA", rb, 1)
    img = img.convert("RGB")
    img.save(out_path, "JPEG", quality=quality)
    os.unlink(raw_path)


# ---------------------------------------------------------------------------
# Chrome connection helpers (inlined to avoid circular deps)
# ---------------------------------------------------------------------------

_port_counter = 0


def _next_base_port() -> int:
    global _port_counter
    _port_counter += 1
    return 12000 + (_port_counter - 1) * 500


async def _launch_chrome(chrome_path: str, port: int) -> tuple:
    """Launch a headless Chrome and return (websocket, proc, user_data_dir)."""
    import websockets

    # Isolated profile per worker. Without --user-data-dir, a launch on a machine that
    # already has Chrome open forwards to the running instance (default profile) instead
    # of starting this headless renderer — navigation/screenshot then hang forever. A
    # unique dir also stops parallel workers from colliding on one profile. See issue #54.
    user_data_dir = tempfile.mkdtemp(prefix=f"pixelshot_chrome_{port}_")
    args = (
        # `--headless=new`: bare `--headless` is deprecated and hangs on modern Chrome.
        [
            chrome_path,
            f"--remote-debugging-port={port}",
            "--headless=new",
            f"--user-data-dir={user_data_dir}",
        ]
        + CHROME_ARGS
        + ["about:blank"]
    )
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for attempt in range(10):
        await asyncio.sleep(1)
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
            return ws, proc, user_data_dir
        except Exception:
            if attempt == 9:
                proc.kill()
                shutil.rmtree(user_data_dir, ignore_errors=True)
                raise ConnectionError(f"Failed to connect to Chrome on port {port}")


class _Conn:
    """Minimal CDP connection with a receive loop."""

    def __init__(self, ws, proc, user_data_dir=None):
        self._ws = ws
        self._proc = proc
        self._user_data_dir = user_data_dir
        self._msg_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._event_listeners: dict[str, list] = {}
        self._recv_task: asyncio.Task | None = None

    def _ensure_recv(self):
        if self._recv_task is None or self._recv_task.done():
            self._recv_task = asyncio.get_event_loop().create_task(self._recv_loop())

    async def _recv_loop(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mid = msg.get("id")
                if mid is not None:
                    fut = self._pending.pop(mid, None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    method = msg.get("method", "")
                    listeners = self._event_listeners.get(method, [])
                    remaining = []
                    for fut, filter_fn in listeners:
                        if fut.done():
                            continue
                        params = msg.get("params", {})
                        matched = filter_fn(params) if filter_fn else True
                        if matched:
                            fut.set_result(params)
                        else:
                            remaining.append((fut, filter_fn))
                    self._event_listeners[method] = remaining
        except Exception:
            exc = ConnectionError("WebSocket receive loop ended")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(exc)
            for listeners in self._event_listeners.values():
                for fut, _ in listeners:
                    if not fut.done():
                        fut.set_exception(exc)

    async def cdp(self, method: str, params: dict | None = None) -> dict:
        self._ensure_recv()
        self._msg_id += 1
        mid = self._msg_id
        msg = {"id": mid, "method": method}
        if params:
            msg["params"] = params
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps(msg))
        return await asyncio.wait_for(fut, timeout=180)

    async def wait_for_event(
        self, method: str, timeout: float = 30.0, filter_fn=None
    ) -> dict:
        self._ensure_recv()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._event_listeners.setdefault(method, []).append((fut, filter_fn))
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            listeners = self._event_listeners.get(method, [])
            self._event_listeners[method] = [
                (f, fn) for f, fn in listeners if f is not fut
            ]
            if not fut.done():
                fut.cancel()
            raise

    async def close(self):
        try:
            await self._ws.close()
        except Exception:
            pass
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._user_data_dir:
            shutil.rmtree(self._user_data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Core render logic
# ---------------------------------------------------------------------------


async def _run_render(
    articles: list[dict],
    output_dir: Path,
    chrome_path: str,
    n_workers: int,
    tile_height: int,
    jpeg_quality: int,
    n_compressors: int,
) -> dict:
    raw_dir = Path("/dev/shm/pixelrag_render/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Semaphore: limit concurrent captures (CPU-bound) to n_workers // 2
    capture_limit = max(1, n_workers // 2)
    capture_sem = asyncio.Semaphore(capture_limit)

    # Compression: dedicated thread with its own multiprocessing.Pool.
    # Tiles are pushed to a thread-safe queue from capture workers.
    # The thread runs pool.starmap in batches, fully independent of asyncio.
    from multiprocessing import Pool as MPPool
    import queue as _queue
    import threading

    metrics = {
        "total_tiles": 0,
        "total_capture_ms": 0.0,
        "errors": 0,
    }

    compress_inbox: _queue.Queue = _queue.Queue()
    compress_done = threading.Event()

    n_cpus = os.cpu_count() or 128
    compress_cores = set(range(max(0, n_cpus - n_compressors), n_cpus))

    def _pool_init():
        try:
            os.sched_setaffinity(0, compress_cores)
        except OSError:
            pass

    def _compressor_thread():
        pool = MPPool(processes=n_compressors, initializer=_pool_init)
        # Warm up: ensure all workers are forked and idle before capture starts
        pool.map(int, range(n_compressors))
        async_results = []
        while True:
            item = compress_inbox.get()  # block until item available
            if item is None:
                break
            async_results.append(pool.apply_async(compress_tile, item))
        # Wait for all remaining
        for ar in async_results:
            try:
                ar.get(timeout=60)
            except Exception:
                pass
        pool.close()
        pool.join()
        compress_done.set()

    compress_thread = threading.Thread(target=_compressor_thread, daemon=True)
    compress_thread.start()

    base_port = _next_base_port()

    # Work-stealing queue
    work_q: asyncio.Queue = asyncio.Queue()
    for art in articles:
        work_q.put_nowait(art)

    # Launch Chrome workers
    connections: list[_Conn] = []
    frame_ids: list[str] = []

    logger.info(
        "Launching %d Chrome workers on ports %d-%d",
        n_workers,
        base_port,
        base_port + n_workers - 1,
    )
    for i in range(n_workers):
        ws, proc, user_data_dir = await _launch_chrome(chrome_path, base_port + i)
        conn = _Conn(ws, proc, user_data_dir)
        connections.append(conn)

    for i, conn in enumerate(connections):
        await conn.cdp("Page.enable")
        await conn.cdp(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": tile_height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        ft = await conn.cdp("Page.getFrameTree")
        frame_ids.append(ft["result"]["frameTree"]["frame"]["id"])

    logger.info("All workers ready. Processing %d articles.", len(articles))
    t_start = time.monotonic()

    async def worker_task(wi: int):
        conn = connections[wi]
        main_fid = frame_ids[wi]

        while True:
            try:
                article = work_q.get_nowait()
            except asyncio.QueueEmpty:
                break

            art_path = article["path"]
            raw_file_val = article.get("file", "")
            target_url = (
                raw_file_val
                if raw_file_val.startswith("http")
                else f"file://{raw_file_val}"
            )

            # Make output tile dir: use path slug
            slug = art_path.replace("/", "_").replace(" ", "_")[:200] or "article"
            tile_dir = output_dir / f"{slug}.tiles"
            tile_dir.mkdir(parents=True, exist_ok=True)

            try:
                # --- NAV (outside semaphore — I/O bound) ---
                nav_event_fut = asyncio.ensure_future(
                    conn.wait_for_event(
                        "Page.frameStoppedLoading",
                        timeout=30.0,
                        filter_fn=lambda p: p.get("frameId") == main_fid,
                    )
                )
                try:
                    await conn.cdp("Page.navigate", {"url": target_url})
                except Exception as e:
                    nav_event_fut.cancel()
                    logger.warning("[w%d] nav failed for %s: %s", wi, art_path, e)
                    metrics["errors"] += 1
                    continue

                try:
                    await nav_event_fut
                except asyncio.TimeoutError:
                    logger.warning(
                        "[w%d] frameStoppedLoading timeout for %s", wi, art_path
                    )
                    metrics["errors"] += 1
                    continue

                # Wait for fonts + images, get page height
                try:
                    r = await conn.cdp(
                        "Runtime.evaluate",
                        {
                            "expression": _WAIT_FONTS_IMGS,
                            "awaitPromise": True,
                            "returnByValue": True,
                        },
                    )
                    page_h = r["result"]["result"]["value"]
                    if not page_h or page_h <= 0:
                        page_h = tile_height
                except Exception:
                    page_h = tile_height

                n_tiles = max(1, (page_h + tile_height - 1) // tile_height)
                n_written = 0
                tile_names = []

                for t in range(n_tiles):
                    clip_h = min(tile_height, page_h - t * tile_height)
                    if clip_h <= 28:
                        break

                    # Scroll + wait in-viewport images (outside semaphore)
                    if t > 0:
                        y = t * tile_height
                        try:
                            await conn.cdp(
                                "Runtime.evaluate",
                                {
                                    "expression": f"""new Promise(resolve => {{
                                    window.scrollTo(0, {y});
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
                                }})""",
                                    "awaitPromise": True,
                                },
                            )
                        except Exception:
                            pass

                    # Acquire semaphore → capture → release (fine-grained)
                    raw_path = str(raw_dir / f"w{wi}_{slug}_{t}.raw")
                    out_path = str(tile_dir / f"tile_{t:04d}.jpg")

                    await capture_sem.acquire()
                    try:
                        t0 = time.monotonic()
                        r = await conn.cdp(
                            "Page.captureScreenshot",
                            {
                                "fromSurface": True,
                                "optimizeForSpeed": True,
                                "rawFilePath": raw_path,
                                "clip": {
                                    "x": 0,
                                    "y": t * tile_height,
                                    "width": VIEWPORT_WIDTH,
                                    "height": clip_h,
                                    "scale": 1,
                                },
                            },
                        )
                        shot_ms = (time.monotonic() - t0) * 1000
                    except Exception as e:
                        logger.warning(
                            "[w%d] capture failed tile %d of %s: %s", wi, t, art_path, e
                        )
                        metrics["errors"] += 1
                        continue
                    finally:
                        capture_sem.release()

                    if "error" in r.get("result", {}):
                        logger.warning(
                            "[w%d] CDP error tile %d of %s: %s",
                            wi,
                            t,
                            art_path,
                            r["result"]["error"],
                        )
                        metrics["errors"] += 1
                        continue

                    metrics["total_capture_ms"] += shot_ms

                    # Enqueue compression (non-blocking — capture continues)
                    compress_inbox.put((raw_path, out_path, jpeg_quality))
                    n_written += 1
                    tile_names.append(f"tile_{t:04d}.jpg")

                # Write manifest
                manifest = {
                    "path": art_path,
                    "url": target_url,
                    "page_height": page_h,
                    "tiles": tile_names,
                    "complete": True,
                }
                with open(tile_dir / "tiles.json", "w") as f:
                    json.dump(manifest, f)

                metrics["total_tiles"] += n_written
                logger.info(
                    "[w%d] %s → %d tiles (%.0f ms capture)",
                    wi,
                    art_path,
                    n_written,
                    shot_ms if n_tiles == 1 else 0,
                )

            except Exception as e:
                logger.warning("[w%d] unexpected error for %s: %s", wi, art_path, e)
                metrics["errors"] += 1

    # Run all workers concurrently
    await asyncio.gather(*[worker_task(i) for i in range(n_workers)])

    capture_wall_s = time.monotonic() - t_start
    total = metrics["total_tiles"]
    capture_tps = total / capture_wall_s if capture_wall_s > 0 else 0.0
    logger.info(
        "Capture done: %d tiles in %.1fs (%.1f tiles/s)",
        total,
        capture_wall_s,
        capture_tps,
    )

    # Wait for compression — run in thread to avoid blocking asyncio event loop
    import threading

    # Signal compression thread to finish, teardown Chrome in parallel
    compress_inbox.put(None)

    for conn in connections:
        try:
            await conn.close()
        except Exception:
            pass

    # Wait for compression to finish (runs in its own thread, no asyncio)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, compress_done.wait, 120)

    wall_s = time.monotonic() - t_start
    tps = total / wall_s if wall_s > 0 else 0.0

    logger.info(
        "Done: %d tiles in %.1fs (%.1f tiles/s, capture=%.1f tiles/s)",
        total,
        wall_s,
        tps,
        capture_tps,
    )
    return {
        "total_tiles": total,
        "wall_s": wall_s,
        "capture_wall_s": capture_wall_s,
        "capture_tiles_per_s": capture_tps,
        "tiles_per_s": tps,
        "errors": metrics["errors"],
        "avg_capture_ms": (metrics["total_capture_ms"] / total if total > 0 else 0.0),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def render_articles(
    articles: list[dict],
    output_dir: str,
    chrome_path: str = None,
    n_workers: int = 48,
    tile_height: int = TILE_HEIGHT,
    jpeg_quality: int = 85,
    n_compressors: int = 4,
) -> dict:
    """Render articles to JPEG tiles with async compression.

    Capture (Chrome → /dev/shm raw BGRA) and compression (raw → JPEG on disk)
    are fully decoupled.  Chrome workers never wait for compression.

    Args:
        articles: List of dicts with keys ``path`` (article ID) and
                  ``file`` (URL, http:// or absolute filesystem path).
        output_dir: Directory for output tile subdirectories.
        chrome_path: Path to Chrome binary.  Auto-detected if None.
        n_workers: Number of parallel Chrome processes (default 48).
        tile_height: Max tile height in pixels (default 8192).
        jpeg_quality: JPEG quality 1–100 (default 85).
        n_compressors: ProcessPoolExecutor workers for compression (default 4).

    Returns:
        dict with keys:
            ``total_tiles``     – number of tiles written
            ``wall_s``          – total wall-clock time in seconds
            ``tiles_per_s``     – throughput
            ``errors``          – count of capture/nav errors
            ``avg_capture_ms``  – average per-tile capture time (ms)
    """
    if not articles:
        return {
            "total_tiles": 0,
            "wall_s": 0.0,
            "tiles_per_s": 0.0,
            "errors": 0,
            "avg_capture_ms": 0.0,
        }

    if chrome_path is None:
        from ..chrome import find_chrome

        chrome_path = find_chrome()

    actual_workers = min(n_workers, len(articles))

    return await _run_render(
        articles=articles,
        output_dir=Path(output_dir),
        chrome_path=chrome_path,
        n_workers=actual_workers,
        tile_height=tile_height,
        jpeg_quality=jpeg_quality,
        n_compressors=n_compressors,
    )
