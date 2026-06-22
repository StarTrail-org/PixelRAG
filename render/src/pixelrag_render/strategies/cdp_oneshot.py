"""One-shot capture strategy.

Each article gets a FRESH headless_shell process: launch → navigate → capture → kill.
No stale state, no IPC race conditions across articles.  The per-process launch
overhead (~0.3–1 s) is hidden by running many processes concurrently.

Key difference from cdp_phased / cdp_sequential:
- Those strategies keep N Chrome processes alive and navigate them repeatedly.
  ForceRedraw IPC latency grows with concurrency because the OS must schedule
  144+ threads (48 Chrome workers × 3 threads each).
- One-shot processes live only for one article.  Each ForceRedraw runs in an
  otherwise idle process, so IPC overhead stays low regardless of concurrency.

Port allocation: each slot in [base_port, base_port+n_workers) is used by at
most one live process.  A semaphore gates concurrent slots so we never try to
bind two processes to the same port.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

from .base import article_url, ArticleCapture, TileCapture
from .connection import CHROME_ARGS, pick_page_ws_url

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875

# How often to poll the /json endpoint while waiting for Chrome to start.
# 100 ms gives ~0.2–0.5 s connect time vs. the 1 s sleep in launch_websocket.
_POLL_INTERVAL = 0.1
_POLL_ATTEMPTS = 60  # 60 × 100 ms = 6 s max before giving up

WAIT_FONTS_IMGS = """new Promise(resolve => {
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
                resolve(body ? Math.min(sh, Math.max(Math.ceil(body.getBoundingClientRect().bottom), 1)) : sh);
            });
        });
    });
})"""


async def _launch_oneshot(
    chrome_path: str,
    port: int,
    headless_shell: bool,
    extra_args: list[str] | None,
):
    """Launch a fresh Chrome/headless_shell process and connect via websocket.

    Returns (WebsocketConnection, proc).  Polls every _POLL_INTERVAL seconds
    (much faster than the 1 s sleep in launch_websocket).
    """
    import websockets
    from .connection import WebsocketConnection

    args = [chrome_path, f"--remote-debugging-port={port}"]
    if not headless_shell:
        args.append("--headless=new")
    args += CHROME_ARGS
    if extra_args:
        args += extra_args
    args += ["about:blank"]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    last_exc: Exception | None = None
    for _ in range(_POLL_ATTEMPTS):
        await asyncio.sleep(_POLL_INTERVAL)
        try:
            data = urllib.request.urlopen(
                f"http://localhost:{port}/json", timeout=2
            ).read()
            targets = json.loads(data)
            ws = await websockets.connect(
                pick_page_ws_url(targets),
                open_timeout=5,
                max_size=50 * 1024 * 1024,
            )
            return WebsocketConnection(ws, proc), proc
        except Exception as e:
            last_exc = e

    proc.kill()
    raise ConnectionError(
        f"Failed to connect to Chrome on port {port} after "
        f"{_POLL_ATTEMPTS * _POLL_INTERVAL:.1f}s: {last_exc}"
    )


def _kill_proc(proc) -> None:
    """Best-effort kill + wait for a subprocess."""
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@dataclass
class CDPOneShotStrategy:
    """One-shot: fresh headless_shell process per article.

    n_workers controls how many articles are captured concurrently (each in its
    own short-lived Chrome process).  The semaphore gates port slot reuse so
    that slot i is never used by two live processes simultaneously.
    """

    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    headless_shell: bool = True
    nav_timeout_ms: int = 2000
    tile_height: int = TILE_HEIGHT
    extra_chrome_args: list = None

    _base_port: int = 9500  # separate range from persistent-pool strategies
    _sem: asyncio.Semaphore = field(default=None, init=False, repr=False)

    # Track (port_slot → proc) so teardown can clean up stragglers.
    _live_procs: dict = field(default_factory=dict, init=False, repr=False)
    _procs_lock: asyncio.Lock = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        hs = " HS" if self.headless_shell else ""
        th = f" h{self.tile_height}" if self.tile_height != TILE_HEIGHT else ""
        return f"{self.n_workers}w {self.fmt} oneshot{hs}{th}"

    async def setup(self) -> None:
        self._sem = asyncio.Semaphore(self.n_workers)
        self._live_procs = {}
        self._procs_lock = asyncio.Lock()
        if self.fmt == "raw":
            os.makedirs("/dev/shm/pixelrag_bench", exist_ok=True)

    async def teardown(self) -> None:
        # Kill any processes that leaked (e.g. due to exceptions).
        async with self._procs_lock:
            procs = list(self._live_procs.values())
            self._live_procs.clear()
        for proc in procs:
            _kill_proc(proc)

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]:
        article_index = {a["path"]: i for i, a in enumerate(articles)}
        all_results: list[ArticleCapture | None] = [None] * len(articles)

        # Assign each article a port slot: slot_index = article_index % n_workers.
        # The semaphore ensures at most n_workers concurrent processes, and each
        # slot is only used by one process at a time.
        async def capture_task(article: dict):
            idx = article_index[article["path"]]
            slot = idx % self.n_workers
            port = self._base_port + slot
            ac = await self._capture_one(article, slot, port)
            all_results[idx] = ac

        await asyncio.gather(*[capture_task(a) for a in articles])
        return [r for r in all_results if r is not None]

    async def _capture_one(self, article: dict, slot: int, port: int) -> ArticleCapture:
        ac = ArticleCapture(article_path=article["path"])
        th = self.tile_height

        # Wait for this port slot to be free.
        t_sem = time.monotonic()
        async with self._sem:
            ac.sem_wait_ms = (time.monotonic() - t_sem) * 1000

            # Launch a fresh process on this port slot.
            try:
                conn, proc = await _launch_oneshot(
                    self.chrome_path,
                    port,
                    headless_shell=self.headless_shell,
                    extra_args=self.extra_chrome_args,
                )
            except Exception as e:
                ac.errors.append(f"launch: {e}")
                return ac

            async with self._procs_lock:
                self._live_procs[slot] = proc

            try:
                await self._capture_article(conn, article, ac, th)
            finally:
                # Close websocket and kill process regardless of outcome.
                try:
                    await conn.close()
                except Exception:
                    _kill_proc(proc)
                async with self._procs_lock:
                    self._live_procs.pop(slot, None)

        return ac

    async def _capture_article(
        self, conn, article: dict, ac: ArticleCapture, th: int
    ) -> None:
        """Navigate, wait for render, capture all tiles.  Mutates ac in place."""
        # === Configure viewport ===
        try:
            await conn.cdp("Page.enable")
            await conn.cdp(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": VIEWPORT_WIDTH,
                    "height": th,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
        except Exception as e:
            ac.errors.append(f"setup cdp: {e}")
            return

        # === Navigate ===
        t_nav = time.monotonic()
        target_url = article_url(article)

        # Use Page.frameStoppedLoading: reliable with --in-process-gpu
        # (Page.frameNavigated has a Chrome bug where it's sometimes not fired).
        nav_event_fut = asyncio.ensure_future(
            conn.wait_for_event("Page.frameStoppedLoading", timeout=30.0)
        )
        try:
            await conn.cdp("Page.navigate", {"url": target_url})
        except Exception as e:
            nav_event_fut.cancel()
            ac.errors.append(f"nav: {e}")
            return

        try:
            await nav_event_fut
        except asyncio.TimeoutError:
            ac.errors.append("nav: frameStoppedLoading timeout (30s)")
            return
        except Exception as e:
            ac.errors.append(f"nav: frameStoppedLoading wait error: {e}")
            return

        # === Wait for fonts + images, measure page height ===
        try:
            wait_expr = WAIT_FONTS_IMGS.replace(
                "setTimeout(r, 2000)", f"setTimeout(r, {self.nav_timeout_ms})"
            )
            r = await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": wait_expr,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = th

        if page_h <= 0:
            page_h = th

        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms
        ac.page_height = page_h
        n_tiles = max(1, (page_h + th - 1) // th)
        ac.n_tiles_expected = n_tiles

        # === Capture tiles ===
        for t in range(n_tiles):
            clip_y = t * th
            clip_h = min(th, page_h - clip_y)
            if clip_h <= 28:
                break

            # Scroll into position and wait for viewport images.
            try:
                await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": f"""new Promise(resolve => {{
                        window.scrollTo(0, {clip_y});
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

            params = {
                "fromSurface": self.from_surface,
                "optimizeForSpeed": True,
                "clip": {
                    "x": 0,
                    "y": clip_y,
                    "width": VIEWPORT_WIDTH,
                    "height": clip_h,
                    "scale": 1,
                },
            }

            raw_path = None
            if self.fmt == "raw":
                raw_path = (
                    f"/dev/shm/pixelrag_bench"
                    f"/os_{article['path'].replace('/', '_')}_{t}.raw"
                )
                params["rawFilePath"] = raw_path
            else:
                params["format"] = self.fmt
                if self.fmt == "jpeg":
                    params["quality"] = self.quality

            t0 = time.monotonic()
            try:
                r = await conn.cdp("Page.captureScreenshot", params)
            except Exception as e:
                ac.errors.append(f"tile {t}: {e}")
                continue
            shot_ms = (time.monotonic() - t0) * 1000
            ac.total_shot_ms += shot_ms

            if "error" in r:
                ac.errors.append(f"tile {t}: {r['error']}")
                continue

            tc = TileCapture(
                shot_ms=shot_ms,
                nav_ms=nav_ms if t == 0 else 0.0,
                tile_index=t,
                clip_y=clip_y,
                clip_h=clip_h,
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)
