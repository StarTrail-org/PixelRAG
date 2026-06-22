"""Two-tab pipelined strategy: overlap nav and capture across articles.

Each Chrome process has 2 tabs. While Tab A captures tiles, Tab B loads
the next article. When Tab A finishes, Tab B is ready to capture.

This overlaps I/O-bound work (nav + fonts + images) with memory-BW-bound
work (readback), using different resources simultaneously within one process.

Effective per-article time: max(nav, capture) instead of nav + capture.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .connection import WebsocketConnection, pick_page_ws_url

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875

WAIT_ALL = """new Promise(resolve => {
    const waitImgs = Promise.all(
        Array.from(document.images)
            .filter(i => !i.complete)
            .map(i => new Promise(r => {
                i.addEventListener('load', r, {once: true});
                i.addEventListener('error', r, {once: true});
            }))
    );
    Promise.all([document.fonts.ready, waitImgs]).then(() => {
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


@dataclass
class CDPPipelinedTabsStrategy:
    """Two-tab ping-pong: overlap nav (I/O) with capture (mem BW).

    Each Chrome process has 2 pages. While one captures, the other loads.
    """

    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    headless_shell: bool = False

    _procs: list = None
    _tab_pairs: list = None  # [(conn_a, conn_b), ...]
    _base_port: int = 9300

    @property
    def name(self) -> str:
        hs = " HS" if self.headless_shell else ""
        return f"{self.n_workers}w {self.fmt} pipe2t{hs}"

    async def setup(self) -> None:
        import subprocess

        self._procs = []
        self._tab_pairs = []

        for i in range(self.n_workers):
            port = self._base_port + i
            args = [
                self.chrome_path,
                f"--remote-debugging-port={port}",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
            if not self.headless_shell:
                args.append("--headless=new")
            args.append("about:blank")
            proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._procs.append(proc)

        await asyncio.sleep(max(5, self.n_workers // 4))

        import urllib.request

        for i in range(self.n_workers):
            port = self._base_port + i
            try:
                # Connect to default page (Tab A)
                data = urllib.request.urlopen(f"http://localhost:{port}/json").read()
                targets = json.loads(data)
                ws_a = await __import__("websockets").connect(
                    pick_page_ws_url(targets),
                    open_timeout=10,
                    max_size=50 * 1024 * 1024,
                )
                conn_a = WebsocketConnection(ws_a, self._procs[i])
                conn_a._msg_id = i * 200000

                # Create Tab B via Target.createTarget on browser ws
                browser_data = urllib.request.urlopen(
                    f"http://localhost:{port}/json/version"
                ).read()
                browser_ws_url = json.loads(browser_data)["webSocketDebuggerUrl"]
                browser_ws = await __import__("websockets").connect(
                    browser_ws_url, open_timeout=10, max_size=50 * 1024 * 1024
                )

                # Create new target
                await browser_ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Target.createTarget",
                            "params": {"url": "about:blank"},
                        }
                    )
                )
                r = json.loads(await asyncio.wait_for(browser_ws.recv(), timeout=10))
                target_id = r["result"]["targetId"]

                # Get the new page's ws url
                await asyncio.sleep(0.5)
                data2 = urllib.request.urlopen(f"http://localhost:{port}/json").read()
                targets2 = json.loads(data2)
                ws_b_url = None
                for t in targets2:
                    if t["id"] == target_id:
                        ws_b_url = t["webSocketDebuggerUrl"]
                        break
                if not ws_b_url:
                    ws_b_url = targets2[-1]["webSocketDebuggerUrl"]

                ws_b = await __import__("websockets").connect(
                    ws_b_url, open_timeout=10, max_size=50 * 1024 * 1024
                )
                conn_b = WebsocketConnection(ws_b, self._procs[i])
                conn_b._msg_id = i * 200000 + 100000

                await browser_ws.close()

                # Setup both tabs
                for conn in [conn_a, conn_b]:
                    await conn.cdp("Page.enable")
                    await conn.cdp(
                        "Emulation.setDeviceMetricsOverride",
                        {
                            "width": VIEWPORT_WIDTH,
                            "height": TILE_HEIGHT,
                            "deviceScaleFactor": 1,
                            "mobile": False,
                        },
                    )

                self._tab_pairs.append((conn_a, conn_b))
            except Exception:
                self._tab_pairs.append(None)

        if self.fmt == "raw":
            os.makedirs("/dev/shm/pixelrag_bench", exist_ok=True)

    async def teardown(self) -> None:
        if self._tab_pairs:
            for pair in self._tab_pairs:
                if pair:
                    for conn in pair:
                        try:
                            await conn._ws.close()
                        except Exception:
                            pass
        if self._procs:
            import signal

            for p in self._procs:
                p.send_signal(signal.SIGTERM)
            await asyncio.sleep(2)
            for p in self._procs:
                try:
                    p.kill()
                except OSError:
                    pass

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]:
        n = len([p for p in self._tab_pairs if p])
        wp = [[] for _ in range(n)]
        article_index = {a["path"]: i for i, a in enumerate(articles)}
        valid_pairs = [p for p in self._tab_pairs if p]
        for i, a in enumerate(articles):
            wp[i % n].append(a)

        all_results = [None] * len(articles)

        async def worker_task(wi):
            conn_a, conn_b = valid_pairs[wi]
            my_articles = wp[wi]
            if not my_articles:
                return

            # Start loading first article on Tab A
            nav_task = asyncio.create_task(self._nav_and_wait(conn_a, my_articles[0]))

            for idx in range(len(my_articles)):
                # Wait for current article to finish loading
                page_h = await nav_task

                current_conn = conn_a if idx % 2 == 0 else conn_b
                next_conn = conn_b if idx % 2 == 0 else conn_a

                # Start loading NEXT article on the OTHER tab (overlapped with capture)
                if idx + 1 < len(my_articles):
                    nav_task = asyncio.create_task(
                        self._nav_and_wait(next_conn, my_articles[idx + 1])
                    )

                # Capture current article tiles
                ac = await self._capture_tiles(
                    current_conn, wi, my_articles[idx], page_h
                )
                all_results[article_index[my_articles[idx]["path"]]] = ac

        await asyncio.gather(
            *[worker_task(i) for i in range(n)], return_exceptions=True
        )
        return [r for r in all_results if r is not None]

    async def _nav_and_wait(self, conn, article: dict) -> int:
        """Phase 1: Navigate + wait for all resources. Returns page_height."""
        await conn.cdp("Page.navigate", {"url": article_url(article)})
        try:
            r = await conn.cdp(
                "Runtime.evaluate",
                {"expression": WAIT_ALL, "awaitPromise": True, "returnByValue": True},
            )
            return r["result"]["result"]["value"] or TILE_HEIGHT
        except Exception:
            return TILE_HEIGHT

    async def _capture_tiles(
        self, conn, wi: int, article: dict, page_h: int
    ) -> ArticleCapture:
        """Phase 2: Pure capture, no waiting."""
        ac = ArticleCapture(article_path=article["path"])
        ac.page_height = page_h
        n_tiles = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)
        ac.n_tiles_expected = n_tiles

        t0 = time.monotonic()
        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break

            if t > 0:
                await conn.cdp(
                    "Runtime.evaluate",
                    {"expression": f"window.scrollTo(0, {t * TILE_HEIGHT})"},
                )
                await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))",
                        "awaitPromise": True,
                    },
                )

            params = {
                "fromSurface": self.from_surface,
                "optimizeForSpeed": True,
                "clip": {
                    "x": 0,
                    "y": t * TILE_HEIGHT,
                    "width": VIEWPORT_WIDTH,
                    "height": clip_h,
                    "scale": 1,
                },
            }
            raw_path = None
            if self.fmt == "raw":
                raw_path = f"/dev/shm/pixelrag_bench/w{wi}_{id(article)}_{t}.raw"
                params["rawFilePath"] = raw_path
            else:
                params["format"] = self.fmt
                if self.fmt == "jpeg":
                    params["quality"] = self.quality

            try:
                r = await conn.cdp("Page.captureScreenshot", params)
            except Exception as e:
                ac.errors.append(f"tile {t}: {e}")
                continue

            shot_ms = (time.monotonic() - t0) * 1000 / (t + 1)

            if "error" in r:
                ac.errors.append(f"tile {t}: {r['error']}")
                continue

            tc = TileCapture(
                shot_ms=shot_ms,
                tile_index=t,
                clip_y=t * TILE_HEIGHT,
                clip_h=clip_h,
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)

        ac.total_shot_ms = (time.monotonic() - t0) * 1000
        return ac
