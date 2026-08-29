"""No-scroll capture strategy.

Uses captureBeyondViewport=True with clip.y to capture different page regions
without scrolling. Saves the scroll + rAF wait overhead per tile.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .connection import launch_playwright, launch_websocket

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875


@dataclass
class CDPNoScrollStrategy:
    """Capture tiles using clip.y offset without scrolling.

    Sets viewport once, then captures each tile region via clip parameter
    with captureBeyondViewport=True. No scrollTo, no rAF waits.
    """

    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    launcher: str = "websocket"
    headless_shell: bool = False

    _connections: list = None
    _base_port: int = 9300

    @property
    def name(self) -> str:
        hs = " HS" if self.headless_shell else ""
        return f"{self.n_workers}w {self.fmt} noscr{hs}"

    async def setup(self) -> None:
        self._connections = []
        if self.launcher == "playwright":
            for _ in range(self.n_workers):
                conn = await launch_playwright(self.chrome_path)
                self._connections.append(conn)
        else:
            for i in range(self.n_workers):
                conn = await launch_websocket(
                    self.chrome_path,
                    self._base_port + i,
                    headless_shell=self.headless_shell,
                )
                self._connections.append(conn)

        for conn in self._connections:
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

        if self.fmt == "raw":
            os.makedirs("/dev/shm/pixelrag_bench", exist_ok=True)

    async def teardown(self) -> None:
        if self._connections:
            for conn in self._connections:
                await conn.close()

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]:
        n = len(self._connections)
        wp = [[] for _ in range(n)]
        article_index = {a["path"]: i for i, a in enumerate(articles)}
        for i, a in enumerate(articles):
            wp[i % n].append(a)

        all_results = [None] * len(articles)

        async def worker_task(wi):
            for article in wp[wi]:
                ac = await self._capture_one(wi, article)
                all_results[article_index[article["path"]]] = ac

        await asyncio.gather(
            *[worker_task(i) for i in range(n)], return_exceptions=True
        )
        return [r for r in all_results if r is not None]

    async def _capture_one(self, wi: int, article: dict) -> ArticleCapture:
        conn = self._connections[wi]
        ac = ArticleCapture(article_path=article["path"])

        t_nav = time.monotonic()
        try:
            await conn.cdp("Page.navigate", {"url": article_url(article)})
        except Exception as e:
            ac.errors.append(f"nav: {e}")
            return ac
        await asyncio.sleep(0.03)
        ac.total_nav_ms = (time.monotonic() - t_nav) * 1000

        try:
            r = await conn.cdp(
                "Runtime.evaluate",
                {"expression": "document.documentElement.scrollHeight"},
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = TILE_HEIGHT

        ac.page_height = page_h
        n_tiles = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)
        ac.n_tiles_expected = n_tiles

        # Pre-scroll warmup: force compositor to rasterize entire page
        if n_tiles > 1:
            for scroll_y in range(0, page_h, 1080):
                await conn.cdp(
                    "Runtime.evaluate",
                    {"expression": f"window.scrollTo(0, {scroll_y})"},
                )
            await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
                    "awaitPromise": True,
                },
            )
            await conn.cdp("Runtime.evaluate", {"expression": "window.scrollTo(0, 0)"})

        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break

            # No scrollTo per tile — just clip with captureBeyondViewport
            params = {
                "fromSurface": self.from_surface,
                "captureBeyondViewport": True,
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
                nav_ms=ac.total_nav_ms if t == 0 else 0.0,
                tile_index=t,
                clip_y=t * TILE_HEIGHT,
                clip_h=clip_h,
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)

        return ac
