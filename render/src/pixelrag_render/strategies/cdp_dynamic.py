"""Dynamic capture strategy — no upfront height measurement.

Scrolls and captures tiles until scroll stops moving. Eliminates the
fonts.ready + scrollHeight measurement overhead (~170ms on maxi ZIM).
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
MAX_TILES = 20  # safety limit


@dataclass
class CDPDynamicStrategy:
    """Capture tiles dynamically — no upfront height measurement.

    1. Navigate, wait for load event only (no fonts.ready)
    2. Capture tile at y=0 immediately
    3. Scroll to next tile, wait 2 rAF, capture
    4. Stop when scroll doesn't move or blank tile detected
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
        return f"{self.n_workers}w {self.fmt} dyn{hs}"

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

        # Minimal wait: just 2 rAF for first paint (no fonts.ready, no height measurement)
        try:
            await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
                    "awaitPromise": True,
                },
            )
        except Exception:
            await asyncio.sleep(0.05)

        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms

        # Dynamic capture: scroll + capture until scroll stops
        prev_scroll_y = -1
        tile_idx = 0

        while tile_idx < MAX_TILES:
            y = tile_idx * TILE_HEIGHT

            # Scroll to tile position
            if tile_idx > 0:
                await conn.cdp(
                    "Runtime.evaluate", {"expression": f"window.scrollTo(0, {y})"}
                )
                await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
                        "awaitPromise": True,
                    },
                )

                # Check if scroll moved
                try:
                    r = await conn.cdp(
                        "Runtime.evaluate", {"expression": "window.scrollY"}
                    )
                    actual_y = r["result"]["result"]["value"]
                except Exception:
                    break

                if int(actual_y) == prev_scroll_y:
                    break  # reached bottom
                prev_scroll_y = int(actual_y)

            # Get current page height for clip (might still be changing, that's ok)
            try:
                r = await conn.cdp(
                    "Runtime.evaluate",
                    {"expression": "document.documentElement.scrollHeight"},
                )
                page_h = r["result"]["result"]["value"]
            except Exception:
                page_h = (tile_idx + 1) * TILE_HEIGHT

            clip_h = min(TILE_HEIGHT, page_h - y)
            if clip_h <= 28:
                break

            ac.page_height = page_h

            params = {
                "fromSurface": self.from_surface,
                "optimizeForSpeed": True,
                "clip": {
                    "x": 0,
                    "y": y,
                    "width": VIEWPORT_WIDTH,
                    "height": clip_h,
                    "scale": 1,
                },
            }

            raw_path = None
            if self.fmt == "raw":
                raw_path = f"/dev/shm/pixelrag_bench/w{wi}_{id(article)}_{tile_idx}.raw"
                params["rawFilePath"] = raw_path
            else:
                params["format"] = self.fmt
                if self.fmt == "jpeg":
                    params["quality"] = self.quality

            t0 = time.monotonic()
            try:
                r = await conn.cdp("Page.captureScreenshot", params)
            except Exception as e:
                ac.errors.append(f"tile {tile_idx}: {e}")
                break
            shot_ms = (time.monotonic() - t0) * 1000
            ac.total_shot_ms += shot_ms

            if "error" in r:
                ac.errors.append(f"tile {tile_idx}: {r['error']}")
                break

            tc = TileCapture(
                shot_ms=shot_ms,
                nav_ms=nav_ms if tile_idx == 0 else 0.0,
                tile_index=tile_idx,
                clip_y=y,
                clip_h=clip_h,
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)

            tile_idx += 1

        ac.n_tiles_expected = len(ac.tiles)
        return ac
