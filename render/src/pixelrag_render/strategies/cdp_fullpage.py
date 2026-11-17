"""Full-page single screenshot + Python crop strategy.

Measures content height at a small viewport (avoids 100vh inflation),
then takes one large screenshot and crops into tiles in Python.
Reduces N CDP calls to 1 per article (for pages <= 16384px).
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass

from PIL import Image

from .base import ArticleCapture, TileCapture, article_url
from .connection import launch_playwright, launch_websocket

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875
MEASURE_HEIGHT = 1080  # small viewport for measuring true content height
MAX_CHROME_HEIGHT = 16384


@dataclass
class CDPFullpageStrategy:
    """Single full-page screenshot, crop tiles in Python.

    1. Set viewport to 1080px, navigate, measure scrollHeight (true height)
    2. Inject CSS to lock viewport-relative units
    3. Resize viewport to content height (capped at 16384)
    4. Single captureScreenshot
    5. Crop into 8192px tiles in Python
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
        return f"{self.n_workers}w {self.fmt} full{hs}"

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

        # Step 1: small viewport, navigate, measure true height
        await conn.cdp(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": MEASURE_HEIGHT,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

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
                {
                    "expression": """(() => {
                    const sh = document.documentElement.scrollHeight;
                    const body = document.body;
                    if (body) {
                        const bottom = Math.ceil(body.getBoundingClientRect().bottom);
                        return Math.min(sh, Math.max(bottom, 1));
                    }
                    return sh;
                })()"""
                },
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = MEASURE_HEIGHT

        ac.page_height = page_h
        ac.n_tiles_expected = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)

        # Step 2: lock viewport-relative units before resize
        await conn.cdp(
            "Runtime.evaluate",
            {
                "expression": """(() => {
                const style = document.createElement('style');
                style.textContent = 'html, body { height: auto !important; min-height: 0 !important; }';
                document.head.appendChild(style);
            })()"""
            },
        )

        # Step 3: resize viewport to content height for capture
        capture_h = min(page_h, MAX_CHROME_HEIGHT)
        await conn.cdp(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": capture_h,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )

        await conn.cdp(
            "Runtime.evaluate",
            {
                "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
                "awaitPromise": True,
            },
        )

        # Step 4: single screenshot
        params = {
            "fromSurface": self.from_surface,
            "optimizeForSpeed": True,
            "clip": {
                "x": 0,
                "y": 0,
                "width": VIEWPORT_WIDTH,
                "height": capture_h,
                "scale": 1,
            },
            "format": "png",
        }

        t0 = time.monotonic()
        try:
            r = await conn.cdp("Page.captureScreenshot", params)
        except Exception as e:
            ac.errors.append(f"fullpage: {e}")
            return ac
        shot_ms = (time.monotonic() - t0) * 1000
        ac.total_shot_ms = shot_ms

        if "error" in r:
            ac.errors.append(f"fullpage: {r['error']}")
            return ac

        # Step 5: crop into tiles (NOT timed)
        full_bytes = base64.b64decode(r["result"]["data"])
        full_img = Image.open(io.BytesIO(full_bytes))
        w, h = full_img.size

        y = 0
        tile_idx = 0
        while y < h:
            tile_h = min(TILE_HEIGHT, h - y)
            if tile_h <= 28:
                break

            tile_img = full_img.crop((0, y, w, y + tile_h))
            buf = io.BytesIO()
            if self.fmt == "jpeg":
                tile_img.convert("RGB").save(buf, "JPEG", quality=self.quality)
            else:
                tile_img.save(buf, "PNG")

            tc = TileCapture(
                image_bytes=buf.getvalue(),
                shot_ms=shot_ms / ac.n_tiles_expected,
                nav_ms=ac.total_nav_ms if tile_idx == 0 else 0.0,
                tile_index=tile_idx,
                clip_y=y,
                clip_h=tile_h,
            )
            ac.tiles.append(tc)
            y += TILE_HEIGHT
            tile_idx += 1

        return ac
