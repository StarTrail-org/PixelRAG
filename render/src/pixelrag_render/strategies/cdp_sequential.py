"""Sequential tile capture strategy.

Supports both websocket and Playwright connections.
Tiles within a page are captured one at a time (sequential).
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .connection import (
    launch_playwright,
    launch_websocket,
)

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875


@dataclass
class CDPSequentialStrategy:
    """Sequential tile capture. One screenshot at a time per page.

    Args:
        chrome_path: Chrome binary path.
        n_workers: Number of Chrome processes.
        fmt: 'png', 'jpeg', or 'raw' (rawFilePath to /dev/shm).
        quality: JPEG quality (ignored for png/raw).
        from_surface: CDP fromSurface parameter.
        launcher: 'websocket' (default, faster) or 'playwright'.
        headless_shell: True if chrome_path is headless_shell (no --headless flag).
    """

    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    launcher: str = "websocket"
    headless_shell: bool = False
    extra_args: list = None
    label: str = ""

    _connections: list = None
    _base_port: int = 9300

    @property
    def name(self) -> str:
        l = "pw" if self.launcher == "playwright" else "ws"
        surface = "" if self.from_surface else " !surface"
        hs = " HS" if self.headless_shell else ""
        tag = f" [{self.label}]" if self.label else ""
        return f"{self.n_workers}w {self.fmt} {l}{hs}{surface}{tag}"

    async def setup(self) -> None:
        self._connections = []
        if self.launcher == "playwright":
            for _ in range(self.n_workers):
                conn = await launch_playwright(self.chrome_path)
                self._connections.append(conn)
        else:
            for i in range(self.n_workers):
                port = self._base_port + i
                conn = await launch_websocket(
                    self.chrome_path,
                    port,
                    headless_shell=self.headless_shell,
                    extra_args=self.extra_args,
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
            self._connections = None

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
                idx = article_index[article["path"]]
                all_results[idx] = ac

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

        # Wait for fonts + eager images (with timeout) + layout, return scrollHeight
        try:
            r = await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": """new Promise(resolve => {
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
                })""",
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = TILE_HEIGHT

        if page_h <= 0:
            page_h = TILE_HEIGHT

        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms

        ac.page_height = page_h
        n_tiles = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)
        ac.n_tiles_expected = n_tiles

        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break

            # Scroll to tile position + wait for lazy images in viewport (with timeout)
            if t > 0:
                y = t * TILE_HEIGHT
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
                clip_y=t * TILE_HEIGHT,
                clip_h=clip_h,
            )

            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])

            ac.tiles.append(tc)

        return ac
