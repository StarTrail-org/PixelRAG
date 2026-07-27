"""Parallel tile capture using directClip.

Scrolls through the page in viewport-sized chunks. Within each chunk,
fires parallel directClip requests for 1024px tiles. Combines scrolling
correctness with parallel readback speed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .connection import launch_playwright, launch_websocket

TILE_HEIGHT_SMALL = 1024  # small tiles for parallel capture
VIEWPORT_HEIGHT = 8192  # viewport size (rasterized area)
VIEWPORT_WIDTH = 875

WAIT_FONTS = """new Promise(resolve => {
    document.fonts.ready.then(() => {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const sh = document.documentElement.scrollHeight;
                const body = document.body;
                resolve(body ? Math.min(sh, Math.max(Math.ceil(body.getBoundingClientRect().bottom), 1)) : sh);
            });
        });
    });
})"""


@dataclass
class CDPDirectClipStrategy:
    """Parallel tile capture via directClip.

    For each viewport-sized chunk of the page:
    1. Scroll to chunk position, wait 2 rAF
    2. Fire N parallel directClip requests (1024px each)
    3. Collect all responses

    directClip reads from the already-rendered frame without modifying
    shared viewport/emulation state, so parallel requests are safe.
    """

    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    launcher: str = "websocket"
    headless_shell: bool = False
    tile_height: int = TILE_HEIGHT_SMALL

    _connections: list = None
    _base_port: int = 9300

    @property
    def name(self) -> str:
        hs = " HS" if self.headless_shell else ""
        return f"{self.n_workers}w {self.fmt} dc{self.tile_height}{hs}"

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
                    "height": VIEWPORT_HEIGHT,
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

        try:
            r = await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": WAIT_FONTS,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = VIEWPORT_HEIGHT

        if page_h <= 0:
            page_h = VIEWPORT_HEIGHT

        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms
        ac.page_height = page_h
        ac.n_tiles_expected = max(
            1, (page_h + self.tile_height - 1) // self.tile_height
        )

        # Process page in viewport-sized chunks
        viewport_y = 0
        global_tile_idx = 0

        while viewport_y < page_h:
            # Scroll to this viewport chunk
            if viewport_y > 0:
                await conn.cdp(
                    "Runtime.evaluate",
                    {"expression": f"window.scrollTo(0, {viewport_y})"},
                )
                await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))",
                        "awaitPromise": True,
                    },
                )

            # Calculate tiles within this viewport chunk
            chunk_end = min(viewport_y + VIEWPORT_HEIGHT, page_h)
            tile_y = viewport_y
            tiles_in_chunk = []

            while tile_y < chunk_end:
                clip_h = min(self.tile_height, chunk_end - tile_y)
                if clip_h <= 28:
                    break
                tiles_in_chunk.append((global_tile_idx, tile_y, clip_h))
                tile_y += self.tile_height
                global_tile_idx += 1

            if not tiles_in_chunk:
                break

            # Fire all directClip requests in parallel
            if hasattr(conn, "_ws"):
                # Websocket: true parallel
                t0 = time.monotonic()
                pending = []
                for tile_idx, ty, ch in tiles_in_chunk:
                    conn._msg_id += 1
                    mid = conn._msg_id
                    params = {
                        "directClip": True,
                        "optimizeForSpeed": True,
                        "clip": {
                            "x": 0,
                            "y": ty,
                            "width": VIEWPORT_WIDTH,
                            "height": ch,
                            "scale": 1,
                        },
                    }
                    if self.fmt == "raw":
                        params["rawFilePath"] = (
                            f"/dev/shm/pixelrag_bench/w{wi}_{tile_idx}.raw"
                        )
                    else:
                        params["format"] = self.fmt
                        if self.fmt == "jpeg":
                            params["quality"] = self.quality

                    await conn._ws.send(
                        json.dumps(
                            {
                                "id": mid,
                                "method": "Page.captureScreenshot",
                                "params": params,
                            }
                        )
                    )
                    pending.append((mid, tile_idx, ty, ch))

                # Collect responses
                mid_to_info = {mid: (ti, ty, ch) for mid, ti, ty, ch in pending}
                collected = {}
                while len(collected) < len(pending):
                    try:
                        r = json.loads(
                            await asyncio.wait_for(conn._ws.recv(), timeout=180)
                        )
                        rid = r.get("id")
                        if rid in mid_to_info:
                            collected[rid] = r
                    except Exception as e:
                        ac.errors.append(f"recv: {e}")
                        break

                shot_ms = (time.monotonic() - t0) * 1000
                ac.total_shot_ms += shot_ms

                # Store tiles in order
                for mid, tile_idx, ty, ch in pending:
                    r = collected.get(mid)
                    if not r or "error" in r:
                        ac.errors.append(
                            f"tile {tile_idx}: {r.get('error') if r else 'no response'}"
                        )
                        continue

                    tc = TileCapture(
                        shot_ms=shot_ms / len(pending),
                        nav_ms=nav_ms if tile_idx == 0 else 0.0,
                        tile_index=tile_idx,
                        clip_y=ty,
                        clip_h=ch,
                    )
                    if self.fmt == "raw":
                        tc.raw_file_path = (
                            f"/dev/shm/pixelrag_bench/w{wi}_{tile_idx}.raw"
                        )
                    else:
                        tc.image_bytes = base64.b64decode(r["result"]["data"])
                    ac.tiles.append(tc)

            else:
                # Playwright: sequential fallback
                for tile_idx, ty, ch in tiles_in_chunk:
                    params = {
                        "directClip": True,
                        "optimizeForSpeed": True,
                        "clip": {
                            "x": 0,
                            "y": ty,
                            "width": VIEWPORT_WIDTH,
                            "height": ch,
                            "scale": 1,
                        },
                    }
                    if self.fmt == "raw":
                        params["rawFilePath"] = (
                            f"/dev/shm/pixelrag_bench/w{wi}_{tile_idx}.raw"
                        )
                    else:
                        params["format"] = self.fmt
                        if self.fmt == "jpeg":
                            params["quality"] = self.quality

                    t0 = time.monotonic()
                    try:
                        r = await conn.cdp("Page.captureScreenshot", params)
                    except Exception as e:
                        ac.errors.append(f"tile {tile_idx}: {e}")
                        continue
                    shot_ms = (time.monotonic() - t0) * 1000
                    ac.total_shot_ms += shot_ms

                    if "error" not in r:
                        tc = TileCapture(
                            shot_ms=shot_ms,
                            nav_ms=nav_ms if tile_idx == 0 else 0.0,
                            tile_index=tile_idx,
                            clip_y=ty,
                            clip_h=ch,
                        )
                        if self.fmt == "raw":
                            tc.raw_file_path = (
                                f"/dev/shm/pixelrag_bench/w{wi}_{tile_idx}.raw"
                            )
                        else:
                            tc.image_bytes = base64.b64decode(r["result"]["data"])
                        ac.tiles.append(tc)

            viewport_y += VIEWPORT_HEIGHT

        return ac
