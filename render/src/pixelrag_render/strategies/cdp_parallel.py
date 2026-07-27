"""Parallel tile capture strategy.

Fires all tile screenshot requests at once per page, collects all responses.
May produce incorrect images due to Chrome's shared viewport state — the
benchmark framework will verify correctness automatically.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .cdp_sequential import TILE_HEIGHT, VIEWPORT_WIDTH
from .connection import launch_playwright, launch_websocket


@dataclass
class CDPParallelStrategy:
    """Fire all tile screenshots simultaneously per page.

    Same setup as CDPSequentialStrategy, but within one page, all tiles
    are sent at once without waiting for each response.
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
        l = "pw" if self.launcher == "playwright" else "ws"
        surface = "" if self.from_surface else " !surface"
        hs = " HS" if self.headless_shell else ""
        return f"{self.n_workers}w {self.fmt} par-{l}{hs}{surface}"

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
                ac = await self._capture_one_parallel(wi, article)
                idx = article_index[article["path"]]
                all_results[idx] = ac

        await asyncio.gather(
            *[worker_task(i) for i in range(n)], return_exceptions=True
        )
        return [r for r in all_results if r is not None]

    async def _capture_one_parallel(self, wi: int, article: dict) -> ArticleCapture:
        conn = self._connections[wi]
        ac = ArticleCapture(article_path=article["path"])

        t_nav = time.monotonic()
        try:
            await conn.cdp("Page.navigate", {"url": article_url(article)})
        except Exception as e:
            ac.errors.append(f"nav: {e}")
            return ac
        await asyncio.sleep(0.03)
        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms

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

        if n_tiles <= 1:
            # Single tile — same as sequential
            return await self._capture_single(conn, wi, article, ac, page_h, nav_ms)

        # Fire all tile requests at once
        # Need raw websocket access for parallel sends
        if not hasattr(conn, "_ws"):
            # Playwright connection — fall back to sequential
            return await self._capture_sequential_fallback(
                conn, wi, article, ac, page_h, nav_ms
            )

        ws = conn._ws
        pending = []
        t0 = time.monotonic()

        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break

            conn._msg_id += 1
            mid = conn._msg_id

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

            await ws.send(
                json.dumps(
                    {
                        "id": mid,
                        "method": "Page.captureScreenshot",
                        "params": params,
                    }
                )
            )
            pending.append((mid, t, clip_h, raw_path))

        # Collect all responses
        mid_to_info = {mid: (t, clip_h, rp) for mid, t, clip_h, rp in pending}
        collected = {}
        while len(collected) < len(pending):
            try:
                r = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                rid = r.get("id")
                if rid in mid_to_info:
                    collected[rid] = r
            except Exception as e:
                ac.errors.append(f"recv: {e}")
                break

        total_shot_ms = (time.monotonic() - t0) * 1000
        ac.total_shot_ms = total_shot_ms

        # Decode in tile order
        for mid, t, clip_h, raw_path in pending:
            r = collected.get(mid)
            if not r or "error" in r:
                ac.errors.append(f"tile {t}: {r.get('error') if r else 'no response'}")
                continue

            tc = TileCapture(
                shot_ms=total_shot_ms / len(pending),
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

    async def _capture_single(self, conn, wi, article, ac, page_h, nav_ms):
        """Single-tile article — same as sequential."""
        clip_h = min(TILE_HEIGHT, page_h)
        params = {
            "fromSurface": self.from_surface,
            "optimizeForSpeed": True,
            "clip": {
                "x": 0,
                "y": 0,
                "width": VIEWPORT_WIDTH,
                "height": clip_h,
                "scale": 1,
            },
        }
        raw_path = None
        if self.fmt == "raw":
            raw_path = f"/dev/shm/pixelrag_bench/w{wi}_{id(article)}_0.raw"
            params["rawFilePath"] = raw_path
        else:
            params["format"] = self.fmt
            if self.fmt == "jpeg":
                params["quality"] = self.quality

        t0 = time.monotonic()
        try:
            r = await conn.cdp("Page.captureScreenshot", params)
        except Exception as e:
            ac.errors.append(f"tile 0: {e}")
            return ac
        shot_ms = (time.monotonic() - t0) * 1000
        ac.total_shot_ms = shot_ms

        if "error" not in r:
            tc = TileCapture(
                shot_ms=shot_ms, nav_ms=nav_ms, tile_index=0, clip_y=0, clip_h=clip_h
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)

        return ac

    async def _capture_sequential_fallback(self, conn, wi, article, ac, page_h, nav_ms):
        """Playwright doesn't expose raw websocket — fall back to sequential."""
        n_tiles = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)
        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break
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

            if "error" not in r:
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
