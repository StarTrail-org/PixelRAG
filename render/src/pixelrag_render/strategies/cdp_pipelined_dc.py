"""Pipelined navigation + directClip capture using 2 tabs per Chrome process.

Key insight: directClip skips ForceRedraw, so it doesn't contend with
navigation/rendering happening in a different tab. This enables overlapping
nav of article N+1 with capture of article N.

Timeline per worker:
  Tab A: [nav1]────────[capture1(dc)]──[nav3]────────[capture3(dc)]──
  Tab B: ──────[nav2]────────[capture2(dc)]──[nav4]────────[capture4(dc)]──
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import subprocess
import urllib.request
from dataclasses import dataclass

from .base import article_url, ArticleCapture, TileCapture
from .connection import WebsocketConnection, pick_page_ws_url

TILE_HEIGHT = 8192
VIEWPORT_WIDTH = 875

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


class TwoTabConnection:
    """Chrome process with two tabs, each independently controllable."""

    def __init__(self, ws_a, ws_b, proc):
        self.tab_a = ws_a
        self.tab_b = ws_b
        self._proc = proc

    async def close(self):
        try:
            await self.tab_a.close()
        except Exception:
            pass
        try:
            await self.tab_b.close()
        except Exception:
            pass


async def launch_two_tab(
    chrome_path: str, port: int, headless_shell: bool = False
) -> TwoTabConnection:
    import websockets

    args = [chrome_path, f"--remote-debugging-port={port}"]
    if not headless_shell:
        args.append("--headless=new")
    args += [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--enable-gpu-rasterization",
        "--force-gpu-rasterization",
        "about:blank",
    ]

    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for attempt in range(10):
        await asyncio.sleep(1)
        try:
            data = urllib.request.urlopen(
                f"http://localhost:{port}/json", timeout=3
            ).read()
            targets = json.loads(data)
            break
        except Exception:
            if attempt == 9:
                proc.kill()
                raise ConnectionError(f"Chrome port {port}")

    ws_url_a = pick_page_ws_url(targets)
    ws_a = await websockets.connect(
        ws_url_a, open_timeout=10, max_size=50 * 1024 * 1024
    )
    tab_a = WebsocketConnection(ws_a, proc)

    r = await tab_a.cdp("Target.createTarget", {"url": "about:blank"})
    new_target_id = r["result"]["targetId"]
    data2 = urllib.request.urlopen(f"http://localhost:{port}/json", timeout=3).read()
    targets2 = json.loads(data2)
    ws_url_b = None
    for t in targets2:
        if t.get("id") == new_target_id:
            ws_url_b = t["webSocketDebuggerUrl"]
            break
    if not ws_url_b:
        for t in targets2:
            if t["webSocketDebuggerUrl"] != ws_url_a:
                ws_url_b = t["webSocketDebuggerUrl"]
                break

    ws_b = await websockets.connect(
        ws_url_b, open_timeout=10, max_size=50 * 1024 * 1024
    )
    tab_b = WebsocketConnection(ws_b, proc)

    return TwoTabConnection(tab_a, tab_b, proc)


@dataclass
class CDPPipelinedDCStrategy:
    chrome_path: str
    n_workers: int
    fmt: str = "jpeg"
    quality: int = 85
    headless_shell: bool = False

    _connections: list = None
    _base_port: int = 9300

    @property
    def name(self) -> str:
        return f"{self.n_workers}w {self.fmt} pipedc"

    @property
    def from_surface(self) -> bool:
        return True

    @property
    def launcher(self) -> str:
        return "websocket"

    async def setup(self) -> None:
        self._connections = []
        for i in range(self.n_workers):
            conn = await launch_two_tab(
                self.chrome_path,
                self._base_port + i,
                headless_shell=self.headless_shell,
            )
            self._connections.append(conn)

        for conn in self._connections:
            for tab in [conn.tab_a, conn.tab_b]:
                await tab.cdp("Page.enable")
                await tab.cdp(
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
            for ac in self._pipeline_worker(wi, wp[wi]):
                ac_result = await ac
                all_results[article_index[ac_result.article_path]] = ac_result

        async def worker_task(wi):  # noqa: F811
            arts = wp[wi]
            conn = self._connections[wi]
            tabs = [conn.tab_a, conn.tab_b]

            for i, article in enumerate(arts):
                tab = tabs[i % 2]
                other = tabs[(i + 1) % 2]

                nav_task = self._navigate(tab, article)
                if i > 0:
                    cap_task = self._capture(other, prev_article, prev_page_h, wi)  # noqa: F821
                    nav_result, cap_result = await asyncio.gather(nav_task, cap_task)
                    all_results[article_index[prev_article["path"]]] = cap_result  # noqa: F821
                else:
                    nav_result = await nav_task

                prev_article = article
                prev_page_h = nav_result

            last_tab = tabs[len(arts) % 2 - 1] if arts else tabs[0]
            if arts:
                last_tab = tabs[(len(arts) - 1) % 2]
                cap_result = await self._capture(
                    last_tab, prev_article, prev_page_h, wi
                )
                all_results[article_index[prev_article["path"]]] = cap_result

        await asyncio.gather(
            *[worker_task(i) for i in range(n)], return_exceptions=True
        )
        return [r for r in all_results if r is not None]

    async def _navigate(self, tab, article: dict) -> int:
        try:
            await tab.cdp("Page.navigate", {"url": article_url(article)})
        except Exception:
            return TILE_HEIGHT

        try:
            r = await tab.cdp(
                "Runtime.evaluate",
                {
                    "expression": WAIT_FONTS_IMGS,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            page_h = r["result"]["result"]["value"]
        except Exception:
            page_h = TILE_HEIGHT

        return max(page_h, 1)

    async def _capture(
        self, tab, article: dict, page_h: int, wi: int
    ) -> ArticleCapture:
        ac = ArticleCapture(article_path=article["path"])
        ac.page_height = page_h
        n_tiles = max(1, (page_h + TILE_HEIGHT - 1) // TILE_HEIGHT)
        ac.n_tiles_expected = n_tiles

        for t in range(n_tiles):
            clip_h = min(TILE_HEIGHT, page_h - t * TILE_HEIGHT)
            if clip_h <= 28:
                break

            if t > 0:
                try:
                    await tab.cdp(
                        "Runtime.evaluate",
                        {
                            "expression": f"""new Promise(resolve => {{
                            window.scrollTo(0, {t * TILE_HEIGHT});
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
                "directClip": True,
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
                r = await tab.cdp("Page.captureScreenshot", params)
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
                nav_ms=0.0,
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
