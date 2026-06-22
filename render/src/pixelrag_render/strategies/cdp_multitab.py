"""Multi-tab igpu strategy: fewer Chrome processes, multiple tabs each.

16 igpu Chrome processes × 3 tabs = 48 effective workers, but only ~160 OS
threads (vs 480 for 48 separate igpu processes). Each tab has its own
WebSocket connection and DevTools session.

Articles are distributed across tabs. Within each Chrome process, tabs are
captured sequentially (no concurrent ForceRedraw within same process).
Cross-process parallelism provides the throughput.
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
from dataclasses import dataclass

from .base import article_url, ArticleCapture, TileCapture
from .connection import WebsocketConnection, CHROME_ARGS, pick_page_ws_url

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


@dataclass
class CDPMultiTabStrategy:
    """Multi-tab igpu: fewer processes, more tabs per process."""

    chrome_path: str
    n_processes: int = 16
    tabs_per_process: int = 3
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    headless_shell: bool = False

    _tabs: list = None  # flat list of (conn, proc_idx) for all tabs
    _procs: list = None
    _base_port: int = 0
    _port_counter: int = 0

    @property
    def n_workers(self):
        return self.n_processes * self.tabs_per_process

    @property
    def name(self) -> str:
        return f"{self.n_processes}p×{self.tabs_per_process}t {self.fmt} multitab"

    def _pick_base_port(self) -> int:
        CDPMultiTabStrategy._port_counter += 1
        return 15000 + (CDPMultiTabStrategy._port_counter - 1) * 500

    async def setup(self) -> None:
        import websockets

        if self._base_port == 0:
            self._base_port = self._pick_base_port()

        self._tabs = []
        self._procs = []

        for pi in range(self.n_processes):
            port = self._base_port + pi
            args = [self.chrome_path, f"--remote-debugging-port={port}"]
            if not self.headless_shell:
                args.append("--headless=new")
            args += CHROME_ARGS + ["--in-process-gpu", "about:blank"]

            proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._procs.append(proc)

            # Connect to first tab
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

            ws0 = await websockets.connect(
                pick_page_ws_url(targets),
                open_timeout=10,
                max_size=50 * 1024 * 1024,
            )
            tab0 = WebsocketConnection(ws0, proc)
            await tab0.cdp("Page.enable")
            await tab0.cdp(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": VIEWPORT_WIDTH,
                    "height": TILE_HEIGHT,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            self._tabs.append((tab0, pi))

            # Create additional tabs
            class NoopProc:
                def send_signal(self, _):
                    pass

                def wait(self, timeout=None):
                    pass

                def kill(self):
                    pass

            for ti in range(1, self.tabs_per_process):
                r = await tab0.cdp("Target.createTarget", {"url": "about:blank"})
                target_id = r["result"]["targetId"]

                data2 = urllib.request.urlopen(
                    f"http://localhost:{port}/json", timeout=3
                ).read()
                targets2 = json.loads(data2)
                ws_url = None
                for t in targets2:
                    if t.get("id") == target_id:
                        ws_url = t["webSocketDebuggerUrl"]
                        break

                if not ws_url:
                    raise ConnectionError(f"Can't find tab {ti} on port {port}")

                ws = await websockets.connect(
                    ws_url, open_timeout=10, max_size=50 * 1024 * 1024
                )
                tab = WebsocketConnection(ws, NoopProc())
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
                self._tabs.append((tab, pi))

        if self.fmt == "raw":
            os.makedirs("/dev/shm/pixelrag_bench", exist_ok=True)

    async def teardown(self) -> None:
        if self._tabs:
            for tab, _ in self._tabs:
                try:
                    await tab.close()
                except Exception:
                    pass
        if self._procs:
            for proc in self._procs:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]:
        article_index = {a["path"]: i for i, a in enumerate(articles)}
        all_results = [None] * len(articles)
        queue = asyncio.Queue()
        for a in articles:
            queue.put_nowait(a)

        async def worker(tab_idx):
            tab, proc_idx = self._tabs[tab_idx]
            while not queue.empty():
                try:
                    article = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    ac = await self._capture_one(tab, tab_idx, article)
                    all_results[article_index[article["path"]]] = ac
                except Exception as e:
                    ac = ArticleCapture(article_path=article["path"])
                    ac.errors.append(f"worker {tab_idx}: {e}")
                    all_results[article_index[article["path"]]] = ac

        await asyncio.gather(*[worker(i) for i in range(len(self._tabs))])
        return [r for r in all_results if r is not None]

    async def _capture_one(self, conn, wi: int, article: dict) -> ArticleCapture:
        ac = ArticleCapture(article_path=article["path"])
        th = TILE_HEIGHT

        t_nav = time.monotonic()
        # Use Page.frameStoppedLoading: reliable with --in-process-gpu
        # (Page.frameNavigated has a Chrome bug where it's sometimes not fired).
        nav_fut = asyncio.ensure_future(
            conn.wait_for_event("Page.frameStoppedLoading", timeout=30)
        )
        try:
            await conn.cdp("Page.navigate", {"url": article_url(article)})
        except Exception as e:
            nav_fut.cancel()
            ac.errors.append(f"nav: {e}")
            return ac

        try:
            await nav_fut
        except asyncio.TimeoutError:
            ac.errors.append("nav: frameStoppedLoading timeout")
            return ac

        try:
            r = await conn.cdp(
                "Runtime.evaluate",
                {
                    "expression": WAIT_FONTS_IMGS,
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

        for t in range(n_tiles):
            clip_h = min(th, page_h - t * th)
            if clip_h <= 28:
                break

            if t > 0:
                try:
                    await conn.cdp(
                        "Runtime.evaluate",
                        {
                            "expression": f"""new Promise(resolve => {{
                            window.scrollTo(0, {t * th});
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
                    "y": t * th,
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
                clip_y=t * th,
                clip_h=clip_h,
            )
            if self.fmt == "raw":
                tc.raw_file_path = raw_path
            else:
                tc.image_bytes = base64.b64decode(r["result"]["data"])
            ac.tiles.append(tc)

        return ac
