"""Phase-separated capture strategy.

Nav (I/O-bound) and capture (CPU-bound) use different resources.
By limiting concurrent captures with a semaphore, we reduce CPU contention
during screenshots while allowing unlimited concurrent navigations.

Workers naturally pipeline: while N workers capture, the rest navigate.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass

from .base import ArticleCapture, TileCapture, article_url
from .connection import launch_websocket

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
class CDPPhasedStrategy:
    """Semaphore-limited concurrent captures.

    n_workers Chrome processes, but only capture_limit capture simultaneously.
    The rest are free to navigate (I/O-bound), reducing CPU contention.
    """

    chrome_path: str
    n_workers: int
    capture_limit: int = 0  # 0 = n_workers // 2
    fmt: str = "jpeg"
    quality: int = 85
    from_surface: bool = True
    launcher: str = "websocket"
    headless_shell: bool = False
    nav_timeout_ms: int = 2000
    tile_height: int = TILE_HEIGHT
    use_direct_clip: bool = False
    extra_chrome_args: list = None

    _connections: list = None
    _base_port: int = 0
    _capture_sem: asyncio.Semaphore = None
    _port_counter: int = 0

    def _pick_base_port(self) -> int:
        """Pick a unique base port to avoid TIME_WAIT conflicts between runs."""
        CDPPhasedStrategy._port_counter += 1
        return 10000 + (CDPPhasedStrategy._port_counter - 1) * 500

    @property
    def name(self) -> str:
        cl = self.capture_limit or self.n_workers // 2
        th = f" h{self.tile_height}" if self.tile_height != TILE_HEIGHT else ""
        dc = " dc" if self.use_direct_clip else ""
        return f"{self.n_workers}w-{cl}c {self.fmt} phased{th}{dc}"

    async def setup(self) -> None:
        cl = self.capture_limit or self.n_workers // 2
        self._capture_sem = asyncio.Semaphore(cl)
        if self._base_port == 0:
            self._base_port = self._pick_base_port()

        self._connections = []
        for i in range(self.n_workers):
            conn = await launch_websocket(
                self.chrome_path,
                self._base_port + i,
                headless_shell=self.headless_shell,
                extra_args=self.extra_chrome_args,
            )
            self._connections.append(conn)

        self._main_frame_ids = []
        for conn in self._connections:
            await conn.cdp("Page.enable")
            await conn.cdp(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": VIEWPORT_WIDTH,
                    "height": self.tile_height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            ft = await conn.cdp("Page.getFrameTree")
            self._main_frame_ids.append(ft["result"]["frameTree"]["frame"]["id"])

        if self.fmt == "raw":
            os.makedirs("/dev/shm/pixelrag_bench", exist_ok=True)

    async def teardown(self) -> None:
        if self._connections:
            for conn in self._connections:
                await conn.close()

    async def capture_articles(self, articles: list[dict]) -> list[ArticleCapture]:
        article_index = {a["path"]: i for i, a in enumerate(articles)}
        all_results = [None] * len(articles)
        queue = asyncio.Queue()
        for a in articles:
            queue.put_nowait(a)

        async def worker_task(wi):
            while not queue.empty():
                try:
                    article = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    ac = await self._capture_one(wi, article)
                    all_results[article_index[article["path"]]] = ac
                except Exception as e:
                    ac = ArticleCapture(article_path=article["path"])
                    ac.errors.append(f"worker {wi}: {e}")
                    all_results[article_index[article["path"]]] = ac

        await asyncio.gather(*[worker_task(i) for i in range(len(self._connections))])
        return [r for r in all_results if r is not None]

    async def _capture_one(self, wi: int, article: dict) -> ArticleCapture:
        conn = self._connections[wi]
        ac = ArticleCapture(article_path=article["path"])
        th = self.tile_height

        # === NAV (no semaphore) ===
        t_nav = time.monotonic()
        target_url = article_url(article)

        # Wait for Page.frameStoppedLoading instead of Page.frameStoppedLoading.
        # With --in-process-gpu and many concurrent instances, Chrome has a bug
        # where Page.frameStoppedLoading is sometimes never fired (even though the
        # page loads correctly).  Page.frameStoppedLoading is always reliable.
        main_fid = (
            self._main_frame_ids[wi] if hasattr(self, "_main_frame_ids") else None
        )
        nav_event_fut = asyncio.ensure_future(
            conn.wait_for_event(
                "Page.frameStoppedLoading",
                timeout=30.0,
                filter_fn=lambda p: main_fid is None or p.get("frameId") == main_fid,
            )
        )
        try:
            await conn.cdp("Page.navigate", {"url": target_url})
        except Exception as e:
            nav_event_fut.cancel()
            ac.errors.append(f"nav: {e}")
            return ac

        try:
            await nav_event_fut
        except asyncio.TimeoutError:
            ac.errors.append("nav: frameStoppedLoading timeout (30s)")
            return ac
        except Exception as e:
            ac.errors.append(f"nav: frameStoppedLoading wait error: {e}")
            return ac

        try:
            if self.nav_timeout_ms == 0:
                # Fast nav: fonts only + single rAF (no image wait)
                fast_expr = """new Promise(r => {
                    document.fonts.ready.then(() => {
                        requestAnimationFrame(() => {
                            const sh = document.documentElement.scrollHeight;
                            const body = document.body;
                            r(body ? Math.min(sh, Math.max(Math.ceil(body.getBoundingClientRect().bottom), 1)) : sh);
                        });
                    });
                })"""
                r = await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": fast_expr,
                        "awaitPromise": True,
                        "returnByValue": True,
                    },
                )
            else:
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

        if self.use_direct_clip or self.extra_chrome_args:
            try:
                await conn.cdp(
                    "Runtime.evaluate",
                    {
                        "expression": "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r))))",
                        "awaitPromise": True,
                    },
                )
            except Exception:
                pass

        nav_ms = (time.monotonic() - t_nav) * 1000
        ac.total_nav_ms = nav_ms
        ac.page_height = page_h
        n_tiles = max(1, (page_h + th - 1) // th)
        ac.n_tiles_expected = n_tiles

        # === PER-TILE: scroll (free) → acquire sem → capture → release sem ===
        for t in range(n_tiles):
            clip_h = min(th, page_h - t * th)
            if clip_h <= 28:
                break

            # Scroll + wait images (outside semaphore — I/O bound)
            if t > 0:
                y = t * th
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

            # Acquire → capture → release (fine-grained)
            await self._capture_sem.acquire()
            try:
                params = {
                    "optimizeForSpeed": True,
                    "clip": {
                        "x": 0,
                        "y": t * th,
                        "width": VIEWPORT_WIDTH,
                        "height": clip_h,
                        "scale": 1,
                    },
                }
                if self.use_direct_clip:
                    params["skipRedraw"] = True
                else:
                    params["fromSurface"] = self.from_surface
                raw_path = None
                if self.fmt == "raw":
                    raw_path = f"/dev/shm/pixelrag_bench/w{wi}_{id(article)}_{t}.raw"
                    params["rawFilePath"] = raw_path
                else:
                    params["format"] = self.fmt
                    if self.fmt == "jpeg":
                        params["quality"] = self.quality

                t0 = time.monotonic()
                r = await conn.cdp("Page.captureScreenshot", params)
                shot_ms = (time.monotonic() - t0) * 1000

            except Exception as e:
                ac.errors.append(f"tile {t}: {e}")
                continue
            finally:
                self._capture_sem.release()

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
