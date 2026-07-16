"""On-demand tile rendering for pixelrag-serve.

When a tile image is not on disk, render the source page from a kiwix ZIM (served
over HTTP by ``kiwix-serve``), slice it into the same 1024px chunks the index was
built on, and return the requested chunk. This removes the dependency on a
materialized multi-TB ``tiles/`` corpus -- only the pages that retrieval actually
returns get rendered, lazily, and then cached on disk.

Pixel-validated against the original pipeline's tiles (mean |diff| ~2/255 on every
referenced chunk), using the exact build config: viewport_width=875, tile_height=8192,
and the shared ``pixelrag_embed.chunk`` slicer (1024px chunks).
"""

import atexit
import glob
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import quote

_render_lock = threading.Lock()  # one Chrome render at a time per process
# Hard timeout (seconds) for a single page render subprocess.
_RENDER_TIMEOUT = float(os.environ.get("PIXELRAG_RENDER_TIMEOUT", "120"))


class OnDemandTiles:
    def __init__(
        self,
        kiwix_url: str,
        book: str,
        cache_dir: str,
        viewport_width: int = 875,
        tile_height: int = 8192,
    ):
        self.kiwix_url = kiwix_url.rstrip("/")
        self.book = book
        self.cache_dir = cache_dir
        self.viewport_width = viewport_width
        self.tile_height = tile_height
        os.makedirs(cache_dir, exist_ok=True)
        # Persistent headless Chrome, reused across renders via --cdp-url. Starting a fresh
        # Chrome per page costs a cold start (seconds locally, tens of seconds on a cold/NFS
        # box); reusing one browser cuts per-page render from ~tens of s to ~1-2s.
        self._chrome_proc = None
        self._cdp_url = None
        self._chrome_udd = None
        self._chrome_lock = threading.Lock()
        atexit.register(self._kill_chrome)

    def _kill_chrome(self) -> None:
        if self._chrome_proc is not None:
            try:
                self._chrome_proc.kill()
            except Exception:
                pass
            self._chrome_proc = None
        self._cdp_url = None
        if self._chrome_udd:
            shutil.rmtree(self._chrome_udd, ignore_errors=True)
            self._chrome_udd = None

    def _ensure_chrome(self) -> str:
        """Start (or restart) the persistent headless Chrome; return its CDP base URL."""
        with self._chrome_lock:
            if (
                self._chrome_proc is not None
                and self._chrome_proc.poll() is None
                and self._cdp_url
            ):
                return self._cdp_url
            self._kill_chrome()
            from pixelrag_render.chrome import find_chrome

            chrome = find_chrome()
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            self._chrome_udd = os.path.join(self.cache_dir, f".chrome_{port}")
            self._chrome_proc = subprocess.Popen(
                [
                    chrome,
                    f"--remote-debugging-port={port}",
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    f"--user-data-dir={self._chrome_udd}",
                    "about:blank",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(base + "/json/version", timeout=2)
                    self._cdp_url = base
                    return base
                except Exception:
                    if self._chrome_proc.poll() is not None:
                        break
                    time.sleep(0.5)
            self._kill_chrome()
            raise RuntimeError("on-demand Chrome failed to start")

    def _article_dir(self, article_id: int) -> str:
        return os.path.join(self.cache_dir, f"{article_id}.png.tiles")

    def chunk_path(
        self, article_id: int, title: str, tile_index: int, chunk_index: int
    ):
        """Path to chunk_{ti}_{ci}.png, rendering+chunking the page on a cache miss."""
        chunk_name = f"chunk_{tile_index:04d}_{chunk_index:02d}.png"
        cpath = os.path.join(self._article_dir(article_id), chunk_name)
        if os.path.exists(cpath):
            return cpath
        if not title:
            return None
        with _render_lock:
            if os.path.exists(cpath):  # filled while we waited for the lock
                return cpath
            try:
                self._render_and_chunk(article_id, title)
            except Exception:
                return None
        return cpath if os.path.exists(cpath) else None

    def _render_and_chunk(self, article_id: int, title: str) -> None:
        from pixelrag_embed.chunk import chunk_article

        url = f"{self.kiwix_url}/content/{self.book}/{quote(title, safe='')}"
        staging = os.path.join(self.cache_dir, f".render_{article_id}")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        # Render in a SEPARATE PROCESS via the pixelshot CLI, attached to the persistent
        # Chrome with --cdp-url (renders in a fresh tab, no per-page cold start). The
        # subprocess is still needed: render_url internally uses asyncio.run() +
        # multiprocessing.Pool (fork), which deadlocks if called a 2nd time in this
        # long-lived serve process — a fresh subprocess per render avoids that.
        cdp_url = self._ensure_chrome()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pixelrag_render.render",
                    url,
                    "--output",
                    staging,
                    "--viewport-width",
                    str(self.viewport_width),
                    "--tile-height",
                    str(self.tile_height),
                    "--cdp-url",
                    cdp_url,
                    "--workers",
                    "1",
                ],
                check=True,
                timeout=_RENDER_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # The shared Chrome may be wedged/dead — drop it so the next render restarts it.
            self._kill_chrome()
            shutil.rmtree(staging, ignore_errors=True)
            return
        dirs = glob.glob(os.path.join(staging, "*.png.tiles"))
        if not dirs:
            shutil.rmtree(staging, ignore_errors=True)
            return
        rendered = dirs[0]  # <sanitized-url>.png.tiles/ (has tiles.json)
        chunk_article(rendered)  # writes chunk_XXXX_YY.png + chunks.json
        dest = self._article_dir(article_id)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(rendered, dest)  # atomic; commit only after chunking succeeds
        shutil.rmtree(staging, ignore_errors=True)
