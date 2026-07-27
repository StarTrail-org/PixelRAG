"""Kiwix ZIM data source — HTML + images served locally, zero network requests."""

import itertools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote

from .base import Document, Source

logger = logging.getLogger(__name__)


class KiwixServeManager:
    """Manages multiple kiwix-serve processes for high-concurrency serving.

    Benchmark results (80 concurrent requests, 2000 articles):
        1x --threads 4:   454 rps, p50=160ms
        2x --threads 4:   720 rps, p50=98ms
        4x --threads 4:  1212 rps, p50=21ms
        8x --threads 4:  2011 rps, p50=20ms

    Multi-process scales linearly because each instance independently
    decompresses ZIM clusters without lock contention.
    """

    _SEARCH_PATHS = (
        str(Path(__file__).resolve().parents[4] / ".local" / "bin" / "kiwix-serve"),
        "/usr/bin/kiwix-serve",
        "/usr/local/bin/kiwix-serve",
    )

    def __init__(
        self,
        zim_path: str,
        base_port: int = 9454,
        num_instances: int = 8,
        threads_per_instance: int = 4,
    ):
        self.zim_path = zim_path
        self.base_port = base_port
        self.num_instances = num_instances
        self.threads_per_instance = threads_per_instance
        self._procs: list[Optional[subprocess.Popen]] = [None] * num_instances
        self._binary = self._find_binary()
        self._port_cycle = itertools.cycle(range(num_instances))
        self._last_request_time = time.time()
        self._ttl_thread: threading.Thread | None = None

    @property
    def ports(self) -> list[int]:
        return [self.base_port + i for i in range(self.num_instances)]

    def next_url(self) -> str:
        """Return base URL for the next instance (round-robin).

        Resets idle timer. If the selected instance is unresponsive, try
        to restart it and fall back to other healthy instances.
        """
        for _ in range(self.num_instances):
            idx = next(self._port_cycle)
            port = self.ports[idx]
            if self._health_check(port):
                return f"http://localhost:{port}"
            logger.warning("kiwix-serve on port %d unresponsive, restarting...", port)
            try:
                self._start_instance(idx)
                return f"http://localhost:{port}"
            except RuntimeError:
                logger.error("Failed to restart kiwix-serve on port %d, skipping", port)
        logger.error(
            "All kiwix-serve instances down, falling back to port %d", self.base_port
        )
        return f"http://localhost:{self.base_port}"

    _KIWIX_TOOLS_VERSION = "3.7.0-2"

    def _find_binary(self) -> str:
        for p in self._SEARCH_PATHS:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        found = shutil.which("kiwix-serve")
        if found:
            return found
        return self._install_kiwix_tools()

    def _install_kiwix_tools(self) -> str:
        """Auto-download kiwix-tools binary."""
        import platform
        import tarfile
        import tempfile
        import urllib.request

        arch = (
            "x86_64"
            if platform.machine() in ("x86_64", "AMD64")
            else platform.machine()
        )
        url = (
            f"https://download.kiwix.org/release/kiwix-tools/"
            f"kiwix-tools_linux-{arch}-{self._KIWIX_TOOLS_VERSION}.tar.gz"
        )

        install_dir = Path(self._SEARCH_PATHS[0]).parent
        install_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading kiwix-tools %s...", self._KIWIX_TOOLS_VERSION)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            with tarfile.open(tmp.name) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("kiwix-serve"):
                        member.name = "kiwix-serve"
                        tar.extract(member, install_dir)
                    elif member.name.endswith("kiwix-manage"):
                        member.name = "kiwix-manage"
                        tar.extract(member, install_dir)
            os.unlink(tmp.name)

        binary = str(install_dir / "kiwix-serve")
        os.chmod(binary, 0o755)
        logger.info("Installed kiwix-serve to %s", binary)
        return binary

    def _health_check(self, port: int) -> bool:
        """Quick HTTP check to see if kiwix-serve is responding."""
        import urllib.request

        try:
            req = urllib.request.Request(f"http://localhost:{port}/", method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False

    def _start_instance(self, idx: int) -> None:
        """Start or restart a single kiwix-serve instance."""
        port = self.ports[idx]
        old = self._procs[idx]
        if old is not None:
            self._kill_proc(old)
            self._procs[idx] = None

        if self._health_check(port):
            logger.info("kiwix-serve already running on port %d (external)", port)
            return

        logger.info(
            "Starting kiwix-serve instance %d on port %d (threads=%d) ...",
            idx,
            port,
            self.threads_per_instance,
        )
        proc = subprocess.Popen(
            [
                self._binary,
                "--port",
                str(port),
                "--threads",
                str(self.threads_per_instance),
                self.zim_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,
        )
        for _ in range(30):
            time.sleep(1)
            if self._health_check(port):
                logger.info(
                    "kiwix-serve instance %d started (pid %d, port %d)",
                    idx,
                    proc.pid,
                    port,
                )
                self._procs[idx] = proc
                self._start_ttl_watcher()
                return
        raise RuntimeError(
            f"kiwix-serve failed to start on port {port} (pid {proc.pid})"
        )

    _TTL_SECONDS = 300  # 5 min idle → auto-stop

    def _start_ttl_watcher(self) -> None:
        """Start background thread that stops kiwix-serve after idle TTL."""
        if self._ttl_thread is not None and self._ttl_thread.is_alive():
            return

        def _watcher() -> None:
            while True:
                time.sleep(60)
                if not any(p is not None for p in self._procs):
                    break
                if time.time() - self._last_request_time > self._TTL_SECONDS:
                    logger.info(
                        "kiwix-serve idle > %ds, auto-stopping", self._TTL_SECONDS
                    )
                    self.stop()
                    break

        self._ttl_thread = threading.Thread(target=_watcher, daemon=True)
        self._ttl_thread.start()

    def touch(self) -> None:
        """Reset the idle timer."""
        self._last_request_time = time.time()

    def ensure_running(self) -> None:
        """Ensure all instances are running, restart any that crashed."""
        self.touch()
        for idx in range(self.num_instances):
            port = self.ports[idx]
            proc = self._procs[idx]
            alive = proc is not None and proc.poll() is None
            if alive and self._health_check(port):
                continue
            if proc is not None:
                logger.warning(
                    "kiwix-serve instance %d (pid %s, port %d) is dead, restarting...",
                    idx,
                    proc.pid,
                    port,
                )
            self._start_instance(idx)

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    def stop(self) -> None:
        for idx, proc in enumerate(self._procs):
            if proc is not None:
                self._kill_proc(proc)
                self._procs[idx] = None

    def __del__(self) -> None:
        self.stop()


import atexit

_active_sources: list["KiwixSource"] = []


class KiwixSource(Source):
    """Data source backed by a local Kiwix ZIM file served via kiwix-serve.

    Both article HTML and embedded images are served from the ZIM archive,
    eliminating all external network requests and Wikimedia rate-limiting.
    """

    _SKIP_PREFIXES = ("_assets_/", "-/", "_/", "_mw_/")
    _SKIP_EXACT = {"-", "mainpage"}

    # Well-known ZIM aliases → download URLs
    _ZIM_CATALOG = {
        "wikipedia-simple": "https://download.kiwix.org/zim/wikipedia/wikipedia_en_simple_all_nopic_2026-05.zim",
        "wikipedia-en": "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_maxi_2025-08.zim",
    }
    _DEFAULT_ZIM_DIR = Path.home() / ".cache" / "pixelrag" / "zim"

    def __init__(
        self,
        zim_path: str = "wikipedia-simple",
        kiwix_serve_url: str = "http://localhost:9454",
        book_name: Optional[str] = None,
        num_kiwix_instances: int = 8,
        **kwargs,
    ):
        self.zim_path = self._resolve_zim(zim_path)
        self._book_name = book_name
        self._article_paths: Optional[list[str]] = None
        self._zim = None
        self._redirect_ids: Optional[set[int]] = None
        from urllib.parse import urlparse

        parsed = urlparse(kiwix_serve_url)
        base_port = parsed.port or 9454
        self._serve_manager = KiwixServeManager(
            str(self.zim_path),
            base_port=base_port,
            num_instances=num_kiwix_instances,
        )
        _active_sources.append(self)

    @classmethod
    def _resolve_zim(cls, zim_path: str) -> Path:
        """Resolve a ZIM path: file path, alias, or URL. Downloads if needed."""
        # 1. Existing file
        p = Path(zim_path).expanduser().resolve()
        if p.exists():
            return p

        # 2. Known alias (e.g. "wikipedia-simple")
        if zim_path in cls._ZIM_CATALOG:
            url = cls._ZIM_CATALOG[zim_path]
            filename = url.rsplit("/", 1)[-1]
            dest = cls._DEFAULT_ZIM_DIR / filename
            if dest.exists():
                logger.info("Using cached ZIM: %s", dest)
                return dest
            return cls._download_zim(url, dest)

        # 3. URL
        if zim_path.startswith("http://") or zim_path.startswith("https://"):
            filename = zim_path.rsplit("/", 1)[-1]
            dest = cls._DEFAULT_ZIM_DIR / filename
            if dest.exists():
                logger.info("Using cached ZIM: %s", dest)
                return dest
            return cls._download_zim(zim_path, dest)

        raise FileNotFoundError(
            f"ZIM not found: {zim_path}\n"
            f"Pass a file path, URL, or alias: {', '.join(cls._ZIM_CATALOG.keys())}"
        )

    @staticmethod
    def _download_zim(url: str, dest: Path) -> Path:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".zim.part")
        logger.info("Downloading: %s", url)

        resp = urllib.request.urlopen(url)
        total = int(resp.headers.get("Content-Length", 0))

        from tqdm import tqdm

        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                ncols=80,
            ) as bar,
        ):
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

        tmp.rename(dest)
        logger.info("Saved: %s (%.0f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    def _get_zim(self):
        if self._zim is None:
            from libzim.reader import Archive

            self._zim = Archive(str(self.zim_path))
        return self._zim

    @property
    def book_name(self) -> str:
        if self._book_name is None:
            self._book_name = self.zim_path.stem
        return self._book_name

    def _is_article_path(self, path: str) -> bool:
        if not path:
            return False
        if any(path.startswith(p) for p in self._SKIP_PREFIXES):
            return False
        if path in self._SKIP_EXACT:
            return False
        if "." in path.rsplit("/", 1)[-1]:
            last_part = path.rsplit("/", 1)[-1]
            ext = last_part.rsplit(".", 1)[-1].lower()
            if ext in {
                "png",
                "jpg",
                "jpeg",
                "gif",
                "svg",
                "webp",
                "ico",
                "css",
                "js",
                "json",
                "woff",
                "woff2",
                "ttf",
                "eot",
                "tif",
                "tiff",
                "bmp",
                "mp3",
                "mp4",
                "ogg",
                "ogv",
                "webm",
                "flac",
                "wav",
                "opus",
                "mid",
            }:
                return False
        return True

    def _cache_path(self) -> Path:
        return Path(str(self.zim_path) + ".articles.json")

    def _load_article_cache(self) -> Optional[list[str]]:
        cache = self._cache_path()
        if not cache.exists():
            return None
        try:
            with open(cache, "r") as f:
                paths = json.load(f)
            logger.info("Loaded %d articles from cache %s", len(paths), cache)
            return paths
        except Exception as e:
            logger.warning("Failed to load article cache: %s", e)
            return None

    def _save_article_cache(self, paths: list[str]) -> None:
        cache = self._cache_path()
        tmp = cache.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(paths, f)
            os.replace(tmp, cache)
            logger.info("Saved article cache (%d paths) to %s", len(paths), cache)
        except Exception as e:
            logger.warning("Failed to save article cache: %s", e)

    def _redirects_cache_path(self) -> Path:
        return Path(str(self.zim_path) + ".redirects.json")

    def _build_redirect_map(self) -> dict[str, str]:
        """Scan articles for client-side redirects not flagged by ZIM.

        Client-side redirects are tiny HTML pages (<1024 bytes) with
        ``<meta http-equiv="refresh" content="0;URL='./Target_Page'">``.
        Cached to ``<zim_path>.redirects.json``.
        """
        cache = self._redirects_cache_path()
        if cache.exists():
            try:
                with open(cache, "r") as f:
                    redirects = json.load(f)
                logger.info("Loaded %d redirects from cache %s", len(redirects), cache)
                return redirects
            except Exception as e:
                logger.warning("Failed to load redirects cache: %s", e)

        paths = self._build_article_list()
        zim = self._get_zim()
        redirects: dict[str, str] = {}
        url_re = re.compile(
            rb"""content\s*=\s*["'][^"']*URL\s*=\s*['"]?([^"'\s>]+)""", re.IGNORECASE
        )

        logger.info("Scanning %d articles for client-side redirects...", len(paths))
        for i, path in enumerate(paths):
            try:
                entry = zim.get_entry_by_path(path)
                item = entry.get_item()
                if item.size > 1024:
                    continue
                content = bytes(item.content)
                if b"http-equiv" not in content or b"refresh" not in content:
                    continue
                m = url_re.search(content)
                if m:
                    target = m.group(1).decode("utf-8", errors="replace")
                    target = target.lstrip("./")
                    if "#" in target:
                        target = target.split("#", 1)[0]
                    redirects[str(i)] = target
            except Exception:
                continue
            if i % 1_000_000 == 0 and i > 0:
                logger.info(
                    "  Scanned %dM / %dM, %d redirects so far",
                    i // 1_000_000,
                    len(paths) // 1_000_000,
                    len(redirects),
                )

        logger.info(
            "Found %d client-side redirects (%.1f%%)",
            len(redirects),
            100 * len(redirects) / max(len(paths), 1),
        )

        tmp = cache.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(redirects, f)
            os.replace(tmp, cache)
            logger.info("Saved redirects cache to %s", cache)
        except Exception as e:
            logger.warning("Failed to save redirects cache: %s", e)

        return redirects

    def _load_redirect_set(self) -> set[int]:
        if self._redirect_ids is not None:
            return self._redirect_ids
        redirects = self._build_redirect_map()
        self._redirect_ids = {int(k) for k in redirects}
        return self._redirect_ids

    def _build_article_list(self) -> list[str]:
        if self._article_paths is not None:
            return self._article_paths
        cached = self._load_article_cache()
        if cached is not None:
            self._article_paths = cached
            return self._article_paths
        zim = self._get_zim()
        logger.info("Building article list from ZIM (%d entries)...", zim.entry_count)
        paths = []
        for i in range(zim.entry_count):
            try:
                entry = zim._get_entry_by_id(i)
                path = entry.path
                if self._is_article_path(path):
                    if not entry.is_redirect:
                        paths.append(path)
            except Exception:
                continue
            if i % 1_000_000 == 0 and i > 0:
                logger.info(
                    "  Scanned %dM / %dM entries, %d articles so far",
                    i // 1_000_000,
                    zim.entry_count // 1_000_000,
                    len(paths),
                )
        self._article_paths = paths
        logger.info("Found %d articles in ZIM", len(paths))
        self._save_article_cache(paths)
        return self._article_paths

    def _path_to_url(self, path: str, base_url: str) -> str:
        """Convert ZIM entry path to kiwix-serve URL with given base."""
        safe_chars = "/:@!$&'()*+,;="
        return f"{base_url}/content/{self.book_name}/{quote(path, safe=safe_chars)}"

    def __iter__(self) -> Iterator[Document]:
        paths = self._build_article_list()
        self._serve_manager.ensure_running()
        redirect_ids = self._load_redirect_set()
        health_interval = 1_000
        yielded = 0
        for i, path in enumerate(paths):
            if i in redirect_ids:
                continue
            title = path.replace("_", " ")
            base_url = self._serve_manager.next_url()
            yield Document(
                id=str(i),
                url=self._path_to_url(path, base_url),
                metadata={"title": title, "type": "kiwix"},
            )
            yielded += 1
            if yielded % health_interval == 0:
                self._serve_manager.ensure_running()

    def __len__(self) -> int:
        return len(self._build_article_list())

    def close(self) -> None:
        self._serve_manager.stop()
        if self in _active_sources:
            _active_sources.remove(self)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "KiwixSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@atexit.register
def _cleanup_sources() -> None:
    for src in _active_sources:
        try:
            src.close()
        except Exception:
            pass
