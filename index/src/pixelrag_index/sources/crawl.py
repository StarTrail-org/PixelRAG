"""Crawl source — discover and index all pages on a website.

Starts from a URL, follows same-domain links up to a configurable depth/limit,
and yields each discovered page as a Document for the index pipeline.

Usage in pixelrag.yaml:
    source:
      type: crawl
      start_url: https://example.com/
      max_pages: 50          # stop after N pages (default: 100)
      max_depth: 3           # max link-following depth (default: 3)
      stay_on_domain: true   # don't follow external links (default: true)
      exclude_patterns:      # skip URLs matching these (optional)
        - /admin/
        - /login
"""

import logging
import re
from collections import deque
from typing import Iterator
from urllib.parse import urljoin, urlparse

from .base import Document, Source

logger = logging.getLogger(__name__)


def _fetch_links(url: str, timeout: int = 10) -> list[str]:
    """Fetch a page and extract all href links."""
    import urllib.request

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PixelRAG-Crawler/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    # Simple regex link extraction (no dependency on bs4/lxml)
    return re.findall(r'href=["\']([^"\']+)["\']', html)


class CrawlSource(Source):
    """Crawl a website starting from a URL, discovering pages via links."""

    def __init__(
        self,
        start_url: str,
        max_pages: int = 100,
        max_depth: int = 3,
        stay_on_domain: bool = True,
        exclude_patterns: list[str] | None = None,
        **kwargs,
    ):
        self.start_url = start_url.rstrip("/")
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.stay_on_domain = stay_on_domain
        self.exclude_patterns = exclude_patterns or []
        self.domain = urlparse(start_url).netloc

        self._pages: list[str] = []
        self._crawl()

    def _should_skip(self, url: str) -> bool:
        """Check if URL should be excluded."""
        for pattern in self.exclude_patterns:
            if pattern in url:
                return True
        # Skip non-HTML resources
        path = urlparse(url).path.lower()
        skip_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".pdf",
            ".css",
            ".js",
            ".ico",
            ".woff",
            ".woff2",
            ".mp4",
            ".zip",
        )
        if any(path.endswith(ext) for ext in skip_exts):
            return True
        # Skip common non-content paths
        skip_paths = (
            "/feed",
            "/wp-json",
            "/xmlrpc",
            "/wp-admin",
            "/wp-login",
            "/wp-content",
            "/trackback",
            "/comments/feed",
        )
        if any(s in url for s in skip_paths):
            return True
        # Skip query-heavy URLs (likely API/dynamic)
        if url.count("?") > 0 and url.count("=") > 2:
            return True
        # Skip fragments
        if "#" in url:
            url = url.split("#")[0]
        return False

    def _normalize(self, url: str) -> str:
        """Normalize URL: strip fragment, trailing slash."""
        url = url.split("#")[0]
        url = url.rstrip("/")
        return url

    def _crawl(self):
        """BFS crawl from start_url."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.start_url, 0)])
        visited.add(self._normalize(self.start_url))

        while queue and len(self._pages) < self.max_pages:
            url, depth = queue.popleft()
            self._pages.append(url)

            if depth >= self.max_depth:
                continue

            links = _fetch_links(url)
            for href in links:
                # Resolve relative URLs
                full_url = urljoin(url, href)
                normalized = self._normalize(full_url)

                # Filter
                if normalized in visited:
                    continue
                if self.stay_on_domain and urlparse(normalized).netloc != self.domain:
                    continue
                if self._should_skip(normalized):
                    continue
                # Only http/https
                if not normalized.startswith(("http://", "https://")):
                    continue

                visited.add(normalized)
                queue.append((normalized, depth + 1))

        logger.info(
            "Crawled %s: discovered %d pages (max_pages=%d, max_depth=%d)",
            self.domain,
            len(self._pages),
            self.max_pages,
            self.max_depth,
        )

    def __iter__(self) -> Iterator[Document]:
        for i, url in enumerate(self._pages):
            slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
            yield Document(
                id=f"crawl_{i:04d}_{slug}",
                url=url,
                metadata={"type": "web", "depth": 0},
            )

    def __len__(self) -> int:
        return len(self._pages)
