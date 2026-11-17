"""Tests for the crawl source adapter (PR #105)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))
from pixelrag_index.sources.crawl import CrawlSource


def _mock_links(url):
    """Fake link graph for testing."""
    graph = {
        "https://example.com": [
            "/about",
            "/services",
            "/contact",
            "https://example.com/blog",
            "https://external.com/other",
            "/assets/logo.png",
            "/style.css",
            "/feed/",
            "/wp-json/api",
        ],
        "https://example.com/about": ["/team", "/about/history"],
        "https://example.com/services": ["/services/cloud", "/services/ai"],
        "https://example.com/blog": ["/blog/post-1", "/blog/post-2"],
        "https://example.com/about/history": ["/about/history/2020"],
    }
    return graph.get(url, [])


@pytest.fixture
def crawl_source():
    with patch("pixelrag_index.sources.crawl._fetch_links", side_effect=_mock_links):
        yield lambda **kwargs: CrawlSource(start_url="https://example.com/", **kwargs)


def test_crawl_discovers_links(crawl_source):
    """Crawler should discover pages reachable from start_url."""
    source = crawl_source(max_pages=50, max_depth=3)
    urls = [doc.url for doc in source]
    assert "https://example.com" in urls
    assert "https://example.com/about" in urls
    assert "https://example.com/services" in urls
    assert len(urls) > 1


def test_crawl_stays_on_domain(crawl_source):
    """External links should not be followed."""
    source = crawl_source(max_pages=50, max_depth=3)
    urls = [doc.url for doc in source]
    assert not any("external.com" in u for u in urls)


def test_crawl_respects_max_pages(crawl_source):
    """Should stop at max_pages limit."""
    source = crawl_source(max_pages=3, max_depth=5)
    assert len(source) == 3


def test_crawl_respects_max_depth(crawl_source):
    """Should not go deeper than max_depth."""
    source = crawl_source(max_pages=50, max_depth=1)
    urls = [doc.url for doc in source]
    # Depth 0 = start, depth 1 = direct links from start
    # /about/history is depth 2, should NOT be included
    assert "https://example.com/about/history" not in urls
    # /about is depth 1, should be included
    assert "https://example.com/about" in urls


def test_crawl_excludes_patterns(crawl_source):
    """Custom exclude_patterns should filter matching URLs."""
    source = crawl_source(max_pages=50, max_depth=3, exclude_patterns=["/blog"])
    urls = [doc.url for doc in source]
    assert not any("/blog" in u for u in urls)


def test_crawl_filters_assets(crawl_source):
    """Static assets (.png, .css, .js) should be skipped."""
    source = crawl_source(max_pages=50, max_depth=3)
    urls = [doc.url for doc in source]
    assert not any(u.endswith(".png") for u in urls)
    assert not any(u.endswith(".css") for u in urls)


def test_crawl_filters_feeds_and_api(crawl_source):
    """WordPress feeds and wp-json should be skipped."""
    source = crawl_source(max_pages=50, max_depth=3)
    urls = [doc.url for doc in source]
    assert not any("/feed" in u for u in urls)
    assert not any("/wp-json" in u for u in urls)


def test_url_not_mangled_by_config():
    """make_source must not mangle URLs with Path().expanduser()."""
    pytest.importorskip("yaml")
    from pixelrag_index.config import make_source

    config = {
        "source": {
            "type": "crawl",
            "start_url": "https://comprinno.net/",
            "max_pages": 1,
            "max_depth": 0,
        }
    }
    with patch("pixelrag_index.sources.crawl._fetch_links", return_value=[]):
        source = make_source(config)
    assert source.start_url == "https://comprinno.net"
    assert source.domain == "comprinno.net"
