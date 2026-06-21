"""Tests for the --cdp-url attach-to-existing-browser path.

These run on a core install (no chrome, no browser): they exercise URL
normalization and the routing logic that decides between attaching to a running
browser and launching a throwaway one — without ever opening a browser.
"""

import sys
from pathlib import Path

import pytest

from pixelrag_render.backends import cdp

_BIN = Path(sys.executable).parent


def test_http_base_normalization():
    f = cdp._http_base_from_cdp_url
    assert f("http://127.0.0.1:9222") == "http://127.0.0.1:9222"
    assert f("http://127.0.0.1:9222/json/version") == "http://127.0.0.1:9222"
    assert f("127.0.0.1:9222") == "http://127.0.0.1:9222"
    assert f("ws://localhost:9222/devtools/browser/abc") == "http://localhost:9222"


def test_cdp_url_routes_to_attach_without_launching(monkeypatch, tmp_path):
    """With cdp_url set, render_urls must take the attach path and never call
    _find_chrome (i.e. never try to launch/auto-install a browser)."""
    called = {}

    async def fake_attached(urls, output_dir, *a, **kw):
        called["attached"] = list(urls)
        return [Path(output_dir) / "x.png.tiles"]

    def boom():
        raise AssertionError("_find_chrome must not run on the attach path")

    monkeypatch.setattr(cdp, "_run_batch_attached", fake_attached)
    monkeypatch.setattr(cdp, "_find_chrome", boom)

    out = cdp.render_urls(
        ["https://example.com"], tmp_path, cdp_url="http://127.0.0.1:9222"
    )
    assert called["attached"] == ["https://example.com"]
    assert out and out[0].name == "x.png.tiles"


def test_env_var_fallback_routes_to_attach(monkeypatch, tmp_path):
    called = {}

    async def fake_attached(urls, output_dir, *a, **kw):
        called["hit"] = True
        return []

    monkeypatch.setattr(cdp, "_run_batch_attached", fake_attached)
    monkeypatch.setattr(
        cdp, "_find_chrome", lambda: pytest.fail("should not launch")
    )
    monkeypatch.setenv("PIXELSHOT_CDP_URL", "http://127.0.0.1:9222")

    cdp.render_urls(["https://example.com"], tmp_path)
    assert called.get("hit") is True


def test_default_path_still_resolves_chrome(monkeypatch, tmp_path):
    """No cdp_url (and no env) → the launch path runs find_chrome as before."""
    monkeypatch.delenv("PIXELSHOT_CDP_URL", raising=False)
    sentinel = RuntimeError("find_chrome reached")

    def boom():
        raise sentinel

    monkeypatch.setattr(cdp, "_find_chrome", boom)
    with pytest.raises(RuntimeError, match="find_chrome reached"):
        cdp.render_urls(["https://example.com"], tmp_path)


def test_cli_help_exposes_cdp_url():
    import subprocess

    r = subprocess.run(
        [str(_BIN / "pixelshot"), "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0
    assert "--cdp-url" in r.stdout
