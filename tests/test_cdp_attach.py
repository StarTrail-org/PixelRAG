"""Tests for the --cdp-url attach-to-existing-browser path.

These run on a core install (no chrome, no browser): they exercise URL
normalization and the routing logic that decides between attaching to a running
browser and launching a throwaway one — without ever opening a browser.
"""

import json
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
    monkeypatch.setattr(cdp, "_find_chrome", lambda: pytest.fail("should not launch"))
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


def test_attach_creates_and_closes_only_its_own_target(monkeypatch, tmp_path):
    """The attach path must create its own target, and on teardown close ONLY
    that target — never a pre-existing one, and never close/kill the browser.

    Mocks the websocket/CDP layer so no browser is needed: a fake ws records
    every CDP method sent on the browser-level connection.
    """
    browser_methods = []  # (method, params) sent on the browser ws

    class FakeBrowserWS:
        async def send(self, raw):
            msg = json.loads(raw)
            browser_methods.append((msg["method"], msg.get("params", {})))
            method = msg["method"]
            result = {}
            if method == "Target.createTarget":
                result = {"targetId": "OUR-TARGET-123"}
            self._reply = {"id": msg["id"], "result": result}

        async def recv(self):
            return json.dumps(self._reply)

        async def close(self):
            pass

    class FakePageWS:
        async def send(self, raw):
            msg = json.loads(raw)
            self._reply = {"id": msg["id"], "result": {}}

        async def recv(self):
            return json.dumps(self._reply)

        async def close(self):
            pass

    async def fake_connect_ws(ws_url):
        return FakeBrowserWS() if ws_url == "BROWSER_WS" else FakePageWS()

    # Pre-existing target plus the one we create — only ours must be closed.
    def fake_fetch_json(url, cdp_url, timeout=5):
        if url.endswith("/json/version"):
            return {"webSocketDebuggerUrl": "BROWSER_WS"}
        return [
            {"id": "PREEXISTING-999", "webSocketDebuggerUrl": "ws://other"},
            {"id": "OUR-TARGET-123", "webSocketDebuggerUrl": "PAGE_WS"},
        ]

    async def fake_capture_url(*a, **kw):
        return 1

    monkeypatch.setattr(cdp, "_connect_ws", fake_connect_ws)
    monkeypatch.setattr(cdp, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(cdp, "capture_url", fake_capture_url)

    out = cdp.render_urls(
        ["https://example.com"], tmp_path, cdp_url="http://127.0.0.1:9222"
    )
    assert out and out[0].name.endswith(".png.tiles")

    methods = [m for m, _ in browser_methods]
    assert "Target.createTarget" in methods
    assert "Target.closeTarget" in methods
    # Never closed/killed the browser.
    assert "Browser.close" not in methods

    # closeTarget targeted ONLY our own created target, never the pre-existing one.
    closed = [p["targetId"] for m, p in browser_methods if m == "Target.closeTarget"]
    assert closed == ["OUR-TARGET-123"]


def test_attach_bad_cdp_url_raises_clean_error(monkeypatch, tmp_path):
    """An unreachable/bad endpoint surfaces a clear RuntimeError, not a raw
    URLError/KeyError traceback."""
    import urllib.error

    def boom(url, timeout=5):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cdp.urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError, match="Could not reach CDP endpoint at"):
        cdp.render_urls(
            ["https://example.com"], tmp_path, cdp_url="http://127.0.0.1:9999"
        )


def test_cli_help_exposes_cdp_url():
    import subprocess

    r = subprocess.run(
        [str(_BIN / "pixelshot"), "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0
    assert "--cdp-url" in r.stdout
