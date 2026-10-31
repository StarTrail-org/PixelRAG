"""Exercise the actual MCP transport and PixelRAG renderer handoff."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import run

SEARCH = {
    "results": [
        {
            "title": "Python documentation",
            "url": "https://docs.python.org/3/",
            "snippet": "Python docs",
        },
        {"url": "file:///etc/passwd"},
        {"url": "http://169.254.169.254/latest/meta-data/"},
    ]
}


def mcp_fixture(requests, structured, fetch_error=False):
    """Answer MCP JSON-RPC like the server, in text JSON or structured form."""

    def respond(request):
        requests.append(request)
        assert str(request.url) == run.MCP_URL
        assert request.headers["User-Agent"] == run.USER_AGENT
        assert request.headers["X-Keenable-Title"] == run.APP_TITLE
        if request.method == "DELETE":
            return httpx.Response(200)
        payload = json.loads(request.content)
        method = payload["method"]
        if "id" not in payload:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ["search_web_pages", "fetch_page_content"]
                ]
            }
        elif method == "tools/call":
            params = payload["params"]
            args = params["arguments"]
            if params["name"] == "search_web_pages":
                assert set(args) == {"query", "output_format", "session_id"}
                assert args["query"] == "Python language documentation"
                assert args["output_format"] == "json"
                data = SEARCH
            else:
                assert set(args) == {"url", "max_chars", "output_format", "session_id"}
                assert args["url"] == "https://docs.python.org/3/"
                assert args["max_chars"] == run.FETCH_MAX_CHARS
                search_call = next(
                    json.loads(r.content)
                    for r in requests
                    if r.content and b"search_web_pages" in r.content
                )
                assert (
                    args["session_id"]
                    == search_call["params"]["arguments"]["session_id"]
                )
                data = {"url": args["url"], "content": "Welcome to Python"}
            if fetch_error and params["name"] == "fetch_page_content":
                result = {
                    "content": [{"type": "text", "text": "Error: fetch failed"}],
                    "isError": True,
                }
            elif structured:
                result = {
                    "content": [{"type": "text", "text": "Title: ..."}],
                    "structuredContent": data,
                    "isError": False,
                }
            else:
                result = {
                    "content": [{"type": "text", "text": json.dumps(data)}],
                    "isError": False,
                }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}
        )

    return respond


def run_search(monkeypatch, structured=False, fetch_error=False):
    requests = []
    client_type = httpx.AsyncClient
    respond = mcp_fixture(requests, structured, fetch_error)

    def client(**kwargs):
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(run.httpx, "AsyncClient", client)
    data = asyncio.run(run.search_pages("Python language documentation", 2, True))
    return data, requests


@pytest.mark.parametrize("structured", [False, True])
def test_mcp_search_fetch_requests(monkeypatch, structured):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    data, requests = run_search(monkeypatch, structured)
    assert [page["url"] for page in data["selected"]] == ["https://docs.python.org/3/"]
    assert data["selected"][0]["snippet"] == "Python docs"
    assert data["search"] == SEARCH["results"]
    assert data["fetch"] == [
        {"url": "https://docs.python.org/3/", "content": "Welcome to Python"}
    ]
    assert len(requests) >= 5
    assert all("X-API-Key" not in r.headers for r in requests)
    assert all("Authorization" not in r.headers for r in requests)


def test_api_key_is_opt_in(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "fixture-key")
    _, requests = run_search(monkeypatch)
    assert all(r.headers["X-API-Key"] == "fixture-key" for r in requests)
    monkeypatch.setenv("KEENABLE_API_KEY", "  ")
    assert "X-API-Key" not in run.request_headers()


def test_failed_fetch_keeps_search_results(monkeypatch):
    data, _ = run_search(monkeypatch, fetch_error=True)
    assert data["selected"][0]["url"] == "https://docs.python.org/3/"
    assert data["fetch"][0]["url"] == "https://docs.python.org/3/"
    assert "fetch failed" in data["fetch"][0]["error"]


def test_selection_and_errors():
    assert run.select_results(
        {
            "results": [
                {"url": "file:///tmp/example"},
                {"url": "https://user:secret@example.org/"},
                {"url": "http://localhost:8000/"},
                {"url": "http://127.0.0.1/"},
                {"url": "http://10.0.0.5/admin"},
                {"url": "http://169.254.169.254/latest/meta-data/"},
                {"url": "http://[::1]/"},
                {"url": "https://example.org/"},
                {"url": "https://example.org/"},
                {"url": "https://93.184.215.14/"},
                {"url": "https://example.net/"},
            ]
        },
        2,
    ) == [{"url": "https://example.org/"}, {"url": "https://93.184.215.14/"}]
    with pytest.raises(RuntimeError, match="tool failed"):
        run.tool_data(SimpleNamespace(isError=True, content=["rate limited"]))
    with pytest.raises(RuntimeError, match="no JSON"):
        run.tool_data(
            SimpleNamespace(isError=False, structuredContent=None, content=[])
        )


def test_public_renderer_handoff(monkeypatch, tmp_path):
    from pixelrag_render import render

    captures = []

    def render_urls(urls, output, **kwargs):
        assert urls == ["https://docs.python.org/3/"]
        assert output.parent == tmp_path / "tiles"
        assert output not in captures
        captures.append(output)
        assert kwargs == {"stems": ["page-00"], "workers": 1, "wait_network_idle": True}
        directory = Path(output) / "page-00.png.tiles"
        directory.mkdir(parents=True)
        (directory / "tile_0000.jpg").write_bytes(b"fixture")
        return [directory]

    monkeypatch.setattr(render, "render_urls", render_urls)
    paths = run.render_pages([{"url": "https://docs.python.org/3/"}], tmp_path)
    assert len(paths) == 1 and Path(paths[0]).is_file()
    # A prior longer page must not add stale tiles to a later capture's manifest.
    (Path(paths[0]).parent / "tile_0001.jpg").write_bytes(b"old page")
    new_paths = run.render_pages([{"url": "https://docs.python.org/3/"}], tmp_path)
    assert len(new_paths) == 1 and new_paths[0] != paths[0]

    monkeypatch.setattr(render, "render_urls", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="no screenshot tiles"):
        run.render_pages([{"url": "https://docs.python.org/3/"}], tmp_path)
