"""Exercise the actual MCP transport and PixelRAG renderer handoff."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import run


def test_mcp_search_fetch_requests(monkeypatch):
    requests = []
    client_type = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        assert str(request.url) == run.MCP_URL
        assert request.headers["User-Agent"] == run.USER_AGENT
        assert "Authorization" not in request.headers
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
                    for name in ["web_search", "web_fetch"]
                ]
            }
        elif method == "tools/call":
            params = payload["params"]
            args = params["arguments"]
            if params["name"] == "web_search":
                assert args["objective"] == "Find documentation"
                assert args["search_queries"] == ["Python language documentation"]
                data = {
                    "results": [
                        {
                            "url": "https://docs.python.org/3/",
                            "excerpts": ["Python docs"],
                        },
                        {"url": "file:///etc/passwd"},
                    ]
                }
            else:
                assert args["urls"] == ["https://docs.python.org/3/"]
                search_call = next(
                    json.loads(r.content)
                    for r in requests
                    if r.content and b"web_search" in r.content
                )
                assert (
                    args["session_id"]
                    == search_call["params"]["arguments"]["session_id"]
                )
                data = {
                    "results": [
                        {"url": args["urls"][0], "excerpts": ["Welcome to Python"]}
                    ]
                }
            result = {
                "content": [{"type": "text", "text": json.dumps(data)}],
                "isError": False,
            }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}
        )

    def client(**kwargs):
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(run.httpx, "AsyncClient", client)
    data = asyncio.run(
        run.search_pages(
            "Find documentation", ["Python language documentation"], 1, True
        )
    )
    assert data["selected"][0]["excerpts"] == ["Python docs"]
    assert data["fetch"]["results"][0]["excerpts"] == ["Welcome to Python"]
    assert len(requests) >= 5


def test_selection_and_errors():
    assert run.select_results(
        {
            "results": [
                {"url": "file:///tmp/example"},
                {"url": "https://user:secret@example.org/"},
                {"url": "https://example.org/"},
                {"url": "https://example.org/"},
                {"url": "https://example.net/"},
            ]
        },
        2,
    ) == [{"url": "https://example.org/"}, {"url": "https://example.net/"}]
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
