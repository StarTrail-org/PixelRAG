"""Discover web pages with keyless Keenable MCP, optionally render screenshots."""

import argparse
import asyncio
import ipaddress
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://api.keenable.ai/mcp"
APP_TITLE = "PixelRAG"
USER_AGENT = "PixelRAG/keenable-search-demo (https://github.com/StarTrail-org/PixelRAG)"
FETCH_MAX_CHARS = 5000


def request_headers() -> dict:
    """Name the app on every request; send a key only if KEENABLE_API_KEY is set."""
    headers = {"User-Agent": USER_AGENT, "X-Keenable-Title": APP_TITLE}
    api_key = os.environ.get("KEENABLE_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def tool_data(result) -> dict:
    """Check MCP errors and accept structured or JSON text tool responses."""
    if result.isError:
        raise RuntimeError(f"Keenable MCP tool failed: {result.content}")
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        if block.type == "text":
            try:
                data = json.loads(block.text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    raise RuntimeError("Keenable MCP returned no JSON result")


def public_host(hostname: str) -> bool:
    """Reject localhost and loopback, private, link-local or reserved IP hosts."""
    hostname = hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(hostname).is_global
    except ValueError:
        return True


def select_results(data: dict, limit: int) -> list[dict]:
    """Keep distinct public HTTP(S) pages in the server's relevance order."""
    selected, seen = [], set()
    for result in data.get("results", []):
        url = result.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        if parsed.username or parsed.password or url in seen:
            continue
        if not public_host(parsed.hostname):
            continue
        seen.add(url)
        selected.append(result)
        if len(selected) == limit:
            break
    return selected


async def search_pages(query: str, limit: int, fetch: bool) -> dict:
    """Discover and call tools over Streamable HTTP; no key unless one is set."""
    session_id = str(uuid.uuid4())
    async with httpx.AsyncClient(headers=request_headers(), timeout=60) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=60)
            ) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                required = {"search_web_pages"}
                if fetch:
                    required.add("fetch_page_content")
                if not required <= tools:
                    raise RuntimeError(
                        f"Keenable MCP is missing tools: {required - tools}"
                    )
                search = tool_data(
                    await session.call_tool(
                        "search_web_pages",
                        {
                            "query": query,
                            "output_format": "json",
                            "session_id": session_id,
                        },
                    )
                )
                selected = select_results(search, limit)
                if not selected:
                    raise RuntimeError("Search returned no usable HTTP(S) pages")
                fetched = None
                if fetch:
                    fetched = []
                    for result in selected:
                        try:
                            page = tool_data(
                                await session.call_tool(
                                    "fetch_page_content",
                                    {
                                        "url": result["url"],
                                        "max_chars": FETCH_MAX_CHARS,
                                        "output_format": "json",
                                        "session_id": session_id,
                                    },
                                )
                            )
                        except RuntimeError as error:
                            page = {"url": result["url"], "error": str(error)}
                        fetched.append(page)
                return {
                    "query": query,
                    "selected": selected,
                    "search": search.get("results", []),
                    "fetch": fetched,
                }


def render_pages(results: list[dict], output: Path) -> list[str]:
    """Feed selected URLs into PixelRAG's public renderer."""
    from pixelrag_render.render import render_urls

    directories = render_urls(
        [result["url"] for result in results],
        output / "tiles" / uuid.uuid4().hex,
        stems=[f"page-{i:02d}" for i in range(len(results))],
        workers=1,
        wait_network_idle=True,
    )
    tiles = [
        str(tile)
        for directory in directories
        for tile in sorted(Path(directory).glob("tile_*.jpg"))
    ]
    if not tiles:
        raise RuntimeError(
            "PixelRAG produced no screenshot tiles; check Chrome and page access"
        )
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language description of the pages")
    parser.add_argument(
        "--limit",
        type=int,
        choices=range(1, 6),
        default=3,
        help="Select 1 to 5 pages (default: 3)",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="Also fetch page text via MCP"
    )
    parser.add_argument(
        "--render", action="store_true", help="Also screenshot selected pages"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("demos/keenable_search/output")
    )
    args = parser.parse_args()
    result = asyncio.run(search_pages(args.query, args.limit, args.fetch))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "results.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    for page in result["selected"]:
        print(f"{page.get('title') or page['url']}\n  {page['url']}")
    if args.render:
        result["tiles"] = render_pages(result["selected"], args.output)
        manifest.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Rendered {len(result['tiles'])} tiles")
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
