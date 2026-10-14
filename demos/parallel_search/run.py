"""Discover web pages with anonymous Parallel MCP, optionally render screenshots."""

import argparse
import asyncio
import json
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "PixelRAG/parallel-search-demo (https://github.com/StarTrail-org/PixelRAG)"


def tool_data(result) -> dict:
    """Check MCP errors and accept structured or JSON text tool responses."""
    if result.isError:
        raise RuntimeError(f"Parallel MCP tool failed: {result.content}")
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
    raise RuntimeError("Parallel MCP returned no JSON result")


def select_results(data: dict, limit: int) -> list[dict]:
    """Keep distinct HTTP(S) pages in the server's relevance order."""
    selected, seen = [], set()
    for result in data.get("results", []):
        url = result.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        if parsed.username or parsed.password or url in seen:
            continue
        seen.add(url)
        selected.append(result)
        if len(selected) == limit:
            break
    return selected


async def search_pages(
    objective: str, queries: list[str], limit: int, fetch: bool
) -> dict:
    """Discover and call tools over Streamable HTTP, without loading credentials."""
    session_id = str(uuid.uuid4())
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=60
    ) as client:
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
                required = {"web_search", "web_fetch"} if fetch else {"web_search"}
                if not required <= tools:
                    raise RuntimeError(
                        f"Parallel MCP is missing tools: {required - tools}"
                    )
                search = tool_data(
                    await session.call_tool(
                        "web_search",
                        {
                            "objective": objective,
                            "search_queries": queries,
                            "session_id": session_id,
                        },
                    )
                )
                selected = select_results(search, limit)
                if not selected:
                    raise RuntimeError("Search returned no usable HTTP(S) pages")
                fetched = None
                if fetch:
                    fetched = tool_data(
                        await session.call_tool(
                            "web_fetch",
                            {
                                "urls": [result["url"] for result in selected],
                                "search_queries": queries,
                                "session_id": session_id,
                            },
                        )
                    )
                return {
                    "objective": objective,
                    "search_queries": queries,
                    "selected": selected,
                    "search": search,
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
    parser.add_argument("objective", help="Information to find")
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Keyword query (repeat for related queries)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        choices=range(1, 6),
        default=3,
        help="Select 1 to 5 pages (default: 3)",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="Also fetch text excerpts via MCP"
    )
    parser.add_argument(
        "--render", action="store_true", help="Also screenshot selected pages"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("demos/parallel_search/output")
    )
    args = parser.parse_args()
    result = asyncio.run(
        search_pages(args.objective, args.query, args.limit, args.fetch)
    )
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
