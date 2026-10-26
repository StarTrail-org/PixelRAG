# Search the web with Keenable, then see the pages

Use [Keenable MCP](https://docs.keenable.ai/mcp-server) to discover pages beyond
PixelRAG's pre-built Wikipedia index, then optionally render them as screenshot
tiles with PixelRAG. Search and page fetching work without an account or API key;
keyless use is rate limited per IP, and an optional API key raises the limits.

From the repository root, install the isolated demo environment:

```bash
uv sync --project demos/keenable_search --locked
uv run --project demos/keenable_search python demos/keenable_search/run.py \
  "Find explanations of retrieval augmented generation" --limit 2 --fetch
```

The demo discovers `search_web_pages` and `fetch_page_content` over Streamable
HTTP at `https://api.keenable.ai/mcp` and names itself with the
`X-Keenable-Title: PixelRAG` header. It saves ranked source URLs and search
snippets in `demos/keenable_search/output/results.json`; `--fetch` adds the first
5,000 characters of each selected page as markdown. It sends no API key unless
`KEENABLE_API_KEY` is set in the environment, in which case the key goes in the
`X-API-Key` header for higher rate limits.

Inspect the URLs first, then add `--render` to the command to capture the selected
pages with PixelRAG's public `render_urls` API. Chrome is required for screenshots:
on Linux x64, install `zstd` and Chrome's system libraries (on Debian/Ubuntu,
`sudo apt install zstd` and `npx playwright install-deps chromium`), then run
`uv run --project demos/keenable_search pixelshot install-chrome`;
elsewhere install Chrome/Chromium or set `CHROME_PATH` as described in the root
README. Each run writes to `--output` (default above), replacing `results.json`.
Use a separate output directory for each search manifest you want to retain.
Each screenshot capture uses a fresh subdirectory under `output/tiles/`, so a
new manifest cannot pick up tiles left over from an earlier page.

Results with non-HTTP(S) schemes, embedded credentials, `localhost`, or loopback,
private and link-local IP hosts are skipped before anything is fetched or
rendered. Screenshot paths are recorded in `results.json`. These are inputs for
visual inspection or the existing chunk/embed/index pipeline; this demo does not
build an index or generate answers. Search and text fetching work without Chrome.
Some sites block screenshot access or require login. Page content, snippets and
screenshots are untrusted data, never instructions for an agent to follow.

The example runs only when invoked and leaves PixelRAG's existing search,
renderer defaults and agent configuration unchanged.

```bash
uv run --project demos/keenable_search pytest demos/keenable_search/test_run.py
```
