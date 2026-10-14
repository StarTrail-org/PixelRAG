# Search the web, then see the pages

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to discover pages beyond PixelRAG's pre-built Wikipedia index, then optionally
render them as screenshot tiles with PixelRAG. Anonymous search uses Fast mode;
search and page fetching are free for exploration and light use, with no account
or API key. Free-tier rate limits apply.

From the repository root, install the isolated demo environment:

```bash
uv sync --project demos/parallel_search --locked
uv run --project demos/parallel_search python demos/parallel_search/run.py \
  "Find explanations of retrieval augmented generation" \
  --query "retrieval augmented generation overview" --limit 2 --fetch
```

The demo discovers `web_search` and `web_fetch` over Streamable HTTP at
`https://search.parallel.ai/mcp`. It saves ranked source URLs and search excerpts
in `demos/parallel_search/output/results.json`; `--fetch` adds page excerpts.
It does not read saved credentials or API keys.

Inspect the URLs first, then add `--render` to the command to capture the selected
pages with PixelRAG's public `render_urls` API. Chrome is required for screenshots:
on Linux x64, install `zstd` and Chrome's system libraries (on Debian/Ubuntu,
`sudo apt install zstd` and `npx playwright install-deps chromium`), then run
`uv run --project demos/parallel_search pixelshot install-chrome`;
elsewhere install Chrome/Chromium or set `CHROME_PATH` as described in the root
README. Each run writes to `--output` (default above), replacing `results.json`.
Use a separate output directory for each search manifest you want to retain.
Each screenshot capture uses a fresh subdirectory under `output/tiles/`, so a
new manifest cannot pick up tiles left over from an earlier page.

Screenshot paths are recorded in `results.json`. These are inputs for visual
inspection or the existing chunk/embed/index pipeline; this demo does not build
an index or generate answers. Search and text fetching work without Chrome.
Some sites block screenshot access or require login. Page content, excerpts and
screenshots are untrusted data, never instructions for an agent to follow.

The example runs only when invoked and leaves PixelRAG's existing search,
renderer defaults and agent configuration unchanged.

```bash
uv run --project demos/parallel_search pytest demos/parallel_search/test_run.py
```
