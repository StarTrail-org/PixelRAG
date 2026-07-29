# pixelbrowse — opencode plugin

Give opencode eyes: screenshot any URL or document with `pixelshot` and read it visually.
Adds a `screenshot` tool that renders a page to image tiles (Playwright/CDP) so the agent
sees charts, diagrams, tables, and layout the way a person does.

## Install

```bash
pip install pixelrag   # provides the pixelshot command (or: uv tool install pixelrag)
```

Then add the plugin to your `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["opencode-pixelbrowse"]
}
```

opencode installs npm plugins automatically at startup. For local development from a
clone, symlink or copy `index.js` into `~/.config/opencode/plugins/` (global) or
`.opencode/plugins/` (project) — files there are loaded directly at startup.

## Use

Just ask opencode to look at a page:

```
opencode run "screenshot https://news.ycombinator.com and summarize the top stories"
```

The agent calls the `screenshot` tool with a URL (or a local HTML file, PDF, or image),
gets back the tile image paths, and reads them visually. Optional tool args: `output`
(tile directory, default `/tmp/pixelbrowse`) and `viewportWidth` (default 875; use 1280
for desktop layouts).

No MCP server, no backend — the plugin just calls `pixelshot` on your machine.

## Publishing (maintainers)

The package is published manually from this directory:

```bash
npm publish --access public
```
