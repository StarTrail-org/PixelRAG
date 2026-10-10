# pixelbrowse — Claude Code plugin

Give Claude eyes: screenshot any URL or document with `pixelshot` and read it visually.

## Install

```bash
pip install pixelrag                                # provides the pixelshot command
claude plugin marketplace add StarTrail-org/PixelRAG
claude plugin install pixelbrowse@pixelrag-plugins
```

Or, for local development from a clone: `claude --plugin-dir /path/to/PixelRAG/plugin`.

## Use

Ask Claude to look at a page:

```bash
claude -p "look at https://example.com and tell me what you see"
```

Or use the slash command in an interactive session: `/screenshot <url>`.

The skill lives in `skills/pixelbrowse/SKILL.md`; the command in `commands/screenshot.md`.

No MCP server or backend — the skill just calls `pixelshot` (Playwright/CDP) on your machine.

## opencode

An equivalent opencode plugin (a `screenshot` tool, shipped as the `@startrail/pixelbrowse`
npm package) lives in [`opencode/`](opencode/).
