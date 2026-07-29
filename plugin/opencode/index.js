import { tool } from "@opencode-ai/plugin"
import { existsSync, readdirSync, statSync } from "node:fs"
import { join } from "node:path"

const DESCRIPTION = `Screenshot a web page or document with pixelshot and read it visually.

Use instead of fetching raw HTML when you need to see what a page looks like,
read visual content (charts, diagrams, infographics, tables), check layouts, or verify UI.

Runs pixelshot (Playwright/CDP) locally and returns the tile image paths — read them
with the Read tool, in order. tile_0000.jpg is the top of the page; higher numbers go
further down. Tiles are rendered at 1568px height for optimal vision-model readability.

If text or details are too small to read, crop the region of interest from a tile and
re-read at full resolution. Pillow is always available (it's a pixelshot dependency):
  python3 -c "from PIL import Image; Image.open('<tile>').crop((x1, y1, x2, y2)).save('/tmp/pixelbrowse/crop.png')"
Crop to roughly 800x800 or smaller for maximum clarity.`

const INSTALL_HINT =
  "`pixelshot` was not found on PATH. Install it (isolated, on PATH): " +
  "`uv tool install pixelrag` (or `pipx install pixelrag`, or `pip install pixelrag`), then retry."

// Newest *.tiles directory under `output` (pixelshot writes <output>/<name>.tiles/tile_NNNN.jpg).
function latestTilesDir(output) {
  if (!existsSync(output)) return null
  const dirs = readdirSync(output)
    .filter((name) => name.endsWith(".tiles"))
    .map((name) => join(output, name))
    .filter((path) => statSync(path).isDirectory())
  if (dirs.length === 0) return null
  return dirs.reduce((a, b) => (statSync(a).mtimeMs >= statSync(b).mtimeMs ? a : b))
}

export const PixelbrowsePlugin = async ({ $ }) => {
  return {
    tool: {
      screenshot: tool({
        description: DESCRIPTION,
        args: {
          target: tool.schema
            .string()
            .describe("URL (http/https), or path to a local HTML file, PDF, or image"),
          output: tool.schema
            .string()
            .optional()
            .describe("Output directory for tiles (default /tmp/pixelbrowse)"),
          viewportWidth: tool.schema
            .number()
            .optional()
            .describe(
              "Viewport width in px (default 875, mobile/article width). Use 1280 for desktop layouts."
            ),
        },
        async execute(args) {
          const output = args.output ?? "/tmp/pixelbrowse"

          try {
            await $`which pixelshot`.quiet()
          } catch {
            return `ERROR: ${INSTALL_HINT}`
          }

          const cliArgs = [args.target, "--output", output, "--tile-height", "1568"]
          if (args.target.startsWith("http://") || args.target.startsWith("https://")) {
            cliArgs.push("--wait-network-idle")
          }
          if (args.viewportWidth) cliArgs.push("--viewport-width", String(args.viewportWidth))

          let result
          try {
            result = await $`pixelshot ${cliArgs}`.quiet()
          } catch (err) {
            return `ERROR: pixelshot failed:\n${err.stderr?.toString() ?? err.message}`
          }

          const tilesDir = latestTilesDir(output)
          if (!tilesDir) {
            return `ERROR: pixelshot ran but produced no tiles in ${output}.\n${result.stdout.toString()}`
          }

          const tiles = readdirSync(tilesDir)
            .filter((name) => /^tile_\d+\.jpg$/.test(name))
            .sort()
            .map((name) => join(tilesDir, name))

          if (tiles.length === 0) return `ERROR: no tile images found in ${tilesDir}`

          return [
            `Rendered ${args.target} to ${tiles.length} tile(s). Read them with the Read tool, in this order:`,
            ...tiles,
          ].join("\n")
        },
      }),
    },
  }
}
