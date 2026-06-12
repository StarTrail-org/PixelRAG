---
name: pixelbrowse
description: |
  Screenshot and visually read any web page or document using pixelshot.
  Use instead of fetching raw HTML when you need to see what a page looks like,
  read visual content (charts, diagrams, infographics), check layouts, or verify UI.
  Triggers: "look at this page", "screenshot", "what does this site look like",
  "check the UI", "read this visually", "view this URL", viewing web content.
allowed-tools: "Bash, Read"
---

# PixelBrowse — Screenshot-based Web Reading

Use `pixelshot` to capture any URL or document as tiled JPEG images, then read the images visually.

Requires the `pixelshot` command on `PATH`. If `pixelshot` is not found, install it
(isolated, on `PATH`): `uv tool install pixelrag` (or `pipx install pixelrag`, or
`pip install pixelrag`) — then retry. Don't go hunting for it in project venvs.

## How to use

```bash
# Screenshot a URL (optimized for Claude's vision: 1568px tile height)
pixelshot <url> --output /tmp/pixelbrowse --tile-height 1568 --wait-network-idle

# Screenshot multiple URLs in parallel
pixelshot <url1> <url2> --output /tmp/pixelbrowse --tile-height 1568 --wait-network-idle --workers 4

# Wider viewport for desktop layouts
pixelshot <url> --output /tmp/pixelbrowse --tile-height 1568 --viewport-width 1280 --wait-network-idle

# Render a PDF
pixelshot document.pdf --output /tmp/pixelbrowse
```

IMPORTANT: Always use `--tile-height 1568` for screenshots you will read visually.
Claude's vision model downscales images with long edge > 1568px (Sonnet/Haiku) or 2576px (Opus).
The default 8192px tile height will be downscaled and text becomes unreadable.

IMPORTANT: Always use `--wait-network-idle` for URLs. Without it, JavaScript-heavy
pages (most modern sites / single-page apps) are captured before they finish rendering
and come back blank or half-empty. It waits for the page's load event plus a brief
network-quiet window so client-rendered content is actually on screen.

After rendering, read the tile images from the output directory to visually understand the content.

## Workflow

1. Run `pixelshot <url> --output /tmp/pixelbrowse --tile-height 1568 --wait-network-idle`
2. Read `/tmp/pixelbrowse/<domain>.png.tiles/tile_0000.jpg` directly (no need to ls — the naming is deterministic)
3. If the page is long, also read tile_0001.jpg, tile_0002.jpg, etc.

Output path pattern: `/tmp/pixelbrowse/<sanitized-url>.png.tiles/tile_NNNN.jpg`
- For `https://news.ycombinator.com` → `/tmp/pixelbrowse/news.ycombinator.com.png.tiles/tile_0000.jpg`
- For `https://example.com/page` → `/tmp/pixelbrowse/example.com_page.png.tiles/tile_0000.jpg`

Do NOT run `ls` — just read tile_0000.jpg. If it doesn't exist, the page had no content.

## Crop & Zoom

If text or details are too small to read, crop the region of interest and re-read at full resolution.
Pillow is always available (it's a pixelshot dependency):

```bash
python3 -c "from PIL import Image; Image.open('<tile_path>').crop((x1, y1, x2, y2)).save('/tmp/pixelbrowse/crop.png')"
```

- Coordinates are in pixels from the top-left corner of the tile
- Crop to roughly 800x800 or smaller for maximum clarity
- You can crop multiple times to inspect different regions
- Read the cropped image with the Read tool just like any other image

Use this whenever you see content but can't make out the details — tables, small labels, fine print, chart axes, etc.

## Tips

- Output is tiled JPEG images — tile_0000.jpg is the top, higher numbers go down the page
- Use `--viewport-width 1280` for desktop layouts, default 875 for mobile/article width
- Supports URLs (http/https), local HTML files, PDFs, and images
- Backend options: `--backend cdp` (default, fastest) or `--backend playwright`
