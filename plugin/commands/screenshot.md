---
name: screenshot
description: Screenshot a URL or document and read it visually
allowed-tools: "Bash, Read"
---

1. Run: `pixelshot "$ARGUMENTS" --output /tmp/pixelbrowse --tile-height 1568`
2. The output tile is at `/tmp/pixelbrowse/<domain>.png.tiles/tile_0000.jpg` — read it directly with the Read tool. Do not ls.
3. If text is too small to read, crop with Pillow (always available — it's a pixelshot dependency):
   `python3 -c "from PIL import Image; Image.open('<tile>').crop((x1,y1,x2,y2)).save('/tmp/pixelbrowse/crop.png')"`
4. Report what you see.
