---
name: screenshot
description: Screenshot a URL or document and read it visually
allowed-tools: "Bash, Read"
---

1. Run: `pixelshot $ARGUMENTS --output /tmp/pixelbrowse --tile-height 1568`
2. Read `/tmp/pixelbrowse/<domain>.png.tiles/tiles.json`, then read every tile it lists with the Read tool (they are `tile_NNNN.jpg` in that same directory, top of the page first). If the manifest says `"complete": false`, the page was only partly captured — report that alongside what you saw.
3. If text is too small to read, crop with Pillow (always available — it's a pixelshot dependency):
   `python3 -c "from PIL import Image; Image.open('<tile>').crop((x1,y1,x2,y2)).save('/tmp/pixelbrowse/crop.png')"`
4. Report what you see.
