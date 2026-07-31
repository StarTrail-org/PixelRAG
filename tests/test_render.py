"""Render smoke test — the README's `pixelshot <page> --output ./tiles` path.

Renders a local HTML file through the default CDP backend (patched headless
Chrome, auto-downloaded and cached in CI) and asserts tile images are produced.
This is the one user-facing capability the README promises, so it is tested end
to end rather than mocked.
"""

import json
from pathlib import Path

from pixelrag_render import render_file


def test_render_local_html_to_tiles(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body><h1>PixelRAG render smoke</h1>"
        + "<p>lorem ipsum dolor sit amet. </p>" * 60
        + "</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out)

    assert dirs, "render_file returned no tile directories"
    tile_dir = Path(dirs[0])
    tiles = sorted(tile_dir.glob("tile_*.jpg"))
    assert tiles, (
        f"no tile images produced in {tile_dir} "
        f"(contents: {[p.name for p in tile_dir.iterdir()]})"
    )


def test_page_taller_than_a_viewport_bounded_body_is_fully_tiled(tmp_path):
    """A page whose body is pinned to the viewport must still tile in full.

    Regression for issue #124. Sites that set ``html, body { height: 100% }``
    (Wikipedia's Vector 2022 skin among them) leave the article content
    overflowing the body box visibly, so ``body.getBoundingClientRect()`` is one
    viewport tall on a 20,000px page. The readiness probe used to clamp the page
    height to that rect, capturing a single tile and reporting the viewport as
    the page height.
    """
    body = "".join(f"<p>line {i:03d}</p>" for i in range(400))
    html = tmp_path / "viewport_bounded_body.html"
    html.write_text(
        '<!DOCTYPE html><html style="height:100%"><body style="height:100%">'
        f"{body}</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out, tile_height=1000, viewport_width=1280)

    tile_dir = Path(dirs[0])
    manifest = json.loads((tile_dir / "tiles.json").read_text())
    tiles = sorted(tile_dir.glob("tile_*.jpg"))

    # 400 paragraphs are several viewports tall whatever the default font is;
    # assert against the viewport rather than a brittle exact pixel count.
    assert manifest["page_height"] > 3000, (
        f"page_height {manifest['page_height']} is about one viewport — the "
        "content below the fold was never measured"
    )
    assert len(tiles) > 1, f"expected multiple tiles, got {[t.name for t in tiles]}"
