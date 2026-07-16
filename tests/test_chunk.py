"""Chunking tests — 2D tiling keeps every chunk within the render width.

The embedder skips any chunk wider than the render width (875px), so the chunker
must split wide tiles (PDFs, landscape pages) along the width as well as the
height. Narrow web tiles (<= viewport_width) must keep their old single-column
height-strip layout unchanged so existing indexes stay reproducible.
"""

import json

from PIL import Image

from pixelrag_embed.chunk import CHUNK_HEIGHT, chunk_article


def _make_tile(dir_path, w, h, viewport_width=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (255, 255, 255)).save(dir_path / "tile_0000.png")
    meta = {"tiles": ["tile_0000.png"], "tile_height": 8192}
    if viewport_width is not None:
        meta["viewport_width"] = viewport_width
    (dir_path / "tiles.json").write_text(json.dumps(meta))
    return dir_path


def _chunks(dir_path):
    return json.loads((dir_path / "chunks.json").read_text())["chunks"]


def test_narrow_tile_unchanged(tmp_path):
    # 875-wide tall web tile: single-column height strips, exactly as before.
    d = _make_tile(tmp_path / "a.png.tiles", 875, 3000)
    chunk_article(str(d))
    cs = _chunks(d)
    assert [c["file"] for c in cs] == [
        "chunk_0000_00.png",
        "chunk_0000_01.png",
        "chunk_0000_02.png",
    ]
    assert [c["y_offset"] for c in cs] == [0, CHUNK_HEIGHT, 2 * CHUNK_HEIGHT]
    assert all(c["x_offset"] == 0 and c["width"] == 875 for c in cs)


def test_short_narrow_tile_copied_whole(tmp_path):
    # <= viewport_width and <= CHUNK_HEIGHT: one chunk, tile copied verbatim.
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    cs = _chunks(d)
    assert len(cs) == 1
    assert cs[0]["file"] == "chunk_0000_00.png"
    assert cs[0]["width"] == 600 and cs[0]["height"] == 500


def test_wide_tile_split_into_columns(tmp_path):
    # Letter PDF at 200 DPI (~1700x2200): 2 columns x 3 rows, all <= 875 wide.
    d = _make_tile(tmp_path / "a.png.tiles", 1700, 2200)
    chunk_article(str(d))
    cs = _chunks(d)
    assert len(cs) == 6
    assert all(c["width"] <= 875 for c in cs)
    # every produced chunk file is on disk at its recorded size
    for c in cs:
        img = Image.open(d / c["file"])
        assert img.size == (c["width"], c["height"])
    # columns are full viewport_width (875) with the remainder last, covering
    # the full width with no gaps
    row0 = sorted((c for c in cs if c["y_offset"] == 0), key=lambda c: c["x_offset"])
    assert [c["width"] for c in row0] == [875, 825]
    assert row0[0]["x_offset"] == 0
    assert row0[-1]["x_offset"] + row0[-1]["width"] == 1700


def test_very_wide_tile_never_exceeds_width(tmp_path):
    # 19-inch landscape figure (3800px): many columns, still all <= 875, no skips.
    d = _make_tile(tmp_path / "a.png.tiles", 3800, 2200)
    chunk_article(str(d))
    cs = _chunks(d)
    assert cs, "wide tile must still produce chunks (not be skipped)"
    assert max(c["width"] for c in cs) <= 875


def test_short_wide_tile_splits_width(tmp_path):
    # Short but wide (was copied whole -> too wide -> skipped). Now splits width.
    d = _make_tile(tmp_path / "a.png.tiles", 1700, 500)
    chunk_article(str(d))
    cs = _chunks(d)
    assert len(cs) == 2
    assert all(c["width"] <= 875 and c["height"] == 500 for c in cs)
