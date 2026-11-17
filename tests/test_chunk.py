"""Chunking tests — 2D tiling keeps every chunk within the render width.

The embedder skips any chunk wider than the render width (875px), so the chunker
must split wide tiles (PDFs, landscape pages) along the width as well as the
height. Narrow web tiles (<= viewport_width) must keep their old single-column
height-strip layout unchanged so existing indexes stay reproducible.
"""

import json

import pytest
from PIL import Image
from pixelrag_embed.chunk import CHUNK_HEIGHT, chunk_article, process_shard


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


@pytest.mark.parametrize("old_width,new_width", [(875, 400), (400, 875)])
def test_force_rechunks_when_viewport_changes(tmp_path, old_width, new_width):
    d = _make_tile(tmp_path / "a.png.tiles", 875, 500, old_width)
    chunk_article(str(d))
    meta = json.loads((d / "tiles.json").read_text())
    meta["viewport_width"] = new_width
    (d / "tiles.json").write_text(json.dumps(meta))

    assert chunk_article(str(d), force=True) is not None
    manifest = json.loads((d / "chunks.json").read_text())
    assert manifest["viewport_width"] == new_width
    cs = manifest["chunks"]
    expected_widths = [400, 400, 75] if new_width == 400 else [875]
    assert [c["width"] for c in cs] == expected_widths
    assert sorted(p.name for p in d.glob("chunk_*.png")) == sorted(
        c["file"] for c in cs
    )
    for c in cs:
        with Image.open(d / c["file"]) as img:
            assert img.size == (c["width"], c["height"])


@pytest.mark.parametrize(
    "field,value", [("page_height", 500), ("tile_height", 4096), ("article_id", 42)]
)
def test_force_refreshes_manifest_metadata(tmp_path, field, value):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    meta = json.loads((d / "tiles.json").read_text())
    meta[field] = value
    (d / "tiles.json").write_text(json.dumps(meta))

    assert chunk_article(str(d), force=True) is not None
    assert json.loads((d / "chunks.json").read_text())[field] == value


def test_force_removes_obsolete_article_id(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    meta = json.loads((d / "tiles.json").read_text())
    (d / "tiles.json").write_text(json.dumps({**meta, "article_id": 42}))
    chunk_article(str(d))
    (d / "tiles.json").write_text(json.dumps(meta))

    assert chunk_article(str(d), force=True) is not None
    assert "article_id" not in json.loads((d / "chunks.json").read_text())


def test_force_skips_unchanged_inputs(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    assert chunk_article(str(d), force=True) is None
    assert before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()
    }


def test_force_treats_explicit_default_width_as_unchanged(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    meta = json.loads((d / "tiles.json").read_text())
    meta["viewport_width"] = 875
    (d / "tiles.json").write_text(json.dumps(meta))
    assert chunk_article(str(d), force=True) is None


def test_non_forced_call_still_reuses_existing_chunks(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 875, 500)
    chunk_article(str(d))
    before = (d / "chunks.json").read_bytes()
    meta = json.loads((d / "tiles.json").read_text())
    meta["viewport_width"] = 400
    (d / "tiles.json").write_text(json.dumps(meta))
    assert chunk_article(str(d)) is None
    assert (d / "chunks.json").read_bytes() == before


def test_force_dry_run_reports_new_layout_without_writes(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 875, 500)
    chunk_article(str(d))
    meta = json.loads((d / "tiles.json").read_text())
    meta["viewport_width"] = 400
    (d / "tiles.json").write_text(json.dumps(meta))
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    result = chunk_article(str(d), force=True, dry_run=True)
    assert result is not None
    assert result["num_chunks"] == 3
    assert result["files_written"] == 0
    assert before == {p.name: p.read_bytes() for p in d.iterdir()}


def test_force_recreates_missing_chunk(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    (d / _chunks(d)[0]["file"]).unlink()
    assert chunk_article(str(d), force=True) is not None
    assert all((d / c["file"]).exists() for c in _chunks(d))


def test_shard_force_propagates_metadata_changes(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 875, 500)
    process_shard(str(tmp_path), progress=False)
    meta = json.loads((d / "tiles.json").read_text())
    meta["viewport_width"] = 400
    (d / "tiles.json").write_text(json.dumps(meta))

    result = process_shard(str(tmp_path), force=True, progress=False)
    assert result["chunked"] == 1
    assert result["chunks"] == 3
    assert [c["width"] for c in _chunks(d)] == [400, 400, 75]


def test_force_refreshes_incompatible_chunk_height(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    manifest = json.loads((d / "chunks.json").read_text())
    manifest["chunk_height"] = 128
    (d / "chunks.json").write_text(json.dumps(manifest))
    assert chunk_article(str(d), force=True) is not None
    assert json.loads((d / "chunks.json").read_text())["chunk_height"] == CHUNK_HEIGHT


def test_force_refreshes_changed_pixels(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    Image.new("RGB", (600, 500), "black").save(d / "tile_0000.png")
    assert chunk_article(str(d), force=True) is not None
    with Image.open(d / _chunks(d)[0]["file"]) as img:
        assert img.getpixel((0, 0)) == (0, 0, 0)


def test_force_keeps_chunks_when_source_tiles_are_missing(tmp_path):
    d = _make_tile(tmp_path / "a.png.tiles", 600, 500)
    chunk_article(str(d))
    (d / "tile_0000.png").unlink()
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert chunk_article(str(d), force=True) is None
    assert before == {p.name: p.read_bytes() for p in d.iterdir()}
