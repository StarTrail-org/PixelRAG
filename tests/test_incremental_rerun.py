"""Incremental re-run safety: stale tile dirs must be re-rendered, not relabeled.

Position indices are assigned by source enumeration order, so they shift when
the source set changes between runs (a file added, removed, or renamed).
_needs_render must detect that an existing {idx}.png.tiles directory was
rendered from a *different* document and re-render it instead of letting the
stamp loop silently pair one document's pixels with another's metadata.
"""

import json

from pixelrag_index.pipelines import _needs_render
from pixelrag_index.sources.base import Document


def _make_tile_dir(tiles_dir, idx, manifest):
    tile_dir = tiles_dir / f"{idx}.png.tiles"
    tile_dir.mkdir(parents=True)
    (tile_dir / "tiles.json").write_text(json.dumps(manifest))
    return tile_dir


def test_missing_dir_needs_render(tmp_path):
    doc = Document(id="a", path="/src/a.md")
    assert _needs_render(tmp_path, 0, doc) is True


def test_matching_source_is_reused(tmp_path):
    _make_tile_dir(tmp_path, 0, {"source": "/src/a.md", "tiles": ["tile_0000.png"]})
    doc = Document(id="a", path="/src/a.md")
    assert _needs_render(tmp_path, 0, doc) is False


def test_legacy_url_field_is_reused(tmp_path):
    # Dirs stamped before the `source` field existed recorded the render URL.
    _make_tile_dir(tmp_path, 0, {"url": "https://example.com/x", "tiles": []})
    doc = Document(id="x", url="https://example.com/x")
    assert _needs_render(tmp_path, 0, doc) is False


def test_shifted_position_forces_rerender_and_removes_stale_dir(tmp_path):
    # Built over [a, c] -> dir 1 holds c. User adds b: position 1 is now b.
    stale = _make_tile_dir(tmp_path, 1, {"source": "/src/c.md", "tiles": []})
    doc_b = Document(id="b", path="/src/b.md")
    assert _needs_render(tmp_path, 1, doc_b) is True
    assert not stale.exists(), "stale dir must be removed before re-render"


def test_corrupt_manifest_forces_rerender(tmp_path):
    tile_dir = tmp_path / "0.png.tiles"
    tile_dir.mkdir()
    (tile_dir / "tiles.json").write_text('{"source": "/src/a.md", "til')  # truncated
    doc = Document(id="a", path="/src/a.md")
    assert _needs_render(tmp_path, 0, doc) is True
    assert not tile_dir.exists()


def test_corrupt_manifest_skips_article_in_chunk_stage(tmp_path):
    # A truncated tiles.json must not crash the chunk stage (it used to raise
    # an uncaught JSONDecodeError and fail the whole build via check=True).
    from pixelrag_embed.chunk import chunk_article

    article_dir = tmp_path / "0.png.tiles"
    article_dir.mkdir()
    (article_dir / "tiles.json").write_text('{"tiles": ["tile_0000.png"')  # truncated
    assert chunk_article(str(article_dir)) is None
