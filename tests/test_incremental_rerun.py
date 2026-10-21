"""Incremental re-run safety: stale tile dirs must be re-rendered, not relabeled.

Position indices are assigned by source enumeration order, so they shift when
the source set changes between runs (a file added, removed, or renamed).
_needs_render must detect that an existing {idx}.png.tiles directory was
rendered from a *different* document and re-render it instead of letting the
stamp loop silently pair one document's pixels with another's metadata.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from pixelrag_index.pipelines import _needs_render
from pixelrag_index.sources.base import Document


def _make_tile_dir(tiles_dir, idx, manifest):
    tile_dir = tiles_dir / f"{idx}.png.tiles"
    tile_dir.mkdir(parents=True)
    (tile_dir / "tiles.json").write_text(json.dumps(manifest))
    for name in manifest.get("tiles", []):
        Image.new("RGB", (32, 32), "red").save(tile_dir / name)
    return tile_dir


def test_missing_dir_needs_render(tmp_path):
    doc = Document(id="a", path="/src/a.md")
    assert _needs_render(tmp_path, 0, doc) is True


@pytest.mark.parametrize("completion", [{"complete": True}, {}])
def test_matching_source_is_reused(tmp_path, completion):
    tile_dir = _make_tile_dir(
        tmp_path,
        0,
        {"source": "/src/a.md", "tiles": ["tile_0000.png"], **completion},
    )
    doc = Document(id="a", path="/src/a.md")
    assert _needs_render(tmp_path, 0, doc) is False
    assert (tile_dir / "tile_0000.png").is_file()


@pytest.mark.parametrize("identity_field", ["source", "url"])
def test_incomplete_matching_capture_is_removed(tmp_path, identity_field):
    doc = Document(id="a", url="https://example.com/a")
    tile_dir = _make_tile_dir(
        tmp_path,
        0,
        {identity_field: doc.url, "tiles": ["tile_0000.png"], "complete": False},
    )
    assert (tile_dir / "tile_0000.png").is_file()
    assert _needs_render(tmp_path, 0, doc) is True
    assert not tile_dir.exists()


def test_legacy_url_field_is_reused(tmp_path):
    # Dirs stamped before the `source` field existed recorded the render URL.
    _make_tile_dir(
        tmp_path, 0, {"url": "https://example.com/x", "tiles": ["tile_0000.png"]}
    )
    doc = Document(id="x", url="https://example.com/x")
    assert _needs_render(tmp_path, 0, doc) is False


def test_incomplete_capture_cleanup_failure_stops_build(tmp_path, monkeypatch):
    from pixelrag_index import pipelines
    from pixelrag_render import render

    doc = Document(id="a", url="https://example.com/a")
    tile_dir = _make_tile_dir(
        tmp_path / "tiles",
        0,
        {"source": doc.url, "tiles": ["tile_0000.png"], "complete": False},
    )

    def fail_cleanup(path, ignore_errors=False):
        assert path == tile_dir
        if not ignore_errors:
            raise PermissionError("cache is not writable")

    renderer = Mock()
    stages = Mock()
    monkeypatch.setattr(pipelines.shutil, "rmtree", fail_cleanup)
    monkeypatch.setattr(pipelines, "make_source", lambda config: iter([doc]))
    monkeypatch.setattr(render, "render_urls", renderer)
    monkeypatch.setattr(pipelines.subprocess, "run", stages)

    with pytest.raises(PermissionError, match="cache is not writable"):
        pipelines.build({"source": {"type": "web"}, "output": str(tmp_path)})
    renderer.assert_not_called()
    stages.assert_not_called()


def test_shifted_position_forces_rerender_and_removes_stale_dir(tmp_path):
    # Built over [a, c] -> dir 1 holds c. User adds b: position 1 is now b.
    stale = _make_tile_dir(
        tmp_path, 1, {"source": "/src/c.md", "tiles": ["tile_0000.png"]}
    )
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


def test_build_regenerates_incomplete_capture_and_chunks(tmp_path, monkeypatch):
    from pixelrag_embed.chunk import chunk_article
    from pixelrag_index import pipelines
    from pixelrag_render import render

    doc = Document(id="a", url="https://example.com/a")
    config = {"source": {"type": "web"}, "output": str(tmp_path)}
    tile_dir = tmp_path / "tiles" / "0.png.tiles"
    chunk_results = []

    def capture(urls, output_dir, *, stems, **kwargs):
        assert urls == [doc.url]
        assert stems == ["0"]
        # Regeneration must remove the old manifest, tiles AND derived chunks.
        assert not tile_dir.exists()
        complete = renderer.call_count > 1
        names = ["tile_0000.png"] if complete else ["tile_0000.png", "tile_0001.png"]
        _make_tile_dir(
            Path(output_dir),
            0,
            {
                "url": doc.url,
                "page_height": 32 if complete else 96,
                "tile_height": 32,
                "viewport_width": 32,
                "tiles": names,
                "complete": complete,
            },
        )
        return [tile_dir]

    def run_stage(cmd, *, check):
        assert check is True
        if cmd[2] == "pixelrag_embed.chunk":
            chunk_results.append(chunk_article(str(tile_dir)))
        # Embedding and index services are outside this regression's scope.

    renderer = Mock(side_effect=capture)
    monkeypatch.setattr(render, "render_urls", renderer)
    monkeypatch.setattr(pipelines, "make_source", lambda config: iter([doc]))
    monkeypatch.setattr(pipelines.subprocess, "run", run_stage)

    pipelines.build(config)
    assert renderer.call_count == 1
    assert chunk_results[-1] is not None
    assert (tile_dir / "chunk_0001_00.png").is_file()

    pipelines.build(config)
    assert renderer.call_count == 2
    assert chunk_results[-1] is not None, "chunk_article must not reuse old chunks"
    assert not (tile_dir / "tile_0001.png").exists()
    assert not (tile_dir / "chunk_0001_00.png").exists()
    chunks = json.loads((tile_dir / "chunks.json").read_text())
    assert [c["file"] for c in chunks["chunks"]] == ["chunk_0000_00.png"]
    assert chunks["article_id"] == 0
    assert json.loads((tile_dir / "tiles.json").read_text())["complete"] is True

    pipelines.build(config)
    assert renderer.call_count == 2
    assert chunk_results[-1] is None
