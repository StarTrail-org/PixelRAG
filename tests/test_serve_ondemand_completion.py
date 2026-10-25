"""Exercise real on-demand chunking/publication without launching Chrome."""

import atexit
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from pixelrag_embed import chunk
from pixelrag_serve import render_ondemand


@pytest.fixture
def capture(tmp_path, monkeypatch):
    od = render_ondemand.OnDemandTiles("http://example.invalid", "book", str(tmp_path))
    manifest = {"text": json.dumps({"tiles": ["tile_0000.png"], "complete": False})}

    def render(args, **kwargs):
        output = Path(args[args.index("--output") + 1]) / "page.png.tiles"
        output.mkdir()
        Image.new("RGB", (16, 16), "green").save(output / "tile_0000.png")
        if manifest["text"] is not None:
            (output / "tiles.json").write_text(manifest["text"])

    renderer = Mock(side_effect=render)
    monkeypatch.setattr(render_ondemand.subprocess, "run", renderer)
    monkeypatch.setattr(od, "_ensure_chrome", Mock(return_value="http://localhost:1"))
    try:
        yield od, manifest, renderer
    finally:
        atexit.unregister(od._kill_chrome)


def _staging(od):
    return Path(od.cache_dir) / ".render_7"


def _prior_cache(od):
    dest = Path(od._article_dir(7))
    dest.mkdir()
    Image.new("RGB", (16, 16), "red").save(dest / "chunk_0000_01.png")
    (dest / "tiles.json").write_text('{"complete": true, "tiles": []}')
    return dest, {p.name: p.read_bytes() for p in dest.iterdir()}


def test_incomplete_capture_is_not_published(capture):
    od, _, _ = capture
    assert od.chunk_path(7, "Example", 0, 0) is None
    assert not Path(od._article_dir(7)).exists()
    assert not _staging(od).exists()


@pytest.mark.parametrize("completion", [True, "missing", None, 0])
def test_complete_and_legacy_capture_publish_real_chunks(capture, completion):
    od, manifest, renderer = capture
    data = {"tiles": ["tile_0000.png"]}
    if completion != "missing":
        data["complete"] = completion
    manifest["text"] = json.dumps(data)
    result = od.chunk_path(7, "Example", 0, 0)
    assert result is not None
    with Image.open(result) as image:
        assert image.getpixel((0, 0)) == (0, 128, 0)
    dest = Path(od._article_dir(7))
    assert json.loads((dest / "tiles.json").read_text()) == data
    assert (dest / "chunks.json").is_file()
    assert not _staging(od).exists()
    assert od.chunk_path(7, "Example", 0, 0) == result
    renderer.assert_called_once()


@pytest.mark.parametrize(
    "manifest_text",
    [
        None,
        "",
        "{",
        "[]",
        "null",
        "false",
        json.dumps({"tiles": ["tile_0000.png"], "complete": False}),
    ],
)
def test_rejection_preserves_prior_cache_and_skips_chunking(
    capture, monkeypatch, manifest_text
):
    od, manifest, _ = capture
    manifest["text"] = manifest_text
    dest, before = _prior_cache(od)
    chunker = Mock(wraps=chunk.chunk_article)
    monkeypatch.setattr(chunk, "chunk_article", chunker)
    assert od.chunk_path(7, "Example", 0, 0) is None
    chunker.assert_not_called()
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before
    assert not _staging(od).exists()


def test_incomplete_capture_can_be_retried_successfully(capture):
    od, manifest, renderer = capture
    assert od.chunk_path(7, "Example", 0, 0) is None
    manifest["text"] = json.dumps({"tiles": ["tile_0000.png"], "complete": True})
    result = od.chunk_path(7, "Example", 0, 0)
    assert result is not None and Path(result).is_file()
    assert renderer.call_count == 2
    assert not _staging(od).exists()


@pytest.mark.parametrize("failure", ["chrome", "renderer", "timeout", "chunker"])
def test_errors_preserve_prior_cache_and_clean_staging(capture, monkeypatch, failure):
    od, manifest, renderer = capture
    dest, before = _prior_cache(od)
    manifest["text"] = json.dumps({"tiles": ["tile_0000.png"], "complete": True})
    if failure == "chrome":
        od._ensure_chrome.side_effect = RuntimeError("Chrome startup failed")
    elif failure in ("renderer", "timeout"):
        render = renderer.side_effect

        def fail(args, **kwargs):
            render(args, **kwargs)
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 1)
            raise subprocess.CalledProcessError(1, args)

        renderer.side_effect = fail
    else:
        real_chunk = chunk.chunk_article

        def fail(article_dir):
            real_chunk(article_dir)
            raise OSError("chunk write failed")

        monkeypatch.setattr(chunk, "chunk_article", fail)
    assert od.chunk_path(7, "Example", 0, 0) is None
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before
    assert not _staging(od).exists()


def test_existing_incomplete_cache_hit_is_not_invalidated(capture):
    # Publication validation is not a migration of previously cached entries.
    od, _, renderer = capture
    dest, _ = _prior_cache(od)
    (dest / "tiles.json").write_text('{"complete": false}')
    assert od.chunk_path(7, "Example", 0, 1) == str(dest / "chunk_0000_01.png")
    renderer.assert_not_called()


def test_renderer_without_output_preserves_prior_cache(capture):
    od, _, renderer = capture
    dest, before = _prior_cache(od)
    renderer.side_effect = None
    assert od.chunk_path(7, "Example", 0, 0) is None
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before
    assert not _staging(od).exists()
