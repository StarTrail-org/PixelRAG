"""Regression tests for bug fixes across configuration, rendering, indexing, and sources."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pixelrag_embed.chunk import chunk_article
from pixelrag_index.config import DEFAULT_CONFIG, load_config
from pixelrag_index.pipelines import _department_of
from pixelrag_index.sources.kiwix import KiwixServeManager
from pixelrag_index.sources.local import LocalSource


def test_load_config_deep_merge(tmp_path):
    """Deep merge ensures single-field overrides do not wipe out sibling default settings."""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        "embed:\n"
        "  device: cpu\n"
        "ingest:\n"
        "  quality: 92\n",
        encoding="utf-8",
    )
    loaded = load_config(str(cfg_file))

    # embed.device is overridden, but embed.model is preserved from DEFAULT_CONFIG
    assert loaded["embed"]["device"] == "cpu"
    assert loaded["embed"]["model"] == DEFAULT_CONFIG["embed"]["model"]

    # ingest.quality is overridden, but ingest.backend and tile_height are preserved
    assert loaded["ingest"]["quality"] == 92
    assert loaded["ingest"]["backend"] == DEFAULT_CONFIG["ingest"]["backend"]
    assert loaded["ingest"]["tile_height"] == DEFAULT_CONFIG["ingest"]["tile_height"]


def test_local_source_generates_valid_as_uri(tmp_path):
    """LocalSource must yield RFC 8089 as_uri() without raw backslashes or unencoded spaces."""
    doc_dir = tmp_path / "my docs"
    doc_dir.mkdir()
    html_file = doc_dir / "index test.html"
    html_file.write_text("<h1>Hello</h1>", encoding="utf-8")

    source = LocalSource(str(doc_dir))
    docs = list(source)
    assert len(docs) == 1
    doc = docs[0]

    assert doc.url.startswith("file:///")
    assert "\\" not in doc.url
    assert " " not in doc.url  # spaces must be percent-encoded
    assert "%20" in doc.url


def test_department_of_resolves_file_uri(tmp_path):
    """_department_of correctly extracts department on all platforms including Windows drive letters."""
    dept_dir = tmp_path / "finance"
    dept_dir.mkdir()
    doc_file = dept_dir / "report 2026.html"
    doc_file.write_text("<p>Finances</p>", encoding="utf-8")

    file_uri = doc_file.resolve().as_uri()
    art = {"url": file_uri}
    department = _department_of(art, str(tmp_path))
    assert department == "finance"


def test_pixelshot_forwards_extract_text_for_html(monkeypatch, tmp_path):
    """pixelshot CLI must forward extract_text flag when given local HTML files."""
    from pixelrag_render import render as render_mod

    html_file = tmp_path / "page.html"
    html_file.write_text("<p>content</p>", encoding="utf-8")

    called_kwargs = {}

    def fake_render_url(url, output_dir, **kwargs):
        called_kwargs.update(kwargs)
        return [tmp_path / "page.png.tiles"]

    monkeypatch.setattr(render_mod, "render_url", fake_render_url)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pixelshot", str(html_file), "-o", str(tmp_path / "tiles"), "--extract-text"],
    )

    render_mod.main()
    assert called_kwargs.get("extract_text") is True


def test_kiwix_kill_proc_cross_platform():
    """KiwixServeManager._kill_proc safely terminates process without os.killpg AttributeError."""
    mgr = KiwixServeManager.__new__(KiwixServeManager)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    # Should not raise AttributeError on Windows where os.killpg is absent
    mgr._kill_proc(mock_proc)
    assert mock_proc.terminate.called or mock_proc.wait.called


def test_chunk_article_converts_jpeg_to_png(tmp_path):
    """Fast-path chunking of JPEG tile into chunk_...png must create a valid PNG."""
    article_dir = tmp_path / "article.png.tiles"
    article_dir.mkdir()

    # Create a small JPEG tile
    img = Image.new("RGB", (800, 600), color="blue")
    tile_name = "tile_0000.jpg"
    img.save(article_dir / tile_name, format="JPEG")

    manifest = {
        "url": "https://example.com/test",
        "tiles": [tile_name],
        "page_height": 600,
        "viewport_width": 875,
    }
    (article_dir / "tiles.json").write_text(json.dumps(manifest))

    res = chunk_article(str(article_dir))
    assert res is not None
    assert res["num_chunks"] == 1

    chunk_file = article_dir / "chunk_0000_00.png"
    assert chunk_file.exists()

    # Verify chunk_file is actually a valid PNG, not raw JPEG bytes
    with Image.open(chunk_file) as chunk_img:
        assert chunk_img.format == "PNG"

    # Verify chunks.json exists and was written
    assert (article_dir / "chunks.json").exists()
