"""Selected PDF pages retain the same zero-based IDs as a full render."""

import json
import sys
from types import ModuleType

import pytest
from PIL import Image
from pixelrag_render import render_pdf


@pytest.mark.parametrize("pages", [None, [1], [3], [4], [2, 3], [1, 3], [2, 4], [4, 2]])
def test_pdf_page_indices_match_full_render(tmp_path, monkeypatch, pages):
    source = tmp_path / "document.pdf"
    source.touch()
    colors = ["red", "green", "blue", "yellow"]
    converter = ModuleType("pdf2image")

    def convert_from_path(**kwargs):
        first = kwargs.get("first_page", 1)
        last = kwargs.get("last_page", 4)
        return [
            Image.new("RGB", (16, 16), colors[i - 1]) for i in range(first, last + 1)
        ]

    converter.convert_from_path = convert_from_path
    monkeypatch.setitem(sys.modules, "pdf2image", converter)
    full_dir = render_pdf(source, tmp_path / "full")[0]
    selected_dir = render_pdf(source, tmp_path / "selected", pages=pages)[0]
    expected_ids = [0, 1, 2, 3] if pages is None else sorted(p - 1 for p in pages)
    expected_files = [f"tile_{i:04d}.jpg" for i in expected_ids]
    manifest = json.loads((selected_dir / "tiles.json").read_text())
    chunks = json.loads((selected_dir / "chunks.json").read_text())["chunks"]

    assert manifest["tiles"] == expected_files
    assert [c["tile_index"] for c in chunks] == expected_ids
    assert [c["file"] for c in chunks] == expected_files
    assert [c["tile"] for c in chunks] == expected_files
    assert sorted(p.name for p in selected_dir.glob("tile_*.jpg")) == expected_files
    for filename in expected_files:
        assert (selected_dir / filename).read_bytes() == (
            full_dir / filename
        ).read_bytes()
