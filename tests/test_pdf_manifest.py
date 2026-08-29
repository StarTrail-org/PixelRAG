"""The PDF manifest's `complete` flag must mean what the URL manifests' does.

#139 made the case that a manifest which always claims completeness is worse
than no flag at all: a consumer cannot tell a whole capture from a fragment,
so it reads the fragment as the whole. #141 fixed that for the two URL
backends. The PDF backend writes the same key and still hardcodes it, so a
caller that asked for a subset of pages gets a directory that claims to be the
entire document.
"""

import json
from pathlib import Path

import pytest
from PIL import Image
from pixelrag_render import render_pdf


@pytest.fixture
def three_page_pdf(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pages = [Image.new("RGB", (612, 792), "white") for _ in range(3)]
    pages[0].save(pdf, save_all=True, append_images=pages[1:])
    return pdf


def _manifest(tile_dirs):
    return json.loads((Path(tile_dirs[0]) / "tiles.json").read_text())


def test_pdf_manifest_reports_a_full_render_as_complete(three_page_pdf, tmp_path):
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50)
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 3
    assert manifest["complete"] is True
    assert manifest["requested_pages"] is None


def test_pdf_manifest_reports_a_page_subset_as_incomplete(three_page_pdf, tmp_path):
    """The whole point of the flag: two of three pages is not the document."""
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[1, 3])
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 2, f"test setup rendered wrong: {manifest}"
    assert manifest["complete"] is False
    assert manifest["requested_pages"] == [1, 3]


def test_pdf_manifest_records_a_full_page_range_as_incomplete(three_page_pdf, tmp_path):
    """An explicit range that happens to cover everything is still a request.

    The caller asked for specific pages; the backend has not checked them
    against the document's length, so it must not claim to have captured the
    document.
    """
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[1, 2, 3])
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 3
    assert manifest["complete"] is False
    assert manifest["requested_pages"] == [1, 2, 3]
