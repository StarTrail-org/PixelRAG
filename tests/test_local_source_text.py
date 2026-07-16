"""Tests for native .md/.txt support in the local source adapter (issue #100)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))
from pixelrag_index.sources.local import LocalSource


@pytest.fixture
def text_dir(tmp_path):
    (tmp_path / "guide.md").write_text("# Guide\nSome markdown content")
    (tmp_path / "notes.txt").write_text("Plain text notes here")
    (tmp_path / "page.html").write_text("<html><body>HTML</body></html>")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "ignored.csv").write_text("a,b,c")
    return tmp_path


def test_local_source_discovers_text_files(text_dir):
    """LocalSource should find .md and .txt alongside .html and .png."""
    source = LocalSource(str(text_dir))
    docs = list(source)
    names = {d.id for d in docs}
    assert "guide" in names  # .md
    assert "notes" in names  # .txt
    assert "page" in names  # .html
    assert "photo" in names  # .png
    assert "ignored" not in names  # .csv not supported


def test_text_files_have_correct_metadata(text_dir):
    source = LocalSource(str(text_dir))
    docs = {d.id: d for d in source}

    # Text files should have type=text and a path (not a URL)
    assert docs["guide"].metadata["type"] == "text"
    assert docs["guide"].path is not None
    assert docs["guide"].url is None

    assert docs["notes"].metadata["type"] == "text"
    assert docs["notes"].path is not None


def test_html_files_have_url(text_dir):
    source = LocalSource(str(text_dir))
    docs = {d.id: d for d in source}

    # HTML files should have a file:// URL
    assert docs["page"].url.startswith("file://")
    assert docs["page"].metadata["type"] == "web"


def test_source_len_includes_text_files(text_dir):
    source = LocalSource(str(text_dir))
    assert len(source) == 4  # .md + .txt + .html + .png (not .csv)
