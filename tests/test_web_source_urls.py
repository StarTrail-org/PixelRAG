"""Regression tests for WebSource URL file parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))

from pixelrag_index.sources.web import WebSource


def _write(tmp_path, content):
    p = tmp_path / "urls.txt"
    p.write_text(content, encoding="utf-8")
    return WebSource(urls_file=str(p))


def test_plain_urls_are_kept(tmp_path):
    src = _write(tmp_path, "https://a.com\nhttps://b.com\n")
    assert src._urls == ["https://a.com", "https://b.com"]


def test_indented_comment_lines_are_skipped(tmp_path):
    src = _write(tmp_path, "https://a.com\n  # indented comment\n\thttps://b.com\n")
    assert src._urls == ["https://a.com", "https://b.com"]


def test_blank_and_full_comment_lines_are_skipped(tmp_path):
    src = _write(tmp_path, "# top comment\n\nhttps://a.com\n   \n# another\n")
    assert src._urls == ["https://a.com"]


def test_surrounding_whitespace_is_stripped(tmp_path):
    src = _write(tmp_path, "  https://a.com  \n")
    assert src._urls == ["https://a.com"]
