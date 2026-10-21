"""Tests for --extract-text hybrid output (issue #93)."""

from pathlib import Path

from pixelrag_render import render_file


def test_extract_text_produces_text_md(tmp_path):
    """With extract_text=True, a text.md file should be created alongside tiles."""
    html = tmp_path / "page.html"
    html.write_text(
        "<html><body>"
        "<h1>Hello World</h1>"
        "<p>This is a test paragraph with important content.</p>"
        "<ul><li>Item one</li><li>Item two</li></ul>"
        "</body></html>"
    )
    out = tmp_path / "tiles"

    dirs = render_file(html, out, extract_text=True)

    assert dirs, "render_file returned no tile directories"
    tile_dir = Path(dirs[0])
    text_file = tile_dir / "text.md"
    assert text_file.exists(), f"text.md not created in {tile_dir}"

    content = text_file.read_text()
    assert "Hello World" in content
    assert "test paragraph" in content
    assert "Item one" in content


def test_no_extract_text_by_default(tmp_path):
    """Without extract_text, no text.md should be created."""
    html = tmp_path / "page.html"
    html.write_text("<html><body><p>Some text</p></body></html>")
    out = tmp_path / "tiles"

    dirs = render_file(html, out)

    tile_dir = Path(dirs[0])
    assert not (tile_dir / "text.md").exists()


def test_extract_text_wins_over_the_turbo_path(tmp_path):
    """A turbo-capable Chrome must not silently swallow --extract-text.

    The turbo backend (fast_cdp) has no text extraction, so requesting it has
    to send the capture down the standard path — the same trade already made
    for wait_network_idle. Without that, anyone whose Chrome is turbo-capable
    (which is what `pixelshot install-chrome` produces) gets tiles and no
    text.md, with nothing said about it.
    """
    html = tmp_path / "page.html"
    html.write_text("<html><body><p>Turbo should not eat this</p></body></html>")

    dirs = render_file(html, tmp_path / "tiles", extract_text=True, turbo=True)

    text_file = Path(dirs[0]) / "text.md"
    assert text_file.exists(), "turbo path silently dropped the text extraction"
    assert "Turbo should not eat this" in text_file.read_text()
