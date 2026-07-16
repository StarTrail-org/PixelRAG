"""Path-traversal hardening for eval/lib/pixel_query.py (issues #78, #79).

`example_id` is interpolated into output filenames; a crafted id like ``../../evil``
must not let the renderer write outside ``output_dir``. These tests assert the
sanitizer neutralizes separators and that constructed output paths stay contained.

The eval lib is a standalone module (not part of the installed ``pixelrag`` wheel),
so it is imported by file path. Only ``_safe_stem`` and path construction are exercised
by default — an integration test that actually renders is skipped unless a TTF font is
available, since rendering requires one.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_PIXEL_QUERY = Path(__file__).resolve().parents[1] / "eval" / "lib" / "pixel_query.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pixel_query", _PIXEL_QUERY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pixel_query = _load_module()

# Ids that try to escape the output directory.
MALICIOUS_IDS = [
    "../evil",
    "../../etc/passwd",
    "a/b/c",
    "/abs/evil",
    "..\\..\\evil",  # Windows-style separators
    "foo/../bar",
]


@pytest.mark.parametrize("bad_id", MALICIOUS_IDS)
def test_safe_stem_has_no_separators(bad_id):
    stem = pixel_query._safe_stem(bad_id)
    assert "/" not in stem
    assert os.sep not in stem
    if os.altsep:
        assert os.altsep not in stem
    assert not os.path.isabs(stem)


@pytest.mark.parametrize("bad_id", MALICIOUS_IDS)
def test_constructed_path_stays_in_output_dir(tmp_path, bad_id):
    # Mirror exactly how the renderers build their paths.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_path = os.path.join(str(out_dir), f"{pixel_query._safe_stem(bad_id)}_query.png")
    resolved = Path(out_path).resolve()
    assert out_dir.resolve() == resolved.parent, (
        f"{bad_id!r} escaped output_dir: resolved to {resolved}"
    )


def test_safe_stem_preserves_normal_ids():
    assert pixel_query._safe_stem("abc123") == "abc123"
    assert pixel_query._safe_stem("example_42") == "example_42"


def _find_font():
    for c in (
        *pixel_query._FONT_CANDIDATES,
        r"C:\Windows\Fonts\arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if os.path.exists(c):
            return c
    return None


@pytest.mark.skipif(_find_font() is None, reason="no TTF font available to render")
def test_render_malicious_id_does_not_escape(tmp_path):
    out_dir = tmp_path / "out"
    sentinel = tmp_path / "evil_query.png"  # where ../evil would land pre-fix
    renderer = pixel_query.PixelQueryRenderer(
        output_dir=str(out_dir), font_path=_find_font()
    )
    written = renderer.render("../evil", "what is the capital of France?")
    written_path = Path(written).resolve()

    assert out_dir.resolve() == written_path.parent, "render escaped output_dir"
    assert written_path.exists()
    assert not sentinel.exists(), "file was written outside output_dir"
