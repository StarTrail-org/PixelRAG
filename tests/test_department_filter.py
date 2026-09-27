"""Build side of the department feature: directory layout -> department name.

The serve side lives in test_department_search.py (API errors) and
test_serve_backends.py (the article_ids pre-filter, both backends). Keeping
them apart matters: this module needs no heavy extras, so gating it on faiss
meant CI — which installs none — skipped these cases too.
"""

import pytest

pipelines = pytest.importorskip("pixelrag_index.pipelines")


# ---------------------------------------------------------------------------
# Build side: _department_of — directory layout → department name
# ---------------------------------------------------------------------------


def test_department_from_subdirectory(tmp_path):
    (tmp_path / "hr").mkdir()
    f = tmp_path / "hr" / "sop_tuyen_dung.pdf"
    f.write_bytes(b"x")
    assert pipelines._department_of({"path": str(f)}, str(tmp_path)) == "hr"


def test_department_from_nested_subdirectory(tmp_path):
    d = tmp_path / "ketoan" / "2026"
    d.mkdir(parents=True)
    f = d / "sop_thanh_toan.pdf"
    f.write_bytes(b"x")
    assert pipelines._department_of({"path": str(f)}, str(tmp_path)) == "ketoan"


def test_file_at_source_root_has_no_department(tmp_path):
    f = tmp_path / "sop.pdf"
    f.write_bytes(b"x")
    assert pipelines._department_of({"path": str(f)}, str(tmp_path)) == ""


def test_department_from_file_url(tmp_path):
    (tmp_path / "it").mkdir()
    f = tmp_path / "it" / "huong dan.html"
    f.write_bytes(b"x")
    art = {"url": f.resolve().as_uri()}  # file:// with percent-encoded space
    assert pipelines._department_of(art, str(tmp_path)) == "it"


def test_web_url_outside_root_and_empty_root(tmp_path):
    assert pipelines._department_of({"url": "https://ex.am/ple"}, str(tmp_path)) == ""
    assert pipelines._department_of({"path": "/elsewhere/f.pdf"}, str(tmp_path)) == ""
    assert pipelines._department_of({"path": "/a/b.pdf"}, "") == ""


# ---------------------------------------------------------------------------
# Serve side: IDSelector filter
# ---------------------------------------------------------------------------
