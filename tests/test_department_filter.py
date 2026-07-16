"""Department metadata (articles.json) + FAISS IDSelector pre-filter."""

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

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")
api = pytest.importorskip("pixelrag_serve.api")

DIM = 8
# 3 articles × 2 vectors; article 0, 2 → hr; article 1 → ketoan
ARTICLE_IDS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
HR_ROWS = {0, 1, 4, 5}


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(7)
    v = rng.standard_normal((6, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _setup_state(index):
    api._state.clear()
    api._state.update(
        {
            "index": index,
            "metadata": {"article_ids": ARTICLE_IDS},
            "dept_to_aids": {"hr": np.array([0, 2]), "ketoan": np.array([1])},
            "dept_positions": {},
        }
    )


def test_department_positions_maps_articles_to_rows():
    index = faiss.IndexFlatIP(DIM)
    index.add(_vectors())
    _setup_state(index)
    assert set(api._department_positions("hr").tolist()) == HR_ROWS
    assert set(api._department_positions("ketoan").tolist()) == {2, 3}


def test_flat_index_filter_restricts_hits():
    vecs = _vectors()
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    _setup_state(index)
    params = api._department_search_params("hr", nprobe=1)
    _, indices = index.search(vecs[:1], 6, params=params)
    hits = {int(i) for i in indices[0] if i != -1}
    assert hits == HR_ROWS  # every hr row returned, nothing else


def test_ivf_index_filter_restricts_hits():
    vecs = _vectors()
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, 1, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = 1
    _setup_state(index)
    params = api._department_search_params("ketoan", nprobe=1)
    _, indices = index.search(vecs[2:3], 6, params=params)
    hits = {int(i) for i in indices[0] if i != -1}
    assert hits == {2, 3}


def test_unknown_department_raises_404():
    index = faiss.IndexFlatIP(DIM)
    index.add(_vectors())
    _setup_state(index)
    with pytest.raises(api.HTTPException) as exc:
        api._department_positions("phong-khong-ton-tai")
    assert exc.value.status_code == 404
    assert "hr" in exc.value.detail  # lists available departments


def test_index_without_department_metadata_raises_400():
    index = faiss.IndexFlatIP(DIM)
    index.add(_vectors())
    _setup_state(index)
    api._state["dept_to_aids"] = {}
    with pytest.raises(api.HTTPException) as exc:
        api._department_positions("hr")
    assert exc.value.status_code == 400
