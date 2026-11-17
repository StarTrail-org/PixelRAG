"""API-level errors for the `department` search filter.

`_department_article_ids` is the only place these two client errors are
produced, and both are reachable from the public /search endpoint. The pre-
filter's search behaviour is covered against both backends in
test_serve_backends.py; this file needs no index, only the state dict.
"""

import pytest

api = pytest.importorskip("pixelrag_serve.api", reason="serve extra not installed")

DEPTS = {"hr": [0, 2], "ketoan": [1]}


@pytest.fixture(autouse=True)
def _state():
    api._state.clear()
    api._state.update({"dept_to_aids": dict(DEPTS)})
    yield
    api._state.clear()


def test_known_department_returns_its_article_ids():
    assert list(api._department_article_ids("hr")) == [0, 2]


def test_unknown_department_raises_404():
    with pytest.raises(api.HTTPException) as exc:
        api._department_article_ids("phong-khong-ton-tai")
    assert exc.value.status_code == 404
    assert "hr" in exc.value.detail  # lists what is available


def test_index_without_department_metadata_raises_400():
    api._state["dept_to_aids"] = {}
    with pytest.raises(api.HTTPException) as exc:
        api._department_article_ids("hr")
    assert exc.value.status_code == 400
