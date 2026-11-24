"""Hybrid /search fuses BM25 into the dense results, including what dense missed.

The point of hybrid retrieval is that BM25 can surface a chunk the vector
search never returned. Those ids come back from `BM25Index.search` with no
SearchHit attached, so the handler has to ask the backend for their metadata
(`hits_for`) before it can serve them. These tests drive the real handler with
a stub backend so that path is exercised rather than assumed.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
from fastapi.testclient import TestClient
from pixelrag_serve import api

DIM = 4
DENSE_VIDS = [0, 1, 2]  # what the vector index returns
BM25_ONLY_VID = 7  # known to BM25 and to the backend, not to the dense search


def _hit(vid, score=1.0):
    return {
        "score": score,
        "vector_id": vid,
        "article_id": vid,
        "tile_index": 0,
        "chunk_index": 0,
        "y_offset": 0,
        "tile_height": 100,
    }


class _StubBackend:
    dimension = DIM

    def __init__(self):
        self.hits_for_calls = []

    def set_nprobe(self, n):
        pass

    def reset_nprobe(self):
        pass

    def raw_search(self, query_vectors, k, **kw):
        return [
            [_hit(v, 1.0 - i * 0.1) for i, v in enumerate(DENSE_VIDS)]
            for _ in query_vectors
        ]

    def hits_for(self, vids):
        self.hits_for_calls.append(list(vids))
        # The backend knows every vector in the index, dense-returned or not.
        return [_hit(int(v), 0.0) for v in vids]


class _StubBM25:
    """Ranks one vector the dense search returned and one it did not."""

    loaded = True

    def __init__(self, ranked):
        self.ranked = ranked

    def search(self, query, k=100):
        return self.ranked


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Hybrid needs the query text, and the API rejects text alongside a
    # pre-computed embedding, so these go through the text path with the
    # encoder stubbed rather than a model loaded.
    monkeypatch.setattr(
        api,
        "_encode_queries",
        lambda queries, instruction=None: np.ones(
            (len(queries), DIM), dtype=np.float32
        ),
    )

    def _make(bm25):
        backend = _StubBackend()
        api._state.clear()
        api._state.update(
            {
                "backend": backend,
                "articles": [f"Article_{i}" for i in range(16)],
                "dimension": DIM,
                "tiles_dir": str(tmp_path),
                "bm25": bm25,
            }
        )
        api._article_pages.cache_clear()
        return TestClient(api.app), backend

    yield _make
    api._state.clear()


def _search(client, hybrid, text="python"):
    resp = client.post(
        "/search",
        json={
            "queries": [{"text": text}],
            "n_docs": 10,
            "hybrid": hybrid,
        },
    )
    assert resp.status_code == 200, resp.text
    return [h["article_id"] for h in resp.json()["results"][0]["hits"]]


def test_bm25_only_hit_reaches_the_results(client):
    c, backend = client(_StubBM25([(BM25_ONLY_VID, 9.0), (0, 1.0)]))
    ids = _search(c, hybrid=True)

    # The whole point: a chunk the dense search never returned is served.
    assert BM25_ONLY_VID in ids
    assert backend.hits_for_calls == [[BM25_ONLY_VID]]


def test_without_hybrid_the_dense_results_are_untouched(client):
    c, backend = client(_StubBM25([(BM25_ONLY_VID, 9.0), (0, 1.0)]))
    ids = _search(c, hybrid=False)

    assert ids == DENSE_VIDS
    assert backend.hits_for_calls == []


def test_dense_only_ids_need_no_metadata_lookup(client):
    # BM25 agrees with the dense search: nothing is missing, so nothing is fetched.
    c, backend = client(_StubBM25([(1, 9.0), (0, 1.0)]))
    ids = _search(c, hybrid=True)

    assert sorted(ids) == sorted(DENSE_VIDS)
    assert backend.hits_for_calls == []


def test_a_query_whose_bm25_search_is_empty_keeps_the_dense_order(client):
    c, backend = client(_StubBM25([]))
    ids = _search(c, hybrid=True)

    # BM25 contributes nothing, so the dense order stands.
    assert ids == DENSE_VIDS
    assert backend.hits_for_calls == []
