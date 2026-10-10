"""Cap on on-demand renders per /search request (DoS hardening).

An on-demand render is a Chrome page load under a process-global lock with a
120s timeout. Without a per-request budget, one `include_images` request could
queue one per hit — at the endpoint's own limits, tens of thousands — and every
other caller's renders stall behind the same lock.

Driven through the real handler with pre-computed embeddings, so no model or
index is needed.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
from fastapi.testclient import TestClient
from pixelrag_serve import api

DIM = 4
N_HITS = 40  # more hits than the budget allows renders


class _CountingOnDemand:
    """Stands in for OnDemandTiles: counts renders, never touches Chrome."""

    def __init__(self, cached: set[int] | None = None):
        self.cached = cached or set()
        self.renders = 0

    def cached_chunk_path(self, article_id, tile_index, chunk_index):
        return f"/cached/{article_id}.png" if article_id in self.cached else None

    def chunk_path(self, article_id, title, tile_index, chunk_index):
        if article_id not in self.cached:
            self.renders += 1


class _StubBackend:
    dimension = DIM

    def set_nprobe(self, n):
        pass

    def reset_nprobe(self):
        pass

    def raw_search(self, query_vectors, k, **kw):
        return [
            [
                {
                    "score": 1.0,
                    "vector_id": i,
                    "article_id": i,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 100,
                }
                for i in range(min(k, N_HITS))
            ]
            for _ in query_vectors
        ]


@pytest.fixture
def client(tmp_path):
    def _make(ondemand):
        api._state.clear()
        api._state.update(
            {
                "backend": _StubBackend(),
                "articles": [f"Article_{i}" for i in range(N_HITS)],
                "dimension": DIM,
                "tiles_dir": str(tmp_path),  # empty: nothing is on disk
                "ondemand": ondemand,
            }
        )
        api._article_pages.cache_clear()
        return TestClient(api.app)

    yield _make
    api._state.clear()


def _search(client, n_docs):
    body = {
        "queries": [{"embedding": np.ones(DIM).tolist()}],
        "n_docs": n_docs,
        "include_images": True,
    }
    resp = client.post("/search", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_renders_are_capped_per_request(client):
    od = _CountingOnDemand()
    body = _search(client(od), N_HITS)
    # Every hit still comes back — only the images past the budget are dropped.
    assert len(body["results"][0]["hits"]) == N_HITS
    assert od.renders == api._MAX_ONDEMAND_RENDERS


def test_budget_spans_all_queries_in_one_request(client):
    od = _CountingOnDemand()
    c = client(od)
    resp = c.post(
        "/search",
        json={
            "queries": [{"embedding": np.ones(DIM).tolist()} for _ in range(4)],
            "n_docs": N_HITS,
            "include_images": True,
        },
    )
    assert resp.status_code == 200, resp.text
    # One budget for the request, not one per query.
    assert od.renders == api._MAX_ONDEMAND_RENDERS


def test_cache_hits_do_not_spend_the_budget(client):
    # Every article already rendered: serving them must cost no renders at all.
    od = _CountingOnDemand(cached=set(range(N_HITS)))
    _search(client(od), N_HITS)
    assert od.renders == 0
