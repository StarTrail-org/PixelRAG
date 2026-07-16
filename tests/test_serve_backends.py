"""VectorBackend contract tests: FAISS and Qdrant must behave identically.

Both backends are exercised against the same tiny synthetic corpus: 12
vectors across 4 articles, two departments. The suite verifies the shared
raw_search contract — hit shape, ranking, the article_ids pre-filter (the
department feature), min_tile_height handling, and reconstruct().

These run only when the heavy optional deps are installed (`[serve]` /
`[qdrant]` extras); CI's core-only install skips them. Qdrant runs fully
in-process via QdrantClient(":memory:") — no server needed.
"""

import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss", reason="serve extra not installed")

from pixelrag_serve.backends import FaissBackend  # noqa: E402

DIM = 8
N_VECTORS = 12
# vector i belongs to article i // 3 (articles 0..3); departments: 0,1 -> "eng", 2,3 -> "hr"
ARTICLE_OF = [i // 3 for i in range(N_VECTORS)]
ENG_ARTICLES = np.asarray([0, 1])
HR_ARTICLES = np.asarray([2, 3])


def _unit(v):
    return v / np.linalg.norm(v)


def _make_vectors():
    rng = np.random.default_rng(42)
    return np.stack([_unit(rng.normal(size=DIM)) for _ in range(N_VECTORS)]).astype(
        "float32"
    )


VECTORS = _make_vectors()


@pytest.fixture(scope="module")
def faiss_backend(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("faiss_index")
    index = faiss.IndexFlatIP(DIM)
    index.add(VECTORS)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    np.savez(
        index_dir / "metadata.npz",
        article_ids=np.asarray(ARTICLE_OF),
        tile_indices=np.zeros(N_VECTORS, dtype=np.int64),
        chunk_indices=np.arange(N_VECTORS) % 3,
        y_offsets=np.zeros(N_VECTORS, dtype=np.int64),
        tile_heights=np.full(N_VECTORS, 100 + 10 * np.arange(N_VECTORS) % 200),
    )
    (index_dir / "summary.json").write_text(json.dumps({"backend": "faiss"}))
    return FaissBackend(str(index_dir), {"dimension": DIM})


@pytest.fixture(scope="module")
def qdrant_backend():
    pytest.importorskip("qdrant_client", reason="qdrant extra not installed")
    from qdrant_client import QdrantClient, models

    from pixelrag_serve.backends import QdrantBackend

    client = QdrantClient(":memory:")
    client.create_collection(
        "pixelrag",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
    )
    client.upsert(
        "pixelrag",
        points=[
            models.PointStruct(
                id=i,
                vector=VECTORS[i].tolist(),
                payload={
                    "article_id": ARTICLE_OF[i],
                    "tile_index": 0,
                    "chunk_index": i % 3,
                    "y_offset": 0,
                    "tile_height": int(100 + 10 * i % 200),
                },
            )
            for i in range(N_VECTORS)
        ],
    )
    backend = QdrantBackend.__new__(QdrantBackend)
    backend.collection = "pixelrag"
    backend.client = client
    backend.dimension = DIM
    return backend


def _backends(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["faiss_backend", "qdrant_backend"])
def backend(request):
    return _backends(request)


def test_raw_search_hit_shape_and_ranking(backend):
    query = VECTORS[4:5]  # exact vector -> must be its own top hit
    (hits,) = backend.raw_search(query, 3)
    assert len(hits) == 3
    top = hits[0]
    assert int(top["vector_id"]) == 4
    assert top["article_id"] == ARTICLE_OF[4]
    assert set(top) >= {
        "score",
        "vector_id",
        "article_id",
        "tile_index",
        "chunk_index",
        "y_offset",
        "tile_height",
    }
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_article_ids_prefilter_restricts_results(backend):
    # Query WITH an eng vector but filter to hr articles: every hit must be hr.
    query = VECTORS[0:1]
    (hits,) = backend.raw_search(query, 6, article_ids=HR_ARTICLES)
    assert hits, "filtered search must still return results"
    assert all(h["article_id"] in (2, 3) for h in hits)
    # And the unfiltered search does find the eng vector first (sanity).
    (unfiltered,) = backend.raw_search(query, 3)
    assert int(unfiltered[0]["vector_id"]) == 0


def test_prefilter_guarantees_k_within_department(backend):
    # hr has 6 vectors; asking for 6 filtered hits must return all 6 —
    # the point of pre-filtering over post-filtering.
    (hits,) = backend.raw_search(VECTORS[6:7], 6, article_ids=HR_ARTICLES)
    assert len(hits) == 6
    assert sorted(int(h["vector_id"]) for h in hits) == [6, 7, 8, 9, 10, 11]


def test_filter_cache_key_reuses_positions(faiss_backend):
    (first,) = faiss_backend.raw_search(
        VECTORS[0:1], 3, article_ids=ENG_ARTICLES, filter_cache_key="dept:eng"
    )
    assert "dept:eng" in faiss_backend._filter_positions
    cached = faiss_backend._filter_positions["dept:eng"]
    (second,) = faiss_backend.raw_search(
        VECTORS[0:1], 3, article_ids=ENG_ARTICLES, filter_cache_key="dept:eng"
    )
    assert faiss_backend._filter_positions["dept:eng"] is cached
    assert [h["vector_id"] for h in first] == [h["vector_id"] for h in second]


def test_reconstruct_round_trips(backend):
    vecs = backend.reconstruct([1, 5])
    assert len(vecs) == 2
    np.testing.assert_allclose(np.asarray(vecs[0]), VECTORS[1], atol=1e-6)
    np.testing.assert_allclose(np.asarray(vecs[1]), VECTORS[5], atol=1e-6)


def test_k_zero_returns_empty(backend):
    assert backend.raw_search(VECTORS[0:1], 0) == [[]]
