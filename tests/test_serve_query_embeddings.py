"""Validate precomputed batches before NumPy conversion or backend access."""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pixelrag_serve import api


@pytest.fixture
def client(monkeypatch, tmp_path):
    backend = Mock()
    backend.raw_search.side_effect = lambda vectors, *args, **kwargs: [
        [] for _ in vectors
    ]
    monkeypatch.setattr(
        api,
        "_state",
        {"dimension": 4, "backend": backend, "tiles_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        api, "_encode_queries", Mock(side_effect=AssertionError("No model needed"))
    )
    with TestClient(api.app, raise_server_exceptions=False) as session:
        yield session, backend


@pytest.mark.parametrize(
    "vectors,wrong_dim",
    [
        ([[1, 0, 0, 0], [1, 0, 0]], 3),
        ([[1, 0, 0], [1, 0, 0, 0]], 3),
        ([[1, 0, 0], [0, 1, 0]], 3),
        ([[]], 0),
        ([[], []], 0),
        ([[1, 0, 0, 0], []], 0),
        ([[1, 0, 0, 0, 0]], 5),
    ],
)
def test_invalid_dimensions_are_client_errors(client, vectors, wrong_dim):
    session, backend = client
    response = session.post(
        "/search", json={"queries": [{"embedding": v} for v in vectors]}
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": f"Embedding dim mismatch: got {wrong_dim}, expected 4."
    }
    assert backend.mock_calls == []


@pytest.mark.parametrize(
    "vectors,expected",
    [
        ([[3, 4, 0, 0]], [[0.6, 0.8, 0, 0]]),
        (
            [[3, 4, 0, 0], [0, 0, -5, 0]],
            [[0.6, 0.8, 0, 0], [0, 0, -1, 0]],
        ),
    ],
)
def test_valid_batches_are_normalized(client, vectors, expected):
    session, backend = client
    response = session.post(
        "/search", json={"queries": [{"embedding": v} for v in vectors]}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == len(vectors)
    backend.raw_search.assert_called_once()
    actual = backend.raw_search.call_args.args[0]
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "queries,detail",
    [
        (
            [{"embedding": [1, 0, 0, 0]}, {"text": "hello"}],
            "Do not mix pre-computed embeddings with text/image queries in one request.",
        ),
        (
            [{"embedding": [1, 0, 0, 0], "text": "hello"}],
            "Pre-computed embeddings cannot be mixed with text/image fields in the same request.",
        ),
    ],
)
def test_mixed_text_and_embeddings_remain_invalid(client, queries, detail):
    session, backend = client
    response = session.post("/search", json={"queries": queries})
    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    assert backend.mock_calls == []


def test_backend_value_error_is_not_reclassified(client):
    session, backend = client
    backend.raw_search.side_effect = ValueError("backend failure")
    response = session.post("/search", json={"queries": [{"embedding": [1, 0, 0, 0]}]})
    assert response.status_code == 500
    backend.raw_search.assert_called_once()
