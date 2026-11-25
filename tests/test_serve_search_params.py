"""Out-of-range /search parameters are client errors, not server errors.

`nprobe` is set straight on the FAISS index. FAISS rejects 0 at search time
and can't store a negative value, so both used to surface as a 500. Values
above nlist are fine: FAISS clamps them, so they must keep working.

`instruction` is embedded with every query, so its length is capped.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pixelrag_serve import api

faiss = pytest.importorskip("faiss", reason="serve extra not installed")

from pixelrag_serve.backends import FaissBackend

DIM = 4
NLIST = 2


@pytest.fixture
def ivf_client(monkeypatch, tmp_path):
    """/search over a real two-list IVF index, with precomputed query vectors."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(8, DIM)).astype("float32")
    faiss.normalize_L2(vectors)
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 1
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    np.savez(
        tmp_path / "metadata.npz",
        article_ids=np.arange(8),
        tile_indices=np.zeros(8, dtype=np.int64),
        chunk_indices=np.zeros(8, dtype=np.int64),
        y_offsets=np.zeros(8, dtype=np.int64),
        tile_heights=np.full(8, 100),
    )
    (tmp_path / "summary.json").write_text('{"backend": "faiss"}')
    backend = FaissBackend(str(tmp_path), {"dimension": DIM})
    monkeypatch.setattr(
        api,
        "_state",
        {
            "dimension": DIM,
            "backend": backend,
            "tiles_dir": str(tmp_path),
            "articles": [f"article {i}" for i in range(8)],
        },
    )
    with TestClient(api.app, raise_server_exceptions=False) as session:
        yield session, backend


def _search(session, **params):
    return session.post(
        "/search", json={"queries": [{"embedding": [1, 0, 0, 0]}], **params}
    )


@pytest.mark.parametrize("nprobe", [0, -1])
def test_nprobe_below_one_is_rejected(ivf_client, nprobe):
    session, backend = ivf_client
    response = _search(session, nprobe=nprobe)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "nprobe"]
    assert backend.index.nprobe == 1  # the shared index was not touched


@pytest.mark.parametrize("nprobe", [None, 1, NLIST, NLIST + 1, 4096])
def test_valid_nprobe_still_searches(ivf_client, nprobe):
    session, backend = ivf_client
    params = {} if nprobe is None else {"nprobe": nprobe}
    response = _search(session, **params)
    assert response.status_code == 200, response.text
    assert len(response.json()["results"][0]["hits"]) > 0
    assert backend.index.nprobe == 1  # the override is reset after the search


@pytest.fixture
def text_client(monkeypatch, tmp_path):
    """/search with a text query; the model is replaced by a stub encoder."""
    backend = Mock()
    backend.raw_search.side_effect = lambda vectors, *args, **kwargs: [
        [] for _ in vectors
    ]
    encode = Mock(return_value=np.zeros((1, DIM), dtype="float32"))
    monkeypatch.setattr(
        api,
        "_state",
        {"dimension": DIM, "backend": backend, "tiles_dir": str(tmp_path)},
    )
    monkeypatch.setattr(api, "_encode_queries", encode)
    with TestClient(api.app, raise_server_exceptions=False) as session:
        yield session, encode


def _text_search(session, **params):
    return session.post("/search", json={"queries": [{"text": "hello"}], **params})


def test_overlong_instruction_is_rejected_before_encoding(text_client):
    session, encode = text_client
    response = _text_search(session, instruction="x" * 513)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "instruction"]
    encode.assert_not_called()


@pytest.mark.parametrize("instruction", [None, "", "x" * 512])
def test_instruction_up_to_the_cap_is_used(text_client, instruction):
    session, encode = text_client
    params = {} if instruction is None else {"instruction": instruction}
    response = _text_search(session, **params)
    assert response.status_code == 200, response.text
    encode.assert_called_once()
    assert encode.call_args.args[1] == instruction
