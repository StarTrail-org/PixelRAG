"""IVFPQ compression path for the FAISS index build (issue #153).

`build_ivf(pq_m>0)` must produce a trained, searchable IndexIVFPQ that keeps the
inner-product metric — `faiss.IndexIVFPQ` defaults to L2, which would silently
wreck ranking on these L2-normalized embeddings.
"""

import json
import sys

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
sys.path.insert(0, "embed/src")
from pixelrag_embed.index import build_ivf  # noqa: E402


def _write_shard(emb_dir, n=1024, dim=32):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # L2-normalized (cosine/IP)
    np.savez(
        emb_dir / "shard_000.npz",
        embeddings=emb,
        article_ids=np.arange(n, dtype=np.int64),
        tile_indices=np.zeros(n, dtype=np.int32),
        chunk_indices=np.arange(n, dtype=np.int32),
        y_offsets=np.zeros(n, dtype=np.int32),
        tile_heights=np.full(n, 100, dtype=np.int32),
    )
    return emb


def test_ivfpq_index_is_built_trained_and_searchable(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    emb = _write_shard(emb_dir, dim=32)
    out = tmp_path / "out"

    build_ivf(str(emb_dir), str(out), nlist=8, nprobe=8, pq_m=8, pq_nbits=4)

    index = faiss.read_index(str(out / "index.faiss"))
    assert isinstance(index, faiss.IndexIVFPQ)
    # The bug in the issue's own sketch: without the metric arg this would be L2.
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert index.ntotal == emb.shape[0]
    _, ids = index.search(emb[:1], 5)
    assert ids.shape == (1, 5) and (ids[0] >= 0).all()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["index_type"] == "ivfpq"
    assert summary["pq_m"] == 8 and summary["pq_nbits"] == 4


def test_pq_m_must_divide_dim(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_shard(emb_dir, dim=32)
    with pytest.raises(ValueError, match="divide"):
        build_ivf(str(emb_dir), str(tmp_path / "out"), nlist=8, pq_m=7)


def test_default_stays_ivfflat(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    _write_shard(emb_dir, dim=32)
    out = tmp_path / "out"
    build_ivf(str(emb_dir), str(out), nlist=8)
    index = faiss.read_index(str(out / "index.faiss"))
    assert isinstance(index, faiss.IndexIVFFlat)
