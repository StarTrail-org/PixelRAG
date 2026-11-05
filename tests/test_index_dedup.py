"""Shard merge keeps every distinct (article_id, tile_index, chunk_index)."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "embed/src")
from pixelrag_embed.index import _load_shards, _merge_all_shards


def _write_shards(emb_dir, parts, dim=4):
    for s, part in enumerate(parts):
        n = len(part)
        np.savez(
            emb_dir / f"shard_{s:05d}.npz",
            embeddings=np.arange(n * dim, dtype=np.float32).reshape(n, dim) + 100 * s,
            article_ids=np.array([t[0] for t in part], dtype=np.int64),
            tile_indices=np.array([t[1] for t in part], dtype=np.int32),
            chunk_indices=np.array([t[2] for t in part], dtype=np.int32),
            y_offsets=np.zeros(n, dtype=np.int32),
            tile_heights=np.full(n, 8, dtype=np.int32),
        )


def _merge(tmp_path, parts):
    _write_shards(tmp_path, parts)
    merged = _merge_all_shards(_load_shards(str(tmp_path)))
    return list(
        zip(
            merged["article_ids"].tolist(),
            merged["tile_indices"].tolist(),
            merged["chunk_indices"].tolist(),
        )
    )


def test_distinct_rows_are_all_kept(tmp_path):
    # Each pair packed to the same key under article_id*1e8 + tile*1e4 + chunk.
    rows = [
        (0, 1, 0),
        (0, 0, 10_000),
        (1, 0, 0),
        (0, 10_000, 0),
        (2, -1, 0),
        (1, 9_999, 0),
        (5, 3, 7),
    ]
    assert _merge(tmp_path, [rows[:4], rows[4:]]) == rows


def test_hashed_article_ids_with_many_tiles_are_kept(tmp_path):
    # embed_cpu derives article ids from 8 hex digits of sha1 (< 2**32) when a
    # tile dir has no manifest id.
    big = 2**32 - 1
    rows = [(big, 10_000, 0), (big, 0, 10_000), (big - 1, 99_999, 3), (big, 0, 0)]
    assert _merge(tmp_path, [rows]) == rows


def test_true_duplicates_are_removed_keeping_first(tmp_path):
    rows = [(3, 0, 0), (3, 0, 1), (4, 2, 0)]
    merged = _merge(tmp_path, [rows, [(3, 0, 1), (9, 0, 0)]])
    assert merged == rows + [(9, 0, 0)]


def test_large_offsets_pack_exactly(tmp_path):
    # Packed range just fits in int64; only the offsets are large.
    top = np.iinfo(np.int64).max - 2**30
    rows = [
        (0, 2**31 - 1, -(2**31)),
        (top, 2**31 - 1, -(2**31)),
        (top, 2**31 - 1, -(2**31)),
    ]
    assert _merge(tmp_path, [rows]) == rows[:2]


def test_unpackable_range_fails_loudly(tmp_path):
    rows = [(2**62, 0, 0), (0, 2**20, 2**20)]
    with pytest.raises(ValueError, match="int64"):
        _merge(tmp_path, [rows])
