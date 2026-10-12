"""Real-module checks for the opt-in study; missing FAISS is a collection error."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "embed/src"))
from pixelrag_embed.index import _merge_all_shards, build_ivf
from pixelrag_serve.backends import FaissBackend

from benchmarks.index_build_memory import generate_shards, validate
from benchmarks.streaming_ivf import FIELDS, build_streaming_ivf


@pytest.fixture(autouse=True)
def single_thread():
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    yield
    faiss.omp_set_num_threads(previous)


@pytest.fixture
def shards(tmp_path):
    source = tmp_path / "shards"
    generate_shards(source, rows=384, dim=16, shard_rows=128, seed=7)
    return source


def test_does_not_allocate_corpus_matrix(shards, tmp_path, monkeypatch):
    original = np.empty

    def guarded_empty(shape, *args, **kwargs):
        assert shape != (384, 16), "builder allocated the full raw input matrix"
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", guarded_empty)
    # Reproducible red/green demonstration against the *real* legacy builder.
    builder = (
        build_ivf
        if os.environ.get("PIXELRAG_STUDY_BASELINE") == "1"
        else build_streaming_ivf
    )
    builder(str(shards), str(tmp_path / "out"), nlist=2, train_sample=96)
    index = faiss.read_index(str(tmp_path / "out/index.faiss"))
    assert index.ntotal == 365


@pytest.mark.parametrize("pq_m", [0, 4])
@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_metadata_order_and_neighbor_alignment(shards, tmp_path, pq_m, metric):
    expected = _merge_all_shards(sorted(shards.glob("*.npz")))
    output = tmp_path / "out"
    build_streaming_ivf(
        shards,
        output,
        nlist=2,
        nprobe=2,
        train_sample=365,
        batch_size=17,
        pq_m=pq_m,
        pq_nbits=4,
        metric=metric,
    )
    index = faiss.read_index(str(output / "index.faiss"))
    assert index.ntotal == len(expected["embeddings"])
    assert index.metric_type == (
        faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    )
    with np.load(output / "metadata.npz") as actual:
        for name, dtype in FIELDS.items():
            np.testing.assert_array_equal(actual[name], expected[name])
            assert actual[name].dtype == dtype
    queries = expected["embeddings"][:16]
    _, ids = index.search(queries, 5)
    assert (ids >= 0).all()
    if not pq_m:
        # Verify vector/metadata alignment, not just the metadata file by itself.
        index.make_direct_map()
        for row in range(index.ntotal):
            np.testing.assert_array_equal(
                index.reconstruct(row), expected["embeddings"][row]
            )
        exact = faiss.IndexFlatIP(16) if metric == "ip" else faiss.IndexFlatL2(16)
        exact.add(expected["embeddings"])
        np.testing.assert_array_equal(ids, exact.search(queries, 5)[1])


def test_legacy_collisions_overflow_and_first_wins_are_preserved(tmp_path):
    source = tmp_path / "shards"
    source.mkdir()
    # (1,10000,0) collides with (2,0,0); (3,0,10000) with (3,1,0).
    # Large article IDs exercise the current int64 arithmetic overflow too.
    for shard in range(2):
        np.savez_compressed(
            source / f"shard_{shard}.npz",
            embeddings=np.arange(24, dtype=np.float32).reshape(6, 4) + shard,
            article_ids=np.array([1, 2, 3, 3, 2**62, 2**62], dtype=np.int64),
            tile_indices=np.array([10000, 0, 0, 1, 0, 0], dtype=np.int32),
            chunk_indices=np.array([0, 0, 10000, 0, 0, 0], dtype=np.int32),
            y_offsets=np.arange(6, dtype=np.int32) + shard * 100,
            tile_heights=np.full(6, 100 + shard, dtype=np.int32),
        )
    expected = _merge_all_shards(sorted(source.glob("*.npz")))
    build_streaming_ivf(source, tmp_path / "out", nlist=1, train_sample=3)
    index = faiss.read_index(str(tmp_path / "out/index.faiss"))
    index.make_direct_map()
    with np.load(tmp_path / "out/metadata.npz") as actual:
        for name in FIELDS:
            np.testing.assert_array_equal(actual[name], expected[name])
    for row in range(index.ntotal):
        np.testing.assert_array_equal(
            index.reconstruct(row), expected["embeddings"][row]
        )


def test_write_failure_does_not_publish_partial_output(shards, tmp_path, monkeypatch):
    def fail_write(index, path):
        Path(path).write_bytes(b"partial")
        raise OSError("injected disk failure")

    monkeypatch.setattr(faiss, "write_index", fail_write)
    with pytest.raises(OSError, match="injected disk failure"):
        build_streaming_ivf(shards, tmp_path / "out", nlist=2, train_sample=96)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".ivf-study-*"))


def test_existing_output_is_untouched(shards, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        build_streaming_ivf(shards, output)
    assert (output / "sentinel").read_text() == "keep"


def test_invalid_dimension_does_not_publish(shards, tmp_path):
    with np.load(shards / "shard_000001.npz") as data:
        values = dict(data)
    values["embeddings"] = values["embeddings"][:, :8]
    np.savez_compressed(shards / "shard_000001.npz", **values)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        build_streaming_ivf(shards, tmp_path / "out", train_sample=96)
    assert not (tmp_path / "out").exists()


def test_empty_shards_fail_cleanly(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    np.savez(
        source / "shard_000.npz",
        embeddings=np.empty((0, 16), dtype=np.float32),
        **{name: np.empty(0, dtype=dtype) for name, dtype in FIELDS.items()},
    )
    with pytest.raises(ValueError, match="No vectors"):
        build_streaming_ivf(source, tmp_path / "out", train_sample=16)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ["train", "add"])
@pytest.mark.parametrize("mutation", ["new_shard", "replace_shard"])
def test_changed_shards_do_not_publish(
    shards, tmp_path, monkeypatch, operation, mutation
):
    original = getattr(faiss.IndexIVFFlat, operation)
    changed = False

    def run_then_change_shards(index, data):
        nonlocal changed
        original(index, data)
        if changed:
            return
        changed = True
        source = shards / "shard_000000.npz"
        if mutation == "new_shard":
            with np.load(source) as archive:
                values = dict(archive)
            values["article_ids"] += 1000
            np.savez_compressed(shards / "shard_000003.npz", **values)
        else:
            # Replacement with the same size/mtime still changes file identity.
            stat = source.stat()
            replacement = tmp_path / "replacement.npz"
            replacement.write_bytes(source.read_bytes())
            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            os.replace(replacement, source)

    monkeypatch.setattr(faiss.IndexIVFFlat, operation, run_then_change_shards)
    with pytest.raises(ValueError, match="Input shards changed"):
        build_streaming_ivf(shards, tmp_path / "out", nlist=2, train_sample=96)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".ivf-study-*"))


def test_sampling_and_addition_are_bounded_and_repeatable(
    shards, tmp_path, monkeypatch
):
    expected = _merge_all_shards(sorted(shards.glob("*.npz")))["embeddings"]
    train = faiss.IndexIVFFlat.train
    add = faiss.IndexIVFFlat.add
    samples, additions = [], []

    def capture_train(index, data):
        samples.append(data.copy())
        train(index, data)

    def capture_add(index, data):
        assert 0 < len(data) <= 17
        additions.append(data.copy())
        add(index, data)

    monkeypatch.setattr(faiss.IndexIVFFlat, "train", capture_train)
    monkeypatch.setattr(faiss.IndexIVFFlat, "add", capture_add)
    for run in range(2):
        additions.clear()
        build_streaming_ivf(
            shards,
            tmp_path / f"out-{run}",
            nlist=2,
            train_sample=96,
            batch_size=17,
            seed=7,
        )
        np.testing.assert_array_equal(np.concatenate(additions), expected)
    np.testing.assert_array_equal(samples[0], samples[1])
    assert samples[0].shape == (96, 16)
    survivors = {row.tobytes() for row in expected}
    sampled = {row.tobytes() for row in samples[0]}
    assert len(sampled) == 96
    assert sampled <= survivors
    # Training must draw beyond the first shard, not just take an early prefix.
    assert sampled & {row.tobytes() for row in expected[-100:]}


@pytest.mark.parametrize("pq_m", [0, 4])
@pytest.mark.parametrize("mmap", [False, True])
def test_output_is_compatible_with_serving_backend(
    shards, tmp_path, monkeypatch, pq_m, mmap
):
    monkeypatch.delenv("PIXELRAG_INDEX_MMAP", raising=False)
    if mmap:
        monkeypatch.setenv("PIXELRAG_INDEX_MMAP", "1")
    expected = _merge_all_shards(sorted(shards.glob("*.npz")))
    output = tmp_path / "out"
    build_streaming_ivf(
        shards,
        output,
        nlist=2,
        nprobe=2,
        train_sample=365,
        pq_m=pq_m,
        pq_nbits=4,
    )
    summary = json.loads((output / "summary.json").read_text())
    backend = FaissBackend(str(output), summary)
    try:
        assert backend.ntotal == len(expected["embeddings"])
        assert backend.dimension == 16
        assert backend.nprobe == 2
        query = expected["embeddings"][:1]
        article = int(expected["article_ids"][0])
        hits = backend.raw_search(query, 5, article_ids=[article])[0]
        assert len(hits) == 1
        hit = hits[0]
        assert hit["vector_id"] == 0
        for stored, served in zip(
            FIELDS,
            ("article_id", "tile_index", "chunk_index", "y_offset", "tile_height"),
        ):
            assert hit[served] == int(expected[stored][0])
    finally:
        backend._meta.close()


def test_small_benchmark_reports_actual_neighbor_count(tmp_path, monkeypatch):
    source = tmp_path / "shards"
    generate_shards(source, rows=4, dim=8, shard_rows=2, seed=7)
    monkeypatch.setattr("pixelrag_embed.index.os.cpu_count", lambda: 1)
    outputs = {name: tmp_path / name for name in ("legacy", "prototype")}
    for name, builder in (("legacy", build_ivf), ("prototype", build_streaming_ivf)):
        builder(source, outputs[name], nlist=1, nprobe=1, train_sample=4)
    result = validate(source, outputs, SimpleNamespace(dim=8, seed=7, threads=1))
    assert result["between_builders_ann_overlap_at_4"] == 1.0
    for name in outputs:
        assert result[name]["ann_overlap_at_4_with_exact"] == 1.0
        assert "ann_overlap_at_10_with_exact" not in result[name]
