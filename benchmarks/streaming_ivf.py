"""Experimental CPU-only builder; deliberately not wired into pixelrag's CLI.

Memory scales with the largest decompressed shard, training reservoir, add batch,
SQLite cache, and final FAISS index, rather than a second corpus-sized matrix.
Input files must be immutable for the duration of the two passes.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import faiss
import numpy as np

FIELDS = {
    "article_ids": np.dtype("int64"),
    "tile_indices": np.dtype("int32"),
    "chunk_indices": np.dtype("int32"),
    "y_offsets": np.dtype("int32"),
    "tile_heights": np.dtype("int32"),
}


def _read_shard(path, dim):
    # mmap_mode cannot mmap members of a compressed .npz archive.
    with np.load(path, allow_pickle=False) as data:
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError(f"Invalid embeddings shape: {path}")
        if dim is not None and vectors.shape[1] != dim:
            raise ValueError(f"Dimension mismatch: {path}")
        metadata = {name: data[name].astype(dtype) for name, dtype in FIELDS.items()}
    if any(values.shape != (len(vectors),) for values in metadata.values()):
        raise ValueError(f"Metadata length mismatch: {path}")
    return vectors, metadata


def _signature(files):
    signature = []
    for path in files:
        stat = path.stat()
        signature.append(
            (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        )
    return signature


def _check_inputs(embeddings_dir, signature):
    # Re-glob: re-statting only the original paths silently ignores new shards.
    files = sorted(Path(embeddings_dir).glob("shard_*.npz"))
    if _signature(files) != signature:
        raise ValueError("Input shards changed during build")


def _pack_metadata(scratch, destination, count):
    # Write .npy members directly into a ZIP_STORED .npz, without mapping/touching
    # all metadata pages. The existing serve backend can read this unchanged.
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for name, dtype in FIELDS.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as target:
                np.lib.format.write_array_header_1_0(
                    target,
                    {"descr": dtype.str, "fortran_order": False, "shape": (count,)},
                )
                with (scratch / f"{name}.raw").open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)


def build_streaming_ivf(
    embeddings_dir,
    output_dir,
    *,
    nlist=64,
    nprobe=8,
    train_sample=8192,
    batch_size=1024,
    seed=0,
    metric="ip",
    pq_m=0,
    pq_nbits=8,
):
    """Two passes: first-wins dedup + reservoir training, then streaming add.

    Retains legacy *packed int64* collisions and overflow, intentionally. This
    is not the collision fix discussed in PR #140. SQLite stores signed keys;
    survivor positions are disk files, not a corpus-sized Python set or list.
    Destination must not exist. Publication is a same-filesystem directory
    rename; concurrent builders targeting the same destination are unsupported.
    """
    if min(nlist, nprobe, train_sample, batch_size) < 1:
        raise ValueError("nlist, nprobe, train_sample and batch_size must be positive")
    if metric not in {"ip", "l2"} or pq_m < 0 or pq_nbits not in {4, 8}:
        raise ValueError("Invalid metric or PQ configuration")
    files = sorted(Path(embeddings_dir).glob("shard_*.npz"))
    if not files:
        raise ValueError("No shard files found")
    signature = _signature(files)
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix=".ivf-study-", dir=output.parent) as temp:
        scratch = Path(temp)
        stage = scratch / "output"
        stage.mkdir()
        count, dim, reservoir = 0, None, None
        db = sqlite3.connect(scratch / "seen.sqlite")
        try:
            db.execute("PRAGMA cache_size=-2048")  # 2 MiB page cache
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE seen (key INTEGER PRIMARY KEY)")
            for shard_id, path in enumerate(files):
                vectors, metadata = _read_shard(path, dim)
                if dim is None:
                    dim = vectors.shape[1]
                    if pq_m and dim % pq_m:
                        raise ValueError("pq_m must divide dimension evenly")
                    reservoir = np.empty((train_sample, dim), dtype=np.float32)
                # Match legacy casts before arithmetic, including int64 wrap.
                keys = (
                    metadata["article_ids"] * 100_000_000
                    + metadata["tile_indices"].astype(np.int64) * 10_000
                    + metadata["chunk_indices"].astype(np.int64)
                )
                selected = []  # bounded by one shard, not the corpus
                for row, key in enumerate(keys):
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO seen VALUES (?)", (int(key),)
                    )
                    if not cursor.rowcount:
                        continue
                    selected.append(row)
                    count += 1
                    slot = count - 1 if count <= train_sample else rng.integers(count)
                    if slot < train_sample:
                        reservoir[slot] = vectors[row]
                db.commit()
                np.save(
                    scratch / f"rows_{shard_id}.npy",
                    np.asarray(selected, dtype=np.int64),
                )
                del vectors, metadata, keys, selected
        finally:
            db.close()
        if count == 0:
            raise ValueError("No vectors after deduplication")
        metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
        quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        nlist = min(nlist, count)
        if pq_m:
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits, metric_type)
        else:
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)
        index.train(reservoir[: min(train_sample, count)])
        del reservoir
        _check_inputs(embeddings_dir, signature)
        # Open only five metadata streams; their size is O(N) on disk.
        with ExitStack() as stack:
            streams = {
                name: stack.enter_context((scratch / f"{name}.raw").open("wb"))
                for name in FIELDS
            }
            for shard_id, path in enumerate(files):
                vectors, metadata = _read_shard(path, dim)
                rows = np.load(scratch / f"rows_{shard_id}.npy", allow_pickle=False)
                for start in range(0, len(rows), batch_size):
                    selection = rows[start : start + batch_size]
                    index.add(np.ascontiguousarray(vectors[selection]))
                    for name, stream in streams.items():
                        stream.write(metadata[name][selection].tobytes())
                del vectors, metadata, rows
        _check_inputs(embeddings_dir, signature)
        if index.ntotal != count:
            raise RuntimeError("Index/metadata row count mismatch")
        index.nprobe = nprobe
        _pack_metadata(scratch, stage / "metadata.npz", count)
        faiss.write_index(index, str(stage / "index.faiss"))
        summary = {
            "backend": "faiss",
            "total_vectors": count,
            "dimension": dim,
            "nlist": nlist,
            "nprobe": nprobe,
            "metric": metric,
            "index_type": "ivfpq" if pq_m else "ivfflat",
            "pq_m": pq_m,
            "pq_nbits": pq_nbits if pq_m else None,
            "index_file": str(output / "index.faiss"),
            "metadata_file": str(output / "metadata.npz"),
            "experimental_builder": "two-pass-reservoir",
            "seed": seed,
        }
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
