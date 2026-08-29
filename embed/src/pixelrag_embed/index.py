#!/usr/bin/env python3
"""Build a vector search index from embedding .npz shards.

Supports multiple backends:
  - ivf (default): FAISS IndexIVFFlat — fast build (~10 min), periodic rebuild for updates
  - diskann: DiskANN disk/memory index — see build_diskann.py

Steps:
  1. Merge all shard .npz files into a unified vectors + metadata
  2. Build the chosen index
  3. Test search

Usage:
    # Build IVF index (default)
    python indexing/build_index.py build \
        --embeddings-dir /opt/dlami/nvme/embeddings \
        --output-dir /opt/dlami/nvme/search_index

    # Build with custom nlist
    python indexing/build_index.py build \
        --embeddings-dir /opt/dlami/nvme/embeddings \
        --output-dir /opt/dlami/nvme/search_index \
        --nlist 8192 --nprobe 128

    # Test search
    python indexing/build_index.py test \
        --index-dir /opt/dlami/nvme/search_index \
        --nprobe 128
"""

import argparse
import json
import os
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np

# Unbuffered print so output shows up in logs/nohup immediately
print = partial(print, flush=True)


def _load_shards(embeddings_dir: str):
    """Load and deduplicate all shard .npz files. Yields (embeddings, metadata) per shard."""
    emb_dir = Path(embeddings_dir)
    shard_files = sorted(emb_dir.glob("shard_*.npz"))
    print(f"Found {len(shard_files)} shard files in {embeddings_dir}")
    if not shard_files:
        print("No shard files found!", file=sys.stderr)
        sys.exit(1)
    return shard_files


def _merge_all_shards(shard_files):
    """Single-pass: concat all shards, then numpy-vectorized global dedup.

    Returns dict of merged arrays + dim.
    """
    t0 = time.time()

    # First pass: quick count + dim check (mmap, no Python loop)
    total_raw = 0
    dim = None
    for sf in shard_files:
        with np.load(sf, mmap_mode="r") as data:
            n, d = data["embeddings"].shape
            if dim is None:
                dim = d
            assert d == dim, f"Dimension mismatch: {sf} has {d}, expected {dim}"
            total_raw += n
    print(f"Total raw vectors: {total_raw:,}, dim: {dim}")
    print(f"Allocating {total_raw * dim * 4 / 1e9:.1f} GB for float32 embeddings...")

    # Allocate output arrays
    all_emb = np.empty((total_raw, dim), dtype=np.float32)
    all_aids = np.empty(total_raw, dtype=np.int64)
    all_tiles = np.empty(total_raw, dtype=np.int32)
    all_chunks = np.empty(total_raw, dtype=np.int32)
    all_yoff = np.empty(total_raw, dtype=np.int32)
    all_theights = np.empty(total_raw, dtype=np.int32)

    # Concat all shards (no per-shard dedup — verified clean)
    row = 0
    for i, sf in enumerate(shard_files):
        with np.load(sf) as data:
            n = data["embeddings"].shape[0]
            all_emb[row : row + n] = data["embeddings"].astype(np.float32)
            all_aids[row : row + n] = data["article_ids"]
            all_tiles[row : row + n] = data["tile_indices"]
            all_chunks[row : row + n] = data["chunk_indices"]
            all_yoff[row : row + n] = data["y_offsets"]
            all_theights[row : row + n] = data["tile_heights"]
            row += n
        if (i + 1) % 100 == 0 or i == len(shard_files) - 1:
            print(
                f"  [{i + 1}/{len(shard_files)}] {row:,} vectors, {time.time() - t0:.0f}s"
            )

    print(f"Concat done: {row:,} vectors in {time.time() - t0:.0f}s")

    # Global dedup: numpy-vectorized unique on (article_id, tile, chunk)
    # Pack into single int64: article_id * 1e8 + tile * 1e4 + chunk
    print("Deduplicating...")
    t1 = time.time()
    keys = (
        all_aids[:row] * 100_000_000
        + all_tiles[:row].astype(np.int64) * 10_000
        + all_chunks[:row].astype(np.int64)
    )
    _, unique_idx = np.unique(keys, return_index=True)
    unique_idx.sort()  # preserve original order
    n_unique = len(unique_idx)
    n_dupes = row - n_unique
    print(
        f"Dedup done: {n_unique:,} unique, {n_dupes:,} duplicates removed in {time.time() - t1:.1f}s"
    )

    if n_dupes > 0:
        return {
            "embeddings": all_emb[unique_idx],
            "article_ids": all_aids[unique_idx],
            "tile_indices": all_tiles[unique_idx],
            "chunk_indices": all_chunks[unique_idx],
            "y_offsets": all_yoff[unique_idx],
            "tile_heights": all_theights[unique_idx],
            "dim": dim,
        }
    else:
        return {
            "embeddings": all_emb[:row],
            "article_ids": all_aids[:row],
            "tile_indices": all_tiles[:row],
            "chunk_indices": all_chunks[:row],
            "y_offsets": all_yoff[:row],
            "tile_heights": all_theights[:row],
            "dim": dim,
        }


def build_ivf(
    embeddings_dir: str,
    output_dir: str,
    nlist: int = 4096,
    nprobe: int = 128,
    train_sample: int = 500_000,
    metric: str = "ip",
    gpu_id: int = -1,
):
    """Build FAISS IVFFlat index.

    Args:
        nlist: number of IVF clusters (default 4096, good for ~30M vectors)
        nprobe: default search nprobe stored in the index
        train_sample: number of vectors to sample for K-means training
        metric: 'ip' (inner product / cosine for L2-normalized vectors) or 'l2'
        gpu_id: GPU to use for training (-1 = CPU only)
    """
    import faiss

    # Use all cores for FAISS CPU operations
    faiss.omp_set_num_threads(os.cpu_count())

    os.makedirs(output_dir, exist_ok=True)

    shard_files = _load_shards(embeddings_dir)

    print("\nMerging and deduplicating shards...")
    merged = _merge_all_shards(shard_files)
    embeddings = merged["embeddings"]
    dim = merged["dim"]
    n = embeddings.shape[0]
    print(f"Final: {n:,} × {dim}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.npz")
    print(f"Saving metadata to {metadata_path}...")
    np.savez(
        metadata_path,
        article_ids=merged["article_ids"],
        tile_indices=merged["tile_indices"],
        chunk_indices=merged["chunk_indices"],
        y_offsets=merged["y_offsets"],
        tile_heights=merged["tile_heights"],
    )

    # Build IVF index
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

    # Train on a sample
    actual_train = min(train_sample, n)
    train_indices = np.random.choice(n, actual_train, replace=False)
    train_data = embeddings[train_indices]

    quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)

    if gpu_id >= 0:
        # GPU-accelerated training: move CPU index to GPU, train, move back
        print(
            f"\nTraining IVF on GPU {gpu_id} (nlist={nlist}) on {actual_train:,} vectors..."
        )
        t0 = time.time()
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        gpu_index.train(train_data)
        print(f"GPU training done in {time.time() - t0:.1f}s")

        # Copy trained state back to CPU index
        print("Copying trained index to CPU...")
        index = faiss.index_gpu_to_cpu(gpu_index)
        del gpu_index, res  # free GPU memory
    else:
        # CPU training
        print(f"\nTraining IVF on CPU (nlist={nlist}) on {actual_train:,} vectors...")
        t0 = time.time()
        index.train(train_data)
        print(f"CPU training done in {time.time() - t0:.1f}s")

    # Add all vectors (CPU — GPU VRAM can't hold 30M × 2048)
    print(f"Adding {n:,} vectors...")
    t0 = time.time()
    batch = 100_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        index.add(embeddings[start:end])
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        print(
            f"  added {end:,}/{n:,} ({elapsed:.0f}s, {rate:.0f} vec/s, ETA {eta:.0f}s)"
        )
    print(f"Add done in {time.time() - t0:.1f}s")

    # Set default nprobe
    index.nprobe = nprobe

    # Save
    index_path = os.path.join(output_dir, "index.faiss")
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)

    # Summary
    summary = {
        "backend": "faiss",
        "total_vectors": n,
        "dimension": dim,
        "nlist": nlist,
        "nprobe": nprobe,
        "metric": metric,
        "index_file": index_path,
        "metadata_file": metadata_path,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    index_size = os.path.getsize(index_path)
    print(
        f"\nDone! Index: {index_size / 1e9:.1f} GB, metadata: {os.path.getsize(metadata_path) / 1e9:.1f} GB"
    )
    print(f"Summary: {summary_path}")


def build_qdrant(
    embeddings_dir: str,
    output_dir: str,
    url: str | None = None,
    collection: str = "pixelrag",
    api_key: str | None = None,
    client_config: dict | None = None,
    metric: str = "ip",
    quantization_config: dict | None = None,
    append: bool = False,
    recreate: bool = False,
    parallel: int = 1,
    batch: int = 1000,
):
    from pydantic import TypeAdapter
    from qdrant_client import QdrantClient, models

    client_options = dict(client_config or {})
    if url:
        client_options["url"] = url
    if api_key:
        client_options["api_key"] = api_key
    if not any(key in client_options for key in ("url", "host", "location", "path")):
        raise SystemExit(
            "Qdrant requires --qdrant-url or an endpoint in --qdrant-client-config"
        )

    client = QdrantClient(**client_options)
    exists = client.collection_exists(collection)
    if exists and not (append or recreate):
        raise ValueError(
            f"collection {collection!r} already exists. Use --append or --recreate"
        )

    os.makedirs(output_dir, exist_ok=True)
    merged = _merge_all_shards(_load_shards(embeddings_dir))
    vectors = np.ascontiguousarray(merged["embeddings"], dtype=np.float32)
    dim = merged["dim"]

    distance = models.Distance.COSINE if metric == "ip" else models.Distance.EUCLID
    quantization = (
        TypeAdapter(models.QuantizationConfig).validate_python(quantization_config)
        if quantization_config
        else None
    )
    if recreate or not exists:
        if exists:
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=dim, distance=distance, on_disk=True
            ),
            quantization_config=quantization,
        )

    # min_tile_height is the only payload filter used during search.
    client.create_payload_index(
        collection, "tile_height", field_schema=models.PayloadSchemaType.INTEGER
    )

    fields = {
        "article_id": merged["article_ids"],
        "tile_index": merged["tile_indices"],
        "chunk_index": merged["chunk_indices"],
        "y_offset": merged["y_offsets"],
        "tile_height": merged["tile_heights"],
    }

    # Qdrant only allows UUIDs and +ve integers as point IDs.
    # Ref: https://qdrant.tech/documentation/manage-data/points/#point-ids
    ids = (
        str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{article_id}:{tile_index}:{chunk_index}"))
        for article_id, tile_index, chunk_index in zip(
            fields["article_id"], fields["tile_index"], fields["chunk_index"]
        )
    )
    payloads = (
        {name: int(values[i]) for name, values in fields.items()}
        for i in range(len(vectors))
    )

    client.upload_collection(
        collection_name=collection,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        parallel=parallel,
        batch_size=batch,
        wait=True,
    )

    total = client.count(collection_name=collection, exact=True).count
    summary = {
        "backend": "qdrant",
        "total_vectors": total,
        "dimension": dim,
        "metric": metric,
        "collection": collection,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Uploaded {total:,} points to '{collection}'")


def test_search(index_dir: str, nprobe: int = 128, k: int = 10):
    """Test search on a built IVF index."""
    import faiss

    index_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.npz")

    print(f"Loading index from {index_path}...")
    t0 = time.time()
    index = faiss.read_index(index_path)
    print(f"Loaded in {time.time() - t0:.1f}s: {index.ntotal:,} vectors")

    index.nprobe = nprobe
    print(f"nprobe={nprobe}")

    # Load metadata
    meta = np.load(metadata_path)
    article_ids = meta["article_ids"]

    # Self-search: query with first vector
    # Extract first vector from the index
    query = index.reconstruct(0).reshape(1, -1)
    print("Query: first vector (self-search, should return itself as #1)")

    t0 = time.time()
    distances, indices = index.search(query, k)
    dt = time.time() - t0

    print(f"\nTop-{k} results ({dt * 1000:.1f}ms):")
    for i in range(k):
        idx = indices[0, i]
        dist = distances[0, i]
        aid = article_ids[idx]
        print(f"  {i + 1}. row={idx}, dist={dist:.6f}, article_id={aid}")


def main():
    parser = argparse.ArgumentParser(
        description="Build vector search index from wiki-screenshot embeddings"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build a vector index")
    p_build.add_argument("--embeddings-dir", default="./data/embeddings")
    p_build.add_argument("--output-dir", default="./output/search_index")
    p_build.add_argument(
        "--nlist", type=int, default=4096, help="Number of IVF clusters (default: 4096)"
    )
    p_build.add_argument(
        "--nprobe",
        type=int,
        default=128,
        help="Default nprobe for search (default: 128)",
    )
    p_build.add_argument(
        "--train-sample",
        type=int,
        default=500_000,
        help="Vectors to sample for training (default: 500k)",
    )
    p_build.add_argument(
        "--metric",
        choices=["ip", "l2"],
        default="ip",
        help="Distance metric (default: ip for cosine/L2-normalized)",
    )
    p_build.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="GPU for K-means training (-1 = CPU only)",
    )
    p_build.add_argument(
        "--backend",
        choices=["faiss", "qdrant"],
        default="faiss",
        help="Index backend (default: faiss)",
    )
    p_build.add_argument(
        "--qdrant-url", default=None, help="Qdrant server/Cloud URL (qdrant backend)"
    )
    p_build.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY"))
    p_build.add_argument(
        "--qdrant-client-config",
        help="Path to a JSON object of QdrantClient constructor arguments",
    )
    p_build.add_argument(
        "--collection", default="pixelrag", help="Qdrant collection name"
    )
    p_build.add_argument(
        "--qdrant-quantization-config",
        help="Qdrant quantization_config JSON for a new or recreated collection",
    )
    qdrant_mode = p_build.add_mutually_exclusive_group()
    qdrant_mode.add_argument(
        "--append",
        action="store_true",
        help="Upsert into an existing Qdrant collection",
    )
    qdrant_mode.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate an existing Qdrant collection",
    )

    # test
    p_test = sub.add_parser("test", help="Test search on built index")
    p_test.add_argument("--index-dir", default="./output/search_index")
    p_test.add_argument("--nprobe", type=int, default=128)
    p_test.add_argument("-k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "build":
        if args.backend == "qdrant":
            client_config = None
            if args.qdrant_client_config:
                with open(args.qdrant_client_config) as f:
                    client_config = json.load(f)
            quantization_config = None
            if args.qdrant_quantization_config:
                with open(args.qdrant_quantization_config) as f:
                    quantization_config = json.load(f)
            build_qdrant(
                args.embeddings_dir,
                args.output_dir,
                url=args.qdrant_url,
                collection=args.collection,
                api_key=args.qdrant_api_key,
                client_config=client_config,
                metric=args.metric,
                quantization_config=quantization_config,
                append=args.append,
                recreate=args.recreate,
            )
        else:
            build_ivf(
                args.embeddings_dir,
                args.output_dir,
                nlist=args.nlist,
                nprobe=args.nprobe,
                train_sample=args.train_sample,
                metric=args.metric,
                gpu_id=args.gpu_id,
            )
    elif args.command == "test":
        test_search(args.index_dir, nprobe=args.nprobe, k=args.k)


if __name__ == "__main__":
    main()
