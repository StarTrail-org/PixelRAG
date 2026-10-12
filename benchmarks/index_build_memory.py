"""Local generated-data benchmark. Run with uv; see index_build_memory.md."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "embed/src"))
from pixelrag_embed.index import _merge_all_shards, build_ivf

from benchmarks.streaming_ivf import FIELDS, build_streaming_ivf


def generate_shards(destination, *, rows=65536, dim=256, shard_rows=4096, seed=2026):
    """Normalized float32 noise, 5% repeated keys, distinct duplicate payloads."""
    destination.mkdir()
    rng = np.random.default_rng(seed)
    for shard, start in enumerate(range(0, rows, shard_rows)):
        size = min(shard_rows, rows - start)
        positions = np.arange(start, start + size, dtype=np.int64)
        aids = positions.copy()
        # Includes duplicates inside a shard and across shard boundaries.
        duplicate = (positions > 0) & (positions % 20 == 0)
        aids[duplicate] -= 1
        vectors = rng.standard_normal((size, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.savez_compressed(
            destination / f"shard_{shard:06}.npz",
            embeddings=vectors,
            article_ids=aids,
            tile_indices=np.zeros(size, dtype=np.int32),
            chunk_indices=np.zeros(size, dtype=np.int32),
            y_offsets=positions.astype(np.int32),
            tile_heights=(100 + positions % 17).astype(np.int32),
        )


def worker(args):
    faiss.omp_set_num_threads(args.threads)
    np.random.seed(args.seed)
    options = dict(
        nlist=args.nlist,
        nprobe=args.nprobe,
        train_sample=args.train_sample,
        pq_m=args.pq_m,
        pq_nbits=4,
    )
    started = time.perf_counter()
    if args.worker == "legacy":
        # Production resets FAISS threads from os.cpu_count(). Keep both runs
        # on the same explicit CPU budget without editing production code.
        with patch("pixelrag_embed.index.os.cpu_count", return_value=args.threads):
            build_ivf(str(args.work_dir / "shards"), str(args.output), **options)
    else:
        build_streaming_ivf(
            args.work_dir / "shards",
            args.output,
            seed=args.seed,
            batch_size=args.batch_size,
            **options,
        )
    elapsed = time.perf_counter() - started
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        unit, rss_bytes = "bytes", raw_rss
    elif sys.platform.startswith("linux"):
        unit, rss_bytes = "KiB", raw_rss * 1024
    else:
        raise RuntimeError("Peak RSS conversion implemented only for macOS and Linux")
    result = {
        "builder": args.worker,
        "build_wall_seconds": elapsed,
        "peak_rss_raw": raw_rss,
        "peak_rss_os_unit": unit,
        "peak_rss_mib": rss_bytes / 1024**2,
        "index_bytes": (args.output / "index.faiss").stat().st_size,
        "metadata_bytes": (args.output / "metadata.npz").stat().st_size,
        "total_output_bytes": sum(p.stat().st_size for p in args.output.iterdir()),
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))


def overlap(left, right):
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(left, right)]))


def validate(source, outputs, args):
    # Deliberately outside timed/RSS-measured builder processes. This oracle
    # allocates the complete fixture, which is why fixture size is capped.
    merged = _merge_all_shards(sorted(source.glob("shard_*.npz")))
    vectors = merged["embeddings"]
    rng = np.random.default_rng(args.seed + 1)
    queries = rng.standard_normal((128, args.dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    faiss.omp_set_num_threads(args.threads)
    exact = faiss.IndexFlatIP(args.dim)
    exact.add(vectors)
    k = min(10, len(vectors))
    _, truth = exact.search(queries, k)
    results, neighbors = {}, {}
    for label, output in outputs.items():
        index = faiss.read_index(str(output / "index.faiss"))
        assert index.ntotal == len(vectors)
        with np.load(output / "metadata.npz") as metadata:
            for name, dtype in FIELDS.items():
                np.testing.assert_array_equal(metadata[name], merged[name])
                assert metadata[name].dtype == dtype
        _, ann = index.search(queries, k)
        # Ignore missing neighbors when computing overlap with truth.
        neighbors[label] = ann
        index.nprobe = index.nlist
        _, exhaustive = index.search(queries, k)
        results[label] = {
            "vector_count": index.ntotal,
            "metadata_alignment": "all five arrays and dtypes equal legacy merge",
            f"ann_overlap_at_{k}_with_exact": overlap(ann, truth),
            f"all_lists_overlap_at_{k}_with_exact": overlap(exhaustive, truth),
        }
        del index
    # Missing IDs must never count as common neighbors.
    results[f"between_builders_ann_overlap_at_{k}"] = float(
        np.mean(
            [
                len((set(a) & set(b)) - {-1}) / k
                for a, b in zip(neighbors["legacy"], neighbors["prototype"])
            ]
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--shard-rows", type=int, default=4096)
    parser.add_argument("--train-sample", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--pq-m", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--worker", choices=["legacy", "prototype"], help=argparse.SUPPRESS
    )
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    for name in (
        "rows",
        "dim",
        "shard_rows",
        "train_sample",
        "batch_size",
        "nlist",
        "nprobe",
        "threads",
        "repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    largest_matrix = max(args.rows, args.shard_rows, args.train_sample) * args.dim * 4
    if largest_matrix > 256 * 1024**2:
        parser.error("Study capped at 256 MiB for raw corpus, shard and reservoir")
    if args.pq_m < 0 or (args.pq_m and args.dim % args.pq_m):
        parser.error("pq-m must be nonnegative and divide dim")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    generate_shards(
        args.work_dir / "shards",
        rows=args.rows,
        dim=args.dim,
        shard_rows=args.shard_rows,
        seed=args.seed,
    )
    report = {
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "cpu_count": os.cpu_count(),
        "runs": [],
    }
    for repeat in range(args.repeats):
        outputs, metrics = {}, {}
        # Alternate order to reduce systematic cache/order bias; no cold-cache claim.
        order = ["legacy", "prototype"] if repeat % 2 == 0 else ["prototype", "legacy"]
        for label in order:
            output = args.work_dir / f"{label}-{repeat}"
            outputs[label] = output
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
                label,
                "--output",
                str(output),
            ]
            with (args.work_dir / f"{label}-{repeat}.log").open("w") as log:
                subprocess.run(
                    command, check=True, stdout=log, stderr=subprocess.STDOUT
                )
            metrics[label] = json.loads(output.with_suffix(".metrics.json").read_text())
        report["runs"].append(
            {
                "metrics": metrics,
                "validation": validate(args.work_dir / "shards", outputs, args),
            }
        )
        (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
