#!/usr/bin/env python3
"""Materialize unique tile images to a local mirror.

Reads retrieval_raw/<split>.jsonl files produced by fetch_top6_retrieval.py,
collects gold + top-6 hit paths, dedups, and fetches each tile once.

For tiles whose shard-relative path exists in the local dataset
(under <dataset-dir>/images/...), hardlinks to the mirror instead of
re-downloading. Everything else is fetched via GET :30895/tile?path=...

Mirror layout preserves shard tree so later compression steps can simply
rewrite a path prefix:
  <output-dir>/tiles/shard_583/shard_00003/<article>.png.tiles/chunk_0000_00.png
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def shard_suffix(p: str) -> str:
    parts = p.split("/")
    for i, x in enumerate(parts):
        if x.startswith("shard_"):
            return "/".join(parts[i:])
    return p


def collect_paths(retrieval_dir: Path, splits: list[str]) -> dict[str, str | None]:
    """Collect every unique absolute path across hit lists + gold suffixes.

    Gold paths use the dataset-relative form ('images/shard_.../chunk.png');
    hits are absolute '/opt/dlami/nvme/kiwix_tiles/shard_.../chunk.png'.
    We normalize everything to shard-suffix for dedup across sources.
    Returns dict mapping shard_suffix_key to preferred_abs_path_for_fetch.
    """
    by_suffix: dict[str, str | None] = {}
    for split in splits:
        p = retrieval_dir / f"{split}.jsonl"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                # gold (will fall back to local if possible)
                gs = r["gold_suffix"]
                if gs not in by_suffix:
                    by_suffix[gs] = None  # local-resolvable
                # hits (absolute)
                for h in r["hits"]:
                    ss = shard_suffix(h["path"])
                    if ss not in by_suffix or by_suffix[ss] is None:
                        by_suffix[ss] = h["path"]
    return by_suffix


def try_local_link(suffix: str, local_root: Path, mirror_root: Path) -> bool:
    """If local_root/images/<suffix> exists, hardlink it into mirror_root/<suffix>.
    Returns True on success."""
    src = local_root / "images" / suffix
    if not src.exists():
        return False
    dst = mirror_root / suffix
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        # cross-device; fall back to copy (only expected when local & mirror are on different filesystems)
        import shutil

        shutil.copy2(src, dst)
    return True


def fetch_tile(
    api_url: str, path: str, dst: Path, timeout: int = 60, retries: int = 3
) -> tuple[bool, str]:
    url = api_url.rstrip("/") + "/tile?" + urllib.parse.urlencode({"path": path})
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return True, ""
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2**attempt)
    return False, f"{last_err}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retrieval-dir",
        default="/scratch/users/zwcolin/cxr_embeds/sft_data/retrieval_raw",
    )
    p.add_argument(
        "--dataset-dir",
        default="/scratch/users/zwcolin/cxr_embeds/external_data/screenshot-training-natural-filtered-v2",
        help="Local dataset root (used to shortcut gold tiles with a hardlink)",
    )
    p.add_argument(
        "--mirror-dir",
        default=None,
        help="Where to write tile mirror; default <retrieval-dir>/tiles",
    )
    p.add_argument("--api-url", default="http://localhost:30895")
    p.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=5000)
    args = p.parse_args()

    retrieval_dir = Path(args.retrieval_dir)
    local_root = Path(args.dataset_dir)
    mirror = Path(args.mirror_dir) if args.mirror_dir else retrieval_dir / "tiles"
    mirror.mkdir(parents=True, exist_ok=True)
    failed_log = retrieval_dir / "failed.txt"

    print(f"Retrieval dir: {retrieval_dir}")
    print(f"Mirror:        {mirror}")
    print(f"Local dataset: {local_root}")
    print(f"Splits:        {args.splits}")
    print(f"Workers:       {args.workers}")
    print()

    print("Collecting unique paths...")
    by_suffix = collect_paths(retrieval_dir, args.splits)
    print(f"  Unique tile suffixes: {len(by_suffix):,}")

    # Split into linkable-from-local vs must-fetch
    need_fetch: list[tuple[str, str]] = []
    linked = 0
    already = 0
    for suffix, abs_fetch in by_suffix.items():
        dst = mirror / suffix
        if dst.exists():
            already += 1
            continue
        if try_local_link(suffix, local_root, mirror):
            linked += 1
            continue
        if abs_fetch is None:
            # gold not in local AND no hit supplied the abs path → fall back by translating suffix
            abs_fetch = f"/opt/dlami/nvme/kiwix_tiles/{suffix}"
        need_fetch.append((suffix, abs_fetch))

    print(f"  Already on mirror: {already:,}")
    print(f"  Hardlinked from local: {linked:,}")
    print(f"  Must fetch via /tile: {len(need_fetch):,}")

    if not need_fetch:
        print("Nothing to fetch. Done.")
        return

    # Fetch with thread pool
    print(f"\nFetching {len(need_fetch):,} tiles with {args.workers} workers...")
    t0 = time.time()
    ok = 0
    fail = 0
    fail_paths = []
    with open(failed_log, "w") as f_fail:

        def _work(item: tuple[str, str]) -> tuple[str, str, bool, str]:
            suffix, abs_path = item
            dst = mirror / suffix
            success, err = fetch_tile(args.api_url, abs_path, dst)
            return suffix, abs_path, success, err

        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work, it) for it in need_fetch]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                suffix, abs_path, success, err = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    fail_paths.append((suffix, abs_path, err))
                    f_fail.write(f"{abs_path}\t{err}\n")
                    f_fail.flush()

                if i % args.progress_every == 0 or i == len(need_fetch):
                    el = time.time() - t0
                    rate = i / max(el, 1e-9)
                    eta = (len(need_fetch) - i) / max(rate, 1e-9) / 60
                    print(
                        f"  [{i}/{len(need_fetch)}] ok={ok} fail={fail} "
                        f"{rate:.1f} tile/s  eta {eta:.1f} min",
                        flush=True,
                    )

    el = time.time() - t0
    print(f"\nDone in {el / 60:.1f} min. ok={ok} fail={fail}")
    if fail:
        print(f"Failed paths logged to: {failed_log}")

    # Final mirror size
    import subprocess

    try:
        du = (
            subprocess.check_output(["du", "-sh", str(mirror)], timeout=120)
            .decode()
            .split()[0]
        )
        print(f"Mirror size: {du}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
