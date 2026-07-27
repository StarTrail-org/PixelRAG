#!/usr/bin/env python3
"""
Screenshot benchmark with correctness verification.

Bench is a clean measurement harness. It takes a strategy object, runs it,
verifies results against cached GT, and dumps config + results.

Strategies live in pixelrag_render.strategies — bench does NOT know about
specific strategy implementations or naming conventions.

Usage (programmatic):
    from pixelrag_render.strategies import CDPSequentialStrategy
    from pixelrag_render.bench.bench_throughput import Bench

    bench = Bench(zim_path="...", chrome_path="...", output_dir="./results")
    strategy = CDPSequentialStrategy(chrome_path=..., n_workers=32, fmt="raw")
    result = await bench.run(strategy)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pixelrag_render.strategies.base import TileCapture
from pixelrag_render.strategies.cdp_sequential import (
    VIEWPORT_WIDTH,
    CDPSequentialStrategy,
)

CORRECT_THRESHOLD = 99.0
JPEG_MAX_MEAN_DIFF = 5.0
LOSSLESS_MAX_MEAN_DIFF = 3.0


# ---------------------------------------------------------------------------
# Article preparation
# ---------------------------------------------------------------------------


def prepare_articles(
    zim_path: str, n: int, seed: int = 42, kiwix_url: str | None = None
) -> list[dict]:
    """Sample articles from ZIM.

    kiwix_url can be:
    - None: write HTML to temp files (file:// mode)
    - "http://host:port": single kiwix-serve instance
    - "http://host:9461,http://host:9462,...": multiple instances (round-robin)
    """
    from urllib.parse import quote

    from libzim.reader import Archive

    archive = Archive(zim_path)

    # Support multiple kiwix URLs (comma-separated)
    kiwix_urls = kiwix_url.split(",") if kiwix_url else []
    # Detect book_name from first URL or ZIM filename
    if kiwix_urls:
        # Extract book_name from URL: http://host:port/content/{book_name}/...
        # For symlinks like wiki_1.zim, book_name = wiki_1
        # We need to figure out the right book_name for each URL
        pass
    book_name = Path(zim_path).stem
    rng = random.Random(seed)
    articles = []
    tried = 0
    while len(articles) < n and tried < n * 20:
        idx = rng.randint(0, archive.all_entry_count - 1)
        tried += 1
        try:
            e = archive._get_entry_by_id(idx)
            if e.is_redirect or e.path.startswith("-/") or len(e.path) <= 2:
                continue
            entry = archive.get_entry_by_path(e.path)
            item = entry.get_item()
            if "html" not in item.mimetype:
                continue
            html = bytes(item.content).decode("utf-8")
            if 'http-equiv="refresh"' in html.lower() or len(html) < 300:
                continue

            if kiwix_urls:
                safe = "/:@!$&'()*+,;="
                # Round-robin across kiwix instances
                base = kiwix_urls[len(articles) % len(kiwix_urls)]
                # Detect book_name from the symlink/ZIM each instance serves
                parts = base.rstrip("/").rsplit(":", 1)
                port = int(parts[1]) if len(parts) > 1 else 9454
                # Each instance may have different book_name (wiki_1, wiki_2, etc.)
                bname = f"wiki_{port - 9460}" if port > 9460 else book_name
                url = f"{base}/content/{bname}/{quote(e.path, safe=safe)}"
                articles.append({"path": e.path, "file": url})
            else:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, dir="/tmp", prefix="bench_"
                )
                tmp.write(html.encode())
                tmp.close()
                articles.append({"path": e.path, "file": tmp.name})
        except Exception:
            continue
    return articles


def cleanup_articles(articles: list[dict]):
    for a in articles:
        if a["file"].startswith("http"):
            continue
        try:
            os.unlink(a["file"])
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Ground truth (cached)
# ---------------------------------------------------------------------------


def gt_cache_key(articles: list[dict], seed: int) -> str:
    paths = sorted(a["path"] for a in articles)
    content = f"seed={seed}\n" + "\n".join(paths)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def generate_ground_truth(
    articles: list[dict],
    chrome_path: str,
    cache_dir: Path,
    seed: int,
    timeout_ms: int = 5000,
) -> dict[str, list[Path]]:
    cache_key = gt_cache_key(articles, seed)
    manifest_path = cache_dir / f"gt_{cache_key}.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        all_exist = all(Path(p).exists() for paths in manifest.values() for p in paths)
        if all_exist:
            result = {k: [Path(p) for p in v] for k, v in manifest.items()}
            total = sum(len(v) for v in result.values())
            print(
                f"Ground truth cache hit: {len(result)} articles, {total} tiles",
                flush=True,
            )
            return result

    cache_dir.mkdir(parents=True, exist_ok=True)
    strategy = _make_gt_strategy(chrome_path, timeout_ms)
    await strategy.setup()
    try:
        results = await strategy.capture_articles(articles)
    finally:
        await strategy.teardown()

    ground_truth = {}
    for ac in results:
        tile_paths = []
        for tc in ac.tiles:
            tile_path = (
                cache_dir
                / f"gt_{cache_key}_{ac.article_path.replace('/', '_')}_{tc.tile_index:02d}.png"
            )
            if tc.image_bytes:
                tile_path.write_bytes(tc.image_bytes)
            tile_paths.append(tile_path)
        ground_truth[ac.article_path] = tile_paths

    manifest = {k: [str(p) for p in v] for k, v in ground_truth.items()}
    manifest_path.write_text(json.dumps(manifest))

    total = sum(len(v) for v in ground_truth.values())
    print(
        f"Ground truth generated: {len(ground_truth)} articles, {total} tiles",
        flush=True,
    )
    return ground_truth


def _make_gt_strategy(chrome_path: str, timeout_ms: int):
    """GT uses the most conservative strategy: 1 worker, PNG, long timeout.

    Uses port 9222 to avoid TIME_WAIT conflicts with test strategies (9300+).
    """
    s = CDPSequentialStrategy(
        chrome_path=chrome_path, n_workers=1, fmt="png", from_surface=True
    )
    s._base_port = 9222
    return s


def validate_gt(ground_truth: dict[str, list[Path]]) -> tuple[int, int, list[str]]:
    """Validate GT tiles are non-degenerate (not blank, not tiny, readable).

    Returns (ok, bad, bad_examples).
    """
    ok = 0
    bad = 0
    examples = []
    for article_path, tile_paths in ground_truth.items():
        for tp in tile_paths:
            try:
                img = Image.open(tp)
                arr = np.array(img)
                if arr.std() < 1.0:
                    bad += 1
                    if len(examples) < 10:
                        examples.append(
                            f"{article_path} {tp.name}: blank (std={arr.std():.1f})"
                        )
                    continue
                ok += 1
            except Exception as e:
                bad += 1
                if len(examples) < 10:
                    examples.append(f"{article_path} {tp.name}: {e}")
    return ok, bad, examples


# ---------------------------------------------------------------------------
# Decode + verify (NOT timed)
# ---------------------------------------------------------------------------


def decode_tile(tc: TileCapture) -> Image.Image | None:
    try:
        if tc.raw_file_path and os.path.exists(tc.raw_file_path):
            data = open(tc.raw_file_path, "rb").read()
            w, h, rb = struct.unpack_from("<III", data, 0)
            img = Image.frombuffer(
                "RGBA", (w, h), data[12:], "raw", "BGRA", rb, 1
            ).convert("RGB")
            return img
        elif tc.image_bytes:
            return Image.open(io.BytesIO(tc.image_bytes)).convert("RGB")
    except Exception:
        return None
    return None


def verify_tile(
    captured: Image.Image, gt_path: Path, is_lossy: bool
) -> tuple[bool, float]:
    gt = Image.open(gt_path).convert("RGB")
    cap_arr = np.array(captured, dtype=np.float32)
    gt_arr = np.array(gt, dtype=np.float32)
    if cap_arr.shape != gt_arr.shape:
        return False, 999.0
    diff = np.abs(cap_arr - gt_arr)
    mean_diff = float(diff.mean())
    threshold = JPEG_MAX_MEAN_DIFF if is_lossy else LOSSLESS_MAX_MEAN_DIFF
    return mean_diff <= threshold, mean_diff


# ---------------------------------------------------------------------------
# Run one strategy: time capture, then verify separately
# ---------------------------------------------------------------------------


async def run_and_verify(strategy, articles, ground_truth) -> dict:
    await strategy.setup()
    t0 = time.monotonic()
    try:
        article_captures = await strategy.capture_articles(articles)
    finally:
        wall_s = time.monotonic() - t0
        await strategy.teardown()

    # --- UNTIMED: decode + verify ---
    tiles_ok = 0
    tiles_bad = 0
    tiles_total = 0
    total_shot_ms = 0.0
    total_nav_ms = 0.0
    total_pixels = 0
    total_height_px = 0
    per_tile_shot_ms = []
    per_tile_nav_ms = []
    bad_examples = []
    is_lossy = strategy.fmt in ("jpeg",)

    for ac in article_captures:
        gt_tiles = ground_truth.get(ac.article_path, [])
        total_height_px += ac.page_height
        total_shot_ms += ac.total_shot_ms
        total_nav_ms += ac.total_nav_ms

        for tc in ac.tiles:
            tiles_total += 1
            total_pixels += VIEWPORT_WIDTH * tc.clip_h
            per_tile_shot_ms.append(tc.shot_ms)
            if tc.nav_ms > 0:
                per_tile_nav_ms.append(tc.nav_ms)

            if tc.tile_index >= len(gt_tiles):
                tiles_bad += 1
                bad_examples.append(f"{ac.article_path} tile {tc.tile_index}: no GT")
                continue

            img = decode_tile(tc)
            if img is None:
                tiles_bad += 1
                bad_examples.append(
                    f"{ac.article_path} tile {tc.tile_index}: decode failed"
                )
                continue

            ok, mean_diff = verify_tile(img, gt_tiles[tc.tile_index], is_lossy)
            if ok:
                tiles_ok += 1
            else:
                tiles_bad += 1
                if len(bad_examples) < 10:
                    bad_examples.append(
                        f"{ac.article_path} tile {tc.tile_index}: mean_diff={mean_diff:.2f}"
                    )

    for ac in article_captures:
        for tc in ac.tiles:
            if tc.raw_file_path:
                try:
                    os.unlink(tc.raw_file_path)
                except OSError:
                    pass

    correct_pct = tiles_ok / tiles_total * 100 if tiles_total > 0 else 0
    tps = tiles_total / wall_s if wall_s > 0 else 0
    ms_per_tile = total_shot_ms / tiles_total if tiles_total > 0 else 0
    articles_per_s = len(article_captures) / wall_s if wall_s > 0 else 0
    mpix_per_s = (total_pixels / 1_000_000) / wall_s if wall_s > 0 else 0
    shot_share = (
        total_shot_ms / (total_shot_ms + total_nav_ms)
        if (total_shot_ms + total_nav_ms) > 0
        else 0
    )

    # Latency percentiles
    sorted_shots = sorted(per_tile_shot_ms) if per_tile_shot_ms else [0]
    sorted_navs = sorted(per_tile_nav_ms) if per_tile_nav_ms else [0]

    def percentile(arr, p):
        idx = int(len(arr) * p / 100)
        return arr[min(idx, len(arr) - 1)]

    return {
        "name": strategy.name,
        "tiles_total": tiles_total,
        "tiles_ok": tiles_ok,
        "tiles_bad": tiles_bad,
        "correct_pct": correct_pct,
        "wall_s": wall_s,
        "tiles_per_s": tps,
        "ms_per_tile": ms_per_tile,
        "articles_per_s": articles_per_s,
        "mpix_per_s": mpix_per_s,
        "height_kpx_per_s": (total_height_px / 1000) / wall_s if wall_s > 0 else 0,
        "shot_pct": shot_share * 100,
        "bad_examples": bad_examples,
        # Latency distribution
        "shot_min": sorted_shots[0],
        "shot_p50": percentile(sorted_shots, 50),
        "shot_p95": percentile(sorted_shots, 95),
        "shot_p99": percentile(sorted_shots, 99),
        "shot_max": sorted_shots[-1],
        "nav_avg": sum(sorted_navs) / len(sorted_navs),
        "nav_p95": percentile(sorted_navs, 95),
    }


# ---------------------------------------------------------------------------
# Bench: clean harness that takes any CaptureStrategy
# ---------------------------------------------------------------------------


class Bench:
    """Benchmark harness. Measures throughput/latency and verifies correctness.

    Usage:
        bench = Bench(zim_path="...", chrome_path="...", output_dir="./results")
        result = await bench.run(strategy, articles=200, seed=42)
    """

    def __init__(
        self,
        zim_path: str,
        chrome_path: str,
        output_dir: str = "./bench_results",
        kiwix_url: str | None = None,
        gt_timeout_ms: int = 5000,
    ):
        self.zim_path = zim_path
        self.chrome_path = chrome_path
        self.output_dir = Path(output_dir)
        self.kiwix_url = kiwix_url
        self.gt_timeout_ms = gt_timeout_ms
        self._articles: list[dict] | None = None
        self._gt: dict[str, list[Path]] | None = None

    def prepare(self, n_articles: int = 200, seed: int = 42) -> list[dict]:
        if self._articles is None:
            self._articles = prepare_articles(
                self.zim_path, n_articles, seed, kiwix_url=self.kiwix_url
            )
        return self._articles

    async def ensure_gt(
        self, n_articles: int = 200, seed: int = 42
    ) -> dict[str, list[Path]]:
        if self._gt is not None:
            return self._gt
        articles = self.prepare(n_articles, seed)
        gt_dir = self.output_dir / "ground_truth"
        self._gt = await generate_ground_truth(
            articles, self.chrome_path, gt_dir, seed, timeout_ms=self.gt_timeout_ms
        )
        ok, bad, examples = validate_gt(self._gt)
        total = ok + bad
        print(f"GT validation: {ok}/{total} OK, {bad} bad", flush=True)
        if examples:
            for ex in examples:
                print(f"  GT BAD: {ex}", flush=True)
        if bad > 0:
            pct = ok / total * 100 if total else 0
            if pct < CORRECT_THRESHOLD:
                raise RuntimeError(
                    f"GT itself is only {pct:.1f}% valid ({bad} bad tiles). "
                    f"Fix image loading or increase gt_timeout_ms."
                )
        return self._gt

    async def run(self, strategy, n_articles: int = 200, seed: int = 42) -> dict:
        articles = self.prepare(n_articles, seed)
        gt = await self.ensure_gt(n_articles, seed)
        result = await run_and_verify(strategy, articles, gt)

        exp = self._build_experiment(strategy, result, n_articles, seed)
        try:
            self._dump_experiment(exp)
        except Exception as e:
            print(f"Warning: failed to save experiment: {e}", flush=True)
        return result

    def _build_experiment(
        self, strategy, result: dict, n_articles: int, seed: int
    ) -> dict:
        config = {
            "strategy_class": type(strategy).__name__,
            "strategy_name": strategy.name,
            "n_workers": getattr(strategy, "n_workers", None),
            "fmt": strategy.fmt,
            "launcher": getattr(strategy, "launcher", None),
            "from_surface": getattr(strategy, "from_surface", None),
            "chrome_path": getattr(strategy, "chrome_path", None),
            "n_articles": n_articles,
            "seed": seed,
            "zim_path": self.zim_path,
            "kiwix_url": self.kiwix_url,
            "gt_timeout_ms": self.gt_timeout_ms,
        }
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": config,
            "results": {k: v for k, v in result.items() if k != "bad_examples"},
            "bad_examples": result.get("bad_examples", []),
        }

    def _dump_experiment(self, exp: dict):
        exp_dir = self.output_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        ts = exp["timestamp"].replace(":", "")
        name = exp["config"]["strategy_name"].replace(" ", "_").replace("/", "-")
        path = exp_dir / f"{ts}_{name}.json"
        path.write_text(json.dumps(exp, indent=2, default=str))
        print(f"Experiment saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_result_line(r: dict) -> str:
    status = "PASS" if r["correct_pct"] >= CORRECT_THRESHOLD else "FAIL"
    ok = f"{r['tiles_ok']}/{r['tiles_total']}"
    return (
        f"  {r['name']:<25} {ok:>7} {r['correct_pct']:>5.1f}% "
        f"{r['tiles_per_s']:>6.1f} {r['ms_per_tile']:>5.0f} "
        f"{r['shot_pct']:>4.0f}%  {status}"
    )


def print_results(results: list[dict]):
    for r in results:
        print(format_result_line(r), flush=True)
        if r["bad_examples"]:
            for ex in r["bad_examples"][:3]:
                print(f"    BAD: {ex}", flush=True)
