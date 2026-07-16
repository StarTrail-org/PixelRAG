"""Retrieval strategies for SimpleQA evaluation.

This module defines retrieval strategies that work with prepared data:
- NaiveRetriever: No retrieval, just pass query to LLM
- ScreenshotRetriever: Use pre-captured screenshot for the example
- TextRetriever: Use pre-fetched or cached text for the example
- VectorRetriever: Search across all screenshots using vector similarity
"""

import base64
import io
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# POST timeout (seconds) for a retrieval call to the search serve. The default suits a fast
# serve; raise it for a slow one — e.g. on-demand render, where a batch can take minutes:
# `PIXELRAG_RETRIEVAL_TIMEOUT=7200`. Otherwise the request times out, the batch is cached
# empty, and the reader silently falls back to closed-book (looks like a bad score, not an error).
_RETRIEVAL_TIMEOUT = float(os.environ.get("PIXELRAG_RETRIEVAL_TIMEOUT", "600"))


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    # Text content (for text-based retrieval)
    text: str | None = None

    # Image paths with scores (for vector retrieval)
    images: list[tuple[str, float]] = field(default_factory=list)

    # Per-image source URLs, aligned with ``images`` when provided.
    image_urls: list[str | None] = field(default_factory=list)

    # Base64 encoded image (for screenshot)
    base64_image: str | None = None

    # Source URL
    source_url: str | None = None

    # Which retrieval type was used
    retrieval_type: str = "naive"

    # Path to pixel query image used for retrieval embedding (rendered card or raw photo)
    pixel_query_path: str | None = None

    # Path to raw species/landmark photo for generation (always the original photo,
    # never the rendered card). If None, falls back to pixel_query_path in build_messages.
    query_image_path: str | None = None

    @property
    def has_content(self) -> bool:
        """Check if retrieval found any content."""
        return bool(self.text or self.images or self.base64_image)


class BaseRetriever(ABC):
    """Base class for retrieval strategies."""

    @abstractmethod
    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve relevant content for the query.

        Args:
            query: The question/query text.
            example: The full example dict (may contain metadata, prepared data, etc.).

        Returns:
            RetrievalResult with retrieved content.
        """
        raise NotImplementedError


# EVQA query image data dirs (iNaturalist 2021, Google Landmarks v2)
_INAT2021_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "inat2021",
)
_LANDMARK_V2_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "landmark_v2",
)


def _get_inat_image_path_for_example(example: dict, tiles_dir: str) -> str | None:
    """Get iNaturalist 2021 query image path. dataset_name must be 'inaturalist'."""
    inat_ids = example.get("inat_image_ids", [])
    if not inat_ids:
        return None
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "inat_images")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    local_path = os.path.join(cache_dir, f"{example_id}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    import shutil

    id_map = TiledQwen3VLEmbeddingRetriever._load_inat2021_mapping()
    for str_id in inat_ids:
        try:
            img_id = int(str_id)
        except ValueError:
            continue
        file_name = id_map.get(img_id)
        if not file_name:
            continue
        src_path = os.path.join(_INAT2021_DATA_DIR, file_name)
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            shutil.copy2(src_path, local_path)
            return local_path
    logger.warning(f"Failed to find iNaturalist image for {example_id}")
    return None


def _get_landmark_image_path_for_example(
    example: dict, tiles_dir: str, quiet: bool = False
) -> str | None:
    """Get Google Landmarks v2 query image path. dataset_name must be 'landmarks'.

    GLDv2 stores images as {split}/{a}/{b}/{c}/{id}.jpg (a,b,c = first 3 chars of id).
    Searches train, index, test in order.
    """
    ids = example.get("dataset_image_ids_parsed", [])
    if not ids:
        return None
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "landmark_images")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    local_path = os.path.join(cache_dir, f"{example_id}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    import shutil

    data_dir = _LANDMARK_V2_DATA_DIR
    for img_id in ids:
        if len(img_id) < 3:
            continue
        # GLDv2 path: {split}/{a}/{b}/{c}/{id}.jpg
        subpath = f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
        for split in ("train", "index", "test"):
            src_path = os.path.join(data_dir, split, subpath)
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                shutil.copy2(src_path, local_path)
                return local_path
    # Fallback: download from train.csv URL (requires data/landmark_v2/train.csv)
    # Try each img_id in order; first URL may be 404, others might work
    for img_id in ids:
        if _try_download_landmark_from_url(example_id, img_id, local_path):
            return local_path
    if not quiet:
        logger.warning(
            f"Failed to find Landmark image for {example_id} (data in {data_dir}?)"
        )
    return None


def _try_download_landmark_from_url(
    example_id: str, img_id: str, local_path: str
) -> bool:
    """Try to download landmark image from train.csv URL. Used when GLDv2 TARs unavailable.

    Returns True if download succeeded and file is valid, False otherwise.
    """
    import urllib.request

    train_csv = os.path.join(_LANDMARK_V2_DATA_DIR, "train.csv")
    if not os.path.exists(train_csv):
        return False
    import csv

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            if row.get("id") == img_id:
                url = row.get("url", "")
                if url:
                    try:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": "PixelRAG-Bot/1.0"}
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = resp.read()
                        if len(data) >= 1000:
                            with open(local_path, "wb") as out:
                                out.write(data)
                            return True
                    except Exception as e:
                        logger.debug(
                            f"URL download failed for {example_id} (img_id={img_id}): {e}"
                        )
                return False
    return False


def _get_query_image_path_for_example(
    example: dict, tiles_dir: str, quiet: bool = False
) -> str | None:
    """Get EVQA query image path. Dispatches by dataset_name: inaturalist | landmarks."""
    ds = (example.get("dataset_name") or "").lower()
    if ds == "inaturalist":
        return _get_inat_image_path_for_example(example, tiles_dir)
    if ds == "landmarks":
        return _get_landmark_image_path_for_example(example, tiles_dir, quiet=quiet)
    # Fallback: try inaturalist (backward compat when dataset_name missing)
    return _get_inat_image_path_for_example(example, tiles_dir)


def _get_all_inat_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL iNaturalist query image paths for an example (not just the first)."""
    inat_ids = example.get("inat_image_ids", [])
    if not inat_ids:
        return []
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "inat_images_multi")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    import shutil

    id_map = TiledQwen3VLEmbeddingRetriever._load_inat2021_mapping()
    paths = []
    for i, str_id in enumerate(inat_ids):
        local_path = os.path.join(cache_dir, f"{example_id}_{i}.jpg")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            paths.append(local_path)
            continue
        try:
            img_id = int(str_id)
        except ValueError:
            continue
        file_name = id_map.get(img_id)
        if not file_name:
            continue
        src_path = os.path.join(_INAT2021_DATA_DIR, file_name)
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            shutil.copy2(src_path, local_path)
            paths.append(local_path)
    return paths


_landmark_url_map_cache: dict[str, str] | None = None


def _load_landmark_url_map() -> dict[str, str]:
    """Load GLDv2 train.csv: img_id -> url. Cached after first call."""
    global _landmark_url_map_cache
    if _landmark_url_map_cache is not None:
        return _landmark_url_map_cache
    import csv

    train_csv = os.path.join(_LANDMARK_V2_DATA_DIR, "train.csv")
    if not os.path.exists(train_csv):
        _landmark_url_map_cache = {}
        return _landmark_url_map_cache
    url_map = {}
    with open(train_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = row.get("id", "").strip()
            url = row.get("url", "").strip()
            if img_id and url:
                url_map[img_id] = url
    _landmark_url_map_cache = url_map
    logger.info(f"Loaded landmark URL map: {len(url_map)} entries")
    return url_map


def _download_landmark_image_by_id(img_id: str, local_path: str) -> bool:
    """Download a landmark image by its GLDv2 ID. Returns True on success."""
    import urllib.request

    url_map = _load_landmark_url_map()
    url = url_map.get(img_id)
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PixelRAG-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) >= 1000:
            with open(local_path, "wb") as out:
                out.write(data)
            return True
    except Exception as e:
        logger.debug(f"Download failed for landmark {img_id}: {e}")
    return False


def _get_all_landmark_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL Google Landmarks query image paths for an example (not just the first)."""
    ids = example.get("dataset_image_ids_parsed", [])
    if not ids:
        return []
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "landmark_images_multi")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    import shutil

    data_dir = _LANDMARK_V2_DATA_DIR
    paths = []
    for i, img_id in enumerate(ids):
        local_path = os.path.join(cache_dir, f"{example_id}_{i}.jpg")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            paths.append(local_path)
            continue
        if len(img_id) < 3:
            continue
        subpath = f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
        found = False
        for split in ("train", "index", "test"):
            src_path = os.path.join(data_dir, split, subpath)
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                shutil.copy2(src_path, local_path)
                paths.append(local_path)
                found = True
                break
        if not found:
            if _download_landmark_image_by_id(img_id, local_path):
                paths.append(local_path)
    return paths


def _get_all_query_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL query image paths for an EVQA example (all available images, not just the first).

    Falls back to the single ``query_image_path`` / ``_get_query_image_path_for_example``
    when the multi-image helpers return nothing (e.g. ``dataset_image_ids_parsed`` lives
    inside ``original_data`` rather than at top level).
    """
    ds = (example.get("dataset_name") or "").lower()
    if ds not in ("inaturalist", "landmarks"):
        od = example.get("original_data", {})
        if isinstance(od, str):
            import ast

            try:
                od = ast.literal_eval(od)
            except Exception:
                od = {}
        ds = (od.get("dataset_name") or "").lower()
    if ds == "inaturalist":
        paths = _get_all_inat_image_paths(example, tiles_dir)
    elif ds == "landmarks":
        paths = _get_all_landmark_image_paths(example, tiles_dir)
    else:
        paths = _get_all_inat_image_paths(example, tiles_dir)
    if not paths:
        single = example.get("query_image_path") or _get_query_image_path_for_example(
            example, tiles_dir, quiet=True
        )
        if single and os.path.exists(single):
            paths = [single]
    return paths


class NaiveRetriever(BaseRetriever):
    """No retrieval - returns empty result, LLM answers from its own knowledge."""

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        return RetrievalResult(retrieval_type="naive")


class EVQANoRetrievalRetriever(BaseRetriever):
    """EVQA without retrieval: query + iNaturalist image only, no Wikipedia tiles.

    Used to test VLM's ability to answer from the species image alone.
    """

    def __init__(self, tiles_dir: str = "tiles/evqa"):
        self.tiles_dir = tiles_dir

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        inat_image_path = _get_query_image_path_for_example(example, self.tiles_dir)
        return RetrievalResult(
            images=[],
            retrieval_type="evqa_no_retrieval_multimodal",
            pixel_query_path=inat_image_path,
            query_image_path=inat_image_path,
        )


def _save_task_query_image(
    example: dict, task_name: str, base_dir: str = "tiles"
) -> str | None:
    """Save query image from any task to disk. Returns path or None.
    Images saved to {base_dir}/{task_name}_images/{example_id}.png
    Works with PIL images, base64 strings, or dict with 'bytes' key.
    """
    img = example.get("image")
    if img is None:
        return None
    example_id = example.get("id", "unknown")
    save_dir = os.path.join(base_dir, f"{task_name}_images")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{example_id}.png")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        if hasattr(img, "save"):
            img.save(out_path, format="PNG")
            return out_path
        if isinstance(img, str):
            raw = (
                img.split(",", 1)[1] if img.startswith("data:") and "," in img else img
            )
            data = base64.b64decode(raw)
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        if isinstance(img, dict) and "bytes" in img:
            b = img["bytes"]
            if b:
                with open(out_path, "wb") as f:
                    f.write(b)
                return out_path
    except Exception as e:
        logger.warning(f"Failed to save {task_name} image for {example_id}: {e}")
    return None


def _save_worldvqa_query_image(example: dict, base_dir: str = "tiles") -> str | None:
    """Save WorldVQA query image to disk. Returns path or None.
    Images saved to {base_dir}/worldvqa_images/{example_id}.png
    """
    img = example.get("image")
    if img is None:
        return None
    example_id = example.get("id", "unknown")
    save_dir = os.path.join(base_dir, "worldvqa_images")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{example_id}.png")

    try:
        if hasattr(img, "save"):
            img.save(out_path, format="PNG")
            return out_path
        if isinstance(img, str):
            raw = (
                img.split(",", 1)[1] if img.startswith("data:") and "," in img else img
            )
            data = base64.b64decode(raw)
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            out_path = os.path.join(save_dir, f"{example_id}{ext}")
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        if isinstance(img, dict) and "bytes" in img:
            b = img["bytes"]
            if b:
                ext = ".png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                out_path = os.path.join(save_dir, f"{example_id}{ext}")
                with open(out_path, "wb") as f:
                    f.write(b)
                return out_path
    except Exception as e:
        logger.warning(f"Failed to save WorldVQA image for {example_id}: {e}")
    return None


def _worldvqa_image_to_base64(img) -> str | None:
    """Convert WorldVQA image (PIL, base64 str, or dict) to base64 string."""
    if img is None:
        return None
    if isinstance(img, str):
        if img.startswith("data:"):
            if "," in img:
                return img.split(",", 1)[1]
        return img
    if hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    if isinstance(img, dict) and "bytes" in img:
        b = img["bytes"]
        return base64.b64encode(b).decode() if b else None
    return None


class WorldVQANoRetrievalRetriever(BaseRetriever):
    """WorldVQA without retrieval: query + image from dataset only.

    WorldVQA images are embedded in the HuggingFace dataset (PIL or base64).
    """

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        img = example.get("image")
        base64_img = _worldvqa_image_to_base64(img)
        return RetrievalResult(
            base64_image=base64_img,
            retrieval_type="worldvqa_no_retrieval",
        )


class ScreenshotRetriever(BaseRetriever):
    """Use screenshot that was prepared in data layer.

    Expects screenshot to be captured beforehand. This retriever just
    loads and encodes the existing screenshot.

    For ground truth evaluation, uses encode_screenshot_for_vlm_async which
    does NOT apply max_height limit. You can control max_pixels to study
    the effect of resize on VLM performance.

    Args:
        screenshot_dir: Directory containing screenshots.
        max_pixels: Maximum pixels before resize. If None, no resize (89M limit).
                    Common values:
                    - None: No resize (let VLM handle it)
                    - 16_777_216 (16M): Qwen3-VL default, ~16K tokens
                    - 4_000_000 (4M): ~4K tokens
                    - 1_000_000 (1M): ~1K tokens
    """

    def __init__(
        self, screenshot_dir: str = "screenshots", max_pixels: int | None = None
    ):
        self.screenshot_dir = screenshot_dir
        self.max_pixels = max_pixels

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import (
            capture_screenshot_async,
            encode_screenshot_for_vlm_async,
            extract_url_from_metadata,
        )

        # Get or capture screenshot
        screenshot_path = await capture_screenshot_async(example, self.screenshot_dir)

        if not screenshot_path:
            return RetrievalResult(
                retrieval_type="screenshot",
                source_url=extract_url_from_metadata(example),
            )

        # Encode to base64 with configurable max_pixels
        base64_image = await encode_screenshot_for_vlm_async(
            screenshot_path, max_pixels=self.max_pixels
        )

        return RetrievalResult(
            base64_image=base64_image,
            source_url=extract_url_from_metadata(example),
            retrieval_type="screenshot",
        )


class TiledScreenshotRetriever(BaseRetriever):
    """Use tiled screenshot from ground truth URL.

    Captures screenshot for the example's URL, splits it into tiles,
    and returns tiles. This is ground truth (not vector search).

    Args:
        max_tiles: Maximum number of tiles to return. If None, returns all tiles.
                   For context-aware limiting, calculate based on model context length.
                   Rough estimate: max_tiles = (context_length - 2000) / tokens_per_tile
                   where tokens_per_tile ≈ 1500-2000 for most VLMs.
    """

    def __init__(
        self,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        max_tiles: int | None = None,
    ):
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tiles = max_tiles
        os.makedirs(tiles_dir, exist_ok=True)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import (
            capture_screenshot_async,
            encode_screenshot_async,
            extract_url_from_metadata,
            split_image_to_tiles,
        )

        # Get or capture screenshot
        screenshot_path = await capture_screenshot_async(example, self.screenshot_dir)

        if not screenshot_path:
            return RetrievalResult(
                retrieval_type="tiled_screenshot",
                source_url=extract_url_from_metadata(example),
            )

        # Split into tiles
        example_id = example.get("id", "unknown")
        example_tiles_dir = os.path.join(self.tiles_dir, example_id)
        tile_paths = split_image_to_tiles(
            screenshot_path,
            example_tiles_dir,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        if not tile_paths:
            # Fall back to full screenshot
            base64_image = await encode_screenshot_async(screenshot_path)
            return RetrievalResult(
                base64_image=base64_image,
                source_url=extract_url_from_metadata(example),
                retrieval_type="tiled_screenshot",
            )

        # Limit tiles if max_tiles is set
        if self.max_tiles is not None and len(tile_paths) > self.max_tiles:
            logger.info(f"Limiting tiles from {len(tile_paths)} to {self.max_tiles}")
            tile_paths = tile_paths[: self.max_tiles]

        # Return tiles as images list (path, score=1.0 for ground truth)
        images = [(path, 1.0) for path in tile_paths]

        return RetrievalResult(
            images=images,
            source_url=extract_url_from_metadata(example),
            retrieval_type="tiled_screenshot",
        )


class TextRetriever(BaseRetriever):
    """Use text content fetched from URL.

    Can use pre-cached text or fetch on demand.
    """

    def __init__(
        self,
        max_chars: int = 50000,
        text_cache: dict | None = None,
        cache_path: str | None = None,
    ):
        self.max_chars = max_chars
        self.text_cache = text_cache
        self.cache_path = cache_path
        self._cache_lock = asyncio.Lock()

    async def _save_to_cache(self, example_id: str, text: str, url: str):
        """Append result to cache file."""
        if not self.cache_path:
            return
        try:
            import json

            async with self._cache_lock:
                with open(self.cache_path, "a") as f:
                    cache_entry = {"id": example_id, "text": text, "url": url}
                    f.write(json.dumps(cache_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import fetch_text_async

        example_id = example.get("id", "")
        was_cached = self.text_cache and example_id in self.text_cache

        text, source_url = await fetch_text_async(
            example, self.max_chars, self.text_cache
        )

        # Save to cache if not already cached
        if not was_cached and text and source_url:
            await self._save_to_cache(example_id, text, source_url)

        return RetrievalResult(
            text=text, source_url=source_url, retrieval_type="text_rag"
        )


class JinaReaderRetriever(BaseRetriever):
    """Use Jina Reader API to fetch clean markdown text from URL.

    Jina Reader (r.jina.ai) converts any URL to LLM-friendly markdown text.
    """

    def __init__(
        self,
        max_chars: int = 50000,
        api_key: str | None = None,
        text_cache: dict | None = None,
        cache_path: str | None = None,
    ):
        self.max_chars = max_chars
        self.api_key = api_key
        self.text_cache = text_cache
        self.cache_path = cache_path
        self._cache_lock = asyncio.Lock()

    async def _save_to_cache(self, example_id: str, text: str, url: str):
        """Append result to cache file."""
        if not self.cache_path:
            return
        try:
            import json

            async with self._cache_lock:
                with open(self.cache_path, "a") as f:
                    cache_entry = {"id": example_id, "text": text, "url": url}
                    f.write(json.dumps(cache_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        import aiohttp
        import asyncio
        from .simpleqa_data import extract_url_from_metadata

        # Check cache first
        example_id = example.get("id", "")
        if self.text_cache and example_id in self.text_cache:
            cached = self.text_cache[example_id]
            text = cached.get("text", "")
            source_url = cached.get("url", "")
            if text:
                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n\n[Content truncated...]"
                return RetrievalResult(
                    text=text, source_url=source_url, retrieval_type="jina_reader"
                )

        target_url = extract_url_from_metadata(example)
        if not target_url:
            return RetrievalResult(
                text="No URL found in metadata.", retrieval_type="jina_reader"
            )

        # Use Jina Reader API with retry logic
        reader_url = f"https://r.jina.ai/{target_url}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        reader_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        # Handle rate limiting (429) with exponential backoff
                        if response.status == 429:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt * 2, 30)  # Max 30 seconds
                                logger.warning(
                                    f"Rate limited (429) for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = f"Jina Reader API rate limited (429) after {max_retries} retries"
                                logger.error(f"{error_msg} for {target_url}")
                                return RetrievalResult(
                                    text=error_msg,
                                    source_url=target_url,
                                    retrieval_type="jina_reader",
                                )

                        # Handle server errors (5xx) with retry
                        if 500 <= response.status < 600:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt, 10)  # Max 10 seconds
                                logger.warning(
                                    f"Server error ({response.status}) for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = (
                                    f"Jina Reader API server error: {response.status}"
                                )
                                logger.error(f"{error_msg} for {target_url}")
                                return RetrievalResult(
                                    text=error_msg,
                                    source_url=target_url,
                                    retrieval_type="jina_reader",
                                )

                        # Handle client errors (4xx) - don't retry for most
                        if response.status == 200:
                            text = await response.text()
                            # Save to cache before truncation
                            await self._save_to_cache(example_id, text, target_url)
                            # Truncate if too long
                            if len(text) > self.max_chars:
                                text = (
                                    text[: self.max_chars]
                                    + "\n\n[Content truncated...]"
                                )
                            return RetrievalResult(
                                text=text,
                                source_url=target_url,
                                retrieval_type="jina_reader",
                            )
                        else:
                            # Other 4xx errors (403, 404, etc.) - don't retry
                            error_msg = f"Jina Reader API error: {response.status}"
                            logger.warning(f"{error_msg} for {target_url}")
                            return RetrievalResult(
                                text=error_msg,
                                source_url=target_url,
                                retrieval_type="jina_reader",
                            )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 10)  # Max 10 seconds
                    logger.warning(
                        f"Timeout for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Jina Reader fetch timeout after {max_retries} retries"
                    logger.error(f"{error_msg} for {target_url}")
                    return RetrievalResult(
                        text=error_msg,
                        source_url=target_url,
                        retrieval_type="jina_reader",
                    )
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 10)  # Max 10 seconds
                    logger.warning(
                        f"Client error for {target_url}: {e}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Jina Reader fetch failed: {e}"
                    logger.error(f"{error_msg} for {target_url}")
                    return RetrievalResult(
                        text=error_msg,
                        source_url=target_url,
                        retrieval_type="jina_reader",
                    )
            except Exception as e:
                error_msg = f"Jina Reader fetch failed: {e}"
                logger.error(f"{error_msg} for {target_url}")
                return RetrievalResult(
                    text=error_msg, source_url=target_url, retrieval_type="jina_reader"
                )

        # Should not reach here, but just in case
        error_msg = f"Jina Reader fetch failed after {max_retries} retries"
        return RetrievalResult(
            text=error_msg, source_url=target_url, retrieval_type="jina_reader"
        )


class WikipediaAPIRetriever(BaseRetriever):
    """Use Wikipedia API to fetch clean article text.

    Extracts Wikipedia page title from URL and fetches content via API.
    Much cleaner and faster than web scraping.
    """

    def __init__(
        self,
        max_chars: int = 50000,
        text_cache: dict | None = None,
        cache_path: str | None = None,
    ):
        self.max_chars = max_chars
        self.text_cache = text_cache
        self.cache_path = cache_path
        self._cache_lock = asyncio.Lock()

    async def _save_to_cache(self, example_id: str, text: str, url: str):
        """Append result to cache file."""
        if not self.cache_path:
            return
        try:
            import json

            async with self._cache_lock:
                with open(self.cache_path, "a") as f:
                    cache_entry = {"id": example_id, "text": text, "url": url}
                    f.write(json.dumps(cache_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _extract_wiki_title(self, url: str) -> str | None:
        """Extract Wikipedia page title from URL."""
        import re
        from urllib.parse import unquote

        # Match patterns like:
        # https://en.wikipedia.org/wiki/Python_(programming_language)
        # https://zh.wikipedia.org/wiki/Artificial_intelligence
        pattern = r"https?://[a-z]{2,3}\.wikipedia\.org/wiki/(.+?)(?:#.*)?$"
        match = re.match(pattern, url)
        if match:
            title = unquote(match.group(1))
            # Replace underscores with spaces
            title = title.replace("_", " ")
            return title
        return None

    def _get_wiki_lang(self, url: str) -> str:
        """Extract Wikipedia language code from URL."""
        import re

        match = re.match(r"https?://([a-z]{2,3})\.wikipedia\.org", url)
        return match.group(1) if match else "en"

    def _html_to_text(self, html: str) -> str:
        """Convert Wikipedia HTML to plain text, preserving table content."""
        import re
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "link", "meta"]):
            tag.decompose()

        # Remove edit section links
        for tag in soup.find_all("span", class_="mw-editsection"):
            tag.decompose()

        # Remove reference numbers [1], [2], etc.
        for tag in soup.find_all("sup", class_="reference"):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _parse_infobox(self, wikitext: str) -> str:
        """Parse infobox from wikitext and convert to plain text."""
        import re

        # Find infobox start
        start = wikitext.find("{{Infobox")
        if start == -1:
            start = wikitext.find("{{infobox")
        if start == -1:
            return ""

        # Count braces to find matching end
        depth = 0
        end = start
        for i in range(start, len(wikitext)):
            if wikitext[i : i + 2] == "{{":
                depth += 1
            elif wikitext[i : i + 2] == "}}":
                depth -= 1
                if depth == 0:
                    end = i + 2
                    break

        infobox_raw = wikitext[start:end]

        # Parse fields
        lines = []
        for match in re.finditer(
            r"\|\s*([^=|]+?)\s*=\s*([^|]*?)(?=\n\s*\||\}\})", infobox_raw, re.DOTALL
        ):
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Skip image-related fields
            if key.lower() in (
                "image",
                "caption",
                "alt",
                "width",
                "height",
                "image_size",
                "imagesize",
            ):
                continue

            # Clean up wikitext markup
            value = re.sub(
                r"\{\{[^}|]*\|([^}]*)\}\}", r"\1", value
            )  # {{template|value}} -> value
            value = re.sub(
                r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", value
            )  # [[link|text]] -> text
            value = re.sub(r"'''?", "", value)  # bold/italic
            value = re.sub(r"<[^>]+>", "", value)  # HTML tags
            value = re.sub(r"\{\{[^}]*\}\}", "", value)  # remaining templates
            value = " ".join(value.split())  # normalize whitespace

            if value:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        import aiohttp
        from .simpleqa_data import extract_url_from_metadata

        # Check cache first
        example_id = example.get("id", "")
        if self.text_cache and example_id in self.text_cache:
            cached = self.text_cache[example_id]
            text = cached.get("text", "")
            source_url = cached.get("url", "")
            if text:
                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n\n[Content truncated...]"
                return RetrievalResult(
                    text=text, source_url=source_url, retrieval_type="wikipedia_api"
                )

        target_url = extract_url_from_metadata(example)
        if not target_url:
            return RetrievalResult(
                text="No URL found in metadata.", retrieval_type="wikipedia_api"
            )

        # Check if it's a Wikipedia URL
        if "wikipedia.org" not in target_url.lower():
            return RetrievalResult(
                text=f"URL is not a Wikipedia page: {target_url}",
                source_url=target_url,
                retrieval_type="wikipedia_api",
            )

        title = self._extract_wiki_title(target_url)
        if not title:
            return RetrievalResult(
                text=f"Could not extract Wikipedia title from: {target_url}",
                source_url=target_url,
                retrieval_type="wikipedia_api",
            )

        lang = self._get_wiki_lang(target_url)
        api_url = f"https://{lang}.wikipedia.org/w/api.php"

        headers = {
            "User-Agent": "SimpleQA-Evaluation/1.0 (https://github.com/example; contact@example.com)"
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Use action=parse to get full HTML (includes tables)
                parse_params = {
                    "action": "parse",
                    "page": title,
                    "prop": "text",
                    "format": "json",
                    "redirects": "1",
                }

                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(
                    api_url, params=parse_params, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status != 200:
                        error_msg = f"Wikipedia API error: {resp.status}"
                        logger.warning(f"{error_msg} for {target_url}")
                        return RetrievalResult(
                            text=error_msg,
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    data = await resp.json()

                    # Check for error
                    if "error" in data:
                        error_msg = data["error"].get("info", "Unknown error")
                        return RetrievalResult(
                            text=f"Wikipedia page not found: {title}",
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    html = data.get("parse", {}).get("text", {}).get("*", "")
                    if not html:
                        return RetrievalResult(
                            text=f"No content found for Wikipedia page: {title}",
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    # Parse HTML to text (includes tables)
                    text = self._html_to_text(html)

                    # Save to cache before truncation
                    await self._save_to_cache(example_id, text, target_url)

                    # Truncate if too long
                    if len(text) > self.max_chars:
                        text = text[: self.max_chars] + "\n\n[Content truncated...]"

                    return RetrievalResult(
                        text=text, source_url=target_url, retrieval_type="wikipedia_api"
                    )
        except Exception as e:
            error_msg = f"Wikipedia API fetch failed: {e}"
            logger.warning(f"{error_msg} for {target_url}")
            return RetrievalResult(
                text=error_msg, source_url=target_url, retrieval_type="wikipedia_api"
            )


class VectorRetriever(BaseRetriever):
    """Retrieve similar screenshots using vector similarity search.

    Uses Jina API for embedding and retrieval across dataset screenshots only.
    """

    def __init__(
        self,
        api_key: str,
        screenshot_dir: str = "screenshots",
        cache_path: str | None = None,
        use_multivector: bool = True,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)

        # Prepare missing screenshots and get file paths
        screenshot_paths = self._prepare_screenshots()

        # Import retrieval system
        try:
            from scripts.jina_retrieval import JinaAPIRetrievalSystem
        except ImportError:
            try:
                from jina_retrieval import JinaAPIRetrievalSystem
            except ImportError:
                raise ImportError("JinaAPIRetrievalSystem not available")

        vector_type = "single vector" if not use_multivector else "multivector"
        logger.info(f"Initializing VectorRetriever with {vector_type} mode")

        self.retrieval_system = JinaAPIRetrievalSystem(
            api_key=api_key,
            use_multivector=use_multivector,
            device="cpu",  # Use CPU to avoid OOM when VLM is on GPU
        )
        # Only embed screenshots for current dataset
        self.retrieval_system.embed_images(
            file_paths=screenshot_paths, cache_path=cache_path
        )
        logger.info(
            f"VectorRetriever ready with {len(self.retrieval_system.image_paths)} images"
        )

    def _prepare_screenshots(self) -> list[str]:
        """Prepare screenshots for dataset and return list of paths."""
        from .simpleqa_data import capture_screenshot_for_example

        screenshot_paths = []
        missing = []

        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        if missing:
            logger.info(
                f"Found {len(missing)} missing screenshots out of {len(self.examples)} total examples"
            )
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            # Use a more robust approach: continue even if some screenshots fail
            success_count = 0
            for ex in missing:
                try:
                    capture_screenshot_for_example(ex, self.screenshot_dir)
                    success_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to capture screenshot for {ex.get('id', 'unknown')}: {e}"
                    )
                    # Continue with next screenshot instead of failing completely
            logger.info(
                f"Screenshots prepared: {success_count}/{len(missing)} successful"
            )
        else:
            logger.info(
                f"All {len(self.examples)} screenshots already exist, skipping preparation"
            )

        # Return only existing screenshots
        return [
            p for p in screenshot_paths if os.path.exists(p) and os.path.getsize(p) > 0
        ]

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                return RetrievalResult(images=results, retrieval_type="vector")
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="vector")


class ColQwenVectorRetriever(BaseRetriever):
    """Retrieve similar screenshots using ColQwen2 LEANN multi-vector retrieval."""

    def __init__(
        self,
        index_path: str,
        screenshot_dir: str = "screenshots",
        model_name: str = "colqwen2",
        search_method: str = "ann",
        first_stage_k: int = 500,
        rebuild_index: bool = False,
        recursive: bool = False,
        top_k: int = 3,
        examples: list[dict] | None = None,
        prepare_screenshots: bool = False,  # ColQwen2 doesn't need to prepare specific screenshots
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)

        # Build list of image paths for the specific examples (only Wikipedia samples)
        image_paths = self._get_example_image_paths()

        if image_paths:
            logger.info(
                f"ColQwen2 will retrieve from {len(image_paths)} images for {len(self.examples)} examples"
            )
        else:
            logger.warning(
                f"No images found for examples, falling back to all images in: {screenshot_dir}"
            )

        # Import ColQwen2 retrieval system
        import sys
        from pathlib import Path

        # Add scripts directory to path for import
        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
        except ImportError:
            try:
                from scripts.colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
            except ImportError:
                raise ImportError(
                    "ColQwenLEANNRetrievalSystem not available. Make sure colqwen_leann_retrieval.py is in the scripts directory."
                )

        logger.info("Initializing ColQwen2 LEANN retrieval system...")
        logger.info(f"Search method: {search_method}")

        # Use filtered image paths if available, otherwise fall back to directory scanning
        if image_paths:
            self.retrieval_system = ColQwenLEANNRetrievalSystem(
                index_path=index_path,
                model_name=model_name,
                search_method=search_method,
                first_stage_k=first_stage_k,
                rebuild_index=rebuild_index,
                custom_image_paths=image_paths,  # Pass specific image paths
            )
        else:
            self.retrieval_system = ColQwenLEANNRetrievalSystem(
                index_path=index_path,
                model_name=model_name,
                search_method=search_method,
                first_stage_k=first_stage_k,
                rebuild_index=rebuild_index,
                custom_folder_path=screenshot_dir,
                custom_folder_recursive=recursive,
            )
        logger.info("ColQwen2 LEANN retrieval system ready")

    def _get_example_image_paths(self) -> list[str]:
        """Get image paths for the specific examples."""
        image_paths = []
        for ex in self.examples:
            example_id = ex.get("id", "")
            if not example_id:
                continue
            path = os.path.join(self.screenshot_dir, f"{example_id}_fullhd.png")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                image_paths.append(path)
        return image_paths

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                return RetrievalResult(images=results, retrieval_type="colqwen_vector")
        except Exception as e:
            logger.warning(f"ColQwen2 vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="colqwen_vector")


def _filter_tiles_by_aspect_ratio(
    tile_paths: list[str], max_aspect_ratio: float = 100.0
) -> list[str]:
    """Filter out tiles with extreme aspect ratios.

    Args:
        tile_paths: List of tile image paths.
        max_aspect_ratio: Maximum allowed aspect ratio (default 100, ColQwen requires < 200).

    Returns:
        Filtered list of tile paths.
    """
    from PIL import Image

    filtered = []
    for tile_path in tile_paths:
        try:
            with Image.open(tile_path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    aspect_ratio = max(w / h, h / w)
                    if aspect_ratio <= max_aspect_ratio:
                        filtered.append(tile_path)
                    else:
                        logger.warning(
                            f"Skipping tile with extreme aspect ratio {aspect_ratio:.2f}: {tile_path}"
                        )
        except Exception as e:
            logger.warning(f"Failed to check tile {tile_path}: {e}")

    return filtered


class TiledVectorRetriever(BaseRetriever):
    """Retrieve similar image tiles using vector similarity search.

    Splits dataset screenshots into fixed-size tiles, embeds each tile,
    and retrieves the most relevant tiles for a query.
    """

    def __init__(
        self,
        api_key: str,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        cache_path: str | None = None,
        use_multivector: bool = True,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping (prioritize Wikipedia URLs)
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)  # Uses Wikipedia-first priority
            if url:
                self.id_to_url[ex_id] = url

        # Prepare screenshots and get tile paths
        tile_paths = self._prepare_screenshots_and_tiles()

        # Import retrieval system
        try:
            from scripts.jina_retrieval import JinaAPIRetrievalSystem
        except ImportError:
            try:
                from jina_retrieval import JinaAPIRetrievalSystem
            except ImportError:
                raise ImportError("JinaAPIRetrievalSystem not available")

        vector_type = "single vector" if not use_multivector else "multivector"
        logger.info(f"Initializing TiledVectorRetriever with {vector_type} mode")

        self.retrieval_system = JinaAPIRetrievalSystem(
            api_key=api_key,
            use_multivector=use_multivector,
            device="cpu",  # Use CPU to avoid OOM when VLM is on GPU
        )
        # Only embed tiles for current dataset
        self.retrieval_system.embed_images(file_paths=tile_paths, cache_path=cache_path)
        logger.info(
            f"TiledVectorRetriever ready with {len(self.retrieval_system.image_paths)} tiles"
        )

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths."""
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing
        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        # Capture missing screenshots
        if missing:
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            for ex in tqdm(missing, desc="Capturing screenshots"):
                capture_screenshot_for_example(ex, self.screenshot_dir)
            logger.info("Screenshots prepared.")

        # Split each screenshot into tiles
        all_tile_paths = []
        logger.info(
            f"Splitting {len(screenshot_paths)} screenshots into tiles (output: {self.tiles_dir})..."
        )
        for screenshot_path in tqdm(screenshot_paths, desc="Splitting tiles"):
            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                tile_paths = split_image_to_tiles(
                    screenshot_path, self.tiles_dir, self.tile_size, self.overlap
                )
                all_tile_paths.extend(tile_paths)

        # Filter out tiles with extreme aspect ratios
        filtered_tile_paths = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} screenshots (filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
        )
        return filtered_tile_paths

    def _extract_urls_from_results(self, results: list) -> str:
        """Extract source URLs from tile paths in results, preserving retrieval order."""
        urls = []
        seen = set()
        for item in results:
            # item is (path, score) tuple
            path = item[0] if isinstance(item, tuple) else item
            # Extract example_id from tile path: {example_id}_fullhd_tile_{x}_{y}.png
            filename = os.path.basename(path)
            # Split by _fullhd_ or just get the first part before _tile_
            if "_tile_" in filename:
                example_id = filename.split("_tile_")[0]
                # Remove _fullhd suffix if present
                if example_id.endswith("_fullhd"):
                    example_id = example_id[:-7]
                if example_id in self.id_to_url:
                    url = self.id_to_url[example_id]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        return ", ".join(urls)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results, source_url=source_url, retrieval_type="tiled_vector"
                )
        except Exception as e:
            logger.warning(f"Tiled vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="tiled_vector")


class TiledColQwenVectorRetriever(BaseRetriever):
    """Retrieve similar image tiles using ColQwen2 LEANN multi-vector retrieval.

    Splits dataset screenshots into fixed-size tiles, embeds each tile with ColQwen2,
    and retrieves the most relevant tiles for a query using LEANN.
    """

    def __init__(
        self,
        index_path: str,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        model_name: str = "colqwen2",
        search_method: str = "ann",
        first_stage_k: int = 500,
        rebuild_index: bool = False,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping (prioritize Wikipedia URLs)
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)  # Uses Wikipedia-first priority
            if url:
                self.id_to_url[ex_id] = url

        # Prepare screenshots and get tile paths
        tile_paths = self._prepare_screenshots_and_tiles()

        # Import ColQwen2 retrieval system
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
        except ImportError:
            try:
                from scripts.colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
            except ImportError:
                raise ImportError("ColQwenLEANNRetrievalSystem not available.")

        logger.info("Initializing TiledColQwen2 LEANN retrieval system...")
        logger.info(f"Search method: {search_method}, tiles: {len(tile_paths)}")

        self.retrieval_system = ColQwenLEANNRetrievalSystem(
            index_path=index_path,
            custom_image_paths=tile_paths,
            model_name=model_name,
            search_method=search_method,
            first_stage_k=first_stage_k,
            rebuild_index=rebuild_index,
        )

        logger.info(
            f"TiledColQwen2 LEANN retrieval system ready with {len(tile_paths)} tiles"
        )

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths."""
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing
        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        # Capture missing screenshots
        if missing:
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            for ex in tqdm(missing, desc="Capturing screenshots"):
                capture_screenshot_for_example(ex, self.screenshot_dir)
            logger.info("Screenshots prepared.")

        # Split each screenshot into tiles
        all_tile_paths = []
        logger.info(
            f"Splitting {len(screenshot_paths)} screenshots into tiles (output: {self.tiles_dir})..."
        )
        for screenshot_path in tqdm(screenshot_paths, desc="Splitting tiles"):
            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                tile_paths = split_image_to_tiles(
                    screenshot_path, self.tiles_dir, self.tile_size, self.overlap
                )
                all_tile_paths.extend(tile_paths)

        # Filter out tiles with extreme aspect ratios
        filtered_tile_paths = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} screenshots (filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
        )
        return filtered_tile_paths

    def _extract_urls_from_results(self, results: list) -> str:
        """Extract source URLs from tile paths in results, preserving retrieval order."""
        urls = []
        seen = set()
        for item in results:
            # item is (path, score) tuple
            path = item[0] if isinstance(item, tuple) else item
            # Extract example_id from tile path: {example_id}_fullhd_tile_{x}_{y}.png
            filename = os.path.basename(path)
            if "_tile_" in filename:
                example_id = filename.split("_tile_")[0]
                if example_id.endswith("_fullhd"):
                    example_id = example_id[:-7]
                if example_id in self.id_to_url:
                    url = self.id_to_url[example_id]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        return ", ".join(urls)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results,
                    source_url=source_url,
                    retrieval_type="tiled_colqwen_vector",
                )
        except Exception as e:
            logger.warning(f"TiledColQwen2 vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="tiled_colqwen_vector")


class TextVectorRetriever(BaseRetriever):
    """Retrieve text using LEANN vector search.

    Uses LEANN's integrated embedding + indexing system for text retrieval.
    Supports various embedding models (Qwen3, nomic-embed-text, OpenAI, etc.)
    """

    def __init__(
        self,
        text_cache: dict,
        index_path: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_mode: str = "sentence-transformers",
        embedding_options: dict | None = None,
        top_k: int = 3,
        rebuild_index: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """Initialize TextVectorRetriever.

        Args:
            text_cache: Dict of {id: {"text": ..., "url": ...}}
            index_path: Path to LEANN index
            embedding_model: Embedding model name (default: Qwen/Qwen3-Embedding-0.6B)
            embedding_mode: Embedding mode (sentence-transformers, openai, gemini, ollama)
            embedding_options: Additional options for embedding (e.g., base_url, api_key for OpenAI-compatible APIs)
            top_k: Number of results to retrieve
            rebuild_index: Force rebuild index even if exists
            chunk_size: Max tokens per chunk (default: 512)
            chunk_overlap: Overlap tokens between chunks (default: 128)
        """
        import sys
        from pathlib import Path as PathLib

        # Add LEANN to path
        leann_path = (
            PathLib(__file__).parent.parent.parent
            / "LEANN"
            / "packages"
            / "leann-core"
            / "src"
        )
        if str(leann_path) not in sys.path:
            sys.path.insert(0, str(leann_path))

        from leann.api import LeannBuilder, LeannSearcher

        self.text_cache = text_cache
        self.top_k = top_k
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Check if index exists
        meta_path = f"{index_path}.meta.json"
        index_exists = os.path.exists(meta_path)

        if rebuild_index or not index_exists:
            logger.info(f"Building LEANN text index at {index_path}...")
            self._build_index(LeannBuilder)
            logger.info(f"LEANN text index built with {len(text_cache)} documents")
        else:
            logger.info(f"Loading existing LEANN text index from {index_path}")

        # Load searcher
        self.searcher = LeannSearcher(index_path)
        logger.info(
            f"TextVectorRetriever ready with {len(text_cache)} documents, top_k={top_k}"
        )

    def _build_index(self, LeannBuilder):
        """Build LEANN index from text_cache with chunking for long texts."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=self.embedding_model,
            embedding_mode=self.embedding_mode,
            embedding_options=self.embedding_options,
            is_recompute=False,  # Store embeddings to avoid recomputing at search time
        )

        # Chunking parameters (from CLI or defaults)
        max_tokens = self.chunk_size
        overlap_tokens = self.chunk_overlap

        # Import tiktoken for accurate chunking
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            enc = None
            logger.warning("tiktoken not available, using character-based chunking")

        chunk_count = 0
        for example_id, data in self.text_cache.items():
            text = data.get("text", "")
            url = data.get("url", "")
            if not text:
                continue

            if enc:
                # Token-based chunking
                tokens = enc.encode(text)
                if len(tokens) <= max_tokens:
                    # Short text, add as single passage
                    builder.add_text(text, metadata={"id": example_id, "url": url})
                    chunk_count += 1
                else:
                    # Long text, chunk it with overlap
                    start = 0
                    chunk_idx = 0
                    while start < len(tokens):
                        end = min(start + max_tokens, len(tokens))
                        chunk_tokens = tokens[start:end]
                        chunk_text = enc.decode(chunk_tokens)

                        chunk_id = f"{example_id}_chunk_{chunk_idx}"
                        builder.add_text(
                            chunk_text,
                            metadata={
                                "id": chunk_id,
                                "original_id": example_id,
                                "url": url,
                                "chunk_idx": chunk_idx,
                            },
                        )
                        chunk_count += 1
                        chunk_idx += 1

                        if end >= len(tokens):
                            break
                        start = end - overlap_tokens  # Overlap
            else:
                # Fallback: character-based chunking (~4 chars per token)
                max_chars = max_tokens * 4
                overlap_chars = overlap_tokens * 4

                if len(text) <= max_chars:
                    builder.add_text(text, metadata={"id": example_id, "url": url})
                    chunk_count += 1
                else:
                    start = 0
                    chunk_idx = 0
                    while start < len(text):
                        end = min(start + max_chars, len(text))
                        chunk_text = text[start:end]

                        chunk_id = f"{example_id}_chunk_{chunk_idx}"
                        builder.add_text(
                            chunk_text,
                            metadata={
                                "id": chunk_id,
                                "original_id": example_id,
                                "url": url,
                                "chunk_idx": chunk_idx,
                            },
                        )
                        chunk_count += 1
                        chunk_idx += 1

                        if end >= len(text):
                            break
                        start = end - overlap_chars

        logger.info(
            f"Created {chunk_count} chunks from {len(self.text_cache)} documents"
        )

        # Build index
        builder.build_index(self.index_path)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve relevant texts using LEANN vector search."""
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            # Run search in executor (LEANN search is sync)
            results = await loop.run_in_executor(
                None,
                lambda: self.searcher.search(
                    query, top_k=self.top_k, recompute_embeddings=False
                ),
            )

            if results:
                # Combine retrieved texts
                texts = []
                urls = []
                for r in results:
                    texts.append(r.text)
                    url = r.metadata.get("url", "") if r.metadata else ""
                    urls.append(url)

                combined_text = "\n\n---\n\n".join(texts)
                combined_urls = ", ".join(u for u in urls if u)

                return RetrievalResult(
                    text=combined_text,
                    source_url=combined_urls,
                    retrieval_type="text_vector",
                )
        except Exception as e:
            logger.warning(f"Text vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="text_vector")


class DsServeRetriever(BaseRetriever):
    """Use ds-serve API for external text augmentation.

    Calls ds-serve search API to retrieve relevant text passages for the query.
    """

    def __init__(
        self, api_url: str = "http://api.ds-serve.org:30888/search", top_k: int = 3
    ):
        self.api_url = api_url
        self.top_k = top_k

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        import aiohttp
        import asyncio

        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                payload = {"query": query}

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            result = await response.json()

                            # Extract passages from response
                            passages = []
                            if "results" in result and "passages" in result["results"]:
                                # passages is a list of lists, get the first list
                                passage_list = (
                                    result["results"]["passages"][0]
                                    if result["results"]["passages"]
                                    else []
                                )

                                # Take top_k passages
                                for i, passage_data in enumerate(
                                    passage_list[: self.top_k]
                                ):
                                    if isinstance(passage_data, dict):
                                        text = passage_data.get(
                                            "text", ""
                                        ) or passage_data.get("center_text", "")
                                        if text:
                                            passages.append(text)

                            # Combine passages into context text
                            if passages:
                                combined_text = "\n\n".join(
                                    [
                                        f"[Passage {i + 1}]\n{text}"
                                        for i, text in enumerate(passages)
                                    ]
                                )

                                return RetrievalResult(
                                    text=combined_text,
                                    source_url=f"ds-serve:{self.api_url}",
                                    retrieval_type="ds_serve",
                                )
                            else:
                                return RetrievalResult(
                                    text="No passages found from ds-serve.",
                                    source_url=f"ds-serve:{self.api_url}",
                                    retrieval_type="ds_serve",
                                )
                        elif response.status == 429:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt * 2, 10)
                                logger.warning(
                                    f"Rate limited (429), waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = f"ds-serve API rate limited after {max_retries} retries"
                                logger.error(error_msg)
                                return RetrievalResult(
                                    text=error_msg, retrieval_type="ds_serve"
                                )
                        else:
                            error_text = await response.text()
                            error_msg = f"ds-serve API error: {response.status} - {error_text[:200]}"
                            logger.error(error_msg)
                            return RetrievalResult(
                                text=error_msg, retrieval_type="ds_serve"
                            )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 5)
                    logger.warning(
                        f"Timeout, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"ds-serve API timeout after {max_retries} retries"
                    logger.error(error_msg)
                    return RetrievalResult(text=error_msg, retrieval_type="ds_serve")
            except Exception as e:
                error_msg = f"ds-serve API call failed: {e}"
                logger.error(error_msg)
                return RetrievalResult(text=error_msg, retrieval_type="ds_serve")

        return RetrievalResult(
            text="ds-serve API call failed after all retries", retrieval_type="ds_serve"
        )


class LocalAPIRetriever(BaseRetriever):
    """Retrieve tiles from a local search API (e.g. localhost:30888/search).

    The API accepts batch queries:
        {"queries": [{"text": "..."}, ...], "n_docs": N}
    and returns:
        {"results": [{"hits": [{"path": ..., "url": ..., "score": ...}, ...]}, ...]}

    Call prefetch(examples) before the main loop to batch all queries in one API
    call. Individual retrieve() calls then return cached results instantly.

    When query_rewrite is enabled, uses an LLM to rewrite questions into
    keyword-rich search queries before retrieval.
    """

    REWRITE_PROMPT = (
        "You are a search query optimizer. Given a trivia/factual question, "
        "rewrite it as a Wikipedia search query that would find the article "
        "containing the answer. Output ONLY the search query, nothing else.\n\n"
        "Rules:\n"
        "- Focus on the key entity or topic the question is about\n"
        "- Include all specific names, dates, awards, events, or other details mentioned\n"
        "- Remove filler words like 'what is', 'who was', 'in which year'\n"
        "- Preserve all proper nouns and technical terms exactly as written\n\n"
        "Question: {question}\n"
        "Search query:"
    )

    def __init__(
        self,
        api_url: str = "http://localhost:30888/search",
        top_k: int = 5,
        batch_size: int = 32,
        query_rewrite: bool = False,
        rewrite_model: str | None = None,
        rewrite_api_base: str | None = None,
        rewrite_api_key: str = "dummy",
        nprobe: int | None = None,
        reranker=None,
        rerank_top_k: int = 3,
        query_image_fn=None,
        multi_image_query: bool = False,
        tiles_dir: str = "tiles/evqa",
        query_instruction: str | None = None,
    ):
        self.api_url = api_url
        self.top_k = top_k
        self.batch_size = batch_size
        self.query_rewrite = query_rewrite
        self.rewrite_model = rewrite_model
        self.rewrite_api_base = rewrite_api_base
        self.rewrite_api_key = rewrite_api_key
        self.nprobe = nprobe
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.query_image_fn = query_image_fn  # callable(example) -> image_path or None
        self.multi_image_query = multi_image_query
        self.tiles_dir = tiles_dir
        self.query_instruction = query_instruction
        self._cache: dict[str, list[dict]] = {}  # example_id -> hits
        self._rewritten_queries: dict[str, str] = {}  # example_id -> rewritten query

    async def _rewrite_queries(self, examples: list[dict]) -> dict[str, str]:
        """Batch-rewrite questions into search queries using an LLM."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.rewrite_api_key,
            base_url=self.rewrite_api_base,
            timeout=60.0,
        )

        rewritten = {}
        sem = asyncio.Semaphore(20)

        async def rewrite_one(ex):
            eid = ex.get("id", "unknown")
            prompt = self.REWRITE_PROMPT.format(question=ex["problem"])
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=self.rewrite_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    rewritten[eid] = resp.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"Query rewrite failed for {eid}: {e}")
                    rewritten[eid] = ex["problem"]  # fallback to original

        await asyncio.gather(*[rewrite_one(ex) for ex in examples])
        return rewritten

    async def prefetch(self, examples: list[dict]):
        """Batch-fetch retrieval results for all examples via the API."""
        import aiohttp

        # Step 1: Query rewriting (if enabled)
        if self.query_rewrite and self.rewrite_model:
            to_rewrite = [
                ex
                for ex in examples
                if ex.get("id", "unknown") not in self._rewritten_queries
            ]
            if to_rewrite:
                logger.info(
                    f"LocalAPIRetriever: rewriting {len(to_rewrite)} queries..."
                )
                self._rewritten_queries.update(await self._rewrite_queries(to_rewrite))
                # Log some examples
                for ex in to_rewrite[:3]:
                    eid = ex.get("id", "unknown")
                    orig = ex["problem"][:60]
                    rewr = self._rewritten_queries.get(eid, "")[:60]
                    logger.info(f"  Rewrite: '{orig}...' -> '{rewr}'")

        # Step 2: Build query list
        queries = []
        example_ids = []

        if self.multi_image_query:
            # Multi-image: send one query per image, track which example each belongs to
            # We'll aggregate after receiving results
            multi_image_groups: dict[
                str, list[int]
            ] = {}  # eid -> list of indices in queries[]
            for ex in examples:
                eid = ex.get("id", "unknown")
                if eid in self._cache:
                    continue
                if self.query_rewrite and eid in self._rewritten_queries:
                    query_text = self._rewritten_queries[eid]
                else:
                    query_text = ex["problem"]

                all_paths = _get_all_query_image_paths(ex, self.tiles_dir)
                if len(all_paths) <= 1:
                    # Single or no image: just use the standard path
                    query_dict = {"text": query_text}
                    if all_paths:
                        import base64

                        with open(all_paths[0], "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                    elif self.query_image_fn:
                        img_path = self.query_image_fn(ex)
                        if img_path and os.path.exists(img_path):
                            import base64

                            with open(img_path, "rb") as f:
                                query_dict["image"] = base64.b64encode(
                                    f.read()
                                ).decode()
                    multi_image_groups[eid] = [len(queries)]
                    queries.append(query_dict)
                    example_ids.append(eid)
                else:
                    # Multiple images: one query per image
                    group_indices = []
                    import base64

                    for img_path in all_paths:
                        query_dict = {"text": query_text}
                        with open(img_path, "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                        group_indices.append(len(queries))
                        queries.append(query_dict)
                        example_ids.append(eid)
                    multi_image_groups[eid] = group_indices
                    logger.info(
                        f"Multi-image query for {eid[:8]}: {len(all_paths)} images"
                    )
        else:
            for ex in examples:
                eid = ex.get("id", "unknown")
                if eid in self._cache:
                    continue
                if self.query_rewrite and eid in self._rewritten_queries:
                    query_text = self._rewritten_queries[eid]
                else:
                    query_text = ex["problem"]
                query_dict = {"text": query_text}
                if self.query_image_fn:
                    img_path = self.query_image_fn(ex)
                    if img_path and os.path.exists(img_path):
                        import base64

                        with open(img_path, "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                queries.append(query_dict)
                example_ids.append(eid)

        if not queries:
            logger.info("LocalAPIRetriever: all examples already cached")
            return

        # Use smaller batches when queries contain images (GPU memory)
        has_images = any("image" in q for q in queries)
        batch_size = min(self.batch_size, 16) if has_images else self.batch_size
        logger.info(
            f"LocalAPIRetriever: prefetching {len(queries)} queries in batches of {batch_size}"
            f"{' (multimodal)' if has_images else ''}"
        )

        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start : batch_start + batch_size]
            batch_ids = example_ids[batch_start : batch_start + batch_size]

            n_docs = self.top_k * 2 if self.multi_image_query else self.top_k
            payload = {
                "queries": batch_queries,
                "n_docs": n_docs,
                "include_images": True,
            }
            if self.nprobe is not None:
                payload["nprobe"] = self.nprobe
            if self.query_instruction is not None:
                payload["instruction"] = self.query_instruction
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=_RETRIEVAL_TIMEOUT),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Local API batch error {response.status}: {error_text[:200]}"
                            )
                            for eid in batch_ids:
                                self._cache[eid] = []
                            continue
                        result = await response.json()
            except Exception as e:
                logger.error(f"Local API batch call failed: {e}")
                for eid in batch_ids:
                    self._cache[eid] = []
                continue

            results_list = result.get("results", [])
            for i, eid in enumerate(batch_ids):
                if i < len(results_list):
                    hits = results_list[i].get("hits", [])
                else:
                    hits = []
                if eid not in self._cache:
                    self._cache[eid] = hits
                else:
                    # Multi-image: accumulate hits from all images for this example
                    self._cache[eid].extend(hits)

            logger.info(
                f"  Batch {batch_start // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}: "
                f"{len(batch_queries)} queries done"
            )

        # Multi-image aggregation: deduplicate and keep max score per tile path
        if self.multi_image_query:
            for eid in list(self._cache.keys()):
                hits = self._cache[eid]
                if not hits:
                    continue
                # Aggregate by path: keep hit with max score
                best_by_path: dict[str, dict] = {}
                for hit in hits:
                    path = hit.get("path", "")
                    score = hit.get("score", 0.0)
                    if path not in best_by_path or score > best_by_path[path].get(
                        "score", 0.0
                    ):
                        best_by_path[path] = hit
                # Sort by score descending, take top_k
                sorted_hits = sorted(
                    best_by_path.values(),
                    key=lambda h: h.get("score", 0.0),
                    reverse=True,
                )
                self._cache[eid] = sorted_hits[: self.top_k]

        logger.info(f"LocalAPIRetriever: prefetch complete, {len(self._cache)} cached")

        # Step 3: Rerank (if reranker provided)
        if self.reranker is not None:
            # Build batch of (query, candidates) for all examples
            batch_inputs = []
            batch_eids = []
            for ex in examples:
                eid = ex.get("id", "unknown")
                hits = self._cache.get(eid, [])
                if not hits:
                    continue
                candidates = []
                for hit in hits:
                    path = hit.get("path", "")
                    score = hit.get("score", 0.0)
                    if path and os.path.exists(path):
                        candidates.append((path, score))
                if not candidates:
                    continue
                batch_inputs.append((ex["problem"], candidates))
                batch_eids.append(eid)

            if batch_inputs:
                all_reranked = self.reranker.rerank_batch(
                    batch_inputs,
                    top_k=self.rerank_top_k,
                )
                # Update cache with reranked results
                for eid, reranked_results in zip(batch_eids, all_reranked):
                    hits = self._cache[eid]
                    path_to_hit = {hit["path"]: hit for hit in hits if "path" in hit}
                    new_hits = []
                    for path, rerank_score in reranked_results:
                        orig_hit = path_to_hit.get(path, {})
                        new_hits.append(
                            {**orig_hit, "path": path, "score": rerank_score}
                        )
                    self._cache[eid] = new_hits
                logger.info(
                    f"LocalAPIRetriever: reranking complete ({len(batch_inputs)} examples)"
                )

    @staticmethod
    @staticmethod
    def _resolve_tile_path(hit: dict, tiles_dir: str | None = None) -> str | None:
        """Resolve tile path from hit, searching local shard dirs if needed."""
        path = hit.get("path", "")
        if path and os.path.exists(path):
            return path
        if not tiles_dir:
            return path if path else None
        article_id = hit.get("article_id")
        tile_index = hit.get("tile_index", 0)
        chunk_index = hit.get("chunk_index", 0)
        if article_id is None:
            return path if path else None
        tiles_dirname = f"{article_id}.png.tiles"
        chunk_name = f"chunk_{tile_index:04d}_{chunk_index:02d}.png"
        shard_size = 8284
        top_shard = article_id // shard_size
        top_shard_dir = os.path.join(tiles_dir, f"shard_{top_shard:03d}")
        if os.path.isdir(top_shard_dir):
            for sub in sorted(os.listdir(top_shard_dir)):
                sub_path = os.path.join(top_shard_dir, sub, tiles_dirname)
                if os.path.isdir(sub_path):
                    full = os.path.join(sub_path, chunk_name)
                    if os.path.exists(full):
                        return full
        flat = os.path.join(tiles_dir, tiles_dirname, chunk_name)
        if os.path.exists(flat):
            return flat
        return path if path else None

    @staticmethod
    def _hits_to_result(
        hits: list[dict], tiles_dir: str | None = None
    ) -> RetrievalResult:
        """Convert API hits to RetrievalResult."""
        if not hits:
            return RetrievalResult(retrieval_type="local_api")

        images = []
        image_urls = []
        urls = []
        seen_urls = set()
        for hit in hits:
            score = hit.get("score", 0.0)
            url = hit.get("url", "")
            path = LocalAPIRetriever._resolve_tile_path(hit, tiles_dir)
            if path and os.path.exists(path):
                images.append((path, score))
                image_urls.append(url or None)
            elif hit.get("image_base64"):
                images.append((hit["image_base64"], score))
                image_urls.append(url or None)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        return RetrievalResult(
            images=images,
            image_urls=image_urls,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="local_api",
        )

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        eid = example.get("id", "unknown")

        # Return cached result if available (from prefetch)
        if eid in self._cache:
            return self._hits_to_result(self._cache[eid], tiles_dir=self.tiles_dir)

        # Fallback: single query (if prefetch wasn't called)
        import aiohttp

        query_dict = {"text": query}
        if self.query_image_fn:
            img_path = self.query_image_fn(example)
            if img_path and os.path.exists(img_path):
                import base64

                with open(img_path, "rb") as f:
                    query_dict["image"] = base64.b64encode(f.read()).decode()
        payload = {"queries": [query_dict], "n_docs": self.top_k}
        if self.nprobe is not None:
            payload["nprobe"] = self.nprobe
        if self.query_instruction is not None:
            payload["instruction"] = self.query_instruction
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        return RetrievalResult(retrieval_type="local_api")
                    result = await response.json()
        except Exception as e:
            logger.error(f"Local API call failed: {e}")
            return RetrievalResult(retrieval_type="local_api")

        hits = result.get("results", [{}])[0].get("hits", [])
        self._cache[eid] = hits
        return self._hits_to_result(hits, tiles_dir=self.tiles_dir)

    async def get_hits(self, query: str, example: dict) -> list[dict]:
        """Return raw per-hit dicts (path/url/score/...) for this example.

        Used by wrappers that need per-hit granularity (e.g. HybridRetriever).
        Uses the same cache as retrieve().
        """
        await self.retrieve(query, example)
        return self._cache.get(example.get("id", "unknown"), [])


class TiledQwen3VLEmbeddingRetriever(BaseRetriever):
    """Retrieves context by searching through image tiles using Qwen3-VL-Embedding.

    Uses single vector embeddings (2048 dim) with cosine similarity for retrieval.

    When *pixel_query_map* is provided the retriever embeds the rendered query
    image (pixel query) instead of the raw text, so retrieval happens entirely
    in pixel space.
    """

    def __init__(
        self,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int | tuple[int, int] = 512,
        overlap: int = 0,
        cache_path: str | None = None,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        top_k: int = 3,
        examples: list[dict] | None = None,
        gpu_ids: list[int] | None = None,
        tensor_parallel_size: int = 1,
        pixel_query_map: dict[str, str] | None = None,
        multimodal_query_text_only: bool = False,
        multimodal_query_image_only: bool = False,
        multi_image_query: bool = False,
        prebuilt_tiles_dir: str | None = None,
        embedding_backend: str = "vllm",  # "vllm", "hf", or "biqwen3"
        peft_adapter: str | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        self.pixel_query_map = pixel_query_map  # example_id -> pixel query image path
        self.multimodal_query_text_only = multimodal_query_text_only
        self.multimodal_query_image_only = multimodal_query_image_only
        self.multi_image_query = multi_image_query
        self.prebuilt_tiles_dir = prebuilt_tiles_dir
        self.embedding_backend = embedding_backend
        self.peft_adapter = peft_adapter
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping and deduplicate by URL
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        seen_urls: dict[str, str] = {}  # url -> first example_id that uses it
        self.url_to_representative_id: dict[
            str, str
        ] = {}  # url -> representative example_id
        dedup_examples = []
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)
            if url:
                self.id_to_url[ex_id] = url
                if url not in seen_urls:
                    seen_urls[url] = ex_id
                    self.url_to_representative_id[url] = ex_id
                    dedup_examples.append(ex)

        logger.info(
            f"Deduplicated {len(self.examples)} examples -> {len(dedup_examples)} unique URLs "
            f"(removed {len(self.examples) - len(dedup_examples)} duplicate pages)"
        )
        self._dedup_examples = dedup_examples

        # Prepare tile paths: prebuilt dir (hard mini-datastore) or Selenium
        if self.prebuilt_tiles_dir:
            tile_paths = self._load_prebuilt_tiles()
        else:
            tile_paths = self._prepare_screenshots_and_tiles()

        # Import Qwen3-VL-Embedding retrieval system
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from qwen3vl_embedding_retrieval import Qwen3VLEmbeddingSystem
        except ImportError:
            try:
                from scripts.qwen3vl_embedding_retrieval import Qwen3VLEmbeddingSystem
            except ImportError:
                raise ImportError("Qwen3VLEmbeddingSystem not available.")

        logger.info("Initializing Qwen3-VL-Embedding retrieval system...")
        logger.info(f"Model: {model_name}, tiles: {len(tile_paths)}, GPUs: {gpu_ids}")
        if self.pixel_query_map:
            logger.info(
                f"Pixel query mode ENABLED ({len(self.pixel_query_map)} queries)"
            )

        self.retrieval_system = Qwen3VLEmbeddingSystem(
            model_name=model_name,
            cache_path=cache_path,
            gpu_ids=gpu_ids,
            tensor_parallel_size=tensor_parallel_size,
            backend=self.embedding_backend,
            peft_adapter=self.peft_adapter,
        )

        # Embed all tiles (batch_size=8 for HF backend to avoid OOM on shared GPUs)
        embed_bs = 8 if self.embedding_backend == "hf" else 32
        self.retrieval_system.embed_images(
            file_paths=tile_paths,
            cache_path=cache_path,
            batch_size=embed_bs,
        )
        logger.info(
            f"Qwen3-VL-Embedding retrieval ready with {len(self.retrieval_system.image_paths)} tiles"
        )

    def _load_prebuilt_tiles(self) -> list[str]:
        """Load ALL .png tiles from a prebuilt tile directory (e.g. hard mini-datastore).

        Loads every tile in the directory — including distractors.
        """
        import glob as _glob

        all_tiles = sorted(_glob.glob(os.path.join(self.prebuilt_tiles_dir, "*.png")))
        filtered = _filter_tiles_by_aspect_ratio(all_tiles)
        logger.info(
            f"prebuilt-tiles: loaded {len(filtered)} tiles from {self.prebuilt_tiles_dir} "
            f"(filtered {len(all_tiles) - len(filtered)} extreme aspect ratio tiles)"
        )
        return filtered

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths.

        Uses deduplicated examples (one per unique URL) to avoid
        duplicate tiles inflating the retrieval index.
        """
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        examples_to_process = self._dedup_examples
        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing (deduplicated)
        for ex in examples_to_process:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        # Capture missing screenshots
        if missing:
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            for ex in tqdm(missing, desc="Capturing screenshots"):
                capture_screenshot_for_example(ex, self.screenshot_dir)
            logger.info("Screenshots prepared.")

        # Split each screenshot into tiles
        all_tile_paths = []
        logger.info(
            f"Splitting {len(screenshot_paths)} unique screenshots into tiles (output: {self.tiles_dir})..."
        )
        for screenshot_path in tqdm(screenshot_paths, desc="Splitting tiles"):
            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                tile_paths = split_image_to_tiles(
                    screenshot_path, self.tiles_dir, self.tile_size, self.overlap
                )
                all_tile_paths.extend(tile_paths)

        # Filter out tiles with extreme aspect ratios
        filtered_tile_paths = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} unique screenshots "
            f"(filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
        )
        return filtered_tile_paths

    def _extract_urls_from_results(self, results: list) -> str:
        """Extract source URLs from tile paths in results, preserving retrieval order."""
        urls = []
        seen = set()
        for item in results:
            # item is (path, score) tuple
            path = item[0] if isinstance(item, tuple) else item
            # Extract example_id from tile path: {example_id}_fullhd_tile_{x}_{y}.png
            filename = os.path.basename(path)
            if "_tile_" in filename:
                example_id = filename.split("_tile_")[0]
                if example_id.endswith("_fullhd"):
                    example_id = example_id[:-7]
                if example_id in self.id_to_url:
                    url = self.id_to_url[example_id]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        return ", ".join(urls)

    # Class-level cache for iNat 2021 image_id -> file_name mapping
    _inat2021_id_map: dict[int, str] | None = None
    INAT2021_DATA_DIR = _INAT2021_DATA_DIR

    @classmethod
    def _load_inat2021_mapping(cls) -> dict[int, str]:
        """Load iNaturalist 2021 competition image_id -> file_name mapping.

        Downloads val.json from the competition S3 bucket if not cached locally.
        """
        if cls._inat2021_id_map is not None:
            return cls._inat2021_id_map

        import json
        import tarfile
        import urllib.request
        from pathlib import Path

        data_dir = Path(cls.INAT2021_DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        val_json = data_dir / "val.json"

        if not val_json.exists():
            tar_path = data_dir / "val.json.tar.gz"
            if not tar_path.exists():
                logger.info("Downloading iNaturalist 2021 val annotations...")
                urllib.request.urlretrieve(
                    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
                    str(tar_path),
                )
            with tarfile.open(str(tar_path), "r:gz") as tf:
                tf.extractall(path=str(data_dir))
            logger.info(f"Extracted iNat 2021 val.json to {val_json}")

        with open(val_json) as f:
            data = json.load(f)

        cls._inat2021_id_map = {img["id"]: img["file_name"] for img in data["images"]}
        logger.info(f"Loaded iNat 2021 mapping: {len(cls._inat2021_id_map)} images")
        return cls._inat2021_id_map

    def _get_inat_image_path(self, example: dict) -> str | None:
        """Get EVQA query image (iNaturalist or Landmarks). Delegates to _get_query_image_path_for_example."""
        return _get_query_image_path_for_example(example, self.tiles_dir)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        # Dispatch to multi-image retrieval if enabled
        if self.multi_image_query:
            return await self.retrieve_multi_image(query, example)
        return await self._retrieve_single(query, example)

    async def _retrieve_single(self, query: str, example: dict) -> RetrievalResult:
        example_id = example.get("id", "")
        loop = asyncio.get_event_loop()

        # Priority: pixel_query_map > iNaturalist image > text-only
        pixel_query_path = None
        if self.pixel_query_map and example_id in self.pixel_query_map:
            pixel_query_path = self.pixel_query_map[example_id]

        # Check for iNaturalist query image (multimodal text+image query)
        inat_image_path = self._get_inat_image_path(example)

        try:
            # Determine query modality
            query_image = None
            if pixel_query_path and os.path.exists(pixel_query_path):
                # Pixel query: image-only (rendered text as image)
                query_image = pixel_query_path
                query_text = None
                retrieval_type = "tiled_qwen3vl_embedding_pixel_query"
            elif self.multimodal_query_text_only:
                # Ablation: text-only (no image)
                query_image = None
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding_multimodal_textonly"
            elif self.multimodal_query_image_only and inat_image_path:
                # Ablation: image-only (no text)
                query_image = inat_image_path
                query_text = None
                retrieval_type = "tiled_qwen3vl_embedding_multimodal_imageonly"
            elif inat_image_path:
                # Multimodal: text + image
                query_image = inat_image_path
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding_multimodal"
            else:
                # Text-only (no query image available)
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding"

            results = await loop.run_in_executor(
                None,
                lambda: self.retrieval_system.search(
                    text=query_text, image=query_image, top_k=self.top_k
                ),
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results,
                    source_url=source_url,
                    retrieval_type=retrieval_type,
                    pixel_query_path=pixel_query_path or inat_image_path,
                    query_image_path=inat_image_path,
                )
            else:
                return RetrievalResult(
                    text="No relevant tiles found via Qwen3-VL-Embedding search",
                    retrieval_type=retrieval_type,
                    pixel_query_path=pixel_query_path or inat_image_path,
                    query_image_path=inat_image_path,
                )
        except Exception as e:
            logger.error(f"Qwen3-VL-Embedding search failed: {e}")
            return RetrievalResult(
                text=f"Qwen3-VL-Embedding retrieval error: {e}",
                retrieval_type="tiled_qwen3vl_embedding",
                pixel_query_path=pixel_query_path or inat_image_path,
                query_image_path=inat_image_path,
            )

    async def retrieve_multi_image(self, query: str, example: dict) -> RetrievalResult:
        """Multi-image retrieval: search with ALL query images, aggregate scores, return top-K.

        For each query image, does a multimodal search (text + image), then combines
        scores across all images using max-score aggregation per tile.
        Falls back to single-image retrieve() if only 0-1 images available.
        """
        all_image_paths = _get_all_query_image_paths(example, self.tiles_dir)
        # Get single image for generation (first available, used in RetrievalResult)
        single_image_path = self._get_inat_image_path(example)

        if len(all_image_paths) <= 1:
            return await self._retrieve_single(query, example)

        example_id = example.get("id", "")
        loop = asyncio.get_event_loop()
        logger.info(
            f"Multi-image retrieval for {example_id}: {len(all_image_paths)} query images"
        )

        try:
            # Score aggregation: for each tile, keep the max score across all query images
            tile_best_score: dict[str, float] = {}

            for img_path in all_image_paths:
                results = await loop.run_in_executor(
                    None,
                    lambda p=img_path: self.retrieval_system.search(
                        text=query, image=p, top_k=self.top_k * 2
                    ),
                )
                for tile_path, score in results:
                    if (
                        tile_path not in tile_best_score
                        or score > tile_best_score[tile_path]
                    ):
                        tile_best_score[tile_path] = score

            # Sort by score descending, take top_k
            sorted_tiles = sorted(
                tile_best_score.items(), key=lambda x: x[1], reverse=True
            )
            top_results = sorted_tiles[: self.top_k]

            retrieval_type = (
                f"tiled_qwen3vl_embedding_multiimage_{len(all_image_paths)}imgs"
            )

            if top_results:
                source_url = self._extract_urls_from_results(top_results)
                return RetrievalResult(
                    images=top_results,
                    source_url=source_url,
                    retrieval_type=retrieval_type,
                    pixel_query_path=single_image_path,
                    query_image_path=single_image_path,
                )
            else:
                return RetrievalResult(
                    text="No relevant tiles found via multi-image search",
                    retrieval_type=retrieval_type,
                    pixel_query_path=single_image_path,
                    query_image_path=single_image_path,
                )
        except Exception as e:
            logger.error(f"Multi-image retrieval failed: {e}")
            return await self._retrieve_single(query, example)


class TextAPIRetriever(BaseRetriever):
    """Retrieve text chunks from the text search API.

    The API accepts:
        POST /search
        {"queries": [{"text": "..."}], "n_docs": N}
    and returns:
        {"results": [{"hits": [{"text": ..., "title": ..., "url": ..., "score": ...}, ...]}]}

    Supports batch prefetch for efficient evaluation.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:30889/search",
        top_k: int = 3,
        batch_size: int = 32,
        nprobe: int | None = None,
        query_instruction: str | None = None,
        reader_top_k: int | None = None,
        query_image_fn=None,
    ):
        self.api_url = api_url
        self.top_k = top_k
        # If reader_top_k is set and < top_k, only the first reader_top_k hits are
        # passed to the reader. Mirrors the image-side reader_top_k slicing in
        # run_naive_simpleqa.py so text + image cells are comparable at fixed k.
        self.reader_top_k = reader_top_k
        self.batch_size = batch_size
        self.nprobe = nprobe
        self.query_instruction = query_instruction
        self.query_image_fn = query_image_fn
        self._cache: dict[str, list[dict]] = {}

    async def prefetch(self, examples: list[dict]):
        """Batch-fetch retrieval results for all examples."""
        import aiohttp

        queries = []
        example_ids = []
        for ex in examples:
            eid = ex.get("id", "unknown")
            if eid in self._cache:
                continue
            query_dict = {"text": ex["problem"]}
            if self.query_image_fn:
                img_path = self.query_image_fn(ex)
                if img_path and os.path.exists(img_path):
                    import base64

                    with open(img_path, "rb") as f:
                        query_dict["image"] = base64.b64encode(f.read()).decode()
            queries.append(query_dict)
            example_ids.append(eid)

        if not queries:
            logger.info("TextAPIRetriever: all examples already cached")
            return

        has_images = any("image" in q for q in queries)
        batch_size = min(self.batch_size, 16) if has_images else self.batch_size
        logger.info(
            f"TextAPIRetriever: prefetching {len(queries)} queries in batches of {batch_size}"
            f"{' (multimodal)' if has_images else ''}"
        )

        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start : batch_start + batch_size]
            batch_ids = example_ids[batch_start : batch_start + batch_size]

            payload = {"queries": batch_queries, "n_docs": self.top_k}
            if self.nprobe is not None:
                payload["nprobe"] = self.nprobe
            if self.query_instruction is not None:
                payload["instruction"] = self.query_instruction
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=_RETRIEVAL_TIMEOUT),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"TextAPI batch error {response.status}: {error_text[:200]}"
                            )
                            for eid in batch_ids:
                                self._cache[eid] = []
                            continue
                        result = await response.json()
            except Exception as e:
                logger.error(f"TextAPI batch call failed: {e}")
                for eid in batch_ids:
                    self._cache[eid] = []
                continue

            results_list = result.get("results", [])
            for i, eid in enumerate(batch_ids):
                if i < len(results_list):
                    self._cache[eid] = results_list[i].get("hits", [])
                else:
                    self._cache[eid] = []

            logger.info(
                f"  Batch {batch_start // self.batch_size + 1}/"
                f"{(len(queries) + self.batch_size - 1) // self.batch_size}: "
                f"{len(batch_queries)} queries done"
            )

        logger.info(f"TextAPIRetriever: prefetch complete, {len(self._cache)} cached")

    @staticmethod
    def _hits_to_result(
        hits: list[dict], max_passages: int | None = None
    ) -> RetrievalResult:
        """Convert text API hits to RetrievalResult.

        If max_passages is set, only the first max_passages hits are joined into
        the reader prompt. The cache itself is not truncated, so the same cached
        hits can serve multiple reader_top_k values.
        """
        if not hits:
            return RetrievalResult(retrieval_type="text_api")

        if max_passages is not None and max_passages < len(hits):
            hits = hits[:max_passages]

        passages = []
        urls = []
        seen_urls = set()
        for hit in hits:
            text = hit.get("text", "")
            url = hit.get("url", "")
            # Option 1 (2026-04-29): no `[title]` prefix on chunks. Title is leaked
            # metadata for entity-answering tasks (often contains the answer outright).
            # Reader sees only the chunk content. URL lives in retrieval_result.source_url
            # for logging/grading but is not injected into the prompt by build_messages.
            if text:
                passages.append(text)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        combined_text = "\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined_text,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="text_api",
        )

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        eid = example.get("id", "unknown")

        if eid in self._cache:
            return self._hits_to_result(
                self._cache[eid], max_passages=self.reader_top_k
            )

        # Fallback: single query
        import aiohttp

        payload = {"queries": [{"text": query}], "n_docs": self.top_k}
        if self.nprobe is not None:
            payload["nprobe"] = self.nprobe
        if self.query_instruction is not None:
            payload["instruction"] = self.query_instruction
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        return RetrievalResult(retrieval_type="text_api")
                    result = await response.json()
        except Exception as e:
            logger.error(f"TextAPI call failed: {e}")
            return RetrievalResult(retrieval_type="text_api")

        hits = result.get("results", [{}])[0].get("hits", [])
        self._cache[eid] = hits
        return self._hits_to_result(hits, max_passages=self.reader_top_k)

    async def get_hits(self, query: str, example: dict) -> list[dict]:
        """Return raw per-hit dicts (title/text/url/score/...) for this example.

        Used by wrappers that need per-chunk granularity (e.g. RenderedTextWrapper).
        Uses the same cache as retrieve().
        """
        await self.retrieve(query, example)
        return self._cache.get(example.get("id", "unknown"), [])


class OCRWrappedRetriever(BaseRetriever):
    """Wraps an image retriever; OCRs retrieved tiles and returns text.

    Ablation A pipeline: image retrieve -> OCR -> text to reader.
    Talks to an OpenAI-compatible chat endpoint (PaddleOCR-VL served via vLLM).
    Caches OCR output to a JSONL file keyed by absolute image path so reruns
    reuse prior work.
    """

    DEFAULT_PROMPT = "OCR this image. Output only the extracted text verbatim, preserving paragraph and line breaks."

    def __init__(
        self,
        base: BaseRetriever,
        ocr_url: str = "http://localhost:8202/v1",
        model: str = "PaddlePaddle/PaddleOCR-VL",
        api_key: str = "dummy",
        cache_path: str = "ocr_cache/paddleocr_vl.jsonl",
        concurrency: int = 16,
        prompt: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 4096,
        reader_top_k: int | None = None,
    ):
        self.base = base
        self.ocr_url = ocr_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.cache_path = cache_path
        self.concurrency = concurrency
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reader_top_k = reader_top_k
        self._cache: dict[str, str] = {}
        self.tiles_dir = getattr(base, "tiles_dir", None)
        self._load_cache()

    def _load_cache(self):
        if not os.path.isfile(self.cache_path):
            return
        import json

        loaded = 0
        try:
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._cache[entry["path"]] = entry["text"]
                    loaded += 1
            logger.info(
                f"OCRWrappedRetriever: loaded {loaded} cached OCR entries from {self.cache_path}"
            )
        except Exception as e:
            logger.warning(
                f"OCRWrappedRetriever: cache load failed ({e}); starting fresh"
            )

    def _append_cache(self, path: str, text: str):
        import json

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"path": path, "text": text}, ensure_ascii=False) + "\n")
        self._cache[path] = text

    async def _ocr_one(self, path: str, session) -> str:
        if path in self._cache:
            return self._cache[path]
        import aiohttp
        import base64

        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            logger.error(f"OCR read failed for {path}: {e}")
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with session.post(
                f"{self.ocr_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"OCR HTTP {resp.status} for {path}: {err[:200]}")
                    return ""
                result = await resp.json()
                text = result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OCR request failed for {path}: {e}")
            return ""
        self._append_cache(path, text)
        return text

    async def _batch_ocr(self, paths: list[str]) -> dict[str, str]:
        import aiohttp

        to_fetch = [p for p in paths if p not in self._cache]
        if not to_fetch:
            return {p: self._cache[p] for p in paths}
        sem = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:

            async def _one(p):
                async with sem:
                    return await self._ocr_one(p, session)

            await asyncio.gather(*[_one(p) for p in to_fetch])
        return {p: self._cache.get(p, "") for p in paths}

    async def prefetch(self, examples: list[dict]):
        """Forward to base's prefetch, then batch-OCR all tiles up front."""
        if hasattr(self.base, "prefetch"):
            await self.base.prefetch(examples)
        all_paths: set[str] = set()
        for ex in examples:
            r = await self.base.retrieve(ex.get("problem", ""), ex)
            images = (
                r.images[: self.reader_top_k]
                if self.reader_top_k is not None
                else r.images
            )
            for p, _ in images:
                all_paths.add(os.path.abspath(p))
        uncached = [p for p in all_paths if p not in self._cache]
        logger.info(
            f"OCRWrappedRetriever: {len(all_paths)} unique tiles across {len(examples)} examples; "
            f"{len(all_paths) - len(uncached)} cached, OCRing {len(uncached)}"
        )
        if uncached:
            await self._batch_ocr(uncached)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        r = await self.base.retrieve(query, example)
        if not r.images:
            return r
        images = (
            r.images[: self.reader_top_k] if self.reader_top_k is not None else r.images
        )
        image_urls = (
            r.image_urls[: self.reader_top_k]
            if self.reader_top_k is not None and r.image_urls
            else list(r.image_urls or [])
        )
        urls: list[str] = []
        seen_urls: set[str] = set()
        for url in image_urls:
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)
        paths = [os.path.abspath(p) for p, _ in images]
        ocr_map = await self._batch_ocr(paths)
        passages = [ocr_map[p].strip() for p in paths if ocr_map.get(p, "").strip()]
        combined = "\n\n---\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined,
            images=[],
            source_url=", ".join(urls) if urls else r.source_url,
            retrieval_type=f"{r.retrieval_type}+ocr",
            pixel_query_path=r.pixel_query_path,
            query_image_path=r.query_image_path,
        )


class RenderedTextWrapper(BaseRetriever):
    """Wraps a text retriever; renders each chunk as an image.

    Ablation B pipeline: text retrieve -> render as Wikipedia-style image -> VLM reader.
    Requires the base retriever to expose get_hits(query, example) returning
    per-hit dicts with keys: title, text, url, score, article_id, chunk_index.
    (TextAPIRetriever satisfies this.)

    Renders are cached on disk at {render_dir}/{article_id}_{chunk_index}.png
    so repeated eval runs don't re-render.
    """

    def __init__(
        self,
        base: BaseRetriever,
        render_dir: str = "rendered_chunks",
        reader_top_k: int | None = None,
    ):
        if not hasattr(base, "get_hits"):
            raise TypeError(
                f"RenderedTextWrapper requires base retriever with get_hits(); "
                f"got {type(base).__name__}"
            )
        self.base = base
        self.render_dir = render_dir
        self.reader_top_k = reader_top_k
        os.makedirs(self.render_dir, exist_ok=True)
        self.tiles_dir = render_dir

    async def prefetch(self, examples: list[dict]):
        if hasattr(self.base, "prefetch"):
            await self.base.prefetch(examples)

    def _render(self, hit: dict) -> str:
        from .text_renderer import render_text_chunk

        article_id = hit.get("article_id", "unknown")
        chunk_index = hit.get("chunk_index", 0)
        out_path = os.path.join(self.render_dir, f"{article_id}_{chunk_index}.png")
        if os.path.isfile(out_path):
            return out_path
        # No-title policy: mirrors `_hits_to_result` (line ~3035) — title/url are
        # leaked metadata for entity-answering tasks and were stripped from the
        # text→text path on 2026-04-29. Apply the same constraint here so
        # rendered and text→text differ only in modality, not in content.
        render_text_chunk(
            text=hit.get("text", ""),
            title=None,
            url=None,
            output_path=out_path,
        )
        return out_path

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        hits = await self.base.get_hits(query, example)
        if not hits:
            return RetrievalResult(retrieval_type="text_api+rendered")
        if self.reader_top_k is not None:
            hits = hits[: self.reader_top_k]
        images: list[tuple[str, float]] = []
        urls: list[str] = []
        seen_urls: set[str] = set()
        for hit in hits:
            if not hit.get("text"):
                continue
            path = self._render(hit)
            images.append((path, float(hit.get("score", 0.0))))
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)
        return RetrievalResult(
            images=images,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="text_api+rendered",
        )


class HybridRetriever(BaseRetriever):
    """Merge image (LocalAPIRetriever) and text (TextAPIRetriever) hits by raw score.

    Both underlying retrievers embed with Qwen3-VL-Embedding-2B against L2-normalized
    FAISS IVFFlat (IP metric) indices, so their per-hit scores are cosine similarities
    on the same scale and directly comparable without any normalization step.

    Each base is called with its own configured top_k, then the combined candidate pool
    is sorted by score desc and the top `top_k` are kept. The reader receives the
    surviving image hits as image inputs and the surviving text hits as a concatenated
    text block in the same prompt — VL-4B handles mixed modality natively.
    """

    def __init__(
        self,
        image_base: "LocalAPIRetriever",
        text_base: "TextAPIRetriever",
        top_k: int = 3,
        reader_top_k: int | None = None,
    ):
        if not hasattr(image_base, "get_hits"):
            raise TypeError(
                f"HybridRetriever.image_base requires get_hits(); got {type(image_base).__name__}"
            )
        if not hasattr(text_base, "get_hits"):
            raise TypeError(
                f"HybridRetriever.text_base requires get_hits(); got {type(text_base).__name__}"
            )
        self.image_base = image_base
        self.text_base = text_base
        self.top_k = top_k
        self.reader_top_k = reader_top_k
        self.tiles_dir = getattr(image_base, "tiles_dir", None)

    async def prefetch(self, examples: list[dict]):
        if hasattr(self.image_base, "prefetch"):
            await self.image_base.prefetch(examples)
        if hasattr(self.text_base, "prefetch"):
            await self.text_base.prefetch(examples)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        image_hits = await self.image_base.get_hits(query, example)
        text_hits = await self.text_base.get_hits(query, example)

        # Tag each hit with its modality, then merge and sort by score desc.
        merged: list[tuple[float, str, dict]] = []
        for h in image_hits:
            score = float(h.get("score", 0.0))
            merged.append((score, "image", h))
        for h in text_hits:
            score = float(h.get("score", 0.0))
            merged.append((score, "text", h))

        merged.sort(key=lambda x: x[0], reverse=True)
        keep_k = self.reader_top_k if self.reader_top_k is not None else self.top_k
        top = merged[:keep_k]

        images: list[tuple[str, float]] = []
        passages: list[str] = []
        urls: list[str] = []
        seen_urls: set[str] = set()

        for score, modality, hit in top:
            url = hit.get("url", "")
            if modality == "image":
                path = hit.get("path", "")
                if path and os.path.exists(path):
                    images.append((path, score))
            else:  # text
                title = hit.get("title", "")
                text = hit.get("text", "")
                if text:
                    header = f"[{title}]" if title else ""
                    passages.append(f"{header}\n{text}" if header else text)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        return RetrievalResult(
            text="\n\n".join(passages) if passages else None,
            images=images,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="hybrid",
        )


class HTMLDOMLookupRetriever(BaseRetriever):
    """Text-retrieve → DOM lookup: retrieve text chunks, then find their HTML context.

    Wraps TextAPIRetriever. For each retrieved text chunk:
    1. Fetches original HTML from kiwix-serve using article_id
    2. Locates the chunk text within the HTML DOM
    3. Extracts the enclosing semantic container (section/table/div)
    4. Returns structured HTML context to the reader

    This gives the reader table/list structure without needing a separate HTML index.
    Falls back to plain text if DOM lookup fails for a chunk.
    """

    KIWIX_BASE = "http://localhost:9454/content/wikipedia_en_all_maxi_2025-08"

    def __init__(
        self,
        text_api_url: str = "http://localhost:30889/search",
        top_k: int = 3,
        nprobe: int | None = None,
        query_instruction: str | None = None,
        reader_top_k: int | None = None,
        query_image_fn=None,
        kiwix_base: str | None = None,
        articles_json: str = "/path/to/data",
        context_mode: str = "section",
        llm_verify: bool = False,
        llm_verify_model: str = "gpt-4.1-mini",
    ):
        import json as _json

        self._text_retriever = TextAPIRetriever(
            api_url=text_api_url,
            top_k=top_k,
            nprobe=nprobe,
            query_instruction=query_instruction,
            reader_top_k=reader_top_k,
            query_image_fn=query_image_fn,
        )
        if kiwix_base:
            self.KIWIX_BASE = kiwix_base
        self.top_k = top_k
        self.reader_top_k = reader_top_k
        self.context_mode = context_mode
        self.llm_verify = llm_verify
        self.llm_verify_model = llm_verify_model

        with open(articles_json) as f:
            self._articles: list[str] = _json.load(f)

        self._html_cache: dict[int, str] = {}

    async def prefetch(self, examples: list[dict]):
        await self._text_retriever.prefetch(examples)

    def _fetch_html(self, article_id: int) -> str | None:
        """Fetch article HTML from kiwix-serve (with caching)."""
        if article_id in self._html_cache:
            return self._html_cache[article_id]

        if article_id >= len(self._articles):
            return None

        import requests
        from urllib.parse import quote

        slug = self._articles[article_id]
        url = f"{self.KIWIX_BASE}/{quote(slug, safe='/:@!$&()*+,;=')}"
        try:
            resp = requests.get(url, timeout=15)
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                return None
            self._html_cache[article_id] = resp.text
            return resp.text
        except Exception:
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for fuzzy DOM matching."""
        import re
        import unicodedata

        text = re.sub(r"[\xa0    ]", " ", text)
        text = re.sub(r"[‐-―−﹘﹣－—–]", "-", text)
        text = re.sub(r" +", " ", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        return text.lower()

    def _dom_lookup(self, html: str, chunk_text: str) -> str | None:
        """Find the contiguous DOM span covering chunk_text, return its HTML.

        Strategy:
        1. Extract search keys from chunk text (table cells + prose fragments)
        2. For each key, find the tightest DOM element and walk up to a
           direct child of mw-parser-output
        3. Return ALL direct children from the first match to the last match
           (inclusive), plus everything in between — this preserves the full
           contiguous region the chunk spans.
        """
        from lxml import html as lxml_html, etree

        tree = lxml_html.fromstring(html)

        keys = self._extract_search_keys(chunk_text)
        if not keys:
            return None

        mw_output = tree.xpath('//div[contains(@class, "mw-parser-output")]')
        if not mw_output:
            return None
        content_root = mw_output[0]
        children = list(content_root)
        if not children:
            return None

        # For each key, find the tightest match and resolve to a
        # direct-child index of mw-parser-output
        matched_child_indices = set()
        SKIP_TAGS = frozenset(
            ("script", "style", "title", "meta", "link", "nav", "header", "footer")
        )

        for key in keys:
            key_norm = self._normalize(key)
            if len(key_norm) < 4:
                continue

            best_elem = None
            best_len = float("inf")

            for elem in content_root.iter():
                if not isinstance(elem, lxml_html.HtmlElement):
                    continue
                if elem.tag in SKIP_TAGS:
                    continue
                try:
                    tc = elem.text_content()
                except Exception:
                    continue
                tc_norm = self._normalize(tc)
                if key_norm in tc_norm and len(tc) < best_len:
                    best_elem = elem
                    best_len = len(tc)

            if best_elem is None:
                continue

            # Walk up from best_elem to find which direct child of content_root
            # contains it
            current = best_elem
            while current is not None:
                parent = current.getparent()
                if parent is None:
                    break
                if parent == content_root:
                    # current is a direct child of mw-parser-output
                    try:
                        idx = children.index(current)
                        matched_child_indices.add(idx)
                    except ValueError:
                        pass
                    break
                current = parent

        if not matched_child_indices:
            return None

        # Return contiguous range from first to last matched child (inclusive)
        first = min(matched_child_indices)
        last = max(matched_child_indices)

        span_elems = children[first : last + 1]

        # Build result: serialize all elements in the span
        parts = []
        for el in span_elems:
            # Strip style/script/navbox noise
            for tag in ("style", "script"):
                for junk in list(el.iter(tag)):
                    if junk.getparent() is not None:
                        junk.getparent().remove(junk)
            if hasattr(el, "xpath"):
                for nav in el.xpath('.//*[contains(@class, "navbox")]'):
                    if nav.getparent() is not None:
                        nav.getparent().remove(nav)
            try:
                parts.append(etree.tostring(el, encoding="unicode", method="html"))
            except Exception:
                continue

        if not parts:
            return None

        html_str = "\n".join(parts)

        # Log oversized results but still return them (caller decides)
        if len(html_str) > self.MAX_CONTAINER_CHARS * 2:
            logger.warning(
                "DOM lookup oversized: %d chars (max %d) for chunk starting with %r",
                len(html_str),
                self.MAX_CONTAINER_CHARS * 2,
                chunk_text[:50],
            )

        # Minimum useful size
        if len(html_str) < 100 and len(chunk_text) > 200:
            return None

        return html_str

    MAX_CONTAINER_CHARS = 8000

    def _find_semantic_container(self, elem) -> "lxml_html.HtmlElement":  # noqa: F821
        """Walk up from matched element to find a meaningful semantic container.

        Hard cap: never return a container with text_content > MAX_CONTAINER_CHARS.
        Stops at mw-parser-output boundary (never returns the whole article).
        """

        SEMANTIC_TAGS = {
            "section",
            "article",
            "table",
            "blockquote",
            "details",
            "figure",
        }
        STOP_CLASSES = {"mw-parser-output", "mw-body-content", "mw-body"}
        MIN_CONTEXT_LEN = 200

        if elem.tag in SEMANTIC_TAGS:
            return elem

        best = elem
        current = elem

        for _ in range(15):
            parent = current.getparent()
            if parent is None:
                break
            # Hard stop: never go past the article content container
            parent_classes = parent.get("class", "")
            if any(sc in parent_classes for sc in STOP_CLASSES):
                # We've reached the article root — use section gathering instead
                if self.context_mode == "section":
                    gathered = self._gather_section_context(current)
                    if gathered is not None:
                        return gathered
                break

            try:
                parent_len = len(parent.text_content())
            except Exception:
                break

            # Prefer semantic tags — even if parent exceeds size cap
            # Bug fix 2: tbody→table jump — don't let size cap block us from
            # reaching a semantic container that's just one level up
            if parent.tag in SEMANTIC_TAGS:
                return parent

            # Stop if parent is too large (but we already checked semantic tags above)
            if parent_len > self.MAX_CONTAINER_CHARS:
                # One more chance: check if grandparent is a semantic tag
                grandparent = parent.getparent()
                if grandparent is not None and grandparent.tag in SEMANTIC_TAGS:
                    return grandparent
                break

            # Accept block containers that are reasonably sized
            if parent_len >= MIN_CONTEXT_LEN:
                best = parent

            current = parent

        return best

    def _gather_section_context(self, elem) -> "lxml_html.HtmlElement":  # noqa: F821
        """Gather all sibling elements within the same h2/h3 section."""
        from lxml import etree

        # Walk up to find direct child of mw-parser-output
        current = elem
        mw_output = None
        while current is not None:
            parent = current.getparent()
            if parent is not None:
                classes = parent.get("class", "")
                if "mw-parser-output" in classes:
                    mw_output = parent
                    break
            current = parent

        if mw_output is None:
            return elem

        # Find the element's position among mw-parser-output children
        children = list(mw_output)
        try:
            idx = children.index(current)
        except ValueError:
            return elem

        # Gather backward until we hit a heading, forward until next heading
        section_elems = [current]

        # Backward
        for i in range(idx - 1, max(idx - 10, -1), -1):
            child = children[i]
            if hasattr(child, "tag") and child.tag in ("h1", "h2", "h3"):
                section_elems.insert(0, child)
                break
            section_elems.insert(0, child)

        # Forward
        for i in range(idx + 1, min(idx + 10, len(children))):
            child = children[i]
            if hasattr(child, "tag") and child.tag in ("h1", "h2", "h3"):
                break
            section_elems.append(child)

        # Build a container div with these elements
        container = etree.Element("div")
        for el in section_elems:
            try:
                container.append(el)
            except Exception:
                pass

        return container

    @staticmethod
    def _extract_search_keys(chunk_text: str) -> list[str]:
        """Extract distinctive search keys from chunk text for DOM matching.

        Detects chunk type (table-heavy vs prose-heavy) and picks the best strategy.
        Returns keys ordered by distinctiveness — first key is tried first in DOM lookup.
        """
        import re

        lines = chunk_text.split("\n")
        # Skip first line if it looks like an article title (short, no pipes, no punctuation)
        # These match <h1> in DOM and cause Bug 1
        if lines and len(lines[0]) < 80 and "|" not in lines[0] and "." not in lines[0]:
            content_lines = lines[1:]
        else:
            content_lines = lines
        table_lines = [l for l in content_lines if "|" in l and "---" not in l]
        prose_lines = [
            l
            for l in content_lines
            if len(l) > 30 and "|" not in l and not l.startswith("- ^")
        ]
        is_table_heavy = (
            len(table_lines) > len(content_lines) * 0.4 if content_lines else False
        )

        keys = []

        if is_table_heavy:
            # Mixed strategy: include keys from BOTH table cells and prose
            # so coverage scorer can find a container spanning both parts.
            cell_candidates = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split("|") if c.strip()]
                for cell in cells:
                    if len(cell) < 5 or len(cell) > 80:
                        continue
                    if cell.lower() in ("yes", "no", "n/a", "none", ""):
                        continue
                    has_code = bool(re.search(r"[A-Z]\d|[a-z]\d{4,}", cell))
                    has_mixed = bool(re.search(r"\d.*[a-zA-Z]|[a-zA-Z].*\d", cell))
                    has_proper = bool(re.search(r"[A-Z][a-z]+\s+[A-Z]", cell))
                    if has_code or has_mixed or has_proper:
                        cell_candidates.insert(0, cell)
                    elif len(cell) > 12:
                        cell_candidates.append(cell)

            # Table cells first (these anchor to the infobox)
            for cc in cell_candidates[:3]:
                if cc not in keys:
                    keys.append(cc)

            # Then prose keys (these anchor to body paragraphs)
            for line in prose_lines[:3]:
                mid = len(line) // 2
                candidate = line[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10 and re.search(r"[a-zA-Z]{4,}", candidate):
                    keys.append(candidate)

        else:
            # Prose-dominant chunk: use prose fragments as primary keys
            for line in prose_lines[:4]:
                mid = len(line) // 2
                candidate = line[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10 and re.search(r"[a-zA-Z]{4,}", candidate):
                    keys.append(candidate)

            # Add table cell values as secondary
            if table_lines:
                for tl in table_lines[:5]:
                    cells = [
                        c.strip()
                        for c in tl.split("|")
                        if c.strip() and len(c.strip()) > 8
                    ]
                    for cell in cells[:1]:
                        if cell not in keys:
                            keys.append(cell)

        # List item content
        if not keys:
            list_lines = [
                l[2:]
                for l in lines
                if l.startswith("- ") and len(l) > 20 and not l.startswith("- ^")
            ]
            for ll in list_lines[:3]:
                mid = len(ll) // 2
                candidate = ll[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10:
                    keys.append(candidate)

        # Fallback
        if not keys and len(chunk_text) > 40:
            candidate = chunk_text[10:50].strip()
            keys.append(candidate)

        return keys

    async def _llm_dom_closure(self, raw_html: str, chunk_text: str) -> str | None:
        """Use an LLM to find the minimal DOM closure containing the chunk text.

        Sends the article HTML (truncated) and the chunk text to the model,
        asks it to return the minimal enclosing HTML subtree.
        """
        import openai

        # Truncate HTML to avoid context limits — keep first 60K chars
        # (most Wikipedia articles are under 100K)
        html_truncated = raw_html[:60000]

        prompt = f"""Given this HTML document and a text chunk extracted from it, find the minimal DOM subtree that contains ALL the text in the chunk. Return ONLY the raw HTML of that subtree, no explanation.

The text chunk (extracted by Trafilatura, so formatting differs from HTML):
---
{chunk_text[:2000]}
---

The HTML document:
---
{html_truncated}
---

Return the minimal HTML subtree containing all the information from the text chunk. Include complete table/list structures if the chunk spans table cells. Return ONLY HTML, no markdown fences."""

        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=self.llm_verify_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )
            return result if "<" in result else None
        except Exception as e:
            logger.warning(f"LLM DOM closure failed: {e}")
            return None

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve text chunks, then do DOM lookup for HTML context."""
        example.get("id", "unknown")

        # Get raw hits from text retriever
        hits = await self._text_retriever.get_hits(query, example)
        if not hits:
            return RetrievalResult(retrieval_type="html_dom_lookup")

        keep_k = self.reader_top_k if self.reader_top_k is not None else self.top_k
        hits = hits[:keep_k]

        passages = []
        urls = []
        seen_urls: set[str] = set()

        for hit in hits:
            article_id = hit.get("article_id")
            chunk_text = hit.get("text", "")
            url = hit.get("url", "")

            html_context = None
            if article_id is not None:
                raw_html = self._fetch_html(int(article_id))
                if raw_html:
                    # Heuristic DOM lookup first
                    html_context = self._dom_lookup(raw_html, chunk_text)

                    # LLM verification/fallback
                    if self.llm_verify and (
                        html_context is None or len(chunk_text) > 500
                    ):
                        llm_result = await self._llm_dom_closure(raw_html, chunk_text)
                        if llm_result:
                            html_context = llm_result

            if html_context:
                passages.append(html_context)
            else:
                passages.append(chunk_text)

            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        # Hard cap per passage. HTML is ~2 chars/token; reader has 65K tokens
        # with ~2K for output + system prompt. Budget ~50K tokens for context
        # = ~100K chars across all passages. Per-passage cap avoids one huge
        # article starving the others.
        MAX_PER_PASSAGE = 30000
        passages = [p[:MAX_PER_PASSAGE] for p in passages]
        MAX_TOTAL_CHARS = 90000
        total = sum(len(p) for p in passages)
        if total > MAX_TOTAL_CHARS:
            per_passage = MAX_TOTAL_CHARS // max(len(passages), 1)
            passages = [p[:per_passage] for p in passages]
            logger.warning(
                "Truncated %d passages from %d to %d total chars",
                len(passages),
                total,
                sum(len(p) for p in passages),
            )

        combined = "\n\n---\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="html_dom_lookup",
        )
