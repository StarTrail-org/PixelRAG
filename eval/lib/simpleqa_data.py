"""Data loading and preprocessing for SimpleQA evaluation.

This module handles all data preparation:
- Loading the SimpleQA dataset
- Extracting URLs from metadata
- Capturing screenshots from URLs
- Fetching text content from URLs
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import urllib.parse

import pandas as pd
import trafilatura

logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================


def load_simpleqa_data(num_examples: int | None = None) -> list[dict]:
    """Load SimpleQA dataset.

    Args:
        num_examples: Optional limit on number of examples to load.

    Returns:
        List of example dictionaries with 'id', 'problem', 'answer', etc.
    """
    logger.info("Loading SimpleQA dataset...")
    try:
        local_path = "evaluation/simple_qa_eval/data/simple_qa_test_set.csv"
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
            df = pd.read_csv(url)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        df = pd.read_csv(url)

    # Ensure stable ordering: reset index to maintain original CSV row order
    df = df.reset_index(drop=True)

    # Generate unique ID from problem text
    df["id"] = df["problem"].apply(
        lambda problem: hashlib.md5(problem.encode()).hexdigest()
    )

    # Convert to list of dicts, maintaining original CSV order
    data = [row.to_dict() for _, row in df.iterrows()]

    if num_examples:
        logger.info(f"Limiting to first {num_examples} examples.")
        data = data[:num_examples]

    logger.info(f"Loaded {len(data)} examples.")
    return data


def load_simpleqa_verified_data(num_examples: int | None = None) -> list[dict]:
    """Load SimpleQA Verified dataset from Hugging Face.

    Args:
        num_examples: Optional limit on number of examples to load.

    Returns:
        List of example dictionaries with 'id', 'problem', 'answer', etc.
        Compatible format with SimpleQA dataset.
    """
    logger.info("Loading SimpleQA Verified dataset...")
    try:
        # Try using datasets library first (recommended)
        try:
            from datasets import load_dataset

            logger.info("Using Hugging Face datasets library...")
            dataset = load_dataset("google/simpleqa-verified", split="eval")
            df = dataset.to_pandas()
        except ImportError:
            logger.warning("datasets library not available, trying alternative methods")
            # Fallback: try Hugging Face datasets-server API
            try:
                import requests

                logger.info("Trying Hugging Face datasets-server API...")
                api_url = "https://datasets-server.huggingface.co/parquet?dataset=google%2Fsimpleqa-verified&config=simpleqa_verified&split=eval"
                response = requests.get(api_url, timeout=60)
                if response.status_code == 200:
                    import io

                    df = pd.read_parquet(io.BytesIO(response.content))
                    logger.info("Successfully loaded via datasets-server API")
                else:
                    raise Exception(
                        f"Failed to download dataset: HTTP {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Failed to load via API: {e}")
                # Last resort: try direct file download
                try:
                    logger.info("Trying direct file download...")
                    # Try parquet file
                    parquet_url = "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/data/eval-00000-of-00001.parquet"
                    df = pd.read_parquet(parquet_url)
                    logger.info("Successfully loaded via direct file download")
                except Exception as e2:
                    logger.error(f"Failed to load via direct download: {e2}")
                    raise Exception(
                        "All methods failed. Please install 'datasets' library: pip install datasets"
                    )
    except Exception as e:
        logger.error(f"Failed to load SimpleQA Verified dataset: {e}")
        raise

    # Ensure stable ordering: reset index to maintain original order
    df = df.reset_index(drop=True)

    # Convert to compatible format with SimpleQA
    # SimpleQA Verified has: original_index, problem, answer, topic, answer_type, multi_step, requires_reasoning, urls
    # SimpleQA has: metadata (with urls), problem, answer, id

    # Generate unique ID from problem text (same as SimpleQA)
    df["id"] = df["problem"].apply(
        lambda problem: hashlib.md5(problem.encode()).hexdigest()
    )

    # Convert urls to list format if it's a string
    def normalize_urls(urls):
        """Normalize URLs to list format."""
        if isinstance(urls, str):
            # Try to parse as list string
            try:
                import ast

                return ast.literal_eval(urls)
            except Exception:
                # Split by comma if it's a comma-separated string
                return [u.strip() for u in urls.split(",") if u.strip()]
        elif isinstance(urls, list):
            return urls
        else:
            return []

    # Normalize URLs column
    if "urls" in df.columns:
        df["urls"] = df["urls"].apply(normalize_urls)
    else:
        df["urls"] = [[]] * len(df)

    # Convert to metadata format compatible with SimpleQA
    def create_metadata(row):
        """Create metadata dict compatible with SimpleQA format."""
        metadata = {
            "topic": str(row.get("topic", "")),
            "answer_type": str(row.get("answer_type", "")),
            "urls": row.get("urls", []),
        }
        if "multi_step" in row and pd.notna(row["multi_step"]):
            metadata["multi_step"] = bool(row["multi_step"])
        if "requires_reasoning" in row and pd.notna(row["requires_reasoning"]):
            metadata["requires_reasoning"] = bool(row["requires_reasoning"])
        if "original_index" in row and pd.notna(row["original_index"]):
            metadata["original_index"] = int(row["original_index"])
        # Convert to string format similar to SimpleQA (using single quotes for Python dict string)
        return str(metadata)

    df["metadata"] = df.apply(create_metadata, axis=1)

    # Convert to list of dicts, maintaining original order
    data = [row.to_dict() for _, row in df.iterrows()]

    if num_examples:
        logger.info(f"Limiting to first {num_examples} examples.")
        data = data[:num_examples]

    logger.info(f"Loaded {len(data)} SimpleQA Verified examples.")
    return data


def load_text_cache(cache_path: str) -> dict:
    """Load pre-fetched text from JSONL file.

    Args:
        cache_path: Path to JSONL file with cached text.

    Returns:
        Dict mapping example ID to cached item.
    """
    logger.info(f"Loading text cache from {cache_path}...")
    cache = {}
    with open(cache_path, "r") as f:
        for line in f:
            item = json.loads(line)
            cache[item["id"]] = item
    logger.info(f"Loaded {len(cache)} cached items.")
    return cache


# ============================================================================
# URL Extraction
# ============================================================================


def extract_url_from_metadata(example: dict) -> str | None:
    """Extract URL from example metadata.

    Args:
        example: Example dict with 'metadata' field.

    Returns:
        Extracted URL or None.
    """
    meta = example.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            try:
                meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError):
                pass

    target_url = None
    if isinstance(meta, dict):
        if "url" in meta:
            target_url = meta["url"]
        elif (
            "urls" in meta and isinstance(meta["urls"], list) and len(meta["urls"]) > 0
        ):
            # Flatten URLs: some entries have multiple URLs concatenated in a single string
            # (separated by newlines OR directly joined like "https://a.comhttps://b.com")
            all_urls = []
            for url_entry in meta["urls"]:
                if isinstance(url_entry, str):
                    # Split on "https://" boundaries to handle concatenated URLs
                    parts = re.split(r"(?=https?://)", url_entry)
                    for part in parts:
                        part = part.strip().rstrip(",'\"").strip("- ").strip()
                        if part and re.match(r"https?://", part):
                            all_urls.append(part)

            # Prefer en.wikipedia.org article URLs (exclude non-English and Category pages)
            wikipedia_urls = [
                u
                for u in all_urls
                if "en.wikipedia.org/wiki/" in u
                and "/Category:" not in u
                and "wikipedia-on-ipfs" not in u.lower()
            ]
            if wikipedia_urls:
                target_url = wikipedia_urls[0]
            else:
                # Secondary: wikimedia.org URLs (e.g., commons.wikimedia.org)
                wikimedia_urls = [u for u in all_urls if "wikimedia.org" in u.lower()]
                target_url = (
                    wikimedia_urls[0]
                    if wikimedia_urls
                    else (all_urls[0] if all_urls else None)
                )

    # Extract first valid URL from the string
    if target_url:
        url_match = re.search(r"https?://[^\s<>\"{}|\\^`\[\]]+", target_url)
        target_url = url_match.group(0) if url_match else None

    # Note: strip URL fragment (#section) so that URLs differing
    # only by anchor are treated as the same page for deduplication and
    # retrieval-accuracy matching.
    if target_url and "#" in target_url:
        target_url = target_url.split("#")[0]

    return target_url


# ============================================================================
# Screenshot Capture
# ============================================================================

# Lazy import screenshot utilities
_capture_screenshot = None
_encode_image = None
_encode_image_for_vlm = None


def _init_screenshot_utils():
    """Initialize screenshot utilities (lazy import)."""
    global _capture_screenshot, _encode_image, _encode_image_for_vlm
    if _capture_screenshot is not None:
        return True

    try:
        from .screenshot import capture_screenshot, encode_image, encode_image_for_vlm

        _capture_screenshot = capture_screenshot
        _encode_image = encode_image
        _encode_image_for_vlm = encode_image_for_vlm
        return True
    except ImportError:
        logger.warning("Screenshot utilities not available")
        return False


def capture_screenshot_for_example(
    example: dict, screenshot_dir: str = "screenshots"
) -> str | None:
    """Capture screenshot for a single example.

    Args:
        example: Example dict with metadata containing URL.
        screenshot_dir: Directory to save screenshots.

    Returns:
        Path to screenshot file, or None if failed.
    """
    if not _init_screenshot_utils():
        return None

    target_url = extract_url_from_metadata(example)
    if not target_url:
        return None

    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_filename = f"{example['id']}_fullhd.png"
    screenshot_path = os.path.join(screenshot_dir, screenshot_filename)

    # Check if valid screenshot already exists
    if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
        logger.debug(f"Screenshot exists: {screenshot_path}")
        return screenshot_path

    # Capture screenshot
    try:
        if _capture_screenshot is None:
            return None
        success = _capture_screenshot(target_url, screenshot_path, True)
        if (
            success
            and os.path.exists(screenshot_path)
            and os.path.getsize(screenshot_path) > 0
        ):
            file_size = os.path.getsize(screenshot_path) // 1024
            logger.info(f"Screenshot saved: {screenshot_path} ({file_size}KB)")
            return screenshot_path
        else:
            logger.warning(
                f"Screenshot failed (no output): {target_url} -> {screenshot_path}"
            )
    except Exception as e:
        logger.error(f"Screenshot error for {target_url}: {e}")

    return None


async def capture_screenshot_async(
    example: dict, screenshot_dir: str = "screenshots"
) -> str | None:
    """Async wrapper for screenshot capture."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, capture_screenshot_for_example, example, screenshot_dir
    )


def encode_screenshot(screenshot_path: str) -> str | None:
    """Encode screenshot to base64.

    Args:
        screenshot_path: Path to screenshot file, or already-encoded base64 string.

    Returns:
        Base64 encoded string, or None if failed.
    """
    if not screenshot_path:
        return None

    if not os.path.exists(screenshot_path):
        if len(screenshot_path) > 500 and "/" not in screenshot_path[:20]:
            return screenshot_path
        return None

    if not _init_screenshot_utils():
        return None

    try:
        if _encode_image is None:
            return None
        return _encode_image(screenshot_path)
    except Exception as e:
        logger.error(f"Image encoding failed for {screenshot_path}: {e}")
        return None


async def encode_screenshot_async(screenshot_path: str) -> str | None:
    """Async wrapper for screenshot encoding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode_screenshot, screenshot_path)


def encode_screenshot_for_vlm(
    screenshot_path: str, max_pixels: int | None = None
) -> str | None:
    """Encode screenshot for VLM ground truth with configurable max_pixels.

    Unlike encode_screenshot(), this function does NOT apply max_height limit.
    You can control max_pixels to study the effect of resize on VLM performance.

    Args:
        screenshot_path: Path to screenshot file.
        max_pixels: Maximum pixels before resize. If None, uses default (89M).
                    Common values:
                    - 16_777_216 (16M): Qwen3-VL default
                    - 12_845_056 (12.8M): Qwen2-VL default
                    - 4_000_000 (4M): ~4000 tokens
                    - 1_000_000 (1M): ~1000 tokens

    Returns:
        Base64 encoded string, or None if failed.
    """
    if not _init_screenshot_utils():
        return None

    if not screenshot_path or not os.path.exists(screenshot_path):
        return None

    try:
        if _encode_image_for_vlm is None:
            return None
        if max_pixels is not None:
            return _encode_image_for_vlm(screenshot_path, max_pixels=max_pixels)
        return _encode_image_for_vlm(screenshot_path)
    except Exception as e:
        logger.error(f"Image encoding (VLM) failed for {screenshot_path}: {e}")
        return None


async def encode_screenshot_for_vlm_async(
    screenshot_path: str, max_pixels: int | None = None
) -> str | None:
    """Async wrapper for VLM screenshot encoding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, encode_screenshot_for_vlm, screenshot_path, max_pixels
    )


# ============================================================================
# Pixel-Compressed Encoding for Generation
# ============================================================================


def make_compressed_encoder(compress_ratio: int, save_dir: str | None = None):
    """Create an image encoder that downscales images before encoding to base64.

    The compression ratio N divides the total pixel count by N, i.e. each
    dimension is scaled by 1/sqrt(N).  For a 1024x1024 tile:
      - ratio  1 ->  1024x1024  (no compression, baseline)
      - ratio  4 ->   512x512
      - ratio  9 ->  ~341x341
      - ratio 16 ->   256x256
      - ratio 25 ->  ~205x205

    Uses LANCZOS resampling (best quality for downscaling).

    Compressed images are saved to ``save_dir`` (if provided) so they can
    be visually inspected later.  The mapping from original path to saved
    compressed path is recorded in ``encoder.compressed_paths`` (a dict
    attached to the returned function object).

    Args:
        compress_ratio: Pixel compression ratio (1 = no compression).
        save_dir: Directory to save compressed images. If None, a default
                  directory ``compressed_tiles_{ratio}x`` is used.

    Returns:
        A function with the same signature as ``encode_screenshot`` that
        first downscales the image, then encodes it to base64.
        The function has an attribute ``compressed_paths: dict[str, str]``
        mapping original_path -> compressed_path.
    """
    if compress_ratio <= 1:
        # No compression – use the normal encoder
        return encode_screenshot

    import math

    scale_factor = 1.0 / math.sqrt(compress_ratio)

    # Set up save directory
    if save_dir is None:
        save_dir = f"compressed_tiles_{compress_ratio}x"
    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f"Pixel compression enabled: ratio={compress_ratio}, "
        f"scale_factor={scale_factor:.4f} per dimension, "
        f"saving compressed images to {save_dir}"
    )

    # Shared dict to track original -> compressed path mapping
    _compressed_paths: dict[str, str] = {}

    def _compressed_encode(screenshot_path: str) -> str | None:
        """Encode image with pixel compression and save to disk."""
        import base64 as _b64
        from io import BytesIO
        from PIL import Image as _Image

        if not screenshot_path or not os.path.exists(screenshot_path):
            return None

        try:
            _Image.MAX_IMAGE_PIXELS = 300_000_000
            with _Image.open(screenshot_path) as img:
                new_w = max(1, int(img.width * scale_factor))
                new_h = max(1, int(img.height * scale_factor))

                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_resized = img.resize((new_w, new_h), _Image.Resampling.LANCZOS)

                # Save compressed image to disk
                basename = os.path.splitext(os.path.basename(screenshot_path))[0]
                compressed_filename = f"{basename}_compress{compress_ratio}x.png"
                compressed_path = os.path.join(save_dir, compressed_filename)
                img_resized.save(compressed_path, format="PNG")
                _compressed_paths[screenshot_path] = compressed_path

                # Encode to base64 from the saved file
                buf = BytesIO()
                img_resized.save(buf, format="PNG")
                return _b64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Compressed encode failed for {screenshot_path}: {e}")
            return None

    # Attach the path mapping dict to the function so callers can access it
    _compressed_encode.compressed_paths = _compressed_paths
    _compressed_encode.compress_ratio = compress_ratio
    _compressed_encode.save_dir = save_dir

    return _compressed_encode


# ============================================================================
# Text Fetching
# ============================================================================


def fetch_webpage_text(url: str, max_chars: int = 50000) -> str | None:
    """Fetch webpage and extract clean text content using trafilatura.

    Args:
        url: URL to fetch.
        max_chars: Maximum characters to return.

    Returns:
        Extracted text content, or None if failed.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning(f"Failed to download {url}")
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if text is None:
            logger.warning(f"Failed to extract text from {url}")
            return None

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Truncate if needed
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"

        return text
    except Exception as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        return None


def fetch_text_for_example(
    example: dict, max_chars: int = 50000, text_cache: dict | None = None
) -> tuple[str | None, str | None]:
    """Fetch text content for a single example.

    Args:
        example: Example dict with metadata containing URL.
        max_chars: Maximum characters to return.
        text_cache: Optional pre-fetched text cache.

    Returns:
        Tuple of (text_content, source_url).
    """
    example_id = example.get("id")

    # Check cache first
    if text_cache and example_id in text_cache:
        cached = text_cache[example_id]
        text = cached.get("text")
        url = cached.get("extracted_url")
        if text:
            return text, url

    # Extract URL and fetch
    target_url = extract_url_from_metadata(example)
    if not target_url:
        return None, None

    text = fetch_webpage_text(target_url, max_chars)
    return text, target_url


async def fetch_text_async(
    example: dict, max_chars: int = 50000, text_cache: dict | None = None
) -> tuple[str | None, str | None]:
    """Async wrapper for text fetching."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, fetch_text_for_example, example, max_chars, text_cache
    )


# ============================================================================
# Image Tiling
# ============================================================================


def split_image_to_tiles(
    image_path: str,
    output_dir: str,
    tile_size: int | tuple[int, int] = 512,
    overlap: int = 0,
) -> list[str]:
    """Split an image into fixed-size tiles.

    Args:
        image_path: Path to the source image.
        output_dir: Directory to save tiles.
        tile_size: Size of each tile. Can be int (square) or tuple (width, height).
        overlap: Overlap between tiles in pixels.

    Returns:
        List of tile file paths.
    """
    from PIL import Image
    import glob

    if not os.path.exists(image_path):
        return []

    os.makedirs(output_dir, exist_ok=True)

    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Check if tiles already exist for this image
    existing_tiles = sorted(
        glob.glob(os.path.join(output_dir, f"{base_name}_tile_*.png"))
    )
    if existing_tiles:
        # Tiles already exist, return them
        return existing_tiles

    # Support both square and rectangular tiles
    if isinstance(tile_size, tuple):
        tile_w, tile_h = tile_size
    else:
        tile_w = tile_h = tile_size

    try:
        Image.MAX_IMAGE_PIXELS = 300_000_000
        img = Image.open(image_path)
        width, height = img.size

        tile_paths = []
        step_x = tile_w - overlap
        step_y = tile_h - overlap

        row = 0
        y = 0
        while y < height:
            col = 0
            x = 0
            while x < width:
                # Calculate tile boundaries
                x2 = min(x + tile_w, width)
                y2 = min(y + tile_h, height)

                # Calculate tile dimensions
                tile_width = x2 - x
                tile_height = y2 - y

                # Skip tiles with extreme aspect ratios (> 10:1)
                # This prevents issues with ColQwen which requires aspect ratio < 200
                if tile_width > 0 and tile_height > 0:
                    aspect_ratio = max(
                        tile_width / tile_height, tile_height / tile_width
                    )
                    if aspect_ratio > 10:
                        col += 1
                        x += step_x
                        if x >= width:
                            break
                        continue

                # Crop tile
                tile = img.crop((x, y, x2, y2))

                # Save tile
                tile_filename = f"{base_name}_tile_{row}_{col}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                tile.save(tile_path)
                tile_paths.append(tile_path)

                col += 1
                x += step_x
                if x >= width:
                    break

            row += 1
            y += step_y
            if y >= height:
                break

        img.close()
        return tile_paths

    except Exception as e:
        logger.warning(f"Failed to split image {image_path}: {e}")
        return []


def prepare_tiles_for_screenshots(
    screenshot_dir: str, tiles_dir: str, tile_size: int = 512, overlap: int = 0
) -> dict[str, list[str]]:
    """Split all screenshots in a directory into tiles.

    Args:
        screenshot_dir: Directory containing full screenshots.
        tiles_dir: Directory to save tiles.
        tile_size: Size of each tile.
        overlap: Overlap between tiles.

    Returns:
        Dict mapping original image path to list of tile paths.
    """
    os.makedirs(tiles_dir, exist_ok=True)

    result = {}
    for filename in os.listdir(screenshot_dir):
        if not filename.endswith(".png"):
            continue

        image_path = os.path.join(screenshot_dir, filename)
        tile_paths = split_image_to_tiles(image_path, tiles_dir, tile_size, overlap)

        if tile_paths:
            result[image_path] = tile_paths
            logger.info(f"Split {filename} into {len(tile_paths)} tiles")

    logger.info(
        f"Total: {sum(len(v) for v in result.values())} tiles from {len(result)} images"
    )
    return result


# ============================================================================
# NQ (Natural Questions) Data Loading
# ============================================================================


def load_nq_data(
    num_examples: int | None = 1000, split: str = "validation"
) -> list[dict]:
    """Load Natural Questions (full) split.

    For validation, follows the short-answer protocol used by our NQ eval:
    keep only examples where >=2 of 5 annotators marked a non-null short
    answer. The train split has a single annotation per example, so train keeps
    examples with a non-null short answer.

    Source: HuggingFace google-research-datasets/natural_questions.
    Reference: Kwiatkowski et al. (2019).

    Args:
        num_examples: Number of examples to return. Default 1000.
        split: HuggingFace split to stream ("train" or "validation").

    Returns:
        List of dicts with id, problem, gold_answers, metadata.
    """
    from datasets import load_dataset
    import html as _html

    if split not in {"train", "validation"}:
        raise ValueError(
            f"Unsupported NQ split: {split!r}. Expected 'train' or 'validation'."
        )

    logger.info(f"Loading NQ {split} split (streaming)...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split=split,
        streaming=True,
    )

    data = []
    for ex in ds:
        # Extract short answers from all 5 annotators
        annotations = ex["annotations"]
        short_answer_texts = set()
        non_null_annotators = 0

        # annotations is a dict with list values (one per annotator)
        num_annotators = len(annotations["id"])
        for i in range(num_annotators):
            texts = annotations["short_answers"][i].get("text", [])
            if texts:
                non_null_annotators += 1
                for t in texts:
                    if t.strip():
                        short_answer_texts.add(t.strip())

        min_non_null = 2 if split == "validation" else 1
        if non_null_annotators < min_non_null:
            continue

        if not short_answer_texts:
            continue

        question_text = ex["question"]["text"]
        doc_url = ex["document"]["url"]
        # Clean up HTML entities in URL (e.g., &amp; -> &)
        doc_url = _html.unescape(doc_url)
        # Normalize NQ URL format: /w/index.php?title=Foo&oldid=123 -> /wiki/Foo
        _title_match = re.search(r"[?&]title=([^&]+)", doc_url)
        if _title_match:
            doc_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(_title_match.group(1), safe='/:(),-')}"

        example = {
            "id": hashlib.md5(question_text.encode()).hexdigest(),
            "problem": question_text,
            "gold_answers": sorted(short_answer_texts),
            "metadata": {
                "urls": [doc_url],
                "dataset": "nq",
                "document_title": ex["document"]["title"],
            },
        }
        data.append(example)

        if num_examples and len(data) >= num_examples:
            break

    filter_desc = (
        ">=2 annotator agreement" if split == "validation" else "non-null short answer"
    )
    logger.info(f"Loaded {len(data)} NQ {split} examples (filtered by {filter_desc}).")
    return data


# ============================================================================
# TriviaQA Data Loading
# ============================================================================


def load_triviaqa_data(num_examples: int | None = 1000) -> list[dict]:
    """Load TriviaQA rc.wikipedia validation split.

    Uses entity_pages.title to construct ground truth Wikipedia URLs.
    gold_answers includes answer.value + answer.aliases (following TriviaQA official eval).

    Source: HuggingFace mandarjoshi/trivia_qa, config rc.wikipedia, validation split.
    Reference: Joshi et al. (2017).

    Args:
        num_examples: Number of examples to return. Default 1000.

    Returns:
        List of dicts with id, problem, gold_answers, metadata.
    """
    from datasets import load_dataset
    import ast as _ast
    from urllib.parse import quote as _url_quote

    logger.info("Loading TriviaQA rc.wikipedia validation split (streaming)...")
    ds = load_dataset(
        "mandarjoshi/trivia_qa",
        "rc.wikipedia",
        split="validation",
        streaming=True,
    )

    data = []
    for ex in ds:
        question = ex["question"]
        answer_obj = ex["answer"]

        # Extract gold answers: value + aliases
        gold_answers = set()
        value = answer_obj.get("value", "")
        if value:
            gold_answers.add(value)

        # aliases is stored as a string repr of a list
        aliases_raw = answer_obj.get("aliases", "")
        if isinstance(aliases_raw, str) and aliases_raw:
            try:
                aliases = _ast.literal_eval(aliases_raw)
                if isinstance(aliases, list):
                    for a in aliases:
                        if a and a.strip():
                            gold_answers.add(a.strip())
            except (ValueError, SyntaxError):
                pass
        elif isinstance(aliases_raw, list):
            for a in aliases_raw:
                if a and a.strip():
                    gold_answers.add(a.strip())

        if not gold_answers:
            continue

        # Construct Wikipedia URL from entity_pages.title
        urls = []
        entity_titles = ex.get("entity_pages", {}).get("title", [])
        if entity_titles:
            for title in entity_titles:
                if title:
                    wiki_url = f"https://en.wikipedia.org/wiki/{_url_quote(title.replace(' ', '_'))}"
                    urls.append(wiki_url)

        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": sorted(gold_answers),
            "question_type": ex.get("question_source", ""),
            "metadata": {
                "urls": urls,
                "dataset": "triviaqa",
                "question_id": ex.get("question_id", ""),
            },
        }
        data.append(example)

        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} TriviaQA examples.")
    return data


# ============================================================================
# NQ-Tables Data Loading
# ============================================================================


def load_nq_tables_data(num_examples: int | None = 1000) -> list[dict]:
    """Load NQ-Tables dev split (table subset of Natural Questions).

    NQ-Tables filters Natural Questions to only keep examples where the gold
    answer resides inside a Wikipedia HTML table. Each example includes the
    full table content (columns + rows) and the Wikipedia URL.

    Source: GCS gs://tapas_models/2021_07_22/nq_tables/interactions/dev.jsonl
    Reference: Herzig et al. (2021), "Open Domain Question Answering over
    Tables via Dense Retrieval" (NAACL 2021).

    Args:
        num_examples: Number of examples to return. Default 1000.

    Returns:
        List of dicts with id, problem, gold_answers, metadata.
    """
    import html as _html

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "nq_tables", "dev.jsonl"
    )
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"NQ-Tables data not found at {data_path}. "
            "Download with: gsutil cp gs://tapas_models/2021_07_22/nq_tables/interactions/dev.jsonl data/nq_tables/"
        )

    logger.info(f"Loading NQ-Tables dev split from {data_path}...")

    import json as _json

    data = []
    with open(data_path) as f:
        for line in f:
            ex = _json.loads(line)
            questions = ex.get("questions", [])
            if not questions:
                continue

            q = questions[0]
            question_text = q.get("originalText", "")
            answer_texts = q.get("answer", {}).get("answerTexts", [])

            if not question_text or not answer_texts:
                continue

            gold_answers = [a.strip() for a in answer_texts if a.strip()]
            if not gold_answers:
                continue

            # Extract Wikipedia URL from table metadata
            table = ex.get("table", {})
            doc_url = table.get("documentUrl", "")
            doc_url = _html.unescape(doc_url)
            # Normalize NQ URL format: /w/index.php?title=Foo&oldid=123 -> /wiki/Foo
            _title_match = re.search(r"[?&]title=([^&]+)", doc_url)
            if _title_match:
                doc_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(_title_match.group(1), safe='/:(),-')}"

            example = {
                "id": ex.get("id", hashlib.md5(question_text.encode()).hexdigest()),
                "problem": question_text,
                "gold_answers": gold_answers,
                "metadata": {
                    "urls": [doc_url] if doc_url else [],
                    "dataset": "nq_tables",
                    "document_title": table.get("documentTitle", ""),
                    "table_id": table.get("tableId", ""),
                },
            }
            data.append(example)

            if num_examples and len(data) >= num_examples:
                break

    logger.info(f"Loaded {len(data)} NQ-Tables examples.")
    return data


# ============================================================================
# Multiple-Choice Reasoning Benchmarks
# ============================================================================

LETTERS = ["A", "B", "C", "D", "E"]


def _format_mc_options(labels: list[str], texts: list[str]) -> str:
    """Format MC options as 'A. text1\nB. text2\n...'"""
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


MC_INSTRUCTION = "Choose the best answer from the options above. Reply with ONLY the letter (e.g. A, B, C, or D)."


def load_piqa_data(num_examples: int | None = None) -> list[dict]:
    """Load PIQA (Physical Intuition QA) validation split.

    2-choice physical commonsense benchmark. Label is 0 or 1.
    Source: HuggingFace `ybisk/piqa`, validation split.

    Returns list of dicts with problem (question only), gold_answers (letter),
    additional_instructions (options + MC instruction), metadata.
    """
    from datasets import load_dataset

    logger.info("Loading PIQA validation split...")
    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")

    data = []
    for ex in ds:
        question = ex["goal"]
        options = [ex["sol1"], ex["sol2"]]
        label = int(ex["label"])
        gold_letter = LETTERS[label]

        options_text = _format_mc_options(LETTERS[:2], options)
        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": [gold_letter],
            "additional_instructions": f"{options_text}\n\n{MC_INSTRUCTION}",
            "metadata": {"dataset": "piqa", "urls": [], "gold_letter": gold_letter},
        }
        data.append(example)
        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} PIQA examples.")
    return data


def load_hellaswag_data(num_examples: int | None = None) -> list[dict]:
    """Load HellaSwag validation split.

    4-choice sentence completion benchmark. Label is "0"-"3".
    Source: HuggingFace `Rowan/hellaswag`, validation split.
    """
    from datasets import load_dataset

    logger.info("Loading HellaSwag validation split...")
    ds = load_dataset(
        "Rowan/hellaswag", split="validation", revision="refs/convert/parquet"
    )

    data = []
    for ex in ds:
        question = ex["ctx"]
        options = ex["endings"]
        label = int(ex["label"])
        gold_letter = LETTERS[label]

        options_text = _format_mc_options(LETTERS[: len(options)], options)
        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": [gold_letter],
            "additional_instructions": f"{options_text}\n\n{MC_INSTRUCTION}",
            "metadata": {
                "dataset": "hellaswag",
                "urls": [],
                "gold_letter": gold_letter,
            },
        }
        data.append(example)
        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} HellaSwag examples.")
    return data


def load_commonsenseqa_data(num_examples: int | None = None) -> list[dict]:
    """Load CommonsenseQA validation split.

    5-choice commonsense reasoning benchmark. answerKey is A-E.
    Source: HuggingFace `tau/commonsense_qa`, validation split.
    """
    from datasets import load_dataset

    logger.info("Loading CommonsenseQA validation split...")
    ds = load_dataset("tau/commonsense_qa", split="validation")

    data = []
    for ex in ds:
        question = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        gold_letter = ex["answerKey"]

        options_text = _format_mc_options(labels, texts)
        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": [gold_letter],
            "additional_instructions": f"{options_text}\n\n{MC_INSTRUCTION}",
            "metadata": {
                "dataset": "commonsense_qa",
                "urls": [],
                "gold_letter": gold_letter,
            },
        }
        data.append(example)
        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} CommonsenseQA examples.")
    return data


def load_openbookqa_data(num_examples: int | None = None) -> list[dict]:
    """Load OpenBookQA test split.

    4-choice science QA benchmark. answerKey is A-D.
    Source: HuggingFace `allenai/openbookqa`, main config, test split.
    """
    from datasets import load_dataset

    logger.info("Loading OpenBookQA test split...")
    ds = load_dataset("allenai/openbookqa", "main", split="test")

    data = []
    for ex in ds:
        question = ex["question_stem"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        gold_letter = ex["answerKey"]

        options_text = _format_mc_options(labels, texts)
        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": [gold_letter],
            "additional_instructions": f"{options_text}\n\n{MC_INSTRUCTION}",
            "metadata": {
                "dataset": "openbookqa",
                "urls": [],
                "gold_letter": gold_letter,
            },
        }
        data.append(example)
        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} OpenBookQA examples.")
    return data


def load_arc_data(
    config: str = "ARC-Challenge", num_examples: int | None = None
) -> list[dict]:
    """Load ARC (AI2 Reasoning Challenge) test split.

    3-5 choice science exam benchmark. answerKey is A-E or 1-5 (normalized to letters).
    Source: HuggingFace `allenai/ai2_arc`, ARC-Challenge or ARC-Easy config, test split.

    Args:
        config: "ARC-Challenge" or "ARC-Easy"
        num_examples: Max examples to return. None = all.
    """
    from datasets import load_dataset

    dataset_name = config.lower().replace("-", "_")
    logger.info(f"Loading ARC {config} test split...")
    ds = load_dataset("allenai/ai2_arc", config, split="test")

    # ARC answerKey can be "1","2","3","4","5" instead of letters
    DIGIT_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    data = []
    for ex in ds:
        question = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        gold_letter = ex["answerKey"]
        gold_letter = DIGIT_TO_LETTER.get(gold_letter, gold_letter)

        options_text = _format_mc_options(labels, texts)
        example = {
            "id": hashlib.md5(question.encode()).hexdigest(),
            "problem": question,
            "gold_answers": [gold_letter],
            "additional_instructions": f"{options_text}\n\n{MC_INSTRUCTION}",
            "metadata": {
                "dataset": dataset_name,
                "urls": [],
                "gold_letter": gold_letter,
            },
        }
        data.append(example)
        if num_examples and len(data) >= num_examples:
            break

    logger.info(f"Loaded {len(data)} ARC {config} examples.")
    return data
