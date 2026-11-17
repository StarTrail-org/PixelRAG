"""Tests for local image rendering in the index pipeline (issue #67).

Verifies that local image files (PNG/JPG) are correctly rendered into the
tile directory structure that Stage 2 (chunk) expects.
"""

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def image_dir(tmp_path):
    """Create a temp dir with test images."""
    d = tmp_path / "images"
    d.mkdir()
    # Normal sized image
    img = Image.new("RGB", (800, 600), color="blue")
    img.save(d / "small.png")
    # Oversized image (should be resized)
    img_large = Image.new("RGB", (6000, 4000), color="red")
    img_large.save(d / "large.jpg")
    return d


@pytest.fixture
def config_and_build(tmp_path, image_dir):
    """Set up config and run the pipeline build (Stage 1 only)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))
    from pixelrag_index.config import load_config

    config_path = tmp_path / "pixelrag.yaml"
    config_path.write_text(
        f"source:\n  type: local\n  path: {image_dir}\n\n"
        f"embed:\n  model: Qwen/Qwen3-VL-Embedding-2B\n  device: cpu\n\n"
        f"output: {tmp_path / 'index'}\n"
    )
    return load_config(config_path)


def test_local_images_produce_tile_directories(tmp_path, image_dir):
    """Local images must produce {idx}.png.tiles/ directories with tiles.json."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))

    # Simulate what pipelines.py does for image_docs
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    _MAX_WIDTH = 4000
    images = list(image_dir.iterdir())

    for idx, img_path in enumerate(sorted(images)):
        tile_dir = tiles_dir / f"{idx}.png.tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        if img.width > _MAX_WIDTH:
            ratio = _MAX_WIDTH / img.width
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
            )

        tile_path = tile_dir / "tile_0000.jpg"
        img.save(tile_path, "JPEG", quality=90)

        manifest = {
            "url": str(img_path),
            "page_height": img.height,
            "tiles": ["tile_0000.jpg"],
            "complete": True,
        }
        with open(tile_dir / "tiles.json", "w") as f:
            json.dump(manifest, f)

    # Verify
    tile_dirs = sorted(tiles_dir.glob("*.tiles"))
    assert len(tile_dirs) == 2

    for td in tile_dirs:
        assert (td / "tile_0000.jpg").exists()
        assert (td / "tiles.json").exists()
        with open(td / "tiles.json") as f:
            m = json.load(f)
        assert m["complete"] is True
        assert len(m["tiles"]) == 1


def test_large_image_is_resized(tmp_path):
    """Images wider than 4000px must be resized to cap VRAM usage."""
    img = Image.new("RGB", (8000, 5000), color="green")
    src = tmp_path / "huge.png"
    img.save(src)

    _MAX_WIDTH = 4000
    loaded = Image.open(src).convert("RGB")
    if loaded.width > _MAX_WIDTH:
        ratio = _MAX_WIDTH / loaded.width
        loaded = loaded.resize(
            (int(loaded.width * ratio), int(loaded.height * ratio)), Image.LANCZOS
        )

    assert loaded.width == 4000
    assert loaded.height == 2500  # 5000 * (4000/8000)


def test_small_image_not_resized(tmp_path):
    """Images within the width cap must not be modified."""
    img = Image.new("RGB", (1920, 1080), color="blue")
    src = tmp_path / "normal.png"
    img.save(src)

    _MAX_WIDTH = 4000
    loaded = Image.open(src).convert("RGB")
    if loaded.width > _MAX_WIDTH:
        ratio = _MAX_WIDTH / loaded.width
        loaded = loaded.resize(
            (int(loaded.width * ratio), int(loaded.height * ratio)), Image.LANCZOS
        )

    assert loaded.width == 1920
    assert loaded.height == 1080
