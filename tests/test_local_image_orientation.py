"""Exercise EXIF orientation through the real local-image build path."""

import json

import pytest
from PIL import Image
from pixelrag_index import pipelines


def _render(src, tmp_path, monkeypatch):
    # Skip the chunk/embed/index subprocesses: no model or search backend needed.
    monkeypatch.setattr(pipelines.subprocess, "run", lambda *args, **kwargs: None)
    output = tmp_path / "index"
    pipelines.build(
        {"source": {"type": "local", "path": str(src)}, "output": str(output)}
    )
    tile_dir = output / "tiles" / "0.png.tiles"
    manifest = json.loads((tile_dir / "tiles.json").read_text())
    assert manifest["tiles"] == ["tile_0000.jpg"]
    assert manifest["source"] == str(src)
    with Image.open(tile_dir / "tile_0000.jpg") as tile:
        assert manifest["page_height"] == tile.height
        assert tile.getexif().get(274, 1) == 1
        return tile.copy()


@pytest.mark.parametrize(
    "orientation,size,corners",
    [
        (None, (80, 40), "RGBY"),
        (1, (80, 40), "RGBY"),
        (2, (80, 40), "GRYB"),
        (3, (80, 40), "YBGR"),
        (4, (80, 40), "BYRG"),
        (5, (40, 80), "RBGY"),
        (6, (40, 80), "BRYG"),
        (7, (40, 80), "YGBR"),
        (8, (40, 80), "GYRB"),
    ],
)
def test_local_image_exif_orientation(
    tmp_path, monkeypatch, orientation, size, corners
):
    colors = {"R": (255, 0, 0), "G": (0, 255, 0), "B": (0, 0, 255), "Y": (255, 255, 0)}
    img = Image.new("RGB", (80, 40))
    for color, box in zip(
        "RGBY", [(0, 0, 40, 20), (40, 0, 80, 20), (0, 20, 40, 40), (40, 20, 80, 40)]
    ):
        img.paste(colors[color], box)
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    src = tmp_path / "photo.jpg"
    img.save(src, exif=exif, quality=100, subsampling=0)
    original = src.read_bytes()

    tile = _render(src, tmp_path, monkeypatch)

    assert tile.size == size
    w, h = size
    for color, point in zip(
        corners,
        [
            (w // 4, h // 4),
            (3 * w // 4, h // 4),
            (w // 4, 3 * h // 4),
            (3 * w // 4, 3 * h // 4),
        ],
    ):
        assert tile.getpixel(point) == pytest.approx(colors[color], abs=10)
    assert src.read_bytes() == original


@pytest.mark.parametrize(
    "stored_size,expected_size", [((80, 4200), (4000, 76)), ((4200, 80), (80, 4200))]
)
def test_width_cap_uses_oriented_dimensions(
    tmp_path, monkeypatch, stored_size, expected_size
):
    src = tmp_path / "large.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", stored_size, "red").save(src, exif=exif)

    tile = _render(src, tmp_path, monkeypatch)

    assert tile.size == expected_size


def test_oriented_transparent_image_keeps_white_background(tmp_path, monkeypatch):
    src = tmp_path / "transparent.png"
    img = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (0, 0, 40, 40))
    exif = Image.Exif()
    exif[274] = 6
    img.save(src, exif=exif)

    tile = _render(src, tmp_path, monkeypatch)

    assert tile.size == (40, 80)
    assert tile.getpixel((20, 20)) == pytest.approx((255, 0, 0), abs=10)
    assert tile.getpixel((20, 60)) == pytest.approx((255, 255, 255), abs=10)
