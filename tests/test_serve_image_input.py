"""Input bounds on the public /search endpoint (OOM/DoS hardening).

`_parse_queries` decodes attacker-controlled base64 images and `SearchRequest`
caps the batch size. A crafted request must not exhaust memory or turn an
uncaught decode error into a 500 — the same public-endpoint hardening as the
n_docs bound.
"""

import base64
import io

import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from pixelrag_serve.api import Query, SearchRequest, _parse_queries


def _png_b64(w: int, h: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_queries_batch_is_capped():
    SearchRequest(queries=[Query(text="ok")])  # small batch fine
    with pytest.raises(ValidationError):
        SearchRequest(queries=[Query(text="x")] * 33)


def test_valid_image_decodes():
    _, images = _parse_queries([Query(image=_png_b64(4, 4))])
    assert images[0] is not None and images[0].mode == "RGB"


def test_corrupt_image_is_400_not_500():
    not_an_image = base64.b64encode(b"hello world, definitely not a png").decode()
    with pytest.raises(HTTPException) as e:
        _parse_queries([Query(image=not_an_image)])
    assert e.value.status_code == 400


def test_oversized_payload_is_400(monkeypatch):
    monkeypatch.setattr("pixelrag_serve.api._MAX_IMAGE_B64_LEN", 16)
    with pytest.raises(HTTPException) as e:
        _parse_queries([Query(image=_png_b64(4, 4))])  # base64 longer than 16
    assert e.value.status_code == 400


def test_decompression_bomb_is_400(monkeypatch):
    # A small image over a lowered pixel cap stands in for a bomb whose header
    # claims a huge canvas — the guard must reject before .convert() allocates.
    monkeypatch.setattr("pixelrag_serve.api._MAX_IMAGE_PIXELS", 100)
    with pytest.raises(HTTPException) as e:
        _parse_queries([Query(image=_png_b64(50, 50))])  # 2500 px > 100
    assert e.value.status_code == 400
