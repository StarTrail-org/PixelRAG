"""Screenshot capture strategies — core render capability, independent of benchmarking."""

from .base import ArticleCapture, CaptureStrategy, TileCapture, article_url
from .cdp_directclip import CDPDirectClipStrategy
from .cdp_oneshot import CDPOneShotStrategy
from .cdp_pertile_imgwait import CDPPerTileImgWaitStrategy
from .cdp_sequential import CDPSequentialStrategy
from .connection import (
    PlaywrightConnection,
    WebsocketConnection,
    launch_playwright,
    launch_websocket,
)

__all__ = [
    "ArticleCapture",
    "CDPDirectClipStrategy",
    "CDPOneShotStrategy",
    "CDPPerTileImgWaitStrategy",
    "CDPSequentialStrategy",
    "CaptureStrategy",
    "PlaywrightConnection",
    "TileCapture",
    "WebsocketConnection",
    "article_url",
    "launch_playwright",
    "launch_websocket",
]
