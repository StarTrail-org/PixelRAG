"""Web/news URL source — reads URLs from a file and yields Documents.

This is a stub implementation. Full download machinery (async HTML fetcher,
SQLite queue, resource rewriting) can be integrated later.

Usage:
    source:
      type: web
      urls_file: /path/to/urls.txt   # one URL per line
      # OR
      preset: news                   # load preset domain limits
"""

from pathlib import Path
from typing import Iterator

from .base import Document, Source

# Cookie banner CSS selectors common across news sites
_COOKIE_BANNER_CSS = """
#sp_message_container, .sp_message_iframe,
.cookie-banner, .cookie-notice, .cookie-consent,
#cookie-law-info-bar, .cc-window, .cc-banner,
#CybotCookiebotDialog, .cookieConsent,
[id*="cookie"], [class*="cookie-banner"], [class*="cookie-notice"],
.gdpr-banner, .gdpr-notice, [id*="gdpr"],
.consent-banner, .consent-overlay,
#didomi-notice, .didomi-popup-notice
{ display: none !important; }
"""

PRESETS: dict[str, dict] = {
    "news": {
        "domain_limits": {
            "www.bbc.com": 10,
            "edition.cnn.com": 10,
            "www.reuters.com": 10,
            "www.theguardian.com": 10,
            "www.nytimes.com": 10,
            "apnews.com": 10,
            "www.aljazeera.com": 10,
            "www.washingtonpost.com": 10,
        },
        "cookie_banner_css": _COOKIE_BANNER_CSS,
    },
}


class WebSource(Source):
    """Source that reads URLs from a plain-text file (one per line).

    Args:
        urls_file: Path to a text file with one URL per line.
        preset: Optional preset name (e.g. "news") to load default config.
        **kwargs: Ignored (for forward compatibility).
    """

    def __init__(
        self,
        urls_file: str | None = None,
        preset: str | None = None,
        **kwargs,
    ):
        self.preset_config = PRESETS.get(preset, {}) if preset else {}
        self._urls: list[str] = []

        if urls_file:
            p = Path(urls_file)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    self._urls = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
            else:
                raise FileNotFoundError(f"urls_file not found: {urls_file}")

    def __iter__(self) -> Iterator[Document]:
        for i, url in enumerate(self._urls):
            yield Document(
                id=f"web_{i:06d}",
                url=url,
                metadata={"type": "web", "source_url": url},
            )

    def __len__(self) -> int:
        return len(self._urls)
