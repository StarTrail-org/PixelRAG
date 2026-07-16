from .base import Document as Document, Source as Source
from .crawl import CrawlSource
from .kiwix import KiwixSource
from .local import LocalSource
from .pdf import PDFSource
from .web import WebSource

SOURCES = {
    "kiwix": KiwixSource,
    "web": WebSource,
    "pdf": PDFSource,
    "local": LocalSource,
    "crawl": CrawlSource,
}
