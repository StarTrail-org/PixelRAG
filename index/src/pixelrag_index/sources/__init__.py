from .base import Document as Document
from .base import Source as Source
from .kiwix import KiwixSource
from .local import LocalSource
from .pdf import PDFSource
from .web import WebSource

SOURCES = {
    "kiwix": KiwixSource,
    "web": WebSource,
    "pdf": PDFSource,
    "local": LocalSource,
}
