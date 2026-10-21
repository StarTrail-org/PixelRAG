"""PDF source — a single .pdf file or a directory tree of them."""

from pathlib import Path
from typing import Iterator

from .base import Document, Source


class PDFSource(Source):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        # A path pointing straight at a .pdf is a single-item source; the glob
        # below only ever matches inside a directory.
        self._files = (
            [self.path]
            if self.path.is_file() and self.path.suffix.lower() == ".pdf"
            else sorted(self.path.glob("**/*.pdf"))
        )

    def __iter__(self) -> Iterator[Document]:
        for pdf in self._files:
            yield Document(id=pdf.stem, path=str(pdf), metadata={"type": "pdf"})

    def __len__(self) -> int:
        return len(self._files)
