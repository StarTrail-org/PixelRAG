"""Local directory source — auto-detect file types."""

from pathlib import Path
from typing import Iterator

from .base import Document, Source

EXTENSIONS = {
    ".pdf": "pdf",
    ".html": "web",
    ".htm": "web",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".md": "text",
    ".txt": "text",
}


class LocalSource(Source):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        self._files = [
            f
            for f in sorted(self.path.rglob("*"))
            if f.is_file() and f.suffix.lower() in EXTENSIONS
        ]

    def __iter__(self) -> Iterator[Document]:
        for f in self._files:
            ftype = EXTENSIONS.get(f.suffix.lower(), "unknown")
            if ftype == "web":
                yield Document(
                    id=f.stem,
                    url=f"file://{f.resolve()}",
                    metadata={"type": ftype},
                )
            elif ftype == "text":
                yield Document(
                    id=f.stem,
                    path=str(f),
                    metadata={"type": "text", "extension": f.suffix.lower()},
                )
            else:
                yield Document(
                    id=f.stem,
                    path=str(f),
                    metadata={"type": ftype},
                )

    def __len__(self) -> int:
        return len(self._files)
