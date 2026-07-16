import os
from typing import Any, Protocol, TypedDict, runtime_checkable

_META_FIELDS = ("article_id", "tile_index", "chunk_index", "y_offset", "tile_height")


class SearchHit(TypedDict):
    score: float
    vector_id: int | str
    article_id: int
    tile_index: int
    chunk_index: int
    y_offset: int
    tile_height: int


@runtime_checkable
class VectorBackend(Protocol):
    name: str
    dimension: int
    nlist: int
    nprobe: int

    @property
    def ntotal(self) -> int: ...

    def set_nprobe(self, n: int | None) -> None: ...

    def reset_nprobe(self) -> None: ...

    def raw_search(
        self,
        query_vectors: Any,
        k: int,
        min_tile_height: int | None = None,
        article_ids: Any = None,
        filter_cache_key: str | None = None,
    ) -> list[list[SearchHit]]: ...

    def reconstruct(self, vids: list[int | str]) -> list[list[float] | None]: ...


class FaissBackend:
    name = "faiss"

    def __init__(self, index_dir: str, summary: dict):
        import logging

        import faiss
        import numpy as np

        self._faiss = faiss
        self._np = np
        logger = logging.getLogger("pixelrag-serve")
        index_path = os.path.join(index_dir, "index.faiss")
        # PIXELRAG_INDEX_MMAP=1 memory-maps the index instead of reading the
        # whole file into RAM — startup is near-instant (no full read of a
        # multi-100G index over NFS) and inverted lists are paged in on demand
        # at query time; the OS page cache keeps hot lists resident.
        if os.environ.get("PIXELRAG_INDEX_MMAP"):
            logger.info("Loading FAISS index from %s (mmap)...", index_path)
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        else:
            logger.info("Loading FAISS index from %s...", index_path)
            self.index = faiss.read_index(index_path)
        self._meta = np.load(os.path.join(index_dir, "metadata.npz"))
        # filter_cache_key -> int64 vector positions for a stable article_ids
        # filter (department filters are reused across every query).
        self._filter_positions: dict[str, Any] = {}
        self.dimension = summary.get("dimension", self.index.d)
        self.nlist = summary.get("nlist", 4096)
        # Flat indexes have no nprobe — only IVF does. Treat it as 0/no-op
        # there (main's pre-backend code isinstance-checked the same way).
        self._default_nprobe = getattr(self.index, "nprobe", 0)
        self._direct_map_built = False

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def nprobe(self) -> int:
        return getattr(self.index, "nprobe", 0)

    def set_nprobe(self, n):
        if n is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = n

    def reset_nprobe(self):
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self._default_nprobe

    def _positions_for(self, article_ids, filter_cache_key):
        """FAISS row positions whose vector belongs to one of article_ids.

        The isin scan is O(ntotal) — cached under filter_cache_key when the
        caller declares the filter stable (e.g. one entry per department).
        """
        if filter_cache_key and filter_cache_key in self._filter_positions:
            return self._filter_positions[filter_cache_key]
        np = self._np
        positions = np.where(np.isin(self._meta["article_ids"], article_ids))[0].astype(
            "int64"
        )
        if filter_cache_key:
            self._filter_positions[filter_cache_key] = positions
        return positions

    def raw_search(
        self,
        query_vectors,
        k,
        min_tile_height=None,
        article_ids=None,
        filter_cache_key=None,
    ):
        # min_tile_height is post-filtered by the API from hit metadata; only
        # the article_ids pre-filter runs inside FAISS (via an IDSelector, so
        # k results are guaranteed when the filtered set has enough vectors).
        if k <= 0:
            return [[] for _ in query_vectors]
        if article_ids is not None:
            faiss = self._faiss
            sel = faiss.IDSelectorBatch(
                self._positions_for(article_ids, filter_cache_key)
            )
            if isinstance(self.index, faiss.IndexIVF):
                params = faiss.SearchParametersIVF(sel=sel, nprobe=self.index.nprobe)
            else:
                params = faiss.SearchParameters(sel=sel)
            distances, indices = self.index.search(query_vectors, k, params=params)
        else:
            distances, indices = self.index.search(query_vectors, k)
        article_ids = self._meta["article_ids"]
        tile_indices = self._meta["tile_indices"]
        chunk_indices = self._meta["chunk_indices"]
        y_offsets = self._meta["y_offsets"]
        tile_heights = self._meta["tile_heights"]
        out = []
        for query_index in range(indices.shape[0]):
            hits = []
            for result_index in range(indices.shape[1]):
                vector_id = int(indices[query_index, result_index])
                if vector_id == -1:
                    continue
                hits.append(
                    {
                        "score": float(distances[query_index, result_index]),
                        "vector_id": vector_id,
                        "article_id": int(article_ids[vector_id]),
                        "tile_index": int(tile_indices[vector_id]),
                        "chunk_index": int(chunk_indices[vector_id]),
                        "y_offset": int(y_offsets[vector_id]),
                        "tile_height": int(tile_heights[vector_id]),
                    }
                )
            out.append(hits)
        return out

    def reconstruct(self, vids):
        # IVF indexes need a direct map before reconstruct(); flat indexes
        # support it natively and have no make_direct_map.
        if not self._direct_map_built and hasattr(self.index, "make_direct_map"):
            self.index.make_direct_map()
            self._direct_map_built = True
        return [self.index.reconstruct(int(v)).tolist() for v in vids]


class QdrantBackend:
    name = "qdrant"
    nlist = 0
    nprobe = 0

    def __init__(
        self, summary, url=None, collection=None, api_key=None, client_config=None
    ):
        from qdrant_client import QdrantClient

        client_options = dict(client_config or {})
        if url:
            client_options["url"] = url
        if api_key:
            client_options["api_key"] = api_key
        if not any(
            key in client_options for key in ("url", "host", "location", "path")
        ):
            raise ValueError(
                "Qdrant requires --qdrant-url or an endpoint in --qdrant-client-config"
            )
        self.collection = collection or summary.get("collection", "pixelrag")
        self.client = QdrantClient(**client_options)
        config = self.client.get_collection(self.collection).config.params.vectors
        self.dimension = config.size

    @property
    def ntotal(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def set_nprobe(self, n):
        pass

    def reset_nprobe(self):
        pass

    @staticmethod
    def _hit(point):
        payload = point.payload
        return {
            "score": float(point.score),
            "vector_id": point.id,
            **{field: int(payload[field]) for field in _META_FIELDS},
        }

    def raw_search(
        self,
        query_vectors,
        k,
        min_tile_height=None,
        article_ids=None,
        filter_cache_key=None,
    ):
        from qdrant_client import models

        if k <= 0:
            return [[] for _ in query_vectors]

        must = []
        if min_tile_height:
            must.append(
                models.FieldCondition(
                    key="tile_height",
                    range=models.Range(gte=min_tile_height),
                )
            )
        if article_ids is not None:
            # Payload pre-filter on article_id. Fine at local-docs scale
            # (departments of 10^2-10^4 articles); a very large filter set
            # would warrant a dedicated payload field instead.
            must.append(
                models.FieldCondition(
                    key="article_id",
                    match=models.MatchAny(any=[int(a) for a in article_ids]),
                )
            )
        query_filter = models.Filter(must=must) if must else None
        requests = [
            models.QueryRequest(
                query=v.tolist(),
                limit=k,
                with_payload=True,
                filter=query_filter,
            )
            for v in query_vectors
        ]
        responses = self.client.query_batch_points(self.collection, requests=requests)
        return [
            [self._hit(point) for point in response.points] for response in responses
        ]

    def reconstruct(self, vids):
        points = self.client.retrieve(self.collection, ids=vids, with_vectors=True)
        vectors = {point.id: point.vector for point in points}
        return [vectors.get(vid) for vid in vids]


def make_backend(args, summary: dict) -> VectorBackend:
    backend = getattr(args, "backend", None) or summary.get("backend", "faiss")

    if backend == "ivf":
        backend = "faiss"
    if backend == "faiss":
        return FaissBackend(args.index_dir, summary)
    if backend == "qdrant":
        return QdrantBackend(
            summary,
            url=getattr(args, "qdrant_url", None),
            collection=getattr(args, "collection", None),
            api_key=getattr(args, "qdrant_api_key", None),
            client_config=getattr(args, "qdrant_client_config", None),
        )
    raise ValueError(f"unknown backend: {backend!r} (expected 'faiss' or 'qdrant')")
