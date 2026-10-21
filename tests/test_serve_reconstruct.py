"""Input bound on the public /reconstruct endpoint (OOM/DoS hardening).

Every id returns a full `dimension`-length float list (~40 KB of JSON at dim
2048) and the first call builds a direct map over the index, so an unbounded
`vector_ids` list is the same class of trivial OOM as the /search bounds.
"""

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
from pixelrag_serve.api import _MAX_RECONSTRUCT_IDS, ReconstructRequest
from pydantic import ValidationError


def test_vector_ids_are_capped():
    ReconstructRequest(vector_ids=list(range(_MAX_RECONSTRUCT_IDS)))
    with pytest.raises(ValidationError):
        ReconstructRequest(vector_ids=list(range(_MAX_RECONSTRUCT_IDS + 1)))
