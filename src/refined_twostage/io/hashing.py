"""Array hashing utilities compatible with the historical twostage repo."""

from __future__ import annotations

import hashlib
import numpy as np


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def short_hash_array(array: np.ndarray, n: int = 16) -> str:
    return sha256_array(np.asarray(array, dtype=np.float64).reshape(-1))[: int(n)]
