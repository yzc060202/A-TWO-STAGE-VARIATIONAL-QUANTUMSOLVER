"""Row/column scaling conventions."""

from __future__ import annotations

import numpy as np


def column_equilibrate(A_raw, b_raw, epsilon: float = 1e-300):
    A_raw = np.asarray(A_raw, dtype=float)
    b_raw = np.asarray(b_raw, dtype=float)
    Dr = np.ones(A_raw.shape[0], dtype=float)
    Dc = 1.0 / np.maximum(np.linalg.norm(A_raw, axis=0), float(epsilon))
    A_p = (Dr[:, None] * A_raw) * Dc[None, :]
    b_p = Dr * b_raw
    return Dr, Dc, A_p, b_p
