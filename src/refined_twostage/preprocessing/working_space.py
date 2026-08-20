"""Right working-space reduction and SVD sign handling."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorkingSpace:
    V64: np.ndarray
    A64: np.ndarray
    singular_values: np.ndarray
    sign_flips: np.ndarray
    canonicalized: bool
    rule: str


def canonicalize_right_singular_vectors(V: np.ndarray):
    V = np.asarray(V, dtype=float).copy()
    flips = np.ones(V.shape[1], dtype=float)
    for j in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0.0:
            V[:, j] *= -1.0
            flips[j] = -1.0
    return V, flips


def right_svd_working_space(A_p, retained_rank: int = 64, canonicalize: bool = True) -> WorkingSpace:
    A_p = np.asarray(A_p, dtype=float)
    _u, s, vt = np.linalg.svd(A_p, full_matrices=False)
    V = vt.T[:, : int(retained_rank)]
    flips = np.ones(V.shape[1], dtype=float)
    if canonicalize:
        V, flips = canonicalize_right_singular_vectors(V)
    return WorkingSpace(V, A_p @ V, np.asarray(s, dtype=float), flips, bool(canonicalize), "right_svd_full_row_space; A64=A_p@V64; all 64 compact directions retained")


def direct_square_working_space(A_p) -> WorkingSpace:
    A_p = np.asarray(A_p, dtype=float)
    n = A_p.shape[1]
    V = np.eye(n, dtype=float)
    return WorkingSpace(V, A_p.copy(), np.linalg.svd(A_p, compute_uv=False), np.ones(n, dtype=float), False, "direct_square; no right-SVD working-space reduction")
