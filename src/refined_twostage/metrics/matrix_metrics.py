"""Singular-value and conditioning diagnostics."""

from __future__ import annotations

from typing import Iterable
import numpy as np

DEFAULT_THRESHOLDS = (1e-10, 1e-12, 1e-14, 1e-16)


def singular_diagnostics(A, thresholds: Iterable[float] = DEFAULT_THRESHOLDS):
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    out = {"shape": list(A.shape), "rank_default_numpy": int(np.linalg.matrix_rank(A)), "sigma_max": float(s[0]) if s.size else 0.0, "sigma_min": float(s[-1]) if s.size else 0.0, "cond2": float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf"), "singular_values": [float(v) for v in s], "compact_nonzero_cond2": float("inf"), "ranks": {}, "effective_condition_numbers": {}}
    if s.size:
        nz = s[s > 0.0]
        if nz.size:
            out["compact_nonzero_cond2"] = float(nz[0] / nz[-1])
        for thr in thresholds:
            rank = int(np.sum(s >= float(thr) * s[0]))
            out["ranks"]["rank@{}".format(thr)] = rank
            out["effective_condition_numbers"]["kappa_eff@{}".format(thr)] = float(s[0] / s[rank - 1]) if rank > 0 else float("inf")
    return out


def rank_at(A, rel_threshold: float = 1e-12) -> int:
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    return 0 if not s.size else int(np.sum(s >= float(rel_threshold) * s[0]))
