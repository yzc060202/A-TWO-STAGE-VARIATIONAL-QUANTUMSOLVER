"""Stage-I metric evaluation."""

from __future__ import annotations

import numpy as np


def stage1_metrics(K, As):
    K = np.asarray(K, dtype=float)
    As = np.asarray(As, dtype=float)
    diff = K - As
    abs_f = float(np.linalg.norm(diff))
    abs_2 = float(np.linalg.norm(diff, 2))
    norm_f = max(float(np.linalg.norm(As)), 1e-300)
    norm_2 = max(float(np.linalg.norm(As, 2)), 1e-300)
    return {"absolute_frobenius_error": abs_f, "absolute_spectral_error": abs_2, "relative_frobenius_loss": float((abs_f / norm_f) ** 2), "e_F": float(abs_f / norm_f), "e_2": float(abs_2 / norm_2)}
