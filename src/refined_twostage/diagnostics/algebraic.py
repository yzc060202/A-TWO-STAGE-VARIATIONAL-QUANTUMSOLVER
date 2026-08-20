"""Algebraic offline diagnostics for Stage-II results."""

from __future__ import annotations

from typing import Any

import numpy as np


def scalar_eliminated_components(A: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compute scalar-eliminated residual quantities without cancellation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    v = A @ y
    q = float(np.vdot(v, v).real)
    b2 = float(np.vdot(b, b).real)
    bnorm = max(float(np.sqrt(max(b2, 0.0))), 1e-300)
    if q <= 1e-300 or b2 <= 1e-300 or not np.isfinite(q):
        alpha = 0.0
        z = np.zeros_like(y, dtype=float)
        residual = -b
        loss_overlap = 1.0
    else:
        c = float(np.vdot(b, v).real)
        alpha = c / q
        z = alpha * y
        residual = alpha * v - b
        loss_overlap = 1.0 - (c * c) / (b2 * q)
    loss_direct = float(np.vdot(residual, residual).real / max(b2, 1e-300))
    return {
        "alpha": float(alpha),
        "z": z,
        "v": v,
        "q": float(q),
        "c": float(np.vdot(b, v).real),
        "b2": float(b2),
        "residual": residual,
        "loss_overlap_formula": float(loss_overlap),
        "loss_direct_residual": float(max(loss_direct, 0.0)),
        "rhs_residual_direct": float(np.linalg.norm(residual) / bnorm),
        "z_norm": float(np.linalg.norm(z)),
    }


def scalar_eliminated_diagnostics(A: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compute alpha, retained solution, loss, and residual using matrix view."""
    comp = scalar_eliminated_components(A, b, y)
    return {
        "alpha": complex(comp["alpha"]),
        "z": comp["z"],
        "loss": float(comp["loss_direct_residual"]),
        "loss_overlap_formula": float(comp["loss_overlap_formula"]),
        "loss_direct_residual": float(comp["loss_direct_residual"]),
        "relative_residual": float(comp["rhs_residual_direct"]),
        "rhs_residual_direct": float(comp["rhs_residual_direct"]),
        "z_norm": float(comp["z_norm"]),
    }
