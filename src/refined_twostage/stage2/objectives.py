"""Stage-II scalar-eliminated direct residual objectives."""

from __future__ import annotations

import math
import numpy as np


def scalar_eliminated_components(A, b, y):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    v = A @ y
    q = float(np.dot(v, v))
    c = float(np.dot(b, v))
    alpha = c / q if q > 1e-300 else 0.0
    r_vec = alpha * v - b
    b_norm2 = max(float(np.dot(b, b)), 1e-300)
    direct_r2 = float(np.dot(r_vec, r_vec) / b_norm2)
    direct_r = float(math.sqrt(max(direct_r2, 0.0)))
    return {"v": v, "q": q, "c": c, "alpha": alpha, "r_vec": r_vec, "direct_r2": direct_r2, "direct_r": direct_r, "z": alpha * y}


def objective_value_and_grad_y(A, b, y, objective="direct_r2", safe_r: float = 1e-15):
    comp = scalar_eliminated_components(A, b, y)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    b_norm2 = max(float(np.dot(b, b)), 1e-300)
    alpha = float(comp["alpha"])
    r_vec = np.asarray(comp["r_vec"], dtype=float)
    if objective == "direct_r2":
        dL_dv = 2.0 * alpha * r_vec / b_norm2
        value = comp["direct_r2"]
    elif objective == "direct_r":
        value = comp["direct_r"]
        dL_dv = np.zeros_like(r_vec) if value <= float(safe_r) else alpha * r_vec / (b_norm2 * value)
    else:
        raise ValueError("unknown objective {}".format(objective))
    return float(value), np.asarray(A.T @ dL_dv, dtype=float), comp


def objective_value_and_grad_theta(A, b, ansatz, theta, objective="direct_r2"):
    y, _ = ansatz.value_and_vjp(theta, np.zeros(ansatz.dimension, dtype=float))
    value, grad_y, comp = objective_value_and_grad_y(A, b, y, objective=objective)
    _state, grad_theta = ansatz.value_and_vjp(theta, grad_y)
    return value, grad_theta, comp
