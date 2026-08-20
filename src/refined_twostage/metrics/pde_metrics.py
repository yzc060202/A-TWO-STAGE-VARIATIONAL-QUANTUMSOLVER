"""Physical PDE validation metrics."""

from __future__ import annotations

import numpy as np
from refined_twostage.problems.common import u_star


def evaluate_physical(pde, feature_bank, coeff, dense_x):
    coeff = np.asarray(coeff, dtype=float)
    dense_x = np.asarray(dense_x, dtype=float)
    pred = feature_bank.values(dense_x) @ coeff
    truth = u_star(dense_x)
    err = pred - truth
    interior = dense_x[1:-1]
    pde_pred = pde.apply_coefficients(feature_bank, coeff, interior)
    pde_truth = pde.forcing(interior)
    boundary_err = feature_bank.values(np.array([-1.0, 1.0])) @ coeff - u_star(np.array([-1.0, 1.0]))
    return {"PDE_relative_L2": float(np.linalg.norm(err) / max(float(np.linalg.norm(truth)), 1e-300)), "PDE_relative_Linf": float(np.max(np.abs(err)) / max(float(np.max(np.abs(truth))), 1e-300)), "max_abs_function_error": float(np.max(np.abs(err))), "dense_PDE_residual": float(np.linalg.norm(pde_pred - pde_truth) / max(float(np.linalg.norm(pde_truth)), 1e-300)), "boundary_error_left": float(abs(boundary_err[0])), "boundary_error_right": float(abs(boundary_err[1])), "boundary_error": float(np.max(np.abs(boundary_err))), "prediction_norm": float(np.linalg.norm(pred))}
