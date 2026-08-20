"""Classical solve diagnostics."""

from __future__ import annotations

import numpy as np
from refined_twostage.metrics.matrix_metrics import singular_diagnostics
from refined_twostage.metrics.pde_metrics import evaluate_physical


def solve_lstsq(A, b, rcond=None):
    z, residuals, rank, svals = np.linalg.lstsq(np.asarray(A, dtype=float), np.asarray(b, dtype=float), rcond=rcond)
    return np.asarray(z, dtype=float), {"solver": "np.linalg.lstsq", "rcond": rcond, "rank": int(rank), "returned_residuals": np.asarray(residuals, dtype=float).tolist(), "singular_values_solver": np.asarray(svals, dtype=float).tolist(), "relative_algebraic_residual": float(np.linalg.norm(np.asarray(A) @ z - np.asarray(b)) / max(float(np.linalg.norm(b)), 1e-300)), "working_coefficient_norm": float(np.linalg.norm(z))}


def classical_diagnostics(label, construction, A, b, raw_coefficients, pde, feature_bank, dense_x):
    z, solve = solve_lstsq(A, b)
    coeff = raw_coefficients(z)
    phys = evaluate_physical(pde, feature_bank, coeff, dense_x)
    mat = singular_diagnostics(A)
    row = {"PDE": pde.name, "pde_key": pde.key, "construction": construction, "label": label, "shape": "{}x{}".format(A.shape[0], A.shape[1]), "rank@1e-10": mat["ranks"]["rank@1e-10"], "rank@1e-12": mat["ranks"]["rank@1e-12"], "rank@1e-14": mat["ranks"]["rank@1e-14"], "rank@1e-16": mat["ranks"]["rank@1e-16"], "cond2": mat["cond2"], "kappa_eff@1e-10": mat["effective_condition_numbers"]["kappa_eff@1e-10"], "kappa_eff@1e-12": mat["effective_condition_numbers"]["kappa_eff@1e-12"], "kappa_eff@1e-14": mat["effective_condition_numbers"]["kappa_eff@1e-14"], "kappa_eff@1e-16": mat["effective_condition_numbers"]["kappa_eff@1e-16"], "classical_residual": solve["relative_algebraic_residual"], "PDE_L2": phys["PDE_relative_L2"], "PDE_Linf": phys["PDE_relative_Linf"], "max_abs_function_error": phys["max_abs_function_error"], "dense_PDE_residual": phys["dense_PDE_residual"], "boundary_error": phys["boundary_error"], "coefficient_norm": float(np.linalg.norm(coeff)), "prediction_norm": phys["prediction_norm"]}
    return z, coeff, row, mat
