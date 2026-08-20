"""Stage-II evaluation helpers."""

from __future__ import annotations

import numpy as np
from refined_twostage.stage2.objectives import scalar_eliminated_components


def stage2_solution_diagnostics(A, b, ansatz, theta, raw_coefficients, pde, feature_bank, dense_x):
    from refined_twostage.metrics.pde_metrics import evaluate_physical

    y = ansatz.state(theta)
    comp = scalar_eliminated_components(A, b, y)
    coeff = raw_coefficients(comp["z"])
    phys = evaluate_physical(pde, feature_bank, coeff, dense_x)
    return {"alpha_star": float(comp["alpha"]), "direct_r2": float(comp["direct_r2"]), "direct_r": float(comp["direct_r"]), "working_residual": float(np.linalg.norm(A @ comp["z"] - b) / max(float(np.linalg.norm(b)), 1e-300)), "PDE_L2": phys["PDE_relative_L2"], "PDE_Linf": phys["PDE_relative_Linf"], "dense_PDE_residual": phys["dense_PDE_residual"], "boundary_error": phys["boundary_error"]}
