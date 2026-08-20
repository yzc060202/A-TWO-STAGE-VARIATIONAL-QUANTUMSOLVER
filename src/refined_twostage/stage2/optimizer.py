"""L-BFGS-B r^2 -> true-r continuation protocol."""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from refined_twostage.stage2.objectives import objective_value_and_grad_theta
from refined_twostage.stage2.vector_exact import VectorExactLossEvaluator

DEFAULT_R2_OPTIONS = {"maxiter": 30000, "maxfun": 60000, "maxcor": 100, "maxls": 150, "ftol": 1e-16, "gtol": 1e-12, "disp": False}
DEFAULT_R_OPTIONS = {"maxiter": 30000, "maxfun": 60000, "maxcor": 100, "maxls": 150, "ftol": 1e-16, "gtol": 1e-12, "disp": False}


def minimize_objective(A, b, ansatz, theta0, objective, options=None):
    history = []
    t0 = time.time()

    def fun(theta):
        value, grad, comp = objective_value_and_grad_theta(A, b, ansatz, theta, objective=objective)
        history.append({"n": len(history) + 1, "objective": objective, "value": float(value), "direct_r": float(comp["direct_r"]), "grad_norm": float(np.linalg.norm(grad)), "elapsed_seconds": float(time.time() - t0)})
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(fun, np.asarray(theta0, dtype=float), method="L-BFGS-B", jac=True, options=options or {})
    value, grad, comp = objective_value_and_grad_theta(A, b, ansatz, result.x, objective=objective)
    return {"theta": np.asarray(result.x, dtype=float), "value": float(value), "grad_norm": float(np.linalg.norm(grad)), "direct_r": float(comp["direct_r"]), "direct_r2": float(comp["direct_r2"]), "history": history, "nit": int(result.nit), "nfev": int(result.nfev), "success": bool(result.success), "message": str(result.message), "runtime_seconds": float(time.time() - t0)}


def lbfgsb_r2_then_r(A, b, ansatz, seed=13379, init_scale=0.70, r2_options=None, r_options=None):
    theta0 = np.random.default_rng(int(seed)).normal(scale=float(init_scale), size=int(ansatz.num_parameters))
    phase1 = minimize_objective(A, b, ansatz, theta0, "direct_r2", DEFAULT_R2_OPTIONS if r2_options is None else r2_options)
    phase2 = minimize_objective(A, b, ansatz, phase1["theta"], "direct_r", DEFAULT_R_OPTIONS if r_options is None else r_options)
    return {"theta0": theta0, "phase1": phase1, "phase2": phase2}


def lbfgsb_fresh_direct_r(A, b, ansatz, seed=13379, init_scale=0.70, options=None):
    """Formal migrated Stage-II protocol: fresh direct-r from theta0."""
    theta0 = np.random.default_rng(int(seed)).normal(scale=float(init_scale), size=int(ansatz.num_parameters))
    evaluator = VectorExactLossEvaluator(A, b, ansatz, objective="direct_r")
    history = []
    t0 = time.time()

    initial_value, _initial_grad = evaluator.value_and_grad(theta0)

    def fun(theta):
        value, grad = evaluator.value_and_grad(theta)
        history.append({
            "n": len(history) + 1,
            "objective": "direct_r",
            "value": float(value),
            "direct_r": float(value),
            "grad_norm": float(np.linalg.norm(grad)),
            "elapsed_seconds": float(time.time() - t0),
        })
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(fun, theta0, method="L-BFGS-B", jac=True, options=DEFAULT_R_OPTIONS if options is None else options)
    final_value, final_grad = evaluator.value_and_grad(result.x)
    return {
        "theta0": theta0,
        "theta": np.asarray(result.x, dtype=float),
        "initial_r": float(initial_value),
        "final_r": float(final_value),
        "value": float(final_value),
        "grad_norm": float(np.linalg.norm(final_grad)),
        "history": history,
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "runtime_seconds": float(time.time() - t0),
        "protocol": "fresh_direct_r",
    }
