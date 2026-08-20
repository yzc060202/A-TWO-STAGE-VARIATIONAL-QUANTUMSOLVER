"""Recovered-style vector-exact Stage-II loss evaluator."""

from __future__ import annotations

import numpy as np


class VectorExactLossEvaluator:
    """Exact scalar-eliminated direct residual evaluator.

    ``objective="direct_r2"`` evaluates the squared relative residual.
    ``objective="direct_r"`` evaluates the residual itself and applies the
    recovered exact VJP path through the ansatz.
    """

    def __init__(self, operator, b, ansatz, *, objective: str = "direct_r2", numerical_guard: float = 1e-15) -> None:
        self.operator = np.asarray(operator, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.ansatz = ansatz
        self.objective = str(objective)
        if self.objective not in {"direct_r2", "direct_r"}:
            raise ValueError("objective must be direct_r2 or direct_r")
        self.numerical_guard = float(numerical_guard)
        self.bnorm2 = float(np.vdot(self.b, self.b).real)
        if self.bnorm2 <= 1e-300:
            raise ValueError("RHS vector must be nonzero")

    def __call__(self, theta):
        return self.value(theta)

    def value(self, theta):
        loss_r2, _grad_r2, r = self._direct_r2_and_grad(theta)
        return float(loss_r2 if self.objective == "direct_r2" else r)

    def value_and_grad(self, theta):
        loss_r2, grad_r2, r = self._direct_r2_and_grad(theta)
        if self.objective == "direct_r2":
            return float(loss_r2), np.asarray(grad_r2, dtype=float)
        if r <= self.numerical_guard:
            return float(r), np.zeros_like(grad_r2)
        return float(r), np.asarray(grad_r2, dtype=float) / (2.0 * float(r))

    def _direct_r2_and_grad(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        y = self.ansatz.state(theta)
        v = self.operator @ y
        c = float(np.vdot(self.b, v).real)
        q = float(np.vdot(v, v).real)
        if q <= 1e-300 or not np.isfinite(q):
            return 1.0, np.zeros_like(theta), 1.0
        alpha = c / q
        residual = alpha * v - self.b
        loss = float(np.vdot(residual, residual).real / self.bnorm2)
        dloss_dv = (2.0 * alpha / self.bnorm2) * residual
        dloss_dy = self.operator.T @ dloss_dv
        _state, grad = self.ansatz.value_and_vjp(theta, dloss_dy)
        return float(loss), np.asarray(grad, dtype=float), float(np.sqrt(max(loss, 0.0)))

