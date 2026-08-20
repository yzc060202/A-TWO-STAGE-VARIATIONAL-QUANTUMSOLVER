"""Shared dataclasses and validation-light types for the two-stage framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def require_real_array(value: np.ndarray, name: str, *, tol: float = 1e-12) -> np.ndarray:
    """Return a float array, rejecting non-negligible imaginary components."""
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        imag_norm = float(np.linalg.norm(np.imag(arr)))
        real_norm = max(float(np.linalg.norm(np.real(arr))), 1.0)
        if imag_norm > tol * real_norm:
            raise ValueError(f"{name} must be real-amplitude for this implementation; imaginary relative norm={imag_norm / real_norm}")
        arr = np.real(arr)
    return np.asarray(arr, dtype=float)


def is_power_of_two(value: int) -> bool:
    """Return whether `value` is a positive power of two."""
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


@dataclass(frozen=True)
class WorkingSpaceProblem:
    """Square real retained-space operator problem consumed by the solver.

    PDE and RFM code construct this object, but the Stage-I and Stage-II solver
    core depends only on the arrays and metadata stored here. For Appendix B.3,
    this is the compact-SVD retained-space operator: the raw 64 x 200 RFM
    system is column-scaled, mapped to right-singular coordinates, and reduced
    by null-space removal to the square SVD-coordinate working operator A64.
    """

    A: np.ndarray
    b: np.ndarray
    scale: float
    physical_dimension: int
    active_dimension: int
    padded_dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        A = require_real_array(self.A, "A")
        b = require_real_array(self.b, "b")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")
        if b.shape != (A.shape[0],):
            raise ValueError(f"b must have shape {(A.shape[0],)}, got {b.shape}")
        if int(self.active_dimension) != A.shape[1]:
            raise ValueError("active_dimension must equal A.shape[1]")
        if int(self.padded_dimension) < int(self.active_dimension):
            raise ValueError("padded_dimension must be >= active_dimension")
        if not is_power_of_two(int(self.padded_dimension)):
            raise ValueError("padded_dimension must be a power of two")
        if float(self.scale) <= 0:
            raise ValueError("scale must be positive")
        norm_scaled = float(np.linalg.norm(A / float(self.scale), 2))
        if norm_scaled > 1.0 + 5e-12:
            raise ValueError(f"||A/scale||_2 must be <= 1, got {norm_scaled}")


@dataclass(frozen=True)
class NativeGate:
    """Primitive native real gate in little-endian qubit order."""

    kind: str
    target: int
    controls: tuple[tuple[int, int], ...] = ()
    theta_index: int = -1
    theta_scale: float = 1.0
    fixed_angle: float = 0.0
    label: str = ""

    @property
    def trainable(self) -> bool:
        """Whether this gate consumes an optimizer parameter."""
        return self.theta_index >= 0


@dataclass(frozen=True)
class CircuitResources:
    """Primitive resource counts under the paper's RY-CNOT convention."""

    parameters: int
    ry_gates: int
    cnot_gates: int
    fixed_ry_gates: int = 0
    total_gates: int = 0

    @property
    def cz_gates(self) -> int:
        return 0

    @property
    def total_two_qubit_gates(self) -> int:
        return int(self.cnot_gates)

    @property
    def two_qubit_gate_count(self) -> int:
        return int(self.total_two_qubit_gates)

    @property
    def native_depth(self) -> int:
        return max(1, int(self.total_gates))

    def to_dict(self) -> dict[str, int]:
        return {
            "parameters": int(self.parameters),
            "ry_gates": int(self.ry_gates),
            "cnot_gates": int(self.cnot_gates),
            "cz_gates": int(self.cz_gates),
            "fixed_ry_gates": int(self.fixed_ry_gates),
            "total_two_qubit_gates": int(self.total_two_qubit_gates),
            "total_gates": int(self.total_gates),
            "native_depth": int(self.native_depth),
        }


@dataclass(frozen=True)
class ExactOverlapResult:
    """Exact probability result for the RHS overlap objective."""

    p_good: float
    p_match: float
    loss: float


@dataclass(frozen=True)
class Stage2Result:
    """Optimization result plus exact offline diagnostics."""

    mode: str
    theta0_hash: str
    theta_hash: str
    optimizer: str
    optimizer_evals: int
    optimizer_success: bool
    stopping_reason: str
    final_loss: float
    diagnostics: dict[str, Any]
    history: tuple[dict[str, Any], ...]
