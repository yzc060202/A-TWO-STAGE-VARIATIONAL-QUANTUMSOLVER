"""Shared native-gate projected-block Stage-I ansatz machinery."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from refined_twostage.circuits.conventions import native_circuit_hash
from refined_twostage.circuits.native_gates import apply_gate_batch, gate_angle, precompute_indices, resource_counts
from refined_twostage.types import CircuitResources, NativeGate


class NativeProjectedBlockAnsatz:
    """Base class for real Ry/CNOT Stage-I projected-block ansatz families."""

    family_name = "native_projected_block"

    def __init__(
        self,
        data_qubits: int,
        repetitions: int,
        gates: list[NativeGate],
        *,
        fixed_ancilla_ry_pi: bool,
        spec_extra: dict[str, Any] | None = None,
    ) -> None:
        self.data_qubits = int(data_qubits)
        self.layers = int(repetitions)
        self.repetitions = int(repetitions)
        self.ancilla = self.data_qubits
        self.num_qubits = self.data_qubits + 1
        self.dimension = 1 << self.data_qubits
        self.full_dimension = 1 << self.num_qubits
        self.fixed_ancilla_ry_pi = bool(fixed_ancilla_ry_pi)
        self._spec_extra = dict(spec_extra or {})
        self._gates = precompute_indices(gates, self.num_qubits)
        self._num_parameters = resource_counts(self._gates).parameters

    @property
    def num_parameters(self) -> int:
        """Number of trainable Ry angles."""
        return self._num_parameters

    def ansatz_spec(self) -> dict[str, Any]:
        """Return a JSON-safe ansatz specification."""
        return {
            "family": self.family_name,
            "data_qubits": int(self.data_qubits),
            "layers": int(self.layers),
            "repetitions": int(self.repetitions),
            "fixed_ancilla_ry_pi": bool(self.fixed_ancilla_ry_pi),
            "native_gate_set": ["ry", "cnot"],
            **self._spec_extra,
        }

    def native_gate_list(self) -> tuple[NativeGate, ...]:
        """Return the primitive native gate list."""
        return self._gates

    def resource_counts(self) -> CircuitResources:
        """Return primitive native resource counts."""
        return resource_counts(self._gates)

    def circuit_hash(self) -> str:
        """Return a stable hash for the native gate list."""
        return native_circuit_hash(self._gates)

    def _input_batch(self, input_state: np.ndarray) -> np.ndarray:
        arr = np.asarray(input_state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"expected input dimension {self.dimension}, got {arr.shape[0]}")
        psi = np.zeros((arr.shape[1], self.full_dimension), dtype=float)
        psi[:, : self.dimension] = arr.T
        return psi

    def apply(self, theta: np.ndarray, input_state: np.ndarray) -> np.ndarray:
        """Apply the full Stage-I circuit to one state or a column batch."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} parameters, got {theta.size}")
        psi = self._input_batch(input_state)
        for gate in self._gates:
            apply_gate_batch(psi, gate, theta)
        return psi[0] if np.asarray(input_state).ndim == 1 else psi.T

    def projected_action(self, theta: np.ndarray, input_state: np.ndarray) -> np.ndarray:
        """Return the ancilla-good branch action."""
        out = self.apply(theta, input_state)
        if np.asarray(input_state).ndim == 1:
            return out[: self.dimension]
        return out[: self.dimension, :]

    def extract_projected_block(self, theta: np.ndarray) -> np.ndarray:
        """Extract the projected block by applying the circuit to basis columns."""
        return self.projected_action(theta, np.eye(self.dimension, dtype=float))

    def squared_frobenius_loss_and_grad(
        self,
        theta: np.ndarray,
        target: np.ndarray,
        chunk_size: int = 8,
        *,
        relative: bool,
        half: bool,
        evaluator_mode: str = "reversible",
    ) -> tuple[float, np.ndarray]:
        """Compute full-basis Frobenius loss and exact analytic gradient."""
        mode = str(evaluator_mode)
        if mode == "reversible":
            del chunk_size
            return self._squared_frobenius_loss_and_grad_full_batch(theta, target, relative=relative, half=half)
        if mode == "full_batch_tape":
            del chunk_size
            return self._squared_frobenius_loss_and_grad_full_batch_tape_reference(theta, target, relative=relative, half=half)
        if mode == "chunked_reference":
            return self._squared_frobenius_loss_and_grad_chunked_reference(theta, target, chunk_size, relative=relative, half=half)
        raise ValueError(f"unknown Stage-I evaluator_mode {mode!r}")

    def _squared_frobenius_loss_and_grad_full_batch(
        self,
        theta: np.ndarray,
        target: np.ndarray,
        *,
        relative: bool,
        half: bool,
    ) -> tuple[float, np.ndarray]:
        """Full-column exact evaluator using a reversible adjoint sweep."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float)
        if theta.size != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} parameters, got {theta.size}")
        if target.shape != (self.dimension, self.dimension):
            raise ValueError(f"expected target shape {(self.dimension, self.dimension)}, got {target.shape}")
        scale = float(np.linalg.norm(target) ** 2 + 1e-300) if relative else 1.0
        half_factor = 0.5 if half else 1.0

        cos_cache = np.cos(0.5 * theta)
        sin_cache = np.sin(0.5 * theta)

        psi = np.zeros((self.dimension, self.full_dimension), dtype=float)
        psi[:, : self.dimension] = np.eye(self.dimension, dtype=float)
        for gate in self._gates:
            idx0 = gate.idx0
            idx1 = gate.idx1
            if gate.kind == "ry":
                if gate.trainable and float(gate.theta_scale) == 1.0:
                    c = float(cos_cache[int(gate.theta_index)])
                    s = float(sin_cache[int(gate.theta_index)])
                else:
                    angle = gate_angle(gate, theta)
                    c = math.cos(0.5 * angle)
                    s = math.sin(0.5 * angle)
                x = psi[:, idx0].copy()
                y = psi[:, idx1].copy()
                psi[:, idx0] = c * x - s * y
                psi[:, idx1] = s * x + c * y
            elif gate.kind == "x":
                x = psi[:, idx0].copy()
                psi[:, idx0] = psi[:, idx1]
                psi[:, idx1] = x
            elif gate.kind == "cz":
                psi[:, idx1] *= -1.0
            else:
                raise ValueError(f"unsupported native gate kind {gate.kind}")

        residual = psi[:, : self.dimension] - target.T
        loss_num = float(np.linalg.norm(residual) ** 2)
        adj = np.zeros_like(psi)
        adj[:, : self.dimension] = (1.0 if half else 2.0) * residual
        grad = np.zeros_like(theta)

        for gate in reversed(self._gates):
            idx0 = gate.idx0
            idx1 = gate.idx1
            if gate.kind == "ry":
                if gate.trainable and float(gate.theta_scale) == 1.0:
                    c = float(cos_cache[int(gate.theta_index)])
                    s = float(sin_cache[int(gate.theta_index)])
                else:
                    angle = gate_angle(gate, theta)
                    c = math.cos(0.5 * angle)
                    s = math.sin(0.5 * angle)
                out0 = psi[:, idx0].copy()
                out1 = psi[:, idx1].copy()
                ga = adj[:, idx0].copy()
                gb = adj[:, idx1].copy()
                if gate.trainable:
                    grad[int(gate.theta_index)] += float(gate.theta_scale) * float(np.sum(ga * (-0.5 * out1) + gb * (0.5 * out0)))
                psi[:, idx0] = c * out0 + s * out1
                psi[:, idx1] = -s * out0 + c * out1
                adj[:, idx0] = c * ga + s * gb
                adj[:, idx1] = -s * ga + c * gb
            elif gate.kind == "x":
                tmp = psi[:, idx0].copy()
                psi[:, idx0] = psi[:, idx1]
                psi[:, idx1] = tmp
                tmp = adj[:, idx0].copy()
                adj[:, idx0] = adj[:, idx1]
                adj[:, idx1] = tmp
            elif gate.kind == "cz":
                psi[:, idx1] *= -1.0
                adj[:, idx1] *= -1.0
            else:
                raise ValueError(f"unsupported native gate kind {gate.kind}")
        return float(half_factor * loss_num / scale), grad / scale

    def _squared_frobenius_loss_and_grad_full_batch_tape_reference(
        self,
        theta: np.ndarray,
        target: np.ndarray,
        *,
        relative: bool,
        half: bool,
    ) -> tuple[float, np.ndarray]:
        """Previous optimized full-batch evaluator retaining every Ry input."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float)
        if theta.size != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} parameters, got {theta.size}")
        if target.shape != (self.dimension, self.dimension):
            raise ValueError(f"expected target shape {(self.dimension, self.dimension)}, got {target.shape}")
        scale = float(np.linalg.norm(target) ** 2 + 1e-300) if relative else 1.0
        half_factor = 0.5 if half else 1.0

        psi = np.zeros((self.dimension, self.full_dimension), dtype=float)
        psi[:, : self.dimension] = np.eye(self.dimension, dtype=float)
        ry_pre: list[tuple[np.ndarray, np.ndarray]] = []

        for gate in self._gates:
            idx0 = gate.idx0
            idx1 = gate.idx1
            if gate.kind == "ry":
                angle = gate_angle(gate, theta)
                c = math.cos(0.5 * angle)
                s = math.sin(0.5 * angle)
                x = psi[:, idx0].copy()
                y = psi[:, idx1].copy()
                if gate.trainable:
                    ry_pre.append((x, y))
                psi[:, idx0] = c * x - s * y
                psi[:, idx1] = s * x + c * y
            elif gate.kind == "x":
                x = psi[:, idx0].copy()
                psi[:, idx0] = psi[:, idx1]
                psi[:, idx1] = x
            elif gate.kind == "cz":
                psi[:, idx1] *= -1.0
            else:
                raise ValueError(f"unsupported native gate kind {gate.kind}")

        residual = psi[:, : self.dimension] - target.T
        loss_num = float(np.linalg.norm(residual) ** 2)
        adj = np.zeros_like(psi)
        adj[:, : self.dimension] = (1.0 if half else 2.0) * residual
        grad = np.zeros_like(theta)
        pre_cursor = len(ry_pre) - 1

        for gate in reversed(self._gates):
            idx0 = gate.idx0
            idx1 = gate.idx1
            if gate.kind == "ry":
                angle = gate_angle(gate, theta)
                c = math.cos(0.5 * angle)
                s = math.sin(0.5 * angle)
                ga = adj[:, idx0].copy()
                gb = adj[:, idx1].copy()
                if gate.trainable:
                    x, y = ry_pre[pre_cursor]
                    pre_cursor -= 1
                    dnew0 = -0.5 * s * x - 0.5 * c * y
                    dnew1 = 0.5 * c * x - 0.5 * s * y
                    grad[int(gate.theta_index)] += float(gate.theta_scale) * float(np.sum(ga * dnew0 + gb * dnew1))
                adj[:, idx0] = c * ga + s * gb
                adj[:, idx1] = -s * ga + c * gb
            elif gate.kind == "x":
                tmp = adj[:, idx0].copy()
                adj[:, idx0] = adj[:, idx1]
                adj[:, idx1] = tmp
            elif gate.kind == "cz":
                adj[:, idx1] *= -1.0
            else:
                raise ValueError(f"unsupported native gate kind {gate.kind}")
        if pre_cursor != -1:
            raise RuntimeError("internal Ry pre-activation cursor mismatch")
        return float(half_factor * loss_num / scale), grad / scale

    def _squared_frobenius_loss_and_grad_chunked_reference(
        self,
        theta: np.ndarray,
        target: np.ndarray,
        chunk_size: int = 8,
        *,
        relative: bool,
        half: bool,
    ) -> tuple[float, np.ndarray]:
        """Compute full-basis Frobenius loss and exact analytic gradient."""
        theta = np.asarray(theta, dtype=float)
        target = np.asarray(target, dtype=float)
        eye = np.eye(target.shape[1], dtype=float)
        scale = float(np.linalg.norm(target) ** 2 + 1e-300) if relative else 1.0
        half_factor = 0.5 if half else 1.0
        grad = np.zeros_like(theta)
        loss_num = 0.0
        for start in range(0, target.shape[1], int(chunk_size)):
            probes = eye[:, start : start + int(chunk_size)]
            tgt = (target @ probes).T
            psi = self._input_batch(probes)
            ry_pre: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for gate in self._gates:
                if gate.kind == "ry" and gate.trainable:
                    ry_pre[id(gate)] = (psi[:, gate.idx0].copy(), psi[:, gate.idx1].copy())
                apply_gate_batch(psi, gate, theta)
            out = psi[:, : self.dimension]
            residual = out - tgt
            loss_num += float(np.linalg.norm(residual) ** 2)
            adj = np.zeros_like(psi)
            adj[:, : self.dimension] = (1.0 if half else 2.0) * residual
            for gate in reversed(self._gates):
                if gate.kind == "ry":
                    angle = gate_angle(gate, theta)
                    c = math.cos(0.5 * angle)
                    s = math.sin(0.5 * angle)
                    idx0 = gate.idx0
                    idx1 = gate.idx1
                    ga = adj[:, idx0].copy()
                    gb = adj[:, idx1].copy()
                    if gate.trainable:
                        x, y = ry_pre[id(gate)]
                        dnew0 = -0.5 * s * x - 0.5 * c * y
                        dnew1 = 0.5 * c * x - 0.5 * s * y
                        grad[int(gate.theta_index)] += float(gate.theta_scale) * float(np.sum(ga * dnew0 + gb * dnew1))
                    adj[:, idx0] = c * ga + s * gb
                    adj[:, idx1] = -s * ga + c * gb
                elif gate.kind == "x":
                    idx0 = gate.idx0
                    idx1 = gate.idx1
                    tmp = adj[:, idx0].copy()
                    adj[:, idx0] = adj[:, idx1]
                    adj[:, idx1] = tmp
                elif gate.kind == "cz":
                    adj[:, gate.idx1] *= -1.0
                else:
                    raise ValueError(f"unsupported native gate kind {gate.kind}")
        return float(half_factor * loss_num / scale), grad / scale

    def loss_and_grad_full_basis(
        self,
        theta: np.ndarray,
        target: np.ndarray,
        chunk_size: int = 8,
        *,
        evaluator_mode: str = "reversible",
    ) -> tuple[float, np.ndarray]:
        """Compute full-basis relative squared Frobenius loss and gradient."""
        return self.squared_frobenius_loss_and_grad(
            theta,
            target,
            chunk_size,
            relative=True,
            half=False,
            evaluator_mode=evaluator_mode,
        )
