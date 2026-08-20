"""Figure-3 Gray-multiplexed RY-CNOT Stage-I block-encoding ansatz."""

from __future__ import annotations

import math

import numpy as np

from refined_twostage.circuits.conventions import GrayTraversalConvention, native_circuit_hash
from refined_twostage.circuits.native_gates import (
    add_gray_multiplexed_ry,
    append_cx,
    apply_gate_batch,
    precompute_indices,
    resource_counts,
)
from refined_twostage.types import CircuitResources, NativeGate


class GrayMultiplexedBlockEncodingAnsatz:
    """Gray-multiplexed RY-CNOT projected-block ansatz from Figure 3."""

    family_name = "gray_multiplexed_ry_cnot"

    def __init__(
        self,
        data_qubits: int,
        layers: int,
        *,
        alternating_orientation: bool = True,
        include_closing_ring: bool = True,
        fixed_ancilla_ry_pi: bool = True,
    ) -> None:
        self.data_qubits = int(data_qubits)
        self.layers = int(layers)
        self.alternating_orientation = bool(alternating_orientation)
        self.include_closing_ring = bool(include_closing_ring)
        self.fixed_ancilla_ry_pi = bool(fixed_ancilla_ry_pi)
        self.ancilla = self.data_qubits
        self.convention = GrayTraversalConvention.paper_default(self.data_qubits)
        self.num_qubits = self.data_qubits + 1
        self.dimension = 1 << self.data_qubits
        self.full_dimension = 1 << self.num_qubits
        gates, params = self._build_gates()
        self._num_parameters = int(params)
        self._gates = precompute_indices(gates, self.num_qubits)

    @property
    def num_parameters(self) -> int:
        """Number of trainable RY parameters."""
        return self._num_parameters

    def _target_order(self, layer: int) -> list[int]:
        bits = list(range(self.data_qubits))
        if self.alternating_orientation and (layer & 1):
            bits.reverse()
        return bits

    def _ring_edges(self, layer: int) -> list[tuple[int, int]]:
        nodes = list(range(self.data_qubits)) + [self.ancilla]
        edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
        if self.alternating_orientation and (layer & 1):
            edges = [(b, a) for a, b in reversed(edges)]
        return edges

    def _build_gates(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = 0
        if self.fixed_ancilla_ry_pi:
            gates.append(NativeGate("ry", self.ancilla, (), -1, 1.0, math.pi, "fixed_input_to_bad_branch"))
        for layer in range(self.layers):
            reverse = bool(self.alternating_orientation and (layer & 1))
            for target in self._target_order(layer):
                controls = [q for q in range(self.data_qubits) if q != target]
                cursor = add_gray_multiplexed_ry(
                    gates,
                    target=target,
                    controls=controls,
                    parameter_cursor=cursor,
                    label=f"stage1_l{layer}:D{target}",
                    reverse=reverse,
                )
            cursor = add_gray_multiplexed_ry(
                gates,
                target=self.ancilla,
                controls=list(range(self.data_qubits)),
                parameter_cursor=cursor,
                label=f"stage1_l{layer}:L",
                reverse=reverse,
            )
            if self.include_closing_ring:
                for control, target in self._ring_edges(layer):
                    append_cx(gates, control, target, f"stage1_l{layer}:ring_{control}_{target}")
        return gates, cursor

    def native_gate_list(self) -> tuple[NativeGate, ...]:
        """Return the primitive native gate list."""
        return self._gates

    def resource_counts(self) -> CircuitResources:
        """Return primitive paper-convention resource counts."""
        return resource_counts(self._gates)

    def circuit_hash(self) -> str:
        """Return native gate-list hash including Gray traversal convention."""
        return native_circuit_hash(self._gates, self.convention)

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

    def loss_and_grad_full_basis(self, theta: np.ndarray, target: np.ndarray, chunk_size: int = 8, evaluator_mode: str = "chunked_reference") -> tuple[float, np.ndarray]:
        """Compute full-basis relative Frobenius loss and analytic gradient."""
        del evaluator_mode
        theta = np.asarray(theta, dtype=float)
        target = np.asarray(target, dtype=float)
        eye = np.eye(target.shape[1], dtype=float)
        denom_total = float(np.linalg.norm(target) ** 2 + 1e-300)
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
            adj[:, : self.dimension] = 2.0 * residual
            for gate in reversed(self._gates):
                if gate.kind == "ry":
                    angle = float(theta[gate.theta_index]) if gate.trainable else float(gate.fixed_angle)
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
                        grad[gate.theta_index] += float(np.sum(ga * dnew0 + gb * dnew1))
                    adj[:, idx0] = c * ga + s * gb
                    adj[:, idx1] = -s * ga + c * gb
                elif gate.kind == "x":
                    idx0 = gate.idx0
                    idx1 = gate.idx1
                    tmp = adj[:, idx0].copy()
                    adj[:, idx0] = adj[:, idx1]
                    adj[:, idx1] = tmp
        return float(loss_num / denom_total), grad / denom_total
