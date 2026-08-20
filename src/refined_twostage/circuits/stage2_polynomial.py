"""Polynomial-size real-amplitude Stage-II ansatz families."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from refined_twostage.circuits.conventions import GrayTraversalConvention, native_circuit_hash
from refined_twostage.circuits.native_gates import append_cx, append_cz, apply_gate_state, native_adjoint_vjp, precompute_indices
from refined_twostage.types import NativeGate


@dataclass(frozen=True)
class PolynomialResources:
    """Resource counts for the polynomial Stage-II circuits."""

    parameters: int
    ry_count: int
    cz_count: int
    cnot_count: int
    two_qubit_gate_count: int
    total_native_gate_count: int
    greedy_native_critical_path_depth: int

    @property
    def native_depth(self) -> int:
        return int(self.greedy_native_critical_path_depth)

    @property
    def ry_gates(self) -> int:
        return int(self.ry_count)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def greedy_native_critical_path_depth(gates) -> int:
    """Compute a greedy native critical-path depth."""
    avail: dict[int, int] = {}
    depth = 0
    for gate in gates:
        qubits = [int(gate.target)] + [int(q) for q, _value in gate.controls]
        layer = max([avail.get(q, 0) for q in qubits] or [0]) + 1
        for q in qubits:
            avail[q] = layer
        depth = max(depth, layer)
    return int(depth)


class PolynomialStage2Ansatz:
    """Common exact statevector interface for the five polynomial families."""

    def __init__(self, family_name: str, data_qubits: int, repetitions: int) -> None:
        self.family_name = str(family_name)
        self.data_qubits = int(data_qubits)
        self.repetitions = int(repetitions)
        self.dimension = 1 << self.data_qubits
        self.convention = GrayTraversalConvention.paper_default(self.data_qubits)
        gates, params = self._build_gates()
        self._num_parameters = int(params)
        self._gates = precompute_indices(gates, self.data_qubits)

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    def native_gate_list(self) -> tuple[NativeGate, ...]:
        return self._gates

    def circuit_hash(self) -> str:
        return native_circuit_hash(self._gates, self.convention)

    def resource_counts(self) -> PolynomialResources:
        ry = sum(1 for gate in self._gates if gate.kind == "ry")
        cz = sum(1 for gate in self._gates if gate.kind == "cz")
        cnot = sum(1 for gate in self._gates if gate.kind == "x")
        return PolynomialResources(
            parameters=self.num_parameters,
            ry_count=int(ry),
            cz_count=int(cz),
            cnot_count=int(cnot),
            two_qubit_gate_count=int(cz + cnot),
            total_native_gate_count=len(self._gates),
            greedy_native_critical_path_depth=greedy_native_critical_path_depth(self._gates),
        )

    def state(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(-1)
        if omega.size != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} parameters, got {omega.size}")
        state = np.zeros(self.dimension, dtype=float)
        state[0] = 1.0
        for gate in self._gates:
            apply_gate_state(state, gate, omega)
        return state

    def value_and_vjp(self, omega: np.ndarray, output_cotangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=float).reshape(-1)
        if omega.size != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} parameters, got {omega.size}")
        state0 = np.zeros(self.dimension, dtype=float)
        state0[0] = 1.0
        state, grad, _input_adj = native_adjoint_vjp(
            state0,
            self._gates,
            omega,
            np.asarray(output_cotangent, dtype=float).reshape(-1),
            num_parameters=self.num_parameters,
        )
        return state, grad

    def state_and_jacobian(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self.state(omega)
        jac = np.zeros((self.dimension, self.num_parameters), dtype=float)
        for row in range(self.dimension):
            cotangent = np.zeros(self.dimension, dtype=float)
            cotangent[row] = 1.0
            _state, grad = self.value_and_vjp(omega, cotangent)
            jac[row, :] = grad
        return state, jac

    def _append_trainable_ry(self, gates: list[NativeGate], qubit: int, cursor: int, label: str) -> int:
        gates.append(NativeGate("ry", int(qubit), (), int(cursor), 1.0, 0.0, label))
        return int(cursor) + 1

    def _append_ry_layer(self, gates: list[NativeGate], cursor: int, label: str) -> int:
        for qubit in range(self.data_qubits):
            cursor = self._append_trainable_ry(gates, qubit, cursor, f"{label}:ry_{qubit}")
        return int(cursor)

    def _build_gates(self) -> tuple[list[NativeGate], int]:
        builders = {
            "vqls_ry_cz": self._build_vqls_ry_cz,
            "alternating_ry_cnot": self._build_alternating_ry_cnot,
            "ring_ry_cnot": self._build_ring_ry_cnot,
            "sequential_sg_mps": self._build_sequential_sg_mps,
            "dense_pairwise_ry_cnot": self._build_dense_pairwise_ry_cnot,
        }
        if self.family_name not in builders:
            raise ValueError(f"unsupported polynomial Stage-II family: {self.family_name}")
        return builders[self.family_name]()

    def _build_vqls_ry_cz(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = self._append_ry_layer(gates, 0, "initial")
        for ell in range(self.repetitions):
            pairs = [(0, 1), (2, 3), (4, 5)] if ell % 2 == 0 else [(1, 2), (3, 4)]
            for q0, q1 in pairs:
                append_cz(gates, q0, q1, f"L{ell}:cz_{q0}_{q1}")
            cursor = self._append_ry_layer(gates, cursor, f"L{ell}:post")
        return gates, cursor

    def _build_alternating_ry_cnot(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = self._append_ry_layer(gates, 0, "initial")
        for ell in range(self.repetitions):
            pairs = [(0, 1), (2, 3), (4, 5)] if ell % 2 == 0 else [(1, 2), (3, 4)]
            reverse = ell % 4 in {2, 3}
            for q0, q1 in pairs:
                control, target = (q1, q0) if reverse else (q0, q1)
                append_cx(gates, control, target, f"L{ell}:cx_{control}_{target}")
            cursor = self._append_ry_layer(gates, cursor, f"L{ell}:post")
        return gates, cursor

    def _build_ring_ry_cnot(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = self._append_ry_layer(gates, 0, "initial")
        forward = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        reverse = [(0, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]
        for ell in range(self.repetitions):
            for control, target in (forward if ell % 2 == 0 else reverse):
                append_cx(gates, control, target, f"L{ell}:ring_cx_{control}_{target}")
            cursor = self._append_ry_layer(gates, cursor, f"L{ell}:post")
        return gates, cursor

    def _build_sequential_sg_mps(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = 0
        for ell in range(self.repetitions):
            pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] if ell % 2 == 0 else [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]
            for i, j in pairs:
                cursor = self._append_trainable_ry(gates, i, cursor, f"L{ell}:pair_{i}_{j}:pre_ry_{i}")
                cursor = self._append_trainable_ry(gates, j, cursor, f"L{ell}:pair_{i}_{j}:pre_ry_{j}")
                append_cx(gates, i, j, f"L{ell}:pair_{i}_{j}:cx_{i}_{j}")
                cursor = self._append_trainable_ry(gates, i, cursor, f"L{ell}:pair_{i}_{j}:post_ry_{i}")
                cursor = self._append_trainable_ry(gates, j, cursor, f"L{ell}:pair_{i}_{j}:post_ry_{j}")
        return gates, cursor

    def _build_dense_pairwise_ry_cnot(self) -> tuple[list[NativeGate], int]:
        gates: list[NativeGate] = []
        cursor = 0
        pairs0 = [(i, j) for i in range(self.data_qubits) for j in range(i + 1, self.data_qubits)]
        for ell in range(self.repetitions):
            pairs = pairs0 if ell % 2 == 0 else [(j, i) for i, j in reversed(pairs0)]
            for i, j in pairs:
                cursor = self._append_trainable_ry(gates, i, cursor, f"L{ell}:pair_{i}_{j}:pre_ry_{i}")
                cursor = self._append_trainable_ry(gates, j, cursor, f"L{ell}:pair_{i}_{j}:pre_ry_{j}")
                append_cx(gates, i, j, f"L{ell}:pair_{i}_{j}:cx_{i}_{j}")
                cursor = self._append_trainable_ry(gates, i, cursor, f"L{ell}:pair_{i}_{j}:post_ry_{i}")
                cursor = self._append_trainable_ry(gates, j, cursor, f"L{ell}:pair_{i}_{j}:post_ry_{j}")
        return gates, cursor
