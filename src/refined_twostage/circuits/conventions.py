"""Explicit circuit convention metadata and hashes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from refined_twostage.io.hashing import sha256_array
from refined_twostage.types import NativeGate


@dataclass(frozen=True)
class GrayTraversalConvention:
    """Paper-consistent little-endian Gray traversal convention."""

    qubit_indexing: str
    basis_endianness: str
    even_target_order: tuple[int, ...]
    odd_target_order: tuple[int, ...]
    reverse_internal_gray_path_on_odd_layers: bool
    simplify_adjacent_cnot_pairs: bool
    resource_count_before_simplification: bool

    @staticmethod
    def paper_default(data_qubits: int) -> "GrayTraversalConvention":
        """Return the current paper-consistent convention."""
        even = tuple(range(int(data_qubits)))
        return GrayTraversalConvention(
            qubit_indexing="data_qubit_0_is_least_significant",
            basis_endianness="little_endian_integer_basis",
            even_target_order=even,
            odd_target_order=tuple(reversed(even)),
            reverse_internal_gray_path_on_odd_layers=True,
            simplify_adjacent_cnot_pairs=False,
            resource_count_before_simplification=True,
        )

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary."""
        return asdict(self)


def native_gate_payload(gates: Iterable[NativeGate], convention: GrayTraversalConvention | None = None) -> list:
    """Return a stable JSON payload for native gates and optional convention."""
    payload = []
    if convention is not None:
        payload.append(("gray_traversal_convention", convention.to_dict()))
    for gate in gates:
        payload.append((gate.kind, gate.target, gate.controls, gate.theta_index, gate.theta_scale, gate.fixed_angle, gate.label))
    return payload


def native_circuit_hash(gates: Iterable[NativeGate], convention: GrayTraversalConvention | None = None) -> str:
    """Return a SHA-256 hash for native gate list plus convention."""
    text = json.dumps(native_gate_payload(gates, convention), sort_keys=True).encode("utf-8")
    return sha256_array(np.frombuffer(text, dtype=np.uint8))
