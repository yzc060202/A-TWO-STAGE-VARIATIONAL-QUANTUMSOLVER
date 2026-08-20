"""Block11 Stage-I family contract backed by the recovered real circuit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from refined_twostage.circuits.stage1_vcbe import VCBEBlock11RealAnsatz


@dataclass(frozen=True)
class Block11Resources:
    parameters: int
    ry_gates: int
    cnot_gates: int
    cz_gates: int
    total_two_qubit_gates: int
    total_gates: int
    native_depth: int
    logical_qubits: int = 6
    ancilla_qubits: int = 1

    def to_dict(self):
        return asdict(self)


def block11_parameter_count(repetitions: int, total_qubits: int = 7) -> int:
    return 2 * (total_qubits * (total_qubits - 1) // 2) * int(repetitions) + total_qubits


def block11_resource_counts(repetitions: int, total_qubits: int = 7) -> Block11Resources:
    m = int(repetitions)
    edges = total_qubits * (total_qubits - 1) // 2
    params = 2 * edges * m + total_qubits
    cnot = edges * m
    return Block11Resources(params, params, cnot, 0, cnot, params + cnot, 14 * m + 1 if total_qubits == 7 else 2 * total_qubits * m + 1)


class Block11ProjectedAnsatz(VCBEBlock11RealAnsatz):
    """Recovered real Ry/CNOT Block11 projected-block ansatz.

    The optional ``dimension`` argument is retained only for backward API
    compatibility. Formal Stage-I Block11 is the 6-data-qubit, 64-dimensional
    recovered circuit.
    """

    family_name = "vcbe_block11_real"

    def __init__(self, repetitions: int, dimension: int = 64) -> None:
        if int(dimension) != 64:
            raise ValueError("recovered Block11 production circuit requires dimension=64")
        super().__init__(data_qubits=6, repetitions=int(repetitions))

    def resource_counts(self):
        return block11_resource_counts(self.repetitions)


def repetitions_for_capacity(target_parameters: int) -> int:
    target = int(target_parameters)
    m = int(round((target - 7) / 42.0))
    if block11_parameter_count(m) != target:
        raise ValueError("target {} is not legal Block11 P=42*M+7".format(target))
    return m
