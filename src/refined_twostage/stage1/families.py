"""Stage-I projected-block family registry.

Formal production families instantiate the recovered 2026-08-19 real
Ry/CNOT projected-block circuits. Dense projected-block surrogate experiments
are archived outside the public production API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from refined_twostage.circuits.stage1_gray_multiplexed import GrayMultiplexedBlockEncodingAnsatz
from refined_twostage.circuits.stage1_vcbe import VCBEBlock15RealAnsatz


@dataclass(frozen=True)
class Stage1Resources:
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


class GrayMultiplexedProjectedAnsatz(GrayMultiplexedBlockEncodingAnsatz):
    family_name = "gray_multiplexed_ry_cnot"

    def __init__(self, repetitions: int, dimension: int = 64) -> None:
        if int(dimension) != 64:
            raise ValueError("recovered Gray production circuit requires dimension=64")
        super().__init__(data_qubits=6, layers=int(repetitions))

    @staticmethod
    def parameter_count(layers: int) -> int:
        return 256 * int(layers)


class Block15ProjectedAnsatz(VCBEBlock15RealAnsatz):
    family_name = "vcbe_block15_real"

    def __init__(self, repetitions: int, dimension: int = 64) -> None:
        if int(dimension) != 64:
            raise ValueError("recovered Block15 production circuit requires dimension=64")
        super().__init__(data_qubits=6, repetitions=int(repetitions))

    @staticmethod
    def parameter_count(repetitions: int) -> int:
        return 14 * int(repetitions) + 7

    def resource_counts(self) -> Stage1Resources:
        m = int(self.repetitions)
        cnot = 7 * m
        params = self.num_parameters
        return Stage1Resources(params, params, cnot, 0, cnot, params + cnot, 5 * m + 1)


def gray_layers_for_capacity(target_parameters: int) -> int:
    target = int(target_parameters)
    if target % 256 != 0:
        raise ValueError("target {} is not legal Gray P=256*L".format(target))
    layers = target // 256
    if layers < 16:
        raise ValueError("Gray dense projected block requires at least P=4096")
    return int(layers)


def block15_repetitions_for_capacity(target_parameters: int) -> int:
    target = int(target_parameters)
    m = int(round((target - 7) / 14.0))
    if Block15ProjectedAnsatz.parameter_count(m) != target:
        raise ValueError("target {} is not legal Block15 P=14*M+7".format(target))
    return int(m)


def nearest_stage1_capacity(family: str, target_parameters: int):
    target = int(target_parameters)
    if family == "gray_multiplexed_ry_cnot":
        layers = max(16, int(round(target / 256.0)))
        return layers, GrayMultiplexedProjectedAnsatz.parameter_count(layers)
    if family == "vcbe_block15_real":
        m = max(293, int(round((target - 7) / 14.0)))
        return m, Block15ProjectedAnsatz.parameter_count(m)
    if family == "vcbe_block11_real":
        from refined_twostage.stage1.block11 import block11_parameter_count

        m = max(98, int(round((target - 7) / 42.0)))
        return m, block11_parameter_count(m)
    raise ValueError("unknown Stage-I family {}".format(family))
