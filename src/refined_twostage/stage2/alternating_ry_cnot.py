"""Exact recovered alternating RY/CNOT Stage-II ansatz adapter."""

from __future__ import annotations

from refined_twostage.stage2.polynomial import PolynomialStage2Ansatz


class AlternatingRyCnotAnsatz(PolynomialStage2Ansatz):
    family_name = "alternating_ry_cnot"

    def __init__(self, layers: int, data_qubits: int = 6) -> None:
        super().__init__("alternating_ry_cnot", int(layers), data_qubits=int(data_qubits))


def layers_for_target_parameters(target_parameters: int) -> int:
    target = int(target_parameters)
    if target % 6 != 0:
        raise ValueError("alternating_ry_cnot target must be divisible by 6")
    layers = target // 6 - 1
    if layers < 0:
        raise ValueError("target too small")
    return int(layers)

