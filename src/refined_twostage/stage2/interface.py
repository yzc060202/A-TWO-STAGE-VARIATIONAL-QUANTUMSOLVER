"""Stage-II factory interface."""

from __future__ import annotations

from refined_twostage.stage2.alternating_ry_cnot import AlternatingRyCnotAnsatz, layers_for_target_parameters
from refined_twostage.stage2.polynomial import PolynomialStage2Ansatz, nearest_legal_capacity


def make_stage2_ansatz(family: str, layers=None, target_parameters=None):
    if family != "alternating_ry_cnot":
        if layers is None:
            if target_parameters is None:
                raise ValueError("provide layers or target_parameters")
            layers = nearest_legal_capacity(family, int(target_parameters))["layers_or_repetitions"]
        return PolynomialStage2Ansatz(family, int(layers))
    if layers is None:
        if target_parameters is None:
            raise ValueError("provide layers or target_parameters")
        layers = layers_for_target_parameters(int(target_parameters))
    return AlternatingRyCnotAnsatz(int(layers))
