"""Stage-II polynomial ansatz adapter over the recovered 2026-08-19 circuit."""

from __future__ import annotations

from refined_twostage.circuits.stage2_polynomial import (
    PolynomialStage2Ansatz as _RecoveredPolynomialStage2Ansatz,
    PolynomialResources,
    greedy_native_critical_path_depth,
)


class PolynomialStage2Ansatz(_RecoveredPolynomialStage2Ansatz):
    """Clean-framework constructor wrapper for recovered Stage-II circuits."""

    def __init__(self, family_name: str, repetitions: int, data_qubits: int = 6) -> None:
        self.layers = int(repetitions)
        super().__init__(family_name, int(data_qubits), int(repetitions))


def parameter_count_for_family(family: str, repetitions: int) -> int:
    repetitions = int(repetitions)
    if family in ("vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot"):
        return 6 * (repetitions + 1)
    if family == "sequential_sg_mps":
        return 20 * repetitions
    if family == "dense_pairwise_ry_cnot":
        return 60 * repetitions
    raise ValueError("unknown Stage-II family {}".format(family))


def nearest_legal_capacity(family: str, target_parameters: int):
    target = int(target_parameters)
    best = None
    upper = max(2, target + 10)
    for reps in range(1, upper + 1):
        params = parameter_count_for_family(family, reps)
        key = (abs(params - target), 0 if params <= target else 1, params, reps)
        if best is None or key < best[0]:
            best = (key, reps, params)
        if params > target and abs(params - target) > max(100, target):
            break
    if best is None:
        raise RuntimeError("could not resolve {}".format(family))
    _key, reps, params = best
    return {"family": family, "target_P": target, "layers_or_repetitions": int(reps), "actual_P": int(params)}


__all__ = [
    "PolynomialStage2Ansatz",
    "PolynomialResources",
    "greedy_native_critical_path_depth",
    "parameter_count_for_family",
    "nearest_legal_capacity",
]
