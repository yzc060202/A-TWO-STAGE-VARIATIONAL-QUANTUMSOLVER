"""Stage-I factory interface."""

from __future__ import annotations

from refined_twostage.stage1.block11 import Block11ProjectedAnsatz, repetitions_for_capacity
from refined_twostage.stage1.families import Block15ProjectedAnsatz, GrayMultiplexedProjectedAnsatz, block15_repetitions_for_capacity, gray_layers_for_capacity


def make_stage1_ansatz(family: str, repetitions=None, target_parameters=None):
    if repetitions is None and target_parameters is None:
        raise ValueError("provide repetitions or target_parameters")
    if family == "vcbe_block11_real":
        reps = repetitions_for_capacity(int(target_parameters)) if repetitions is None else int(repetitions)
        return Block11ProjectedAnsatz(reps)
    if family == "vcbe_block15_real":
        reps = block15_repetitions_for_capacity(int(target_parameters)) if repetitions is None else int(repetitions)
        return Block15ProjectedAnsatz(reps)
    if family in ("gray_multiplexed_ry_cnot", "fable_inspired_gray"):
        layers = gray_layers_for_capacity(int(target_parameters)) if repetitions is None else int(repetitions)
        return GrayMultiplexedProjectedAnsatz(layers)
    raise ValueError("unknown Stage-I family {}".format(family))
