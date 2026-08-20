"""Real-valued VCBE-inspired Stage-I projected-block ansatz families."""

from __future__ import annotations

import math
try:
    from typing import Literal
except ImportError:  # Python 3.7 runtime compatibility for this recovered copy.
    from typing_extensions import Literal

from refined_twostage.circuits.native_gates import append_cx
from refined_twostage.circuits.stage1_native_projected import NativeProjectedBlockAnsatz
from refined_twostage.types import NativeGate


def _fixed_ancilla_gate(data_qubits: int) -> NativeGate:
    return NativeGate("ry", int(data_qubits), (), -1, 1.0, math.pi, "fixed_input_to_bad_branch")


def _trainable_ry(gates: list[NativeGate], target: int, cursor: int, label: str) -> int:
    gates.append(NativeGate("ry", int(target), (), int(cursor), 1.0, 0.0, label))
    return int(cursor) + 1


def _append_real_rcn(
    gates: list[NativeGate],
    q_left: int,
    q_right: int,
    cursor: int,
    label: str,
    *,
    control: int,
    target: int,
) -> int:
    """Append the real restriction of the paper RCN block.

    Paper Figure 6 defines RCN as Ry(top), Rx(top), Ry(bottom), Rz(bottom),
    then CNOT top -> bottom.  Section 3.1.2/Appendix A set Rx/Rz angles to
    zero in the real ansatz, leaving one Ry on each qubit plus the CNOT.
    CNOT direction is explicit.  The two Ry parameters are placed on
    ``q_left`` and ``q_right`` in that order, independent of which qubit is the
    CNOT control.
    """
    left = int(q_left)
    right = int(q_right)
    control = int(control)
    target = int(target)
    if control == target:
        raise ValueError("real RCN CNOT control and target must differ")
    if control not in {left, right} or target not in {left, right}:
        raise ValueError("real RCN CNOT endpoints must match the parameterized edge")
    cursor = _trainable_ry(gates, left, cursor, f"{label}:real_rcn_left_ry")
    cursor = _trainable_ry(gates, right, cursor, f"{label}:real_rcn_right_ry")
    append_cx(gates, control, target, f"{label}:real_rcn_cnot_{control}_{target}")
    return cursor


def _append_final_ry_layer(gates: list[NativeGate], qubits: list[int], cursor: int, label: str) -> int:
    for q in qubits:
        cursor = _trainable_ry(gates, q, cursor, f"{label}:final_ry_q{q}")
    return cursor


def vcbe_block3_rounds(total_qubits: int) -> list[list[tuple[int, int]]]:
    """Return the Figure-7 Block-3 two-round linear matching schedule."""
    even = [(i, i + 1) for i in range(0, int(total_qubits) - 1, 2)]
    odd = [(i, i + 1) for i in range(1, int(total_qubits) - 1, 2)]
    return [even, odd]


def vcbe_block15_rounds(total_qubits: int) -> list[list[tuple[int, int]]]:
    """Return the Figure-7 Block-15 circular optimal-depth schedule."""
    total = int(total_qubits)
    rounds = vcbe_block3_rounds(total)
    rounds.append([(0, total - 1)])
    return rounds


def vcbe_block9_rounds(data_qubits: int) -> list[list[tuple[int, int]]]:
    """Return the Block-9 ancilla-centered star schedule."""
    ancilla = int(data_qubits)
    return [[(ancilla, q)] for q in range(int(data_qubits))]


def vcbe_block11_rounds(total_qubits: int) -> list[list[tuple[int, int]]]:
    """Return a deterministic round-robin edge coloring of K_N for odd N.

    For N=7 this gives seven rounds, each with three disjoint pairs and one
    idle qubit.  The unordered edge set is K_7 exactly once per repetition.
    """
    total = int(total_qubits)
    if total % 2 != 1:
        raise ValueError("Block-11 round-robin helper currently expects odd total qubits")
    rounds: list[list[tuple[int, int]]] = []
    half = total // 2
    for idle in range(total):
        matching = []
        for offset in range(1, half + 1):
            a = (idle + offset) % total
            b = (idle - offset) % total
            matching.append((min(a, b), max(a, b)))
        rounds.append(matching)
    return rounds


class VCBEBlock3RealAnsatz(NativeProjectedBlockAnsatz):
    """Real-valued VCBE Appendix-A Block 3: linear parallel RCN layers."""

    family_name = "vcbe_block3_real"

    def __init__(self, data_qubits: int, repetitions: int) -> None:
        gates: list[NativeGate] = []
        cursor = 0
        total = int(data_qubits) + 1
        qubits = list(range(total))
        for rep in range(int(repetitions)):
            for ridx, matching in enumerate(vcbe_block3_rounds(total)):
                for q0, q1 in matching:
                    cursor = _append_real_rcn(
                        gates,
                        q0,
                        q1,
                        cursor,
                        f"vcbe_b3_m{rep}:round{ridx}_edge_{q0}_{q1}",
                        control=q0,
                        target=q1,
                    )
        cursor = _append_final_ry_layer(gates, qubits, cursor, "vcbe_b3")
        super().__init__(
            data_qubits,
            repetitions,
            gates,
            fixed_ancilla_ry_pi=False,
            spec_extra={
                "vcbe_paper_block": 3,
                "paper_sources": ["Section 3.1", "Section 3.1.2", "Figure 1", "Figure 6", "Figure 7"],
                "gate_definition": "Each real RCN is Ry(top), Ry(bottom), CNOT(top->bottom); final Ry layer on all qubits.",
                "parallel_rounds": vcbe_block3_rounds(total),
                "formal_parameter_formula": "P_B3(M)=2*(N-1)*M+N; for N=7, P=12M+7",
                "cnot_formula": "CNOT_B3(M)=(N-1)*M; for N=7, CNOT=6M",
            },
        )


class VCBEBlock15RealAnsatz(NativeProjectedBlockAnsatz):
    """Real-valued VCBE Appendix-A Block 15: circular optimal-depth RCN layers."""

    family_name = "vcbe_block15_real"

    def __init__(self, data_qubits: int, repetitions: int) -> None:
        gates: list[NativeGate] = []
        cursor = 0
        total = int(data_qubits) + 1
        qubits = list(range(total))
        rounds = vcbe_block15_rounds(total)
        for rep in range(int(repetitions)):
            for ridx, matching in enumerate(rounds):
                for q0, q1 in matching:
                    label = f"vcbe_b15_m{rep}:round{ridx}_edge_{q0}_{q1}"
                    cursor = _append_real_rcn(gates, q0, q1, cursor, label, control=q0, target=q1)
        cursor = _append_final_ry_layer(gates, qubits, cursor, "vcbe_b15")
        super().__init__(
            data_qubits,
            repetitions,
            gates,
            fixed_ancilla_ry_pi=False,
            spec_extra={
                "vcbe_paper_block": 15,
                "paper_sources": ["Appendix A", "Figure 6", "Figure 7"],
                "gate_definition": "Real RCN on each ring edge; wraparound edge follows Figure-7 downward CNOT q0->qN-1; final Ry layer on all qubits.",
                "parallel_rounds": rounds,
                "formal_parameter_formula": "P_B15(M)=2*N*M+N; for N=7, P=14M+7",
                "cnot_formula": "CNOT_B15(M)=N*M; for N=7, CNOT=7M",
            },
        )


class VCBEBlock9RealAnsatz(NativeProjectedBlockAnsatz):
    """Real-valued VCBE Appendix-A Block 9: ancilla-centered star."""

    family_name = "vcbe_block9_real"

    def __init__(self, data_qubits: int, repetitions: int) -> None:
        gates: list[NativeGate] = []
        cursor = 0
        total = int(data_qubits) + 1
        ancilla = int(data_qubits)
        qubits = list(range(total))
        rounds = vcbe_block9_rounds(data_qubits)
        for rep in range(int(repetitions)):
            for ridx, matching in enumerate(rounds):
                for q0, q1 in matching:
                    cursor = _append_real_rcn(
                        gates,
                        q0,
                        q1,
                        cursor,
                        f"vcbe_b9_m{rep}:round{ridx}_edge_{q0}_{q1}",
                        control=ancilla,
                        target=q1,
                    )
        cursor = _append_final_ry_layer(gates, qubits, cursor, "vcbe_b9")
        super().__init__(
            data_qubits,
            repetitions,
            gates,
            fixed_ancilla_ry_pi=False,
            spec_extra={
                "vcbe_paper_block": 9,
                "paper_sources": ["Appendix A", "Figure 6", "Figure 7"],
                "gate_definition": "Ancilla-centered real RCN star; each CNOT is ancilla->data; final Ry layer on all qubits.",
                "parallel_rounds": rounds,
                "formal_parameter_formula": "P_B9(M)=2*(N-1)*M+N; for N=7, P=12M+7",
                "cnot_formula": "CNOT_B9(M)=(N-1)*M; for N=7, CNOT=6M",
            },
        )


class VCBEBlock11RealAnsatz(NativeProjectedBlockAnsatz):
    """All-to-all real-RCN ansatz adapted from VCBE Block 11.

    The unordered Block-11 topology is retained, while disjoint pair
    interactions are scheduled in deterministic parallel rounds for resource
    accounting. This is an implementation adaptation, not a verbatim
    finite-depth gate-order reproduction of Figure 7.
    """

    family_name = "vcbe_block11_real"

    def __init__(self, data_qubits: int, repetitions: int) -> None:
        gates: list[NativeGate] = []
        cursor = 0
        total = int(data_qubits) + 1
        qubits = list(range(total))
        rounds = vcbe_block11_rounds(total)
        for rep in range(int(repetitions)):
            for ridx, matching in enumerate(rounds):
                for q0, q1 in matching:
                    cursor = _append_real_rcn(
                        gates,
                        q0,
                        q1,
                        cursor,
                        f"vcbe_b11_m{rep}:round{ridx}_edge_{q0}_{q1}",
                        control=q0,
                        target=q1,
                    )
        cursor = _append_final_ry_layer(gates, qubits, cursor, "vcbe_b11")
        super().__init__(
            data_qubits,
            repetitions,
            gates,
            fixed_ancilla_ry_pi=False,
            spec_extra={
                "vcbe_paper_block": 11,
                "paper_sources": ["Appendix A", "Figure 6", "Figure 7"],
                "gate_definition": "All-to-all real RCN on every unordered qubit pair; deterministic round-robin parallel scheduling; low-index->high-index CNOT orientation; final Ry layer on all qubits.",
                "parallel_rounds": rounds,
                "formal_parameter_formula": "P_B11(M)=2*binom(N,2)*M+N; for N=7, P=42M+7",
                "cnot_formula": "CNOT_B11(M)=binom(N,2)*M; for N=7, CNOT=21M",
            },
        )


class AlternatingLocalAnsatz(NativeProjectedBlockAnsatz):
    """Previous Stage-I alternating-local control family."""

    family_name = "alternating_local"

    def __init__(self, data_qubits: int, repetitions: int) -> None:
        gates: list[NativeGate] = [_fixed_ancilla_gate(data_qubits)]
        cursor = 0
        dq = int(data_qubits)
        qubits = list(range(dq + 1))
        match_a = [(0, 1), (2, 3), (4, 5)]
        match_b = [(1, 2), (3, 4), (5, dq)]
        for rep in range(int(repetitions)):
            reverse = bool(rep & 1)
            for q in qubits:
                cursor = _trainable_ry(gates, q, cursor, f"alternating_local_r{rep}:A_ry_q{q}")
            for i, j in match_a:
                control, target = (j, i) if reverse else (i, j)
                append_cx(gates, control, target, f"alternating_local_r{rep}:A_cx_{control}_{target}")
            for q in qubits:
                cursor = _trainable_ry(gates, q, cursor, f"alternating_local_r{rep}:B_ry_q{q}")
            for i, j in match_b:
                control, target = (j, i) if reverse else (i, j)
                append_cx(gates, control, target, f"alternating_local_r{rep}:B_cx_{control}_{target}")
        super().__init__(
            data_qubits,
            repetitions,
            gates,
            fixed_ancilla_ry_pi=True,
            spec_extra={"layout": "q0-q1-q2-q3-q4-q5-qa", "formal_parameter_formula": "P_alt(L)=2*(n+1)*L; for n=6, P=14L"},
        )


def make_vcbe_stage1_ansatz(
    family: Literal["vcbe_block3_real", "vcbe_block15_real", "vcbe_block9_real", "vcbe_block11_real", "alternating_local"],
    data_qubits: int,
    repetitions: int,
):
    """Construct one VCBE-threshold-study ansatz by family name."""
    if family == "vcbe_block3_real":
        return VCBEBlock3RealAnsatz(data_qubits, repetitions)
    if family == "vcbe_block15_real":
        return VCBEBlock15RealAnsatz(data_qubits, repetitions)
    if family == "vcbe_block9_real":
        return VCBEBlock9RealAnsatz(data_qubits, repetitions)
    if family == "vcbe_block11_real":
        return VCBEBlock11RealAnsatz(data_qubits, repetitions)
    if family == "alternating_local":
        return AlternatingLocalAnsatz(data_qubits, repetitions)
    raise ValueError(f"unknown Stage-I threshold-study family {family!r}")
