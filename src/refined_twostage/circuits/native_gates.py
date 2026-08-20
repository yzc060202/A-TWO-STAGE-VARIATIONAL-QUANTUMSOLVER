"""Native RY-CNOT gate-list simulation in little-endian order."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

import numpy as np

from refined_twostage.types import CircuitResources, NativeGate


def gray(value: int) -> int:
    """Return the binary reflected Gray code integer."""
    return int(value) ^ (int(value) >> 1)


def precompute_indices(gates: Iterable[NativeGate], num_qubits: int) -> tuple[NativeGate, ...]:
    """Attach `idx0` and `idx1` arrays to native gates via dataclass replace."""
    dim = 1 << int(num_qubits)
    states = np.arange(dim, dtype=np.int64)
    out: list[NativeGate] = []
    for gate in gates:
        mask = ((states >> int(gate.target)) & 1) == 0
        for q, value in gate.controls:
            mask &= ((states >> int(q)) & 1) == int(value)
        new_gate = replace(gate)
        object.__setattr__(new_gate, "idx0", states[mask])
        object.__setattr__(new_gate, "idx1", states[mask] | (1 << int(gate.target)))
        out.append(new_gate)
    return tuple(out)


def gate_angle(gate: NativeGate, theta: np.ndarray) -> float:
    """Resolve a gate's runtime angle."""
    if gate.trainable:
        return float(gate.theta_scale * theta[int(gate.theta_index)])
    return float(gate.fixed_angle)


def apply_ry_to_batch(psi: np.ndarray, idx0: np.ndarray, idx1: np.ndarray, angle: float) -> None:
    """Apply an RY rotation in-place to a batch of statevectors."""
    c = math.cos(0.5 * float(angle))
    s = math.sin(0.5 * float(angle))
    x = psi[:, idx0].copy()
    y = psi[:, idx1].copy()
    psi[:, idx0] = c * x - s * y
    psi[:, idx1] = s * x + c * y


def apply_x_to_batch(psi: np.ndarray, idx0: np.ndarray, idx1: np.ndarray) -> None:
    """Apply a controlled X/CNOT-style swap in-place to a batch."""
    x = psi[:, idx0].copy()
    psi[:, idx0] = psi[:, idx1]
    psi[:, idx1] = x


def apply_cz_to_batch(psi: np.ndarray, idx1: np.ndarray) -> None:
    """Apply a primitive controlled-Z phase flip in-place to a batch."""
    psi[:, idx1] *= -1.0


def apply_gate_batch(psi: np.ndarray, gate: NativeGate, theta: np.ndarray) -> None:
    """Apply one native gate to a batch of statevectors."""
    idx0 = getattr(gate, "idx0")
    idx1 = getattr(gate, "idx1")
    if gate.kind == "ry":
        apply_ry_to_batch(psi, idx0, idx1, gate_angle(gate, theta))
    elif gate.kind == "x":
        apply_x_to_batch(psi, idx0, idx1)
    elif gate.kind == "cz":
        apply_cz_to_batch(psi, idx1)
    else:
        raise ValueError(f"unsupported native gate kind {gate.kind}")


def apply_gate_state(state: np.ndarray, gate: NativeGate, theta: np.ndarray) -> None:
    """Apply one native gate to a single statevector."""
    batch = state.reshape(1, -1)
    apply_gate_batch(batch, gate, theta)


def apply_native_gates_state(state: np.ndarray, gates: Iterable[NativeGate], theta: np.ndarray) -> np.ndarray:
    """Apply native gates to one real statevector and return the output."""
    out = np.asarray(state, dtype=float).reshape(-1).copy()
    theta = np.asarray(theta, dtype=float).reshape(-1)
    for gate in gates:
        apply_gate_state(out, gate, theta)
    return out


def native_adjoint_vjp(
    initial_state: np.ndarray,
    gates: Iterable[NativeGate],
    theta: np.ndarray,
    output_cotangent: np.ndarray,
    *,
    num_parameters: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return output state, gradient, and input cotangent for a real native circuit.

    The cotangent convention is real Euclidean: if a scalar objective has
    derivative `output_cotangent` with respect to the final state, this returns
    derivatives with respect to trainable RY angles and to the input state.
    """
    gate_tuple = tuple(gates)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    psi = np.asarray(initial_state, dtype=float).reshape(-1).copy()
    states = [psi.copy()]
    for gate in gate_tuple:
        apply_gate_state(psi, gate, theta)
        states.append(psi.copy())
    nparams = int(num_parameters if num_parameters is not None else (theta.size if theta.size else 0))
    grad = np.zeros(nparams, dtype=float)
    adj = np.asarray(output_cotangent, dtype=float).reshape(-1).copy()
    for k in range(len(gate_tuple) - 1, -1, -1):
        gate = gate_tuple[k]
        idx0 = gate.idx0
        idx1 = gate.idx1
        if gate.kind == "ry":
            angle = gate_angle(gate, theta)
            c = math.cos(0.5 * angle)
            s = math.sin(0.5 * angle)
            ga = adj[idx0].copy()
            gb = adj[idx1].copy()
            if gate.trainable:
                pre = states[k]
                x = pre[idx0]
                y = pre[idx1]
                dnew0 = -0.5 * s * x - 0.5 * c * y
                dnew1 = 0.5 * c * x - 0.5 * s * y
                grad[int(gate.theta_index)] += float(gate.theta_scale) * float(np.sum(ga * dnew0 + gb * dnew1))
            adj[idx0] = c * ga + s * gb
            adj[idx1] = -s * ga + c * gb
        elif gate.kind == "x":
            tmp = adj[idx0].copy()
            adj[idx0] = adj[idx1]
            adj[idx1] = tmp
        elif gate.kind == "cz":
            adj[idx1] *= -1.0
        else:
            raise ValueError(f"unsupported native gate kind {gate.kind}")
    return states[-1], grad, adj


def resource_counts(gates: Iterable[NativeGate]) -> CircuitResources:
    """Count trainable parameters, RY gates, fixed RY gates, and CNOT gates."""
    gate_tuple = tuple(gates)
    params = 0
    ry = 0
    fixed_ry = 0
    cnot = 0
    for gate in gate_tuple:
        if gate.kind == "ry":
            ry += 1
            if gate.trainable:
                params = max(params, int(gate.theta_index) + 1)
            else:
                fixed_ry += 1
        elif gate.kind == "x":
            if len(gate.controls) != 1:
                raise ValueError("resource convention only supports primitive CNOT")
            cnot += 1
        elif gate.kind == "cz":
            pass
    return CircuitResources(
        parameters=int(params),
        ry_gates=int(ry),
        cnot_gates=int(cnot),
        fixed_ry_gates=int(fixed_ry),
        total_gates=len(gate_tuple),
    )


def add_gray_multiplexed_ry(
    gates: list[NativeGate],
    *,
    target: int,
    controls: list[int],
    parameter_cursor: int,
    label: str,
    reverse: bool = False,
) -> int:
    """Append a Gray-code decomposed uniformly controlled RY block.

    The appended primitives are unconstrained RY rotations on the target,
    interleaved with CNOTs from the changed Gray-code control bit to target.
    """
    controls = [int(q) for q in controls if int(q) != int(target)]
    count = 1 << len(controls)
    order = range(count - 1, -1, -1) if reverse else range(count)
    previous = None
    for step, item in enumerate(order):
        code = gray(int(item))
        if previous is not None:
            changed = previous ^ code
            bit = int(math.log2(changed)) if changed else 0
            control = controls[bit]
            gates.append(NativeGate("x", int(target), ((int(control), 1),), -1, 1.0, 0.0, f"{label}:gray_cx_{step}_{control}_{target}"))
        gates.append(NativeGate("ry", int(target), (), int(parameter_cursor), 1.0, 0.0, f"{label}:gray_ry_{step}"))
        parameter_cursor += 1
        previous = code
    return int(parameter_cursor)


def append_cx(gates: list[NativeGate], control: int, target: int, label: str) -> None:
    """Append one primitive CNOT."""
    gates.append(NativeGate("x", int(target), ((int(control), 1),), -1, 1.0, 0.0, label))


def append_cz(gates: list[NativeGate], q0: int, q1: int, label: str) -> None:
    """Append one primitive CZ."""
    gates.append(NativeGate("cz", int(q1), ((int(q0), 1),), -1, 1.0, 0.0, label))
