"""Common manufactured solution and PDE definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


def u_star(x):
    t = np.pi * (np.asarray(x, dtype=float) + 0.05)
    return 0.5 * (np.sin(5.0 * t) + np.sin(t)) + 2.0


def u_x(x):
    t = np.pi * (np.asarray(x, dtype=float) + 0.05)
    return 0.5 * np.pi * (5.0 * np.cos(5.0 * t) + np.cos(t))


def u_xx(x):
    t = np.pi * (np.asarray(x, dtype=float) + 0.05)
    return -0.5 * np.pi ** 2 * (25.0 * np.sin(5.0 * t) + np.sin(t))


@dataclass(frozen=True)
class PdeSpec:
    key: str
    name: str
    operator: str
    forcing: Callable
    feature_rows: Callable
    apply_coefficients: Callable


def pde_specs() -> List[PdeSpec]:
    return [
        PdeSpec("poisson", "1D Poisson", "-u''", lambda x: -u_xx(x), lambda fb, x: -fb.second_derivatives(x), lambda fb, c, x: (-fb.second_derivatives(x)) @ c),
        PdeSpec("reaction_diffusion", "1D reaction-diffusion", "-0.1u'' + u", lambda x: -0.1 * u_xx(x) + u_star(x), lambda fb, x: -0.1 * fb.second_derivatives(x) + fb.values(x), lambda fb, c, x: (-0.1 * fb.second_derivatives(x) + fb.values(x)) @ c),
        PdeSpec("convection_diffusion", "1D convection-diffusion", "-0.1u'' + u'", lambda x: -0.1 * u_xx(x) + u_x(x), lambda fb, x: -0.1 * fb.second_derivatives(x) + fb.first_derivatives(x), lambda fb, c, x: (-0.1 * fb.second_derivatives(x) + fb.first_derivatives(x)) @ c),
    ]


def helmholtz_control_spec() -> PdeSpec:
    return PdeSpec("helmholtz_control", "Helmholtz control", "u'' + 4u", lambda x: u_xx(x) + 4.0 * u_star(x), lambda fb, x: fb.second_derivatives(x) + 4.0 * fb.values(x), lambda fb, c, x: (fb.second_derivatives(x) + 4.0 * fb.values(x)) @ c)


def pde_map() -> Dict[str, PdeSpec]:
    mapping = {p.key: p for p in pde_specs()}
    mapping[helmholtz_control_spec().key] = helmholtz_control_spec()
    return mapping


def get_pde(key: str) -> PdeSpec:
    mapping = pde_map()
    if key not in mapping:
        raise KeyError("unknown PDE key {!r}; choices are {}".format(key, sorted(mapping)))
    return mapping[key]


MANUFACTURED_SOLUTION_ID = "0.5*(sin(5*pi*(x+0.05))+sin(pi*(x+0.05)))+2"
