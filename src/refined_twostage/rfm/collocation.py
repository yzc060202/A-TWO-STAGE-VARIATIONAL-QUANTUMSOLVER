"""Recovered 64-row collocation rule."""

from __future__ import annotations

import numpy as np


def linspace_64_including_endpoints(n_total: int = 64) -> np.ndarray:
    return np.linspace(-1.0, 1.0, int(n_total))


def row_types_for_points(points: np.ndarray):
    n = int(np.asarray(points).size)
    if n < 3:
        raise ValueError("need at least two boundary rows and one interior row")
    return ["dirichlet_left"] + ["pde_interior"] * (n - 2) + ["dirichlet_right"]
