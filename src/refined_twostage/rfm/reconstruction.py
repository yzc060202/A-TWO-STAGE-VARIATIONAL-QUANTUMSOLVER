"""Coefficient reconstruction maps."""

from __future__ import annotations

import numpy as np


def raw_coefficients_from_working(Dc, V64, z):
    return np.asarray(Dc, dtype=float) * (np.asarray(V64, dtype=float) @ np.asarray(z, dtype=float))


def raw_coefficients_direct(Dc, z):
    return np.asarray(Dc, dtype=float) * np.asarray(z, dtype=float)
