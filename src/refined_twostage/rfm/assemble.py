"""Assembly of recovered RFM systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from refined_twostage.io.hashing import sha256_array
from refined_twostage.metrics.diagnostics import classical_diagnostics
from refined_twostage.metrics.matrix_metrics import singular_diagnostics
from refined_twostage.preprocessing.scaling import column_equilibrate
from refined_twostage.preprocessing.working_space import direct_square_working_space, right_svd_working_space
from refined_twostage.problems.common import MANUFACTURED_SOLUTION_ID, get_pde, u_star
from refined_twostage.rfm.collocation import linspace_64_including_endpoints, row_types_for_points
from refined_twostage.rfm.features import SineFeatureBank, sample_uniform_sine_features
from refined_twostage.rfm.reconstruction import raw_coefficients_direct, raw_coefficients_from_working


@dataclass
class RFMSystem:
    pde: object
    feature_bank: SineFeatureBank
    points: np.ndarray
    dense_x: np.ndarray
    row_types: list
    A_raw: np.ndarray
    b_raw: np.ndarray
    Dr: np.ndarray
    Dc: np.ndarray
    A_p: np.ndarray
    b_p: np.ndarray
    V64: np.ndarray
    A64: np.ndarray
    As: np.ndarray
    s_A: float
    singular_values_Ap: np.ndarray
    singular_values_A64: np.ndarray
    construction: str
    working_rule: str
    svd_canonicalized: bool
    svd_sign_flips: np.ndarray

    def raw_coefficients(self, z):
        if self.construction in ("DIRECT64_REGENERATED", "PREFIX64_FROM_200_DIAGNOSTIC"):
            return raw_coefficients_direct(self.Dc, z)
        return raw_coefficients_from_working(self.Dc, self.V64, z)

    def raw_rectangular_coefficients(self, y):
        return raw_coefficients_direct(self.Dc, y)

    def exact_working_diagnostics(self):
        return classical_diagnostics("working", self.construction, self.A64, self.b_p, self.raw_coefficients, self.pde, self.feature_bank, self.dense_x)

    def raw_rectangular_diagnostics(self):
        return classical_diagnostics("raw_scaled_rectangular", "HISTORICAL_NEW_RAW_64x200", self.A_p, self.b_p, self.raw_rectangular_coefficients, self.pde, self.feature_bank, self.dense_x)

    def identity(self):
        return {
            "pde_key": self.pde.key,
            "pde_name": self.pde.name,
            "operator": self.pde.operator,
            "manufactured_solution": MANUFACTURED_SOLUTION_ID,
            "domain": [-1.0, 1.0],
            "feature_formula": "phi_j(x)=sin(alpha*(x*w_j+b_j))",
            "feature_count": int(self.feature_bank.w.size),
            "feature_seed": int(self.feature_bank.seed),
            "feature_rng": "np.random.default_rng(seed)",
            "feature_rng_draw_order": "draw all w Uniform[-1,1], then all b Uniform[-1,1]",
            "feature_alpha": float(self.feature_bank.alpha),
            "feature_hash": sha256_array(np.vstack([self.feature_bank.w, self.feature_bank.b])),
            "collocation_rule": "np.linspace(-1,1,64), first/last rows Dirichlet",
            "collocation_hash": sha256_array(self.points),
            "row_ordering": "row0 Dirichlet left, rows1..62 PDE interior, row63 Dirichlet right",
            "row_counts": {name: int(self.row_types.count(name)) for name in sorted(set(self.row_types))},
            "scaling": "Dr=I; Dc_j=1/max(||A_raw[:,j]||_2,1e-300); A_p=Dr*A_raw*Dc; b_p=Dr*b_raw",
            "reconstruction_map": "c_raw=Dc*(V64@z)" if self.construction == "HISTORICAL_NEW_64x200_TO_64x64" else "c_raw=Dc*z",
            "working_rule": self.working_rule,
            "svd_sign_canonicalized": bool(self.svd_canonicalized),
            "s_A": float(self.s_A),
            "A_raw_hash": sha256_array(self.A_raw),
            "b_raw_hash": sha256_array(self.b_raw),
            "Dr_hash": sha256_array(self.Dr),
            "Dc_hash": sha256_array(self.Dc),
            "A_p_hash": sha256_array(self.A_p),
            "b_p_hash": sha256_array(self.b_p),
            "V64_hash": sha256_array(self.V64),
            "A64_hash": sha256_array(self.A64),
            "As_hash": sha256_array(self.As),
            "raw_matrix": singular_diagnostics(self.A_raw),
            "scaled_matrix": singular_diagnostics(self.A_p),
            "working_matrix": singular_diagnostics(self.A64),
        }


def assemble_raw_system(pde, feature_bank, points):
    points = np.asarray(points, dtype=float)
    A = pde.feature_rows(feature_bank, points).copy()
    b = pde.forcing(points).copy()
    phi = feature_bank.values(points)
    A[0, :] = phi[0, :]
    A[-1, :] = phi[-1, :]
    b[0] = u_star(np.array([points[0]]))[0]
    b[-1] = u_star(np.array([points[-1]]))[0]
    return A, b, row_types_for_points(points)


def build_rfm_system(pde_key: str, n_features: int = 200, seed: int = 7061015, alpha: float = 50.0, dense_points: int = 4096, construction: Optional[str] = None, source_bank: Optional[SineFeatureBank] = None, canonicalize_svd: bool = True, scale_safety: float = 1.05) -> RFMSystem:
    pde = get_pde(pde_key)
    if source_bank is not None:
        fb = source_bank.prefix(n_features)
        if construction is None:
            construction = "PREFIX64_FROM_200_DIAGNOSTIC"
    else:
        fb = sample_uniform_sine_features(n_features, seed=seed, alpha=alpha)
        if construction is None:
            construction = "HISTORICAL_NEW_64x200_TO_64x64" if int(n_features) > 64 else "DIRECT64_REGENERATED"
    points = linspace_64_including_endpoints(64)
    dense_x = np.linspace(-1.0, 1.0, int(dense_points))
    A_raw, b_raw, row_types = assemble_raw_system(pde, fb, points)
    Dr, Dc, A_p, b_p = column_equilibrate(A_raw, b_raw)
    if int(n_features) > 64:
        ws = right_svd_working_space(A_p, retained_rank=64, canonicalize=canonicalize_svd)
    else:
        ws = direct_square_working_space(A_p)
    s_A = float(scale_safety) * float(np.linalg.norm(ws.A64, 2))
    As = ws.A64 / s_A
    return RFMSystem(pde, fb, points, dense_x, row_types, A_raw, b_raw, Dr, Dc, A_p, b_p, ws.V64, ws.A64, As, s_A, ws.singular_values if int(n_features) > 64 else np.linalg.svd(A_p, compute_uv=False), np.linalg.svd(ws.A64, compute_uv=False), construction, ws.rule, ws.canonicalized, ws.sign_flips)
