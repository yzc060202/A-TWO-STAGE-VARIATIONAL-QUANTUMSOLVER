import numpy as np

from refined_twostage.metrics.pde_metrics import evaluate_physical
from refined_twostage.rfm.assemble import build_rfm_system


def test_reconstruction_matches_feature_evaluation():
    bundle = build_rfm_system("convection_diffusion", 200)
    z = np.random.default_rng(10).normal(size=64)
    coeff = bundle.raw_coefficients(z)
    grid = np.linspace(-1, 1, 17)
    np.testing.assert_allclose(bundle.feature_bank.values(grid) @ coeff, bundle.feature_bank.values(grid) @ (bundle.Dc * (bundle.V64 @ z)))
    metrics = evaluate_physical(bundle.pde, bundle.feature_bank, coeff, bundle.dense_x)
    assert "PDE_relative_L2" in metrics
