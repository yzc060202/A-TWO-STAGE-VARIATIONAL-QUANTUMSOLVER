import numpy as np

from refined_twostage.stage1.block11 import Block11ProjectedAnsatz
from refined_twostage.stage1.interface import make_stage1_ansatz


def test_stage1_full_basis_gradient_fd():
    ansatz = Block11ProjectedAnsatz(repetitions=1)
    rng = np.random.default_rng(5)
    target = rng.normal(scale=0.01, size=(64, 64))
    theta = rng.normal(scale=0.1, size=ansatz.num_parameters)
    loss, grad = ansatz.loss_and_grad_full_basis(theta, target)
    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    h = 1e-6
    lp, _ = ansatz.loss_and_grad_full_basis(theta + h * direction, target)
    lm, _ = ansatz.loss_and_grad_full_basis(theta - h * direction, target)
    assert abs(loss) >= 0
    assert abs((lp - lm) / (2 * h) - np.dot(grad, direction)) < 1e-7


def test_stage1_campaign_families_gradient_fd():
    rng = np.random.default_rng(123)
    target = rng.normal(size=(64, 64)) / 100.0
    for family in ["gray_multiplexed_ry_cnot", "vcbe_block11_real", "vcbe_block15_real"]:
        ansatz = make_stage1_ansatz(family, repetitions=1)
        theta = rng.normal(scale=0.02, size=ansatz.num_parameters)
        direction = rng.normal(size=theta.size)
        direction /= np.linalg.norm(direction)
        loss, grad = ansatz.loss_and_grad_full_basis(theta, target)
        h = 1e-6
        lp, _ = ansatz.loss_and_grad_full_basis(theta + h * direction, target)
        lm, _ = ansatz.loss_and_grad_full_basis(theta - h * direction, target)
        fd = (lp - lm) / (2.0 * h)
        analytic = float(np.dot(grad, direction))
        rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
        assert loss >= 0.0
        assert ansatz.extract_projected_block(theta).shape == (64, 64)
        assert rel <= 1e-6
