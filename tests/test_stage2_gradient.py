import numpy as np

from refined_twostage.stage2.alternating_ry_cnot import AlternatingRyCnotAnsatz
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.objectives import objective_value_and_grad_theta


def test_ansatz_norm_and_vjp_fd():
    ansatz = AlternatingRyCnotAnsatz(layers=1)
    rng = np.random.default_rng(8)
    theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
    cot = rng.normal(size=ansatz.dimension)
    state, grad = ansatz.value_and_vjp(theta, cot)
    assert abs(np.linalg.norm(state) - 1.0) < 1e-12
    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    h = 1e-6
    sp = ansatz.state(theta + h * direction)
    sm = ansatz.state(theta - h * direction)
    assert abs(np.dot(cot, (sp - sm) / (2 * h)) - np.dot(grad, direction)) < 1e-7


def test_stage2_theta_gradient_fd():
    ansatz = AlternatingRyCnotAnsatz(layers=1)
    rng = np.random.default_rng(9)
    A = rng.normal(size=(64, 64))
    b = rng.normal(size=64)
    theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    value, grad, _ = objective_value_and_grad_theta(A, b, ansatz, theta, objective="direct_r2")
    h = 1e-6
    vp, _, _ = objective_value_and_grad_theta(A, b, ansatz, theta + h * direction, objective="direct_r2")
    vm, _, _ = objective_value_and_grad_theta(A, b, ansatz, theta - h * direction, objective="direct_r2")
    assert value >= 0
    assert abs((vp - vm) / (2 * h) - np.dot(grad, direction)) < 1e-6


def test_stage2_five_family_norm_vjp_and_direct_r_gradients():
    rng = np.random.default_rng(19)
    A = rng.normal(size=(64, 64))
    b = rng.normal(size=64)
    families = ["vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot", "sequential_sg_mps", "dense_pairwise_ry_cnot"]
    for family in families:
        ansatz = make_stage2_ansatz(family, layers=1)
        theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
        cot = rng.normal(size=ansatz.dimension)
        state, vjp = ansatz.value_and_vjp(theta, cot)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12
        direction = rng.normal(size=theta.size)
        direction /= np.linalg.norm(direction)
        h = 1e-6
        sp = ansatz.state(theta + h * direction)
        sm = ansatz.state(theta - h * direction)
        fd_action = float(np.dot(cot, (sp - sm) / (2.0 * h)))
        analytic_action = float(np.dot(vjp, direction))
        assert abs(fd_action - analytic_action) / max(abs(fd_action), abs(analytic_action), 1e-12) <= 1e-6
        for objective in ["direct_r2", "direct_r"]:
            value, grad, _ = objective_value_and_grad_theta(A, b, ansatz, theta, objective=objective)
            vp, _, _ = objective_value_and_grad_theta(A, b, ansatz, theta + h * direction, objective=objective)
            vm, _, _ = objective_value_and_grad_theta(A, b, ansatz, theta - h * direction, objective=objective)
            fd = (vp - vm) / (2.0 * h)
            analytic = float(np.dot(grad, direction))
            rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
            assert value >= 0.0
            assert rel <= 1e-6
