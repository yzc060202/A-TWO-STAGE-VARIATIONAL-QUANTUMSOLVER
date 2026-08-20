import numpy as np

from refined_twostage.stage2.objectives import objective_value_and_grad_y, scalar_eliminated_components


def test_direct_r2_and_direct_r_are_not_aliased():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(5, 5))
    b = rng.normal(size=5)
    y = rng.normal(size=5)
    comp = scalar_eliminated_components(A, b, y)
    assert np.isclose(comp["direct_r"], np.sqrt(comp["direct_r2"]))
    assert not np.isclose(comp["direct_r"], comp["direct_r2"])


def test_direct_r2_and_r_gradients_fd_y():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, 4))
    b = rng.normal(size=6)
    y = rng.normal(size=4)
    direction = rng.normal(size=4)
    direction /= np.linalg.norm(direction)
    for objective in ("direct_r2", "direct_r"):
        value, grad, _ = objective_value_and_grad_y(A, b, y, objective=objective)
        h = 1e-6
        vp, _, _ = objective_value_and_grad_y(A, b, y + h * direction, objective=objective)
        vm, _, _ = objective_value_and_grad_y(A, b, y - h * direction, objective=objective)
        assert value >= 0
        assert abs((vp - vm) / (2 * h) - np.dot(grad, direction)) < 1e-6
