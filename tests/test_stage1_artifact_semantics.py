import numpy as np

from refined_twostage.io.hashing import sha256_array
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize


def test_stage1_optimizer_reports_best_and_last_theta_separately():
    ansatz = make_stage1_ansatz("vcbe_block11_real", target_parameters=3997)
    target = np.random.default_rng(44).normal(scale=0.1, size=(64, 64))
    result = adam_optimize(ansatz, target, steps=4, lr=0.01, history_interval=1)

    assert result["theta_final_policy"] == "best_theta"
    np.testing.assert_array_equal(result["theta"], result["theta_best"])
    assert 1 <= result["best_step"] <= 4
    assert result["last_step"] == 4

    best_loss, best_grad = ansatz.loss_and_grad_full_basis(result["theta_best"], target)
    last_loss, last_grad = ansatz.loss_and_grad_full_basis(result["theta_last"], target)

    assert result["best_loss"] == best_loss
    assert result["final_loss"] == best_loss
    assert result["last_loss"] == last_loss
    assert np.isclose(result["best_gradient_norm"], np.linalg.norm(best_grad), rtol=0.0, atol=1e-14)
    assert np.isclose(result["final_gradient_norm"], np.linalg.norm(best_grad), rtol=0.0, atol=1e-14)
    assert np.isclose(result["last_gradient_norm"], np.linalg.norm(last_grad), rtol=0.0, atol=1e-14)


def test_saved_k_and_ahat_semantics_from_theta_and_scale():
    ansatz = make_stage1_ansatz("vcbe_block15_real", target_parameters=5005)
    theta = np.random.default_rng(123).normal(scale=0.02, size=ansatz.num_parameters)
    scale = 3.25

    k_from_theta = ansatz.extract_projected_block(theta)
    ahat_from_k = scale * k_from_theta

    np.testing.assert_array_equal(ahat_from_k, scale * ansatz.extract_projected_block(theta))
    assert sha256_array(k_from_theta) == sha256_array(ansatz.extract_projected_block(theta))


def test_projected_block_independent_of_scoring_target_and_parameter_sensitive():
    ansatz = make_stage1_ansatz("gray_multiplexed_ry_cnot", target_parameters=4096)
    rng = np.random.default_rng(321)
    theta = rng.normal(scale=0.02, size=ansatz.num_parameters)
    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    target_a = rng.normal(size=(64, 64))
    target_b = rng.normal(size=(64, 64))

    k_before_a = ansatz.extract_projected_block(theta)
    ansatz.loss_and_grad_full_basis(theta, target_a)
    ansatz.loss_and_grad_full_basis(theta, target_b)
    k_after = ansatz.extract_projected_block(theta)
    k_perturbed = ansatz.extract_projected_block(theta + 1e-4 * direction)

    np.testing.assert_array_equal(k_before_a, k_after)
    assert np.linalg.norm(k_perturbed - k_before_a) > 0.0


def test_distinct_stage1_runs_do_not_share_artifacts_by_capacity():
    gray = make_stage1_ansatz("gray_multiplexed_ry_cnot", target_parameters=4096)
    block11 = make_stage1_ansatz("vcbe_block11_real", target_parameters=5005)

    theta_gray = np.random.default_rng(1).normal(size=gray.num_parameters)
    theta_block11 = np.random.default_rng(2).normal(size=block11.num_parameters)

    k_gray = gray.extract_projected_block(theta_gray)
    k_block11 = block11.extract_projected_block(theta_block11)

    assert gray.num_parameters != block11.num_parameters
    assert sha256_array(k_gray) != sha256_array(k_block11)
