import sys
from pathlib import Path

import numpy as np
import pytest

from refined_twostage.rfm.assemble import build_rfm_system
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.vector_exact import VectorExactLossEvaluator


ROOT = Path(__file__).resolve().parents[2]
RECOVERED = ROOT.parent / "recovered_20260819_twostage_framework"
REFERENCE = RECOVERED / "reference_20260819" / "final_refined_stage2_and_crosspde_20260819_045316"


pytestmark = pytest.mark.provenance


def _add_recovered_path():
    src = str(RECOVERED / "src")
    if not Path(src).exists():
        pytest.skip("external recovered_20260819_twostage_framework artifact is not present")
    if src not in sys.path:
        sys.path.insert(0, src)


def _require_reference_file(path: Path) -> None:
    if not path.exists():
        pytest.skip("external historical reference artifact is not present: {}".format(path))


def test_migrated_block11_projected_block_equals_recovered():
    _add_recovered_path()
    from twostage.circuits.stage1_vcbe import make_vcbe_stage1_ansatz

    rng = np.random.default_rng(101)
    theta = rng.normal(scale=0.02, size=49)
    migrated = make_stage1_ansatz("vcbe_block11_real", repetitions=1)
    recovered = make_vcbe_stage1_ansatz("vcbe_block11_real", 6, 1)

    np.testing.assert_allclose(migrated.extract_projected_block(theta), recovered.extract_projected_block(theta), atol=0.0, rtol=0.0)


def test_migrated_stage1_gradient_equals_recovered_and_fd():
    _add_recovered_path()
    from twostage.circuits.stage1_vcbe import make_vcbe_stage1_ansatz

    rng = np.random.default_rng(102)
    migrated = make_stage1_ansatz("vcbe_block11_real", repetitions=1)
    recovered = make_vcbe_stage1_ansatz("vcbe_block11_real", 6, 1)
    theta = rng.normal(scale=0.02, size=migrated.num_parameters)
    target = rng.normal(scale=0.01, size=(64, 64))

    loss_m, grad_m = migrated.loss_and_grad_full_basis(theta, target, chunk_size=8, evaluator_mode="full_batch_tape")
    loss_r, grad_r = recovered.loss_and_grad_full_basis(theta, target, chunk_size=8, evaluator_mode="full_batch_tape")

    np.testing.assert_allclose(loss_m, loss_r, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(grad_m, grad_r, rtol=0.0, atol=0.0)

    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    h = 1e-6
    lp, _ = migrated.loss_and_grad_full_basis(theta + h * direction, target, chunk_size=8, evaluator_mode="full_batch_tape")
    lm, _ = migrated.loss_and_grad_full_basis(theta - h * direction, target, chunk_size=8, evaluator_mode="full_batch_tape")
    assert abs((lp - lm) / (2.0 * h) - np.dot(grad_m, direction)) <= 1e-6


def test_migrated_stage2_state_equals_recovered():
    _add_recovered_path()
    from twostage.circuits.stage2_polynomial import PolynomialStage2Ansatz as RecoveredStage2

    rng = np.random.default_rng(103)
    for family in ["vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot", "sequential_sg_mps", "dense_pairwise_ry_cnot"]:
        migrated = make_stage2_ansatz(family, layers=1)
        recovered = RecoveredStage2(family, 6, 1)
        theta = rng.normal(scale=0.2, size=migrated.num_parameters)
        cot = rng.normal(size=migrated.dimension)
        np.testing.assert_allclose(migrated.state(theta), recovered.state(theta), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(migrated.value_and_vjp(theta, cot)[1], recovered.value_and_vjp(theta, cot)[1], atol=0.0, rtol=0.0)


def test_direct_r_value_and_gradient_independent_reference_fd():
    rng = np.random.default_rng(104)
    ansatz = make_stage2_ansatz("alternating_ry_cnot", layers=1)
    A = rng.normal(size=(64, 64))
    b = rng.normal(size=64)
    theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
    evaluator = VectorExactLossEvaluator(A, b, ansatz, objective="direct_r")
    value, grad = evaluator.value_and_grad(theta)

    y = ansatz.state(theta)
    v = A @ y
    alpha = float(np.dot(b, v) / np.dot(v, v))
    expected = np.linalg.norm(alpha * v - b) / np.linalg.norm(b)
    assert abs(value - expected) <= 1e-14

    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    h = 1e-6
    vp, _ = evaluator.value_and_grad(theta + h * direction)
    vm, _ = evaluator.value_and_grad(theta - h * direction)
    assert abs((vp - vm) / (2.0 * h) - np.dot(grad, direction)) <= 1e-6


def test_new_generator_matches_frozen_pre_svd_arrays_and_reconstruction():
    _require_reference_file(REFERENCE / "cross_pde_final" / "poisson" / "arrays.npz")
    frozen = np.load(REFERENCE / "cross_pde_final" / "poisson" / "arrays.npz")
    bundle = build_rfm_system("poisson", 200, canonicalize_svd=False)

    for name, actual in [
        ("A_raw", bundle.A_raw),
        ("b_raw", bundle.b_raw),
        ("A_p", bundle.A_p),
        ("b_p", bundle.b_p),
        ("Dc", bundle.Dc),
        ("w", bundle.feature_bank.w),
        ("feature_b", bundle.feature_bank.b),
        ("x64", bundle.points),
        ("dense_x", bundle.dense_x),
    ]:
        np.testing.assert_allclose(actual, frozen[name], rtol=0.0, atol=3e-13)

    z = np.random.default_rng(105).normal(size=64)
    np.testing.assert_allclose(bundle.raw_coefficients(z), bundle.Dc * (bundle.V64 @ z), rtol=0.0, atol=0.0)


def test_dense_surrogate_absent_from_formal_stage1_call_path():
    for family, kwargs in [
        ("vcbe_block11_real", {"repetitions": 1}),
        ("vcbe_block15_real", {"repetitions": 1}),
        ("gray_multiplexed_ry_cnot", {"repetitions": 1}),
    ]:
        ansatz = make_stage1_ansatz(family, **kwargs)
        assert "Surrogate" not in type(ansatz).__name__
        assert "surrogate" not in getattr(ansatz, "family_name", "").lower()
