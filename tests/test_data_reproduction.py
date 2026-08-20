import numpy as np

from refined_twostage.io.hashing import sha256_array
from refined_twostage.rfm.assemble import assemble_raw_system, build_rfm_system
from refined_twostage.rfm.features import sample_uniform_sine_features


def test_feature_and_collocation_reproducibility():
    bundle = build_rfm_system("poisson", 200, canonicalize_svd=True)
    assert sha256_array(np.vstack([bundle.feature_bank.w, bundle.feature_bank.b])) == "07b91674c760f972fa819184cc2a669b8db4475f425e83896bafc20c1e4d2df1"
    assert sha256_array(bundle.points) == "a388ee8aaa279e706481bd27c90bafaff43c145c27444a2f05cc1b892a34f7b9"
    assert bundle.row_types[0] == "dirichlet_left"
    assert bundle.row_types[-1] == "dirichlet_right"
    assert bundle.row_types.count("pde_interior") == 62


def test_reproduction_fingerprints_are_numerically_recovered():
    expected = {"poisson": 7.508839752368795e-10, "reaction_diffusion": 3.979894827674283e-10, "convection_diffusion": 6.435457078047263e-10}
    for key, l2 in expected.items():
        bundle = build_rfm_system(key, 200, canonicalize_svd=True)
        _z, _c, row, _mat = bundle.exact_working_diagnostics()
        assert abs(row["PDE_L2"] - l2) < 1e-10
        assert row["boundary_error"] < 5e-12


def test_boundary_rows_replace_operator_rows():
    fb = sample_uniform_sine_features(8)
    bundle = build_rfm_system("poisson", 8, construction="DIRECT64_REGENERATED")
    A, _b, row_types = assemble_raw_system(bundle.pde, fb, bundle.points)
    np.testing.assert_allclose(A[0], fb.values(np.array([-1.0]))[0])
    np.testing.assert_allclose(A[-1], fb.values(np.array([1.0]))[0])
    assert row_types[1] == "pde_interior"
