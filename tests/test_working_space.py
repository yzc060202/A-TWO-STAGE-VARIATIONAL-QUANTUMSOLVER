import numpy as np

from refined_twostage.preprocessing.scaling import column_equilibrate
from refined_twostage.preprocessing.working_space import canonicalize_right_singular_vectors, right_svd_working_space
from refined_twostage.rfm.assemble import build_rfm_system


def test_Dr_Dc_identity_and_Ap_formula():
    bundle = build_rfm_system("poisson", 200)
    Dr, Dc, A_p, b_p = column_equilibrate(bundle.A_raw, bundle.b_raw)
    np.testing.assert_allclose(Dr, np.ones(64))
    np.testing.assert_allclose(Dc, bundle.Dc)
    np.testing.assert_allclose(A_p, bundle.A_p)
    np.testing.assert_allclose(b_p, bundle.b_p)


def test_right_working_space_identity_and_reconstruction():
    bundle = build_rfm_system("reaction_diffusion", 200, canonicalize_svd=True)
    np.testing.assert_allclose(bundle.A64, bundle.A_p @ bundle.V64, atol=1e-13, rtol=1e-13)
    z = np.random.default_rng(3).normal(size=64)
    coeff = bundle.raw_coefficients(z)
    np.testing.assert_allclose(bundle.A_raw @ coeff, bundle.A_p @ (bundle.V64 @ z), atol=1e-10, rtol=1e-10)


def test_svd_sign_canonicalization():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 12))
    ws = right_svd_working_space(A, retained_rank=8, canonicalize=True)
    for j in range(ws.V64.shape[1]):
        i = int(np.argmax(np.abs(ws.V64[:, j])))
        assert ws.V64[i, j] >= 0
    V2, flips = canonicalize_right_singular_vectors(-ws.V64)
    assert np.all(flips == -1)
    np.testing.assert_allclose(V2, ws.V64)
