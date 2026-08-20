import json
from pathlib import Path

import numpy as np
import pytest

from refined_twostage.io.hashing import sha256_array
from refined_twostage.problems.common import get_pde, u_star, u_xx
from refined_twostage.rfm.assemble import build_rfm_system


CAMPAIGN_ROOT = Path(__file__).resolve().parents[2] / "results" / "direct64_full_twostage_campaign_20260820_204937"
pytestmark = pytest.mark.provenance


def test_helmholtz_forcing_and_boundary_definition():
    pde = get_pde("helmholtz_control")
    x = np.linspace(-0.9, 0.9, 17)
    np.testing.assert_allclose(pde.forcing(x), u_xx(x) + 4.0 * u_star(x))
    endpoints = np.array([-1.0, 1.0])
    assert np.all(np.isfinite(pde.forcing(endpoints)))


def test_helmholtz_frozen_matrix_hash_reproduction():
    manifest_path = CAMPAIGN_ROOT / "01_frozen_direct64" / "helmholtz_control" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("external frozen Direct64 Helmholtz campaign artifact is not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = build_rfm_system("helmholtz_control", 64, construction="DIRECT64_REGENERATED")
    assert sha256_array(bundle.A_raw) == manifest["hashes"]["A_raw"]
    assert sha256_array(bundle.A_p) == manifest["hashes"]["A_p"]
    assert sha256_array(bundle.b_p) == manifest["hashes"]["b_p"]
    assert sha256_array(bundle.Dc) == manifest["hashes"]["Dc"]
