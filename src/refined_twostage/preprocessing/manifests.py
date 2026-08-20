"""Machine-readable provenance manifests."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from refined_twostage.io.artifacts import environment_record, write_json


def bundle_manifest(bundle, extra=None):
    manifest = bundle.identity()
    _z, _c, row, _mat = bundle.exact_working_diagnostics()
    manifest.update({"exact_classical_residual": row["classical_residual"], "exact_classical_PDE_L2": row["PDE_L2"], "exact_classical_PDE_Linf": row["PDE_Linf"], "exact_classical_dense_PDE_residual": row["dense_PDE_residual"], "exact_classical_boundary_error_max": row["boundary_error"], "exact_classical_coefficient_norm": row["coefficient_norm"], "environment": environment_record()})
    if extra:
        manifest.update(extra)
    return manifest


def save_frozen_bundle(bundle, out_dir: Path, extra_manifest=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A_raw.npy", bundle.A_raw)
    np.save(out_dir / "b_raw.npy", bundle.b_raw)
    np.save(out_dir / "Dr.npy", bundle.Dr)
    np.save(out_dir / "Dc.npy", bundle.Dc)
    np.save(out_dir / "A_p.npy", bundle.A_p)
    np.save(out_dir / "b_p.npy", bundle.b_p)
    np.save(out_dir / "V64.npy", bundle.V64)
    np.save(out_dir / "A64.npy", bundle.A64)
    np.save(out_dir / "As.npy", bundle.As)
    np.savez_compressed(out_dir / "feature_parameters.npz", w=bundle.feature_bank.w, b=bundle.feature_bank.b, alpha=np.array([bundle.feature_bank.alpha], dtype=float), seed=np.array([bundle.feature_bank.seed], dtype=np.int64))
    np.save(out_dir / "collocation_points.npy", bundle.points)
    write_json(out_dir / "row_types.json", {"row_types": bundle.row_types, "counts": bundle.identity()["row_counts"]})
    write_json(out_dir / "reconstruction_metadata.json", {"map": bundle.identity()["reconstruction_map"], "working_rule": bundle.working_rule, "svd_sign_canonicalized": bundle.svd_canonicalized, "svd_sign_flips": bundle.svd_sign_flips})
    np.save(out_dir / "singular_values_Ap.npy", bundle.singular_values_Ap)
    np.save(out_dir / "singular_values_A64.npy", bundle.singular_values_A64)
    manifest = bundle_manifest(bundle, extra_manifest)
    write_json(out_dir / "manifest.json", manifest)
    return manifest
