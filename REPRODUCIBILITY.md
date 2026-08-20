# Reproducibility Guide

## Python Version

The code is packaged with `requires-python >= 3.7`. The current development and
cleanup pass was run with the local Anaconda Python available on Windows.

## Dependencies

Install with:

```text
python -m pip install -e .
```

Core dependencies are NumPy, SciPy, and PyYAML. No external experiment results
are required for the default test suite.

## Run Tests

```text
python -m pytest -q
```

Historical provenance tests are optional:

```text
python -m pytest -q -m provenance
```

Those tests compare against external recovered artifacts and skip when the
artifacts are not present.

## Reproduce Section 4.2

Formal configuration:

```text
configs/paper/direct64_helmholtz.yaml
```

Command:

```text
python scripts/run_direct64_helmholtz_main_text.py
```

Expected scale:

```text
Stage-I e_F: about 1e-14
Stage-I e_2: about 1e-14
Stage-II residual: about 1e-6 to 1e-7 for strong runs
Final PDE L2: about 1e-6 to 1e-5
```

Small numerical differences are expected across BLAS/SciPy builds. Bitwise
hashes are useful provenance diagnostics, not the sole success criterion.

## Reproduce Section 4.3

Formal configuration:

```text
configs/paper/direct64_crosspde.yaml
```

Command:

```text
python scripts/run_parallel_crosspde_twostage.py --phase results/direct64_crosspde_section43 --pdes poisson reaction_diffusion convection_diffusion --workers 3 --feature-count 64 --stage2-family alternating_ry_cnot --stage2-target-p 2400
```

For a single PDE:

```text
python scripts/run_single_crosspde_twostage.py --phase results/direct64_poisson_demo --pdes poisson --workers 1 --feature-count 64 --stage2-family alternating_ry_cnot --stage2-target-p 2400
```

Expected cross-PDE scale:

```text
Stage-I e_F: about 1e-14
Stage-I e_2: about 1e-14
Stage-II residual: about 1e-6 to 1e-5, depending on family/capacity
Final PDE L2: about 1e-6 to 1e-5
```

## Output Directory Structure

Typical outputs are:

```text
results/<phase>/
  <pde>/
    <pde>/
      stage1_block11_M143/
        result.json
        Ahat.npy
        training_history.csv
        stage1_checkpoint_latest.npz
      stage2_learned_Ahat/
        <family>/P<target>/
          RESULT.json
          final_theta.npy
          trajectory.csv
          progress_status.json
```

Generated outputs are ignored by Git.

## Random Seeds

```text
feature seed: 7061015
feature alpha: 50
Stage-I theta0 seed: 2026072202
Stage-I theta0 scale: 0.02
Stage-II theta0 seed: 13379
Stage-II theta0 scale: 0.70
```

The cross-PDE Direct64 study uses the same 64 sine-feature realization and
collocation points across Poisson, reaction-diffusion, and
convection-diffusion.

## Optional Historical Provenance

Historical 64x200 -> 64x64 reproduction utilities live under
`configs/historical/` and `archive/historical_reproduction/`. They are retained
for provenance only and are not the current main-text Direct64 workflow.

Dense projected-block surrogate debugging code is archived under
`archive/internal_diagnostics/` and is not used in the paper experiments.
