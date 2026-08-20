# Direct64 Two-Stage RFM Reproduction Framework

This repository contains the public reproduction code for the paper's Direct64
two-stage random-feature-method experiments. The main workflow is:

```text
RFM Direct64 numerical system
        |
        v
Stage-I projected-block compilation
        |
        v
freeze learned operator
        |
        v
Stage-II optimization
        |
        v
reconstructed PDE solution
```

The scientific production path uses real quantum-circuit projected-block
families for Stage I and real polynomial ansatz circuits for Stage II. It does
not use dense projected-block surrogates, r^2 warm starts, historical theta
warm starts, or SVD/right-working-space reductions for the current main-text
Direct64 experiments.

## Method Overview

Each 1D PDE is represented by a random feature method system

```text
A_raw c_raw ~= b_raw
```

with 64 collocation rows and 64 random sine features:

```text
phi_j(x) = sin(alpha * (x*w_j + b_j))
w_j ~ Uniform[-1,1], all w drawn first
b_j ~ Uniform[-1,1], all b drawn second
seed = 7061015
alpha = 50
collocation = np.linspace(-1, 1, 64)
row 0 = left Dirichlet, rows 1..62 = PDE residual, row 63 = right Dirichlet
```

Preprocessing is

```text
Dr = I
Dc_j = 1 / max(||A_raw[:,j]||_2, 1e-300)
A_p = Dr A_raw Dc
b_p = Dr b_raw
```

For Direct64, `A_raw`, `A_p`, and `A64` are all `64 x 64`; `A64=A_p`.
There is no SVD, no nontrivial `V64`, and no classical right-working-space
reduction.

## Stage I

Stage I compiles the scaled projected block

```text
K_theta ~= A64 / s_A
s_A = 1.05 * ||A64||_2
L_I = ||K_theta - A64/s_A||_F^2 / ||A64/s_A||_F^2
```

The paper Direct64 cross-PDE line uses `vcbe_block11_real` with `M=143`,
`P=6013`, and real Ry/CNOT projected-block semantics. The formal optimizer is
full-basis Adam with seed `2026072202`, initialization scale `0.02`, and 1000
production steps.

## Stage II

Stage II uses a fresh initialization and optimizes the true direct residual:

```text
v = Ahat y(omega)
alpha* = (b^T v) / (v^T v)
r = ||alpha* v - b|| / ||b||
```

The formal protocol is:

```text
objective: direct_r
initialization: fresh
optimizer: L-BFGS-B
theta0 seed: 13379
theta0 scale: 0.70
warm_start: false
```

The implementation still includes `direct_r2` and an optional historical
`r2_then_r` optimizer for diagnostics/provenance, but those are not the formal
main-text Direct64 Stage-II protocol.

## Installation

Use an editable install from a fresh clone:

```text
python -m pip install -e .
```

The package depends on NumPy, SciPy, and PyYAML. The code uses a standard
`src/` layout; the old top-level import shim has been removed.

## Tests

Run the self-contained test suite:

```text
python -m pytest -q
```

Optional historical provenance tests are marked with `provenance`. They skip
automatically if the external recovered repository or frozen campaign artifacts
are not present:

```text
python -m pytest -q -m provenance
```

## Reproduce Section 4.2

The Helmholtz Direct64 architecture study is driven by:

```text
python scripts/run_direct64_helmholtz_main_text.py
```

The corresponding formal configuration is:

```text
configs/paper/direct64_helmholtz.yaml
```

## Reproduce Section 4.3

Run the three Direct64 cross-PDE problems in parallel:

```text
python scripts/run_parallel_crosspde_twostage.py --phase results/direct64_crosspde_section43 --pdes poisson reaction_diffusion convection_diffusion --workers 3 --feature-count 64 --stage2-family alternating_ry_cnot --stage2-target-p 2400
```

Run one PDE:

```text
python scripts/run_single_crosspde_twostage.py --phase results/direct64_poisson_demo --pdes poisson --workers 1 --feature-count 64 --stage2-family alternating_ry_cnot --stage2-target-p 2400
```

Try another Stage-II family from a completed Stage-I directory:

```text
python scripts/continue_stage2_from_theta.py --phase results/direct64_crosspde_section43/poisson --pde poisson --out-dir results/direct64_crosspde_section43/poisson/poisson/stage2_learned_Ahat/dense_pairwise_ry_cnot/P2400 --family dense_pairwise_ry_cnot --target-p 2400 --feature-count 64 --history-interval 25
```

Useful Stage-II families are:

```text
vqls_ry_cz
alternating_ry_cnot
ring_ry_cnot
sequential_sg_mps
dense_pairwise_ry_cnot
```

## Repository Layout

```text
src/refined_twostage/        library code
configs/paper/               formal paper configurations
configs/diagnostics/         optional diagnostic configurations
configs/historical/          historical reproduction configurations
scripts/                     paper reproduction and utility scripts
tests/                       self-contained unit and integration tests
tests/provenance/            optional historical artifact checks
archive/                     retained internal/historical utilities
```

## Outputs

Generated outputs are written under `results/` and are intentionally ignored by
Git. They include arrays, checkpoints, optimizer trajectories, JSON/CSV tables,
and Markdown reports needed for local reproduction.

## Historical Provenance

Historical 64x200 -> 64x64 utilities are retained in `configs/historical/` and
`archive/historical_reproduction/` for provenance. They are not part of the
current main-text Direct64 experiments.

Dense projected-block surrogate debugging code is retained only in
`archive/internal_diagnostics/`. It is not used in the paper experiments.

## License

No license has been selected yet. See `LICENSE_PLACEHOLDER.md`.
