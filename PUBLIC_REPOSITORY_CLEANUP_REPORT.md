# Public Repository Cleanup Report

## Scope

This cleanup prepared the upload package as a public Direct64 two-stage RFM
reproduction repository. It did not change the scientific Direct64 generator,
Stage-I circuit mathematics, Stage-II ansatz mathematics, or production
experiment selection.

## Files Changed

- `README.md`: rewritten around the current Direct64 main-text workflow and
  fresh Stage-II `direct_r` protocol.
- `REPRODUCIBILITY.md`: added installation, tests, Section 4.2/4.3 commands,
  random seeds, output layout, and expected metric scales.
- `LICENSE_PLACEHOLDER.md`: added because no license has been selected.
- `.gitignore`: expanded to exclude generated results, checkpoints, caches,
  build artifacts, IDE files, logs, and array/checkpoint data.
- `pyproject.toml`: made pytest configuration authoritative and added the
  `provenance` marker.
- `pytest.ini`: removed obsolete `python_paths = src` configuration.
- `refined_twostage/`: removed the top-level shim; public usage is standard
  `src/` layout with `pip install -e .`.
- `configs/`: split into `paper/`, `diagnostics/`, and `historical/`.
- `scripts/README.md`: added script classification.
- `archive/internal_diagnostics/`: moved internal diagnostics and surrogate-era
  validation scripts here.
- `archive/historical_reproduction/`: moved historical 64x200 reproduction
  helper scripts here.
- `tests/provenance/`: moved tests that require external historical artifacts
  and made them skip cleanly when those artifacts are unavailable.
- `src/refined_twostage/cli.py`: removed a personal absolute historical path,
  added a formal fresh `direct_r` Stage-II path, and made smoke Stage-I/Stage-II
  commands construct Direct64 systems directly.
- `src/refined_twostage/stage1/families.py`: removed the dense projected-block
  surrogate class from the public production API.

## Rationale

- Default pytest must be self-contained in a fresh clone.
- The public README should describe the current Direct64 paper experiments, not
  the historical 64x200 -> 64x64 reproduction path.
- Formal Stage II is fresh `direct_r` optimization, with no `direct_r2` warm
  start and no historical theta warm start.
- Dense projected-block surrogate code had internal diagnostic value, but it is
  not part of the paper experiments or public production API.
- Historical code remains available under `archive/` instead of being deleted.

## Moved To Historical/Archive

- `scripts/validate_stage1_training.py` -> `archive/internal_diagnostics/validate_stage1_training.py`
- `scripts/write_partial_continuation_report.py` -> `archive/internal_diagnostics/write_partial_continuation_report.py`
- `scripts/continue_direct64_campaign.py` -> `archive/internal_diagnostics/continue_direct64_campaign.py`
- `scripts/compare_direct64.py` -> `archive/historical_reproduction/compare_direct64.py`
- `scripts/reproduce_crosspde_data.py` -> `archive/historical_reproduction/reproduce_crosspde_data.py`
- `configs/reproduce_crosspde_20260819.yaml` -> `configs/historical/reproduce_crosspde_20260819.yaml`
- `configs/stage2_alternating_r2_to_r.yaml` -> `configs/historical/stage2_alternating_r2_to_r.yaml`
- `configs/direct64_same_generator.yaml` -> `configs/diagnostics/direct64_same_generator.yaml`

## Provenance-Only Tests

- `tests/provenance/test_migration_recovered_real.py`
- `tests/provenance/test_helmholtz_pdespec.py`

These tests require the external recovered 2026-08-19 repository or frozen
Direct64 campaign artifacts. They are marked `@pytest.mark.provenance` and skip
when those artifacts are absent. They are retained because they document
migration and historical provenance.

## Pytest Status

Before cleanup:

```text
python -m pip install -e .
success

python -m pytest -q
5 failed, 26 passed, 18 warnings
```

The failures were caused by missing external historical artifacts:

- `results/direct64_full_twostage_campaign_20260820_204937/...`
- `../recovered_20260819_twostage_framework/...`

After cleanup:

```text
python -m pip install -e .
success

python -m pytest -q
26 passed, 5 skipped, 18 warnings
```

The remaining warnings are third-party SciPy/NumPy deprecation warnings from
the local environment. The previous pytest configuration warning is gone.

## README Changes

The README now foregrounds:

- Direct64 numerical system with 64 collocation rows and 64 sine features.
- Stage-I real projected-block compilation.
- Frozen learned operator.
- Stage-II fresh `direct_r` L-BFGS-B optimization.
- Section 4.2 and Section 4.3 reproduction commands.
- Historical 64x200 utilities as provenance only.
- Dense surrogate debugging as archive only.

## Formal Reproduction Commands

Section 4.2:

```text
python scripts/run_direct64_helmholtz_main_text.py
```

Section 4.3:

```text
python scripts/run_parallel_crosspde_twostage.py --phase results/direct64_crosspde_section43 --pdes poisson reaction_diffusion convection_diffusion --workers 3 --feature-count 64 --stage2-family alternating_ry_cnot --stage2-target-p 2400
```

## Formal Stage-II Protocol

```text
objective: direct_r
initialization: fresh
optimizer: L-BFGS-B
theta0 seed: 13379
theta0 scale: 0.70
warm_start: false
```

The `direct_r2` and `r2_then_r` implementations remain available for optional
diagnostics and historical provenance, but are not formal paper configs.

## Surrogate Status

The dense projected-block surrogate is no longer exposed from the public
production Stage-I API. Historical surrogate/debugging material is retained in
`archive/internal_diagnostics/` with a README stating that it is not used in the
paper experiments.

## Remaining Author Decision

- `LICENSE`: no license has been selected. `LICENSE_PLACEHOLDER.md` was added.
  The authors must choose and approve a license such as MIT, BSD-3-Clause, or
  Apache-2.0 before publication.

No blocker was found that would require changing the paper's scientific
implementation.

## Git Diff Stat

Generated with `git diff --no-index --stat` against the pre-cleanup zip:

```text
33 files changed, 468 insertions(+), 168 deletions(-)
```

High-level stat:

```text
.gitignore                                      |   8 +
LICENSE_PLACEHOLDER.md                          |   8 +
README.md                                       | 178 +++++++++++++--------
REPRODUCIBILITY.md                              | 134 ++++++++++++++++
archive/historical_reproduction/README.md       |   8 +
archive/internal_diagnostics/README.md          |   9 ++
configs/README.md                               |   8 +
configs/paper/direct64_crosspde.yaml            |  12 ++
configs/paper/direct64_helmholtz.yaml           |  26 +++
configs/paper/direct64_stage2_fresh_direct_r.yaml | 30 ++++
pyproject.toml                                  |   5 +-
pytest.ini                                      |   3 -
refined_twostage/__init__.py                    |  11 --
scripts/README.md                               |  11 ++
src/refined_twostage/cli.py                     |  77 ++++++---
src/refined_twostage/stage1/families.py         |  44 +----
tests/provenance/*                              |  26 +--
```

## Suggested First Public Commit Message

```text
Prepare public Direct64 two-stage RFM reproduction repository
```

## Suggested Public File Tree

```text
README.md
REPRODUCIBILITY.md
REPORTING_STANDARD.md
LICENSE_PLACEHOLDER.md
pyproject.toml
.gitignore
src/
configs/
  paper/
  diagnostics/
  historical/
scripts/
tests/
  provenance/
archive/
  internal_diagnostics/
  historical_reproduction/
```
