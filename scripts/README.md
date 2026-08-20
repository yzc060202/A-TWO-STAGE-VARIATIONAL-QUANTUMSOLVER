# Scripts

Paper reproduction and utility entry points:

- `run_direct64_helmholtz_main_text.py`: Section 4.2 Direct64 Helmholtz architecture study.
- `run_parallel_crosspde_twostage.py`: Section 4.3 Direct64 cross-PDE runs, one worker per PDE.
- `run_single_crosspde_twostage.py`: one Direct64 cross-PDE run.
- `continue_stage2_from_theta.py`: utility for Stage-II continuation or additional family sweeps from an existing Stage-I result.
- `run_stage1.py`, `run_stage2.py`, `run_full_twostage.py`: small CLI wrappers using `configs/paper/`.

Historical reproduction and internal diagnostics are retained under `archive/`.
