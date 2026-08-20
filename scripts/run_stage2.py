from refined_twostage.cli import main

raise SystemExit(main(["run-stage2", "--config", "configs/paper/direct64_stage2_fresh_direct_r.yaml", "--pde", "poisson"]))
