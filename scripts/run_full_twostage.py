from refined_twostage.cli import main

raise SystemExit(main(["run-two-stage", "--config", "configs/paper/direct64_crosspde.yaml", "--pde", "poisson"]))
