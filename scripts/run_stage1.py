from refined_twostage.cli import main

raise SystemExit(main(["run-stage1", "--config", "configs/paper/stage1_block11_direct64.yaml", "--pde", "poisson"]))
