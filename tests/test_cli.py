from refined_twostage.cli import main


def test_cli_nonzero_for_missing_config():
    assert main(["reproduce-data", "--config", "configs/does_not_exist.yaml"]) != 0
