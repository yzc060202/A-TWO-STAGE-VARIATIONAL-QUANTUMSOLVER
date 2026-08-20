from refined_twostage.stage1.block11 import block11_resource_counts
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.families import nearest_stage1_capacity
from refined_twostage.stage2.alternating_ry_cnot import AlternatingRyCnotAnsatz
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.polynomial import nearest_legal_capacity


def test_block11_resource_formulas():
    r119 = block11_resource_counts(119)
    r143 = block11_resource_counts(143)
    assert r119.parameters == 5005
    assert r119.cnot_gates == 2499
    assert r143.parameters == 6013
    assert r143.cnot_gates == 3003
    assert r143.native_depth == 2003


def test_alternating_resource_formulas():
    ansatz = AlternatingRyCnotAnsatz(layers=399)
    r = ansatz.resource_counts()
    assert r.parameters == 2400
    assert r.cnot_count == 998
    assert r.native_depth == 799


def test_stage1_campaign_resource_formulas():
    cases = [
        ("gray_multiplexed_ry_cnot", 4096, 4096, 4096, 0),
        ("gray_multiplexed_ry_cnot", 5120, 5120, 5120, 0),
        ("gray_multiplexed_ry_cnot", 6144, 6144, 6144, 0),
        ("vcbe_block11_real", 3997, 3997, 1995, 0),
        ("vcbe_block11_real", 5005, 5005, 2499, 0),
        ("vcbe_block11_real", 6013, 6013, 3003, 0),
        ("vcbe_block15_real", 3997, 3997, 1995, 0),
        ("vcbe_block15_real", 5005, 5005, 2499, 0),
        ("vcbe_block15_real", 5999, 5999, 2996, 0),
    ]
    for family, target, params, cnot, cz in cases:
        ansatz = make_stage1_ansatz(family, target_parameters=target)
        r = ansatz.resource_counts()
        assert ansatz.num_parameters == params
        assert r.cnot_gates == cnot
        assert r.cz_gates == cz
        assert r.total_two_qubit_gates == cnot + cz
        assert r.native_depth > 0


def test_stage2_capacity_mapping_and_resources():
    targets = [1200, 1800, 2400, 3000, 4200, 6000]
    families = ["vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot", "sequential_sg_mps", "dense_pairwise_ry_cnot"]
    for family in families:
        for target in targets:
            row = nearest_legal_capacity(family, target)
            ansatz = make_stage2_ansatz(family, layers=row["layers_or_repetitions"])
            r = ansatz.resource_counts()
            assert ansatz.num_parameters == row["actual_P"]
            assert r.two_qubit_gate_count == r.cnot_count + r.cz_count
            assert r.native_depth > 0
    assert nearest_legal_capacity("sequential_sg_mps", 1800)["actual_P"] == 1800
    assert nearest_legal_capacity("dense_pairwise_ry_cnot", 1800)["actual_P"] == 1800
