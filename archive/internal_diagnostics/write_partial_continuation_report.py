"""Write the centralized Chinese partial continuation report."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "direct64_full_twostage_campaign_20260820_204937"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def md_table(rows, fields):
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    stage1_rows = read_csv(CAMPAIGN / "05_helmholtz_stage1" / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.csv")
    stage1_selected = read_json(CAMPAIGN / "05_helmholtz_stage1" / "STAGE1_SELECTION_DECISION.json")["selected"]
    stage2_path = CAMPAIGN / "06_helmholtz_stage2" / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.json"
    stage2_rows = read_json(stage2_path).get("rows", []) if stage2_path.exists() else []

    stage1_fields = [
        "batch",
        "public_name",
        "capacity",
        "actual_P",
        "total_two_qubit_gates",
        "native_depth",
        "e_F",
        "e_2",
        "learned_LS_PDE_L2",
        "learned_LS_residual",
        "runtime_seconds",
    ]
    stage2_fields = [
        "family",
        "target_P",
        "actual_P",
        "layers_or_repetitions",
        "CZ",
        "CNOT",
        "total_two_qubit_gates",
        "native_depth",
        "initial_r",
        "final_r",
        "PDE_L2",
        "runtime_seconds",
        "termination_message",
    ]
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    stage1_table = md_table(stage1_rows, stage1_fields)
    stage2_table = md_table(stage2_rows, stage2_fields) if stage2_rows else "暂无正式完成的 Stage-II 行。"

    report = f"""# Direct64 Full Two-Stage Campaign 阶段性汇总报告

生成时间：{now}

## 结论摘要

本轮继续任务已完成能力补齐、测试门、Helmholtz Stage-I 九组生产训练，并启动 Helmholtz Stage-II formal fresh direct-r 深度扫描。由于 Stage-II 单个 depth 自然收敛耗时约 11-21 分钟，完整五家族加扩展 depth 预计为多小时作业；为避免留下无人监控后台训练，本轮已主动停止未完成的 `vqls_ry_cz/P2400` partial run。已完成的 `P1200` 与 `P1800` 均有正式 `RESULT.json`，可由续跑脚本自动跳过。

当前状态不是完整 campaign 终点；它是一个可复现、可续跑、不中断已完成结果的阶段性检查点。

## 当前 campaign root

`{CAMPAIGN}`

## 冻结数据与合法性

冻结 Direct64 数据未重新生成、未改 feature bank、seed、alpha、collocation、PDE 定义、scaling 或 reconstruction。根据能力完成阶段与交接记录，common feature hash、collocation hash 以及四个 PDE 的 `A_raw/A_p/b_p` hash 验证均为 PASS。

## 能力补齐与测试门

- Helmholtz control 已作为 first-class `PdeSpec` 接入。
- Stage-I 已支持 `gray_multiplexed_ry_cnot`、`vcbe_block11_real`、`vcbe_block15_real`。
- Stage-II 已支持五家族：`vqls_ry_cz`、`alternating_ry_cnot`、`ring_ry_cnot`、`sequential_sg_mps`、`dense_pairwise_ry_cnot`。
- Stage-II formal objective 保持 fresh `direct_r`，没有替换为 `r^2 -> r`。
- 最新 pytest：`21 passed, 18 warnings in 1.10 seconds`（发生在 Stage-II 向量化与断点续跑补丁之后）。

## 本轮额外实现修正

- `src/refined_twostage/stage2/polynomial.py`：RY/CX/CZ 作用改为预计算索引的 NumPy 向量化实现，数学语义不变。
- `scripts/continue_direct64_campaign.py`：Stage-II depth job 如果已有 `RESULT.json` 会自动加载并跳过，支持断点续跑。
- `src/refined_twostage/stage2/polynomial.py`：尝试加入 numba JIT 后发现本机 `numba 0.39.0` 与当前 NumPy 组合会触发运行时错误；因此 JIT 改为 `REFINED_TWOSTAGE_ENABLE_NUMBA=1` 显式 opt-in，默认关闭，测试门恢复 PASS。

## Helmholtz Stage-I 九组结果

{stage1_table}

## Stage-I 选择

选择规则：best residual tier，然后最少 total two-qubit gates、native depth、P、max(e_F,e_2)。

选中：`{stage1_selected["family"]}` / `{stage1_selected["capacity"]}`，actual_P `{stage1_selected["actual_P"]}`，total_two_qubit `{stage1_selected["total_two_qubit_gates"]}`，native_depth `{stage1_selected["native_depth"]}`，e_F `{stage1_selected["e_F"]}`，e_2 `{stage1_selected["e_2"]}`。

选中 Ahat：`{stage1_selected["result_directory"]}\\Ahat.npy`

## Helmholtz Stage-II 已完成正式行

{stage2_table}

说明：`vqls_ry_cz/P2400` 曾启动但未自然终止，本轮已停止；该 partial 不作为正式结果行，不进入选择。

## 尚未完成

- Helmholtz Stage-II：`vqls_ry_cz/P2400` 及后续 `P3000`、必要时 `P4200/P6000`。
- 其他 Stage-II 家族：`alternating_ry_cnot`、`ring_ry_cnot`、`sequential_sg_mps`、`dense_pairwise_ry_cnot` 的 depth ladder。
- Stage-II per-family winner 与 global winner 尚未冻结。
- `07_selected_pipeline/SELECTED_TWOSTAGE_CONFIG.*` 尚未生成，因为 Stage-II global winner 未产生。
- 三个 cross-PDE end-to-end 生产训练尚未启动。
- campaign 终态 `08_cross_pde/*` 尚未生成。

## 可续跑命令

从 framework root 执行：

```powershell
cd <repository-root>
python scripts\\continue_direct64_campaign.py stage2
```

该命令会跳过已有正式 `RESULT.json` 的 `vqls_ry_cz/P1200` 与 `P1800`，从后续未完成 depth 继续。当前脚本仍按 formal `fresh_direct_r_lbfgsb_refined_v2` 配置运行：seed `13379`、theta0 `Normal(0,0.70^2)`、L-BFGS-B、`maxiter=30000`、`maxfun=60000`、`maxcor=100`、`maxls=150`、`ftol=1e-16`、`gtol=1e-12`。

## 关键产物路径

- 能力报告：`{CAMPAIGN}\\04_capability_completion\\CAPABILITY_COMPLETION_REPORT.md`
- 迁移来源报告：`{CAMPAIGN}\\04_capability_completion\\CIRCUIT_MIGRATION_PROVENANCE.md`
- Stage-I 汇总：`{CAMPAIGN}\\05_helmholtz_stage1\\HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.csv`
- Stage-I 选择：`{CAMPAIGN}\\05_helmholtz_stage1\\STAGE1_SELECTION_DECISION.json`
- Stage-II 已完成汇总：`{CAMPAIGN}\\06_helmholtz_stage2\\HELMHOLTZ_STAGE2_DEPTH_RESULTS.csv`
- Stage-II P1200：`{CAMPAIGN}\\06_helmholtz_stage2\\vqls_ry_cz\\P1200\\RESULT.json`
- Stage-II P1800：`{CAMPAIGN}\\06_helmholtz_stage2\\vqls_ry_cz\\P1800\\RESULT.json`

## 当前可声明与不可声明

可以声明：冻结数据未改；能力门 PASS；Stage-I 九组完成并选中 Block11 P5005；Stage-II formal direct-r 已完成 VQLS P1200/P1800 两行，目标值均与显式残差一致。

不能声明：Stage-II 家族选择完成；全两阶段最终配置冻结；Poisson/reaction/convection 的 end-to-end 生产结果完成；完整 Master-v2 campaign 完成。
"""

    status = {
        "generated_at": now,
        "campaign_root": str(CAMPAIGN),
        "frozen_hash_status": "PASS from capability/audit records; data not regenerated in this turn",
        "capability_completion": "PASS",
        "latest_pytest": "21 passed, 18 warnings in 1.10 seconds",
        "stage1_selected": stage1_selected,
        "stage2_completed_rows": stage2_rows,
        "incomplete": [
            "helmholtz_stage2_remaining_depths_and_families",
            "selected_pipeline",
            "cross_pde_production",
            "final_campaign_reports",
        ],
        "stopped_background_training": "vqls_ry_cz/P2400 partial stopped; no RESULT.json counted",
        "resume_command": "python scripts\\continue_direct64_campaign.py stage2",
    }

    (CAMPAIGN / "PARTIAL_CONTINUATION_STATUS_REPORT.md").write_text(report, encoding="utf-8")
    (CAMPAIGN / "PARTIAL_CONTINUATION_STATUS_REPORT.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (CAMPAIGN / "EXPERIMENT_RESULT_REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "EXPERIMENT_RESULT_REPORT.md").write_text(report, encoding="utf-8")
    print(CAMPAIGN / "PARTIAL_CONTINUATION_STATUS_REPORT.md")
    print(CAMPAIGN / "PARTIAL_CONTINUATION_STATUS_REPORT.json")


if __name__ == "__main__":
    main()
