# Default Experiment Reporting Standard

以后每次实验完成，默认写集中式 Markdown 结果报告，优先命名为 `EXPERIMENT_RESULT_REPORT.md`。

报告必须包含：

1. 任务目标、实验范围、明确未做的事情。
2. 数据来源、生成器、seed、collocation、scaling、working-space、reconstruction。
3. 泄漏/筛选/调参/正则化/fallback 审计。
4. 主要数值表：rank、condition number、effective condition number、classical residual、PDE L2/Linf、dense residual、boundary error。
5. 与历史 fingerprint 或 baseline 的差异和原因。
6. 测试命令、退出码、通过/失败摘要。
7. 结论、风险、不能过度声称的点。
8. 结果根、报告、数组、manifest、CLI 命令路径。

如果用户没有另行指定，报告使用中文，并在框架顶层保留一份统一总报告。
