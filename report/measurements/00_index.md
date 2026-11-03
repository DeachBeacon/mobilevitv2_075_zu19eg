# 实测记录索引

本目录按时间/主题记录每次优化的数据与命令。所有板端数字来自 `summary.json`，
PC 数字只作参考。

| 编号 | 文件 | 内容 |
| --- | --- | --- |
| 00 | 本文件 | 索引 |
| 01 | `01_baseline_and_numpy_blas.md` | 19 s 的根因（无 BLAS）、NumPy 1.19.5+OpenBLAS 安装与验证 |
| 02 | `02_split_analysis.md` | 各块耗时结构、为什么暂不再切分、可选方案 |
| 03 | `03_tail_benchmark.md` | 板端后段 606.6 ms、分块/分算子数据、`--fast-tail` |
| 04 | `04_end_to_end_analysis.md` | 两段端到端 25.5×、准确率逐类分析、后续优化排序 |
| 05 | `05_tail_fusion.md` | GroupNorm/注意力融合原理、公式、参考代码 |
| 06 | `06_fusion_operations.md` | 旧两段包的操作手册（改哪些文件、何时跑） |
| 07 | `07_progress_evidence.md` | 19 s→0.6 s 证据链、当晚报错（pip/DPU/路径） |
| 08 | `08_deploy_best_fusion.md` | 五段部署（0.5557 s）的融合接入、命令、预期 |
| 09 | `09_board_validation.md` | 2026-11-01 实测：0.529 s、84.33%、DPU 输入范围、下一步 |
| 10 | `10_pure_ps_guide.md` | 纯 PS（全 NumPy、FP32）上板步骤与对比口径 |

参考：

- `../model_versions.md`：模型版本配套；
- `../paths.md`：历史路径与脚本限制。

## 关键数字速查

| 阶段 | 板端单张 | 备注 |
| --- | ---: | --- |
| 纯 PS（未切分、未优化） | **33.0 s** | 全 CPU 基线；FP32 91.33%（PC 参考） |
| 切一刀（front DPU，仍无 BLAS） | 19.0 s | 1.7× |
| 两段 + OpenBLAS + 图折叠 | 0.745 s | 44.3× |
| 五段 + 融合 + CLE（最终） | **0.527 s** | 300 张，87.00% |
