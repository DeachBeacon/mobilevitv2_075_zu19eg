# report：设计报告、协作记录与实测数据

> 历史资料：保留原测量口径与过程描述。当前提交版的目录、模型选择和操作命令以根目录 README 为准；旧 ONNX 分支和旧 MV2_4 对照包不随本次提交。
| 文件 | 内容 |
| --- | --- |
| `design_report.md` | 设计报告（背景、原理、软硬件划分、优化过程、结果、复现） |
| `llm_collaboration.md` | 大模型协作记录（提示词类别、纠错轨迹、工作流、未采纳方向） |
| `visualization_system.md` | 板端 miniDP 1080P 实时可视化系统（队友 2）：渲染引擎、看板、SOP、实测 87.00% / 525.92 ms / ~1.90 FPS |
| `software_system.md` | 软件系统报告正文（1.1/1.2/1.3/2.3.1/2.3.2，含 9 幅配图）：PC→Docker→板端全链路与函数级说明 |
| `software_system_readme.md` | 软件系统报告包的说明（配图、字数、口径） |
| `software_system_sources.md` | 软件系统报告的核对依据（对话、代码文件、数据来源） |
| `software_system_check.json` | 字数与文件检查结果 |
| `figures/` | 软件系统 9 幅配图（PNG + SVG）与配图总览 |
| `build_report_assets.py` | 生成/检查报告配图的脚本 |
| `measurements/` | 10 篇分阶段实测记录（含全部数据表与命令） |
| `model_versions.md` | 模型版本与配套关系（PT/TF/ONNX/PB/ELF/SO） |
| `paths.md` | 原脚本路径限制与新旧位置对照 |
| `upload_checklist.md` | 按官方上传要求逐项对照（视频/报告/代码 RAR/授权书/照片/海报） |

## 本次提交版本

- 先读根目录 README.md 和 [本次验证范围](release_validation.md)。
- [实现与硬件证据报告](../build/implementation_report.md)列出固定平台配置、三段真实编译证据及尚缺的整设计资源/时序报告。
- [本次改动记录](release_changes.json)包含命名迁移和新增证据的来源与哈希。
- 最终原始逐图结果以 `data/results/final_20261103/visual_summary.json` 为准：261/300、87.00%、平均 526.4251 ms。下列分阶段资料保留原口径，不代表本次重新测量。

## 阅读顺序

1. `measurements/00_index.md`：总索引；
2. `measurements/01_baseline_and_numpy_blas.md`：为什么 19 s、怎么修 BLAS；
3. `measurements/03_tail_benchmark.md`、`05_tail_fusion.md`：后段优化；
4. `measurements/08_deploy_best_fusion.md`、`09_board_validation.md`：最终五段与板端实测；
5. `measurements/10_pure_ps_guide.md`：纯 PS 对照。

## 数据口径

- 板端数字来自 `summary.json` 的 `mean_seconds`（300 张、warmup 1）；
- PC 数字只作量级参考，不能直接当板端结论；
- INT8 与 FP32 应在相同样本、预处理和标签下对比并明确数值精度；不能把 PC 浮点与板端 INT8 耗时混作同一测量。
