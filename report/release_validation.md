# 本次新副本的验证范围

本工程基于原 `submission` 副本整理，原 `src` 和 `submission` 工程保留不动。命名仅保留 `README.md`、`SKILL.md` 两类约定入口，其余名称全部为小写英文、数字、下划线、连字符及扩展名所需的点；隐藏配置文件的前导点保留。

## 当前验证与证据入口

| 环节 | 验证方式 / 记录 |
|---|---|
| 主机 PT→H5→PB→六段模型导出与预存 SO 组装 | `../build/mobilevitv2_axu19eg_release/audit/validate_local.py`；本次结果 `../build/mobilevitv2_axu19eg_release/local_validation.json` |
| 三个最终 kernel 的日志、ELF、SO 对应关系 | `../build/audit_evidence.py`；`../build/reports/evidence_validation.json` |
| 全包文件、命名、相对路径、Python/Bash 语法、技能包哈希 | `../build/verify_submission.py --shell`；`submission_validation.json` |
| 纯 PS 六段数值冒烟 | `ps_reference_smoke.json`；这是主机合成输入，不是板端性能或精度测试 |
| 历史最终板端输出 | `../data/results/final_20261103/visual_summary.json`，261/300、87.00%、逐图平均 526.4251 ms |
| 硬件配置与模型侧实现说明 | `../build/implementation_report.md` |

本次没有重新执行真实 Vitis 量化、编译、交叉链接或 DPU 板端推理，没有执行板端显示界面。保留的编译日志、ELF、SO 和板端结果属于历史真实产物，不能称作新环境全链路复现。

## 仍待完成

1. 在匹配的 Vitis AI 1.2/Runtime 环境完成量化、编译、链接，并保存本次新日志和产物。
2. 在实际板上核对镜像、DPU 配置、时钟、运行库及 BLAS，运行完整 300 图并保留结果与输入范围报告。
3. 索取与该镜像对应的整设计综合/实现资源报告和时序报告，或取得赛事方允许固定平台替代提交的明确说明。
4. 若要报告泛化精度，另做独立测试集；现有校准与测试共享图片。

无新测量之前，原始结果只作为历史基准，不覆盖或冒充本次重新测量。
