# 大模型协作记录索引

> 历史资料：保留原测量口径与过程描述。当前提交版的目录、模型选择和操作命令以根目录 README 为准；旧 ONNX 分支和旧 MV2_4 对照包不随本次提交。
## 原模型部署与技能沉淀

原始聊天：[拆分中间MV2并部署到DPU](codex://threads/01a19204-10ca-7fb0-bf71-ba6e78ae509d)。这是原始协作记录入口，本文件是索引与概要，不是完整聊天导出。

协作围绕将后端两个 MV2 拆出到 DPU、保留 Transformer 的 NumPy 执行、检查量化输入范围与分段接口、使用 BN/CLE/偏置校正优化精度展开。PC 转换、Docker 量化/编译、Runtime 链接分开执行；最终流程固化在 `build/mobilevitv2_axu19eg_release/`。分段与 CLE 的适用条件、失败经验、校验脚本和历史验证记录沉淀在根目录 `skill/`，详见 `skill/sources.json` 与两个技能的 references。

## 2026-11-03 工程整理

用户指定：上板使用队友 `deploy_best_fused`，PT→H5→PB→SO 使用 `mobilevitv2_axu19eg_release`。用户确认本轮只整理原 src 内容，按用途归入工作区根目录；既有历史目录不动。

本轮完成文件盘点、dupscan 字节重复扫描、依赖审查、可逆搬移、路径修订、README 更新及主机重建检查。正式上板代码和三个 SO 保持原样；转换组装改为引用这一份队友运行代码。没有执行真实板端测试，历史 261/300 结果保留原始 JSON。

具体证据见整理记录、`ORGANIZATION_validation.json`、构建 `local_validation.json` 与 backup 的逐文件搬移表。未虚构综合报告、PYNQ 实板验证或新的板端性能。
