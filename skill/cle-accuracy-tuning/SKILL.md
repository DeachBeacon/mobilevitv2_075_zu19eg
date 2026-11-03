---
name: cle-accuracy-tuning
description: "从普通 DPU SO 对应的原始浮点 PB 出发，验证 BN 折叠、CLE、限幅与偏置校正候选，经真实量化评估后重新编译链接可追溯 SO；适用于量化精度优化，不能仅凭 SO 二进制恢复原始模型，也不保证 CLE 必然提高精度。"
---

# 从模型重建精度优化 SO

先读 [README.md](README.md)。SO 是基线和最终产物，优化作用于浮点图及量化过程。只有 SO 时先收集原始 PB、量化/编译配置和样本；可以检查 ELF/内嵌 kernel，但不能承诺从二进制无损恢复 FP32 权重。

## 固定实验口径

1. 确认用户希望优化的段、必须保留的 DPU/PS 段、实际运行库、数据/预处理和验收指标。用户说“两个 MV2 保持 DPU”时，固定它们的真实量化 PB、SO、配置和 PS 实现，按完整混合链路选型。
2. 为原始 FP32、现用 INT8、候选建立独立目录和文件哈希；核对量化 PB ↔ ELF ↔ SO ↔ runtime 配置的来源。不要根据 `_cle`、`_bn` 或文件名判断权重/精度。
3. 先复算同一数据、同一预处理和同一后端下的基线。用相同捕获输入逐段比较原生量化与板端输出，分开解码、PS 浮点、量化、运行时舍入差异；旧实验百分比只能作为有来源的观察。

## 逐步搜索

- 按 [references/workflow.md](references/workflow.md) 操作。只对受支持的冻结 CNN 子图应用 BN/CLE；动态 GN/LN、注意力、ReLU6/门控等不满足本辅助算法的条件时，先切分或编写有等价证明的专用变换。
- 用 `scripts/cle_graph.py` 生成 BN-only、CLE、限幅候选。输入/输出通道尺度固定为 1；残差支路共享缩放组；无安全自由度的组保持固定。参数化缩放幅度，并同时观察偏置、激活及权重量化范围。
- 每个基础候选先通过保存 PB 的 FP32 等价检查。BN-only 是对照而非精度更高的默认答案。
- 在正确工具环境用真实 FixNeuron/目标运行时评估，读取**最终量化器之后**的输出。不能把 FixNeuron 改成 Identity 或用 `np.round` 近似结果来宣称 INT8 精度。
- 对逐层/输出通道误差有偏的候选，再做少量偏置校正。已有最终偏置时同步 eval/deploy 且由原生量化器计算新偏置；最终卷积无偏置时优先添加浮点补偿后重新量化，不能猜 bpos。补偿后的量化模型不再声称与原浮点模型等价。
- 中间段用实际部署前缀产生的特征校准。候选前端改变后要检查两个固定 DPU 的输入范围；用户要求它们固定时，对越界候选排除或征求改变范围的明确指示，不自行修改这些后端模型。

## 量化、选择、交付

Vitis AI 1.x/DNNDK 的元数据问题读 [references/legacy-vitis.md](references/legacy-vitis.md)。`scripts/quant_boundary.py` 可只读核对输入 FixNeuron、Placeholder、输入 Pad 和首层卷积；修改时输出新 PB，随后重新原生评估、编译、链接并使旧缓存失效。案例位置 5/0 不是通用参数。

用统一报告契约记录完整链路、数据与预处理哈希、固定后端哈希、真实后端、范围策略和逐样本结果；`scripts/rank_candidates.py` 可拒绝口径混用、算术不一致及旧文件，并在并列时保留基线。`front_only_accuracy` 等子段分数只在用户明确选择该口径时作为主指标。

只编译验收候选，确认目标架构、实际节点和 CPU/DPU kernel 分工，链接匹配架构的 SO；完整更新配置及配套 JSON/NPZ/运行脚本。优化过的其他段逐文件保持不变。交付已有授权范围内的结果，不额外设置每一步都询问的审批流程。

保存本机 FP32、Docker 原生、板端与独立集的验证层级。没有真实结果时标“待验证”；无候选优于基线时保留基线并报告失败观察。使用 [assets/report-template.md](assets/report-template.md) 写结论，历史来源读 [references/case-evidence.md](references/case-evidence.md)。
