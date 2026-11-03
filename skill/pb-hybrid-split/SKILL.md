---
name: pb-hybrid-split
description: "按用户给定的 PS/PL 或 CPU/DPU 分工切分 TensorFlow 冻结 PB，检查张量端口、跨段依赖和数值等价，交付独立子图与运行接口；适用于模型分段部署，不用于仅改量化精度或直接切分编译后的 SO。"
---

# PB 分段与异构部署

先读 [README.md](README.md) 确认辅助脚本能力；本技能处理数值型、无状态的 frozen GraphDef。用户要求只给方案时先给具体切分方案；要求执行时完成导出与验证，不停在建议。

## 确定边界

1. 确认原始 FP32 PB、真实输入/输出张量、预处理、样本、目标设备和工具版本。路径以当前 PC/WSL/容器中的实际文件为准，不按目录名字推断版本或补删 `driver`。
2. 明确用户要求保留哪些 DPU、PS 阶段；优化/切分子段时仍保留其指定的完整链路。缺少关键边界或样本时询问最小缺项，同时继续只读图检查。
3. 用 `scripts/pb_partition.py inspect` 收集操作、形状与端口。根据实际版本的算子支持和编译结果决定 DPU 候选，不把“量化完成”当作“能编译”，不通过删除 Const/Transpose 掩盖图语义。
4. 先列接口表：每段设备、每个输入/输出的完整 `node:port`、shape、dtype、layout、生产者和消费者。残差/分支跨边界时显式传入多个张量，不强行压成单输入。

## 实施并证明

- 按 [references/workflow.md](references/workflow.md) 生成计划；用 `pb_partition.py split` 保存独立 PB、接口和哈希。辅助脚本保留控制边并按端口重写数据边；检测到未声明的存活输入时先修正计划。
- 用相同预处理后的 NPZ 输入运行原图及**重新读取的**子图链，验证全部边界和最终输出。保存容差、样本清单及哈希；只对明确分类输出报告类别一致率，不以 argmax 一致代替张量等价。
- 板端没有 TensorFlow 时，先确认实际 CPU 后端。可选的 `export_numpy.py` / `numpy_runtime.py` 只支持单输入/输出 FP32 及其有限算子集合；不支持的运算必须实现并验证或改用用户允许的后端，不能跳过。
- 切分产物、JSON/NPZ、调度代码、量化回调和配置作为配套整体交付。中间 DPU 的校准输入来自实际前缀的中间特征，不重新按 RGB 归一化。
- 将 PC 图变换、Vitis 量化/编译、Runtime 链接、板端执行分工写清，按用户的环境约束安排。未实际执行的层级标为待验证。

## 部署适配

DNNDK/VART 等适配按实际运行库生成，节点名以编译日志和运行时查询为准。形状相同不证明通道布局相同；FP32 取回 API 已反量化时不再除输出尺度。

面向 PYNQ 时读 [references/pynq.md](references/pynq.md)，使用参数化接口契约和 `scripts/tensor_contract.py`。Overlay、DMA、缓存一致性和 TLAST 与具体硬件契约对齐；本包没有自动编程 PL 的脚本。

遇到问题读 [references/pitfalls.md](references/pitfalls.md)。历史效果、失败来源和证明边界见 [references/case-evidence.md](references/case-evidence.md)，案例节点与尺寸不能作为别的模型的默认值。

完成时交付：可执行命令、PB/可选 JSON/NPZ、接口/哈希清单、黄金参考验证、实际后端与未完成层级、切分开销及失效条件。不要声称主机 FP32 通过就等于板端 INT8 精度或速度通过。
