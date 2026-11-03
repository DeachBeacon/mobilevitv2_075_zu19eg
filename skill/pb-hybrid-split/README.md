# pb-hybrid-split：按需求拆分 PB

这是可交给 Agent 的工作流，也提供可直接运行的图切分与校验工具。入口为 [SKILL.md](SKILL.md)。脚本不依赖本队工程，不包含模型权重、特定板卡路径或必须使用的切点。

## 适用范围与失效条件

| 项目 | 能力与边界 |
|---|---|
| 输入 | TensorFlow frozen GraphDef `.pb`，数值输入，无变量/状态；按完整张量名确定边界 |
| 切分 | 多段、多输入/输出、非零输出端口；显式保留跳连；PS/DPU 标签来自用户计划 |
| 校验 | 保存 PB 重新读取，原图/分段逐边界与最终输出比较，记录形状、dtype、误差和哈希 |
| 可选 NumPy | 单输入/输出 FP32、静态接口；算子范围见 `scripts/numpy_export_core.py`，未知算子直接失败 |
| 不直接适用 | 只有 SO、训练/有状态图、未冻结变量、依赖 TF2 专用算子、无法提供正确输入样本 |
| 硬件边界 | 不自动证明 DPU 支持，不生成任意 CPU 算子的实现，不保证减少往返后会更快 |

PC 工具已使用 Python 3.6 / TF 1.15.2 / NumPy 1.19.5 检查。NumPy 后端沿用历史 NumPy 1.17 已有 API，不导入 TensorFlow；板卡实际依赖仍需确认。

## 给 Agent 的使用方法

将本文件夹复制到支持技能的 Agent 技能目录，或直接提供本文件夹并让 Agent 读 `SKILL.md`。Codex 可用 `$pb-hybrid-split`，其他 Agent 无需依赖 Codex 的安装机制。

```text
请使用 pb-hybrid-split。原始模型是 model.pb，输入和输出请先检查。
我的硬件/工具链是……；希望……部分在 PL，其余在 PS。
这些阶段必须保留原后端：……。预处理与样本目录是……。
请交付子图、接口清单、数值验证和分环境执行命令。
```

## 直接运行工具

以下命令在本技能目录执行，`python` 应指向已安装 TF 1.15 的环境；也可用脚本绝对路径。先复制 `assets/partition.example.json` 并按实际 `inspect` 输出改张量名；示例节点不是自动识别结果。样本是预处理完成的 NPZ，键为原图完整输入名，如 `image:0`。

```bash
python scripts/pb_partition.py inspect --source model.pb --report inventory.json
python scripts/pb_partition.py split --source model.pb --plan partition.json --out build/split1
python scripts/pb_partition.py verify --source model.pb --parts build/split1 --samples samples --report build/split1/verification.json
```

输出包括每段 PB 和 `manifest.json`。输出目录必须是新目录，不覆盖原模型。多输入阶段必须显式列出所有存活输入，否则会报边界不闭合。

PS 端没有 TensorFlow 且图符合有限 NumPy 算子集合时：

```bash
python scripts/export_numpy.py --pb build/split1/tail.pb --input ps_features:0 --output logits:0 --out build/numpy/tail
python scripts/pb_partition.py verify --source model.pb --parts build/split1 --samples samples --numpy-dir build/numpy --report build/split1/numpy_verification.json
```

板端只需要 `numpy_runtime.py`、`graph.json`、`weights.npz` 和调度器。调用 `NumpySegment(graph_path).predict(features)`；输入是实际反量化特征，不额外做 RGB 预处理。多输入 PS 子图可使用 TensorFlow/其他 CPU 后端，或由 Agent 扩展并验证 NumPy 适配器。

## 已验证效果与经验来源

- 所给历史 Markdown 中，MobileViTv2 两段 PB 在 10 张真实图片上串联 logits 误差为 0；NumPy 后段最大误差约 `3.82e-6`，类别一致，并做过阻断 TensorFlow 导入的独立运行检查。
- 旧量化部署图存在 30 个未完成融合的 Transpose、动态归一化/注意力以及 6 处 Reshape 属性差异，促成了“检查语义与支持范围，再切出 CNN”的流程。
- 本次通用工具另做非 MobileViT 图的多端口、残差闭包、源文件保持和 NumPy 校验。[本次验证摘要](references/validation-summary.json) 随本技能保留，完整执行记录在技能包根目录 `validation/`；不能据此宣称任意模型或板卡都通过。

具体出处和失败到规则的对应关系见 [案例证据](references/case-evidence.md) 和 [踩坑表](references/pitfalls.md)。完整操作见 [工作流](references/workflow.md)。

## 通用 PYNQ 适配

[PYNQ 指南](references/pynq.md) 与 `tensor_contract.py` 使用参数化 shape/dtype/scale，不绑定 MobileViT、AXU19EG 或 IP 名称，换题目可复用接口检查流程。它不下载 bitstream、不写寄存器、不发起 DMA；本包未宣称完成通用 PYNQ 实板验证。
