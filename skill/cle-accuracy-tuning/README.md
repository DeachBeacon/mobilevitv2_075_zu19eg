# cle-accuracy-tuning：从原模型重建优化精度 SO

入口为 [SKILL.md](SKILL.md)。日常说的“普通 SO 改成 CLE SO”，执行上是：找回对应原始浮点 PB → BN 折叠/CLE 等候选 → 校准量化 → 真实链路评估 → 编译 ELF → 链接新 SO。

## 适用范围与失效条件

| 项目 | 能力与边界 |
|---|---|
| 输入资料 | 原始浮点 PB、对应现用 SO/PB/配置、真实样本、预处理与目标工具/硬件；只有 SO 时不能无损恢复原始模型 |
| 浮点助手 | 单输入 CNN 子图，NHWC Conv2D / DepthwiseConv2dNative，直接推理 BN、已有 BiasAdd、ReLU/Identity/Pad/Add 残差 |
| 不支持的 CLE 结构 | 动态 GN/LN、注意力、ReLU6/非齐次门控、训练 BN、被其他消费者共享且无法安全折叠的 BN、复杂仿射组合 |
| 量化助手 | `quant_boundary.py` 只适配 legacy Vitis/DNNDK 8-bit 输入元数据；VART/XIR/其他后端需按其实际格式适配 |
| 排序助手 | 已完成的分类测量报告；同数据、预处理、固定模型、运行后端、范围策略与评估链路 |
| 失效判据 | 基础候选 FP32 不等价、输入越界、固定模型被改、原生与板端偏差未定位、无候选改善、工具/板卡不匹配 |

脚本处理浮点候选、输入元数据和报告一致性；模型特定的原生完整链路评估器及板端适配器由 Agent 按实际接口生成并验证。本包不提供把任意 SO 一键变得更准的承诺。

## 给 Agent 的使用方法

```text
请使用 cle-accuracy-tuning。现用 SO 和对应原始 PB/量化目录是……。
保持这些后端与 SO 不变：……；只优化……。
主指标是完整……链路的准确率，样本/预处理/验收条件是……。
请先核对基线，再做少量候选；交付可追溯的新 SO 与板端包。
```

目录可直接复制到 Agent 技能目录；Codex 用 `$cle-accuracy-tuning`。也可让其他 Agent 直接读取 `SKILL.md`。

## 可运行的浮点变换

使用 TF 1.15 环境，以下命令在本技能目录执行，或使用脚本绝对路径。样本 NPZ 的键必须是该段真实输入，如 `features:0`；内部段不能改喂 RGB 图。下面对三个**不同新目录**执行，`input/output` 以实际模型为准：

```bash
python scripts/cle_graph.py --source segment.pb --input features:0 --output output:0 --samples features --variant bn --out candidates/bn
python scripts/cle_graph.py --source segment.pb --input features:0 --output output:0 --samples features --variant cle --out candidates/cle
python scripts/cle_graph.py --source segment.pb --input features:0 --output output:0 --samples features --variant bounded --scale-cap 32 --bias-cap 8 --activation-cap 8 --out candidates/bounded32
```

每个目录输出 `float_model.pb`、参数范围、缩放组和 FP32 等价报告。默认阈值是示范值，可按模型设定；它们不能替代 INT8 或最终任务指标。无安全自由度的自耦合残差组固定为 1。

## 量化与编译

按 [工作流](references/workflow.md) 和 [legacy Vitis 说明](references/legacy-vitis.md) 操作。量化/编译模板位于 `assets/vitis12_pipeline.sh`；通过环境变量传入真实模型、校准回调、输入输出、shape、架构、DCF、kernel 和构建目录。它不填写本队路径或 tensor 名。

模板分开执行 `quantize`、`compile`、`link`。编译前要求 Agent 提供已测评候选的选择文件与评估产物哈希；这一检查证明文件对应关系，不证明报告已真实运行，更不能跳过原生评估。

输入量化检查示例：

```bash
python scripts/quant_boundary.py --eval-pb quant/quantize_eval_model.pb --deploy-pb quant/deploy_model.pb --input features:0 --samples features --report input_audit.json
```

只有经过范围分析才指定 `--apply-position`，并配合 `--out quant_fixed` 输出新 PB。改变输入位置会使后续缓存、报告和 ELF/SO 失效，必须重评估、重编译、重链接；不通过改 JSON 假装原 SO 的尺度改变。

将真实评估结果规范化为 `assets/normalized_report.example.json` 格式，再用：

```bash
python scripts/rank_candidates.py --config ranking.json --out selection.json
```

示例报告标为 `example_not_measured`，不能直接用于选型。分类之外的任务应由 Agent 定义正确指标和等价的一致性检查。

## 已验证效果与失败来源

当前对话里，两个 MV2 DPU 方案板端由 85% 降到 79%，BN-only 对照又降到 28%；因此不能把“不做 CLE”当默认修复。首个 MV2 偏置经 CLE 放大约 `9.92 → 268.98`、偏置量化步长为 2，是检查线索而非已证明的唯一原因。

用户提供的同口径 Docker 原生结果中，原前端与第二个 MV2 固定，第一个 MV2 输出偏置校正使结果从 240/300 到 250/300，增加 10 张；这是**原生测量报告**，不能改写成板端已达到 83.33%，也不能与另一个口径的 85% 或 FP32 91.3% 直接相减。

前端基础候选的历史 300 样本 FP32 等价检查通过，但整个三段 DPU 新精度在本对话中未得到板端确认。详见 [案例证据](references/case-evidence.md)。[本次验证摘要](references/validation-summary.json) 随本技能保留，完整主机测试记录另记于技能包根目录 `validation/`。

本技能不绑定某一板卡。PYNQ 自定义加速器可复用黄金参考、边界契约、固定后端和测量分层原则；本辅助量化元数据脚本并不直接适配任意 PYNQ IP 或新 VART 格式。
