# 可复用技能包：模型分段与量化精度优化

本目录包含 **3 个技能**：前两个从《排查 VAI 编译 Transpose 报错》记录与部署/精度对照中提炼；第三个是 PS 后段（CPU 侧）的测量与加速，来自本队 33 s → 0.527 s 的实践。没有把个人路径、模型节点或某块板卡当成使用前提。

| 技能 | 适用场景 | 使用入口 | 主要失效条件 |
|---|---|---|---|
| [pb-hybrid-split](pb-hybrid-split/README.md) | 按用户要求将冻结 PB 拆为 CPU/DPU 或 PS/PL 多段，保留残差依赖，证明分段数值一致 | [SKILL.md](pb-hybrid-split/SKILL.md)；`$pb-hybrid-split` | 非冻结/有状态图、边界依赖不闭合、未知 CPU 算子、无法提供正确输入样本；硬件支持仍需实际编译 |
| [cle-accuracy-tuning](cle-accuracy-tuning/README.md) | 从现用 SO 对应的浮点模型出发，比较 BN/CLE/限幅/偏置校正，按完整部署链路重新生成精度优化 SO | [SKILL.md](cle-accuracy-tuning/SKILL.md)；`$cle-accuracy-tuning` | 只有 SO 没有原始模型、变换不等价、比较口径不同、固定后端改变、输入越界、没有更好候选、工具链不匹配 |
| [ps-tail-optimization](ps-tail-optimization/README.md) | ARM 板上 NumPy 后段提速：测 BLAS、折叠 BN/Bias、GroupNorm/注意力融合、纯 PS 基线 | [SKILL.md](ps-tail-optimization/SKILL.md)；`$ps-tail-optimization` | 板端无 BLAS（im2col 可能更慢）、换模型后 GroupNorm 结构不同、输出节点未重映射、FP32 与 INT8 精度不可直接比 |

“SO 优化”需要重新处理原模型、量化、编译和链接，不是修改 SO 二进制或增加 `_cle` 文件后缀。CLE 不保证提精度；未改善时保留现用基线。

## PS 后段优化（个人贡献，第三个技能）

这一条来自本队 CPU 侧的实测，和上面两个“切分/量化”技能互补：

- **33 s → 19 s**：切一刀（front 上 DPU），仍无 BLAS；
- **19 s → 0.745 s**：板端 NumPy 无 BLAS，换 1.19.5 + OpenBLAS 后后段 0.606 s；
- **0.745 s → 0.63 s**：`--fast-tail` 折叠 11 个 BN、39 个 BiasAdd、155 个 Identity，卷积改 im2col；
- **0.63 s → 0.527 s**：21 处 GroupNorm + 9 处可分离注意力融合 + CLE 精度修正，logits 差 1.5e-6；
- **纯 PS 基线**：把 DPU 前段也导出为 NumPy，FP32 91.33%、PC 0.20–0.31 s/张，用于量化“切 DPU 的收益”。

脚本在 [ps-tail-optimization/scripts](ps-tail-optimization/scripts)，分阶段记录在
[ps-tail-optimization/references](ps-tail-optimization/references)。所有板端数字以
`summary.json` 为准，PC 数字只作量级参考。

## 陌生队伍如何使用

1. 将所需技能的整个文件夹复制到 Agent 的技能目录，或将其交给 Agent，要求先读 `SKILL.md`。每个文件夹的 README 都给出输入清单和调用示例；不依赖另一个技能才能运行。
2. 提供自己的源模型、输入/输出及预处理、真实样本、硬件/运行库与工具版本，说明要修改和保留哪些阶段。未知节点可先检查图，不需要照抄案例切点。
3. Agent 按需读取 `references/`，复制 `assets/` 模板，调用 `scripts/`。工具参数应来自实际模型和环境；示例 JSON 的名字和阈值不是推荐默认配置。
4. 按保存模型的黄金参考 → 真实量化后端 → 编译/链接 → 板端 → 独立集分层验收。在用户只要求方案时先交方案，在要求执行时完成授权范围内的产物与验证。

工具按环境分工：图操作使用 Python 3.6、TensorFlow 1.15、NumPy；范围检查和无 TF 推理只需 NumPy；报告与 ELF/SO 检查使用标准库；Vitis 编译模板在匹配的 Docker/Runtime 环境运行。TF2、新 VART/XIR、ARM32 或其他量化格式需单独适配，不能只替换文件后缀。

## 提交内容与评分项对应

| 经验形态 | 包内内容 | 复用方式 |
|---|---|---|
| 提示词工作流 | 两个 `SKILL.md`、README 调用示例、`references/workflow.md` | 用户提供自己的模型/目标，Agent 按实际接口推进 |
| 案例模板 | 分段计划、边界契约、实验契约、测量报告、验收报告、Vitis 1.2 流水线模板 | 替换模型和环境参数，保留一致性约束 |
| 校验脚本 | PB 多端口切分/串联校验、有限算子 NumPy 后端、CLE 等价检查、输入范围核对、同口径排序、ELF/SO 来源检查 | 按每个 README 的命令直接运行；不支持时明确失败 |
| 踩坑清单 | Transpose/Reshape、残差输入、容器路径、量化尺度、内嵌 kernel、缓存和 Python 路径语义 | 由错误现象定位应检查的事实及恢复方式 |
| 通用 PYNQ 内容 | [PYNQ PS/PL 工作流](pb-hybrid-split/references/pynq.md)、`tensor_contract.py` 与接口示例 | 不绑定题目、IP 名、寄存器地址或板卡；按实际 bit/hwh 和 IP 协议适配 |

PYNQ 部分是可迁移工作流和主机接口校验工具，**没有声称完成未知 PYNQ 板卡的 DMA 或控制时序验证**，也没有自动加载硬件的脚本。DNNDK 案例不能当成 PYNQ 实板证据。

## 已验证的效果及证明范围

本次交付在 Python 3.6.10 / TensorFlow 1.15.2 / NumPy 1.19.5 主机环境完成：

- 13 项回归检查全部通过：多端口、多输入残差、漏输入拒绝、未知算子拒绝、BN/CLE/限幅等价、量化元数据一致性、报告口径和旧文件拒绝等。
- 用现存 MobileViT 原始模型及 10 张真实图片复核：保存的两段 PB 串联与原模型输出最大误差为 0；可选 NumPy 后端最大绝对误差约 `5.12e-6`。NumPy 后端另通过独立进程禁止导入 TensorFlow 的检查。
- 现存 ELF/SO 的 DPU 数据段一致；现存前端和两个 MV2 的输入量化标签核对一致。这是只读文件检查，没有执行编译器或 DPU。

历史证据中，同口径 Docker 原生测量的第一个 MV2 偏置校正候选由 `240/300` 到 `250/300`；这是经验来源，**不是本技能在任意模型上的收益保证，也不是板端已达到 83.33%**。BN-only 的 28% 失败反馈也保留在案例说明，避免把某种变换当作必然更优。

完整记录见 [host_validation.json](validation/host_validation.json)，两个技能各有本次验证摘要及历史案例证据。未执行的新候选 Docker 量化、板端精度/速度、独立数据集与 PYNQ 实机验证保持待验证。脚本检查报告的算术和来源一致性，不能替代真实硬件测量或认证其真实性。

## 复跑验证与文件校验

在本包的上级目录、匹配 TF 1.15 环境中运行：

```bash
python -B skill/tests/validate_bundle.py --report validation_recheck.json
```

该命令自建小型图和临时样本，不需要本队权重、板卡或图片。10 图案例复核是可选项，需使用者自己持有原工程：

```bash
python -B skill/tests/validate_bundle.py --real-case-root /path/to/original/compile_tools --report validation_case_recheck.json
```

包内不附模型权重、用户图片、完整聊天或个人账号路径。来源与改编记录见 [sources.json](sources.json)；`sha256sums.json` 记录提交文件哈希，自身不纳入哈希清单。每个技能可单独复制使用；整体测试和完整验证记录在本目录保留。
