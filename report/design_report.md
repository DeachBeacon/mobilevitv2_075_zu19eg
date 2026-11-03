# 设计报告：MobileViTv2-0.75 在固定 DPU 平台上的混合部署

> 本报告采用当前正式六段方案，分阶段数据保留历史测量口径。实现与硬件证据见 `build/implementation_report.md`；本次实际验证范围见 `report/release_validation.md`。本文为工程包 Markdown 正文，不能代替按赛事模板排版并单独上传的设计报告 PDF。

## 1. 选题背景与创新点

MobileViT 系列把 CNN 的局部性与 Transformer 的全局建模结合，在移动端精度/算力比上很有
竞争力；但本模型的“卷积 + 注意力 + GroupNorm”混合结构对 FPGA/DPU 部署不友好：
纯 DPU 方案受限于注意力算子的量化与支持度，纯 CPU 方案又太慢。

本工程选择 **DPU + PS 混合**：把卷积密集、量化友好的前段放 DPU（INT8），把
MobileViT 的注意力/归一化后段放 PS（NumPy + OpenBLAS），并按“纯卷积块”继续切分，
把可量化的部分搬回 DPU。创新点：

1. **按算子特性而不是按层数切分**：切点落在 `stack3_block2_pre_2_conv`，前段 969.67 MOPS
   全部是卷积；后段 0.543 GMAC 里注意力/归一化占 22%，留在 CPU 更合适。
2. **多段切分**：把 `stack4_block1`、`stack5_block1` 两个纯卷积块再切回 DPU，形成
   六段顺序调度链（front → ps1 → mv2_4 → ps2 → mv2_5 → ps3）。历史材料中的“五段”沿用旧命名，不代表实际只有五段。
3. **PS 侧的工程优化**：OpenBLAS NumPy、BN/BiasAdd 折叠、GroupNorm/注意力融合，
   把后段从（纯 PS 的 33 s → 切分后 19 s）一路降到 0.43 s。
4. **可复现的对照**：同时提供纯 PS（FP32）版本，能量化“切 DPU”的收益与精度代价。

## 2. 设计原理与功能框图

```text
输入 256×256×3
   │
   ▼
[front DPU]  stem … stack3_block2_pre        (INT8, 969.67 MOPS)
   │  1×32×32×96
   ▼
[ps1 NumPy]  stack3_block2/3（2 个 MobileViT block）
   │  1×32×32×192
   ▼
[mv2_4 DPU]  stack4_block1（纯卷积）          (INT8)
   │  1×16×16×288
   ▼
[ps2 NumPy]  stack4_block2…5（4 个 block）
   │  1×16×16×288
   ▼
[mv2_5 DPU]  stack5_block1（纯卷积）          (INT8)
   │  1×8×8×384
   ▼
[ps3 NumPy]  stack5_block2/3 + 分类头
   │  1×10
   ▼
logits
```

接口：DPU 段之间用 FP32 HWC 张量传输（板端 1–2 ms 量级）；PS 段用 NHWC float32 数组，
与 NumPy 图的 input/output 张量名一一对应（见 `board/deploy_best_fused/runtime_config.json`）。

## 3. 软硬件划分依据

| 段 | 内容 | 计算量 | 放置 | 理由 |
| --- | --- | ---: | --- | --- |
| front | stem…stack3_block2_pre | 969.67 MOPS | DPU | 纯卷积，INT8 友好 |
| ps1 | 2 个 MobileViT block | — | PS | 含注意力/GroupNorm，DPU 支持差 |
| mv2_4 | stack4_block1 | — | DPU | 纯卷积，可量化 |
| ps2 | 4 个 block | — | PS | 同上 |
| mv2_5 | stack5_block1 | — | DPU | 纯卷积 |
| ps3 | 2 个 block + head | 0.543 GMAC（整后段） | PS | 注意力 + 分类头，head 仅 0.2% 时间 |

早期方案的对比数据（按块计时、为什么不再切 head 等）见
`report/measurements/02_split_analysis.md`、`03_tail_benchmark.md`。

## 4. 优化过程

| 阶段 | 动作 | 板端单张 | 备注 |
| --- | --- | ---: | --- |
| 0 | 纯 PS（未切分、未优化，NumPy 1.17 无 BLAS） | **33.0 s** | 全 CPU 基线；FP32 91.33%（PC 参考） |
| 1 | 切一刀：front 上 DPU（仍无 BLAS） | 19.0 s | 1.7×；后段仍跑标量循环 |
| 2 | + OpenBLAS + 图折叠（BN/Bias/Identity、im2col） | 0.745 s | 44.3× |
| 3 | 五段切分（mv2_4/mv2_5 回 DPU） | 0.556 s | 59.4× |
| 4 | + GroupNorm/注意力融合 + CLE 精度修正 | **0.527 s** | **62.7×**，87.00% |

> 33.0 s 是“未切分、未优化”的纯 PS 基线；19.0 s 是只切前段、仍无 BLAS 的中间点。主要收益来自 OpenBLAS（19→0.745 s）与五段切分 + 融合（0.745→0.527 s）。

关键数据：

- 后段分块：`stack4_block1` 16.9%、`stack3_block3` 14.7%、`stack5_block1` 10.7%、
  `stack3_block2` 9.5%、`io/reshape` 8.8%（见 03）；
- 融合后 logits 与 fast 版差 `2.2e-6`，无精度损失（见 05/09）；
- DPU 输入范围检查：三个 kernel 301 次调用 0 越界、0 clip（见 09）。

## 5. 结果与对比

| 版本 | 精度 | 单张 | 相对纯 PS |
| --- | ---: | ---: | ---: |
| 纯 PS（未切分、未优化） | FP32 91.33%* | 33.0 s | 1.0× |
| 切一刀（front DPU，无 BLAS） | INT8 85.0% | 19.0 s | 1.7× |
| + OpenBLAS + 图折叠 | 85.0% | 0.745 s | 44.3× |
| 五段 DPU+PS | 84.33% | 0.556 s | 59.4× |
| + 融合 + CLE（最终） | INT8 87.00% | **0.527 s** | **62.7×** |

\* 纯 PS 的 91.33% 为 PC 端 FP32 同图参考；300 张同时是校准集，87.00% 偏乐观。

准确率分析：最终 87.00%（261/300），FP32 参考 91.33%，量化损失约 4.3 个点。
早期五段版（84.33%）的主要误差是 `c9→c0`（8 张）、`c7→c0`（5 张）、`c1→c0`（4 张）；
经 CLE / 偏置修正后各类召回趋于均衡（见 09 与 `visualization_system.md`）。

### 5.1 板端可视化系统（队友实现）

队友 2 在 `board/run_visual_board.py` 上实现了 miniDP 直连 1080P 显示的分栏渲染与
全局性能看板：逐张显示分类/置信度/各阶段耗时/Top-3，推理结束后驻留总结看板 30 秒。
本包最终逐图原始记录 `data/results/final_20261103/visual_summary.json` 包含 **261/300、87.00%**，逐图 `total_ms` 平均 **526.4251 ms**，最小 518.58 ms、最大 583.55 ms。本文以这份可复核数据为最终结果依据；`report/visualization_system.md` 保留的 525.92 ms 属于其原记录口径，不与该逐图均值混用。此次整理未重新执行板端推理或显示界面。

## 6. 大模型协作（摘要）

完整记录见 `report/llm_collaboration.md`。要点：

- 用大模型做**数据分析 + 代码生成 + 排错**三类任务；
- 关键纠错：pip 旧版不认 manylinux2014、DPU kernel 加载失败的分层定位、
  FastTail 折叠后输出节点 KeyError、融合模式匹配；
- 所有模型给出的数字都要求回板/回 PC 复测，未复测的只作“预期”。

## 7. 技能包

见 `skill/README.md`：已包含 PB 切分、CLE 精度优化、PS 后段优化三个可复用技能及各自脚本、适用条件、验证范围和经验来源。通用技能的主机验证不代表在任意板卡完成真实部署。

## 7.1 软件系统章节（队友）

函数级实现、PC→Docker→板端全链路与 9 幅配图见 `report/software_system.md`；
核对依据见 `report/software_system_sources.md`。

## 8. 复现说明

见根目录 `README.md` 的“直接上板”和“从 PT 重建”；最短路径：

```bash
# 板上
cd board/deploy_best_fused
export LD_LIBRARY_PATH="$PWD/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders --mv2-backend dpu --fuse \
  --output-dir batch_final_300
```

## 9. 局限与展望

1. 准确率：INT8 87.00%（早期 84.33%），仍可用 QAT/校准进一步优化；
2. 速度：ps1+ps2 仍占后段 82%，可继续切分或换 ONNX Runtime；
3. 泛化：300 张同时是校准集，需独立测试集验证；
4. 工程：换 bitstream/镜像后需重新确认 `.so`、DPU 架构与运行库配套。
5. 硬件证据：HWH 配置和模型编译资料已归档，整设计资源与时序报告仍待补齐；未获得赛事方替代提交许可，不宣称该项已满足。
