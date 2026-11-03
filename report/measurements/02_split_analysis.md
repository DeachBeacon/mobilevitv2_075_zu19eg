# 再切分提速：工作区整理与可行性分析

> 性能口径：完整基线从 **33.0 s**（未切分、未优化纯 PS）开始；本文的 19 s 是“切一刀后、仍无 BLAS”的中间点，0.745 s 是加 OpenBLAS 后，最终 0.527 s（87.00%）。


日期：2026-10-30　对象：`MobileViTv2_075_ZU19EG`（新整理后的工作区）
输入事实：板端单图约 **19 s**；INT8 上板准确率 **29/32**；当前切分 = L=2 MobileViT block
之前（不含）在 DPU，其余在 PS。

---

## 1. 工作区现状

顶层只有 `README.md`、`code/`、`compile_tools/` 两个目录，结构和用途如下：

| 位置 | 内容 | 关键文件 |
| --- | --- | --- |
| `code/pytorch` | 训练、ONNX、量化记录 | `onnx/int8_best/model_int8_qdq.onnx`（4485 张 99.22%，软件评估） |
| `code/tensorflow` | TF 推理、PB 导出、GN 折叠 | `pb/inference`、`pb/deploy` |
| `code/dpu/split` | 切分与 NumPy 后段导出 | `split_model.py`、`export_numpy.py`、`verify_split.py` |
| `code/dpu/baseline`、`cle` | 原量化/编译/链接；CLE 四步脚本 | `1_quantize…`、`equalize_front.py` |
| `compile_tools/models/front_baseline` | 原前段：FP32、INT8、ELF/SO、日志 | kernel `mobilevitv2_075_front`；**969.67 MOPS**；边界 `input_1` → `stack3_block2_pre_2_conv`（32×32×96） |
| `compile_tools/models/front_cle` | BN 折叠 + 通道均衡后的前段 | FP32 已验证 300/300、logits 误差 1.3e-5；**尚未量化/编译** |
| `compile_tools/models/tail_shared` | 共用后段 | `ps_tail_numpy.json`（941 节点）、NPZ（8.4 MB）、`ps_tail_fp32.pb` |
| `compile_tools/deploy_in_board` | 板端运行时与样例 | `runtime/`（单图、批量、检查、NumPy 执行器）、`baseline/model`、`baseline/samples` |
| `compile_tools/diagnostics` | 量化精度排查与实验 | `FixNeuron原生结果分析_20261029.md`、`量化精度下降排查_20261029.md` |
| `compile_tools/calibration/images` | 300 张校准图（c0–c9） | — |
| `compile_tools/docs` | 路径与版本说明 | `PATHS.md`、`model_versions.md` |

模型总计算量约 **1.51 GMAC**：前段 0.970 GMAC（64%）在 DPU，后段 0.543 GMAC（36%）在 PS。

---

## 2. 先说结论

1. **把 L=3 之后的“最后两部分”放 DPU，几乎不会提速。** 尾段实测：最后的 1×1 卷积 +
   全局池化 + 线性分类头只占后段 **0.00% MACs / 0.07% 时间**。为它新增一个 DPU kernel，
   收益 ≤0.1%，还要多一次量化与边界，性价比不成立。
2. **真正值得动的 DPU 段只有两个纯卷积堆叠**：`stack4_block1`（19.4% 时间）和
   `stack5_block1`（8.9% 时间），合计 **28.3%**。即使全部搬到 DPU，后段 19 s → 约 13.6 s。
3. **19 s 本身不正常。** 如果这 19 s 主要花在后段，相当于 0.543 GMAC 跑出
   **约 57 MFLOP/s**，比 ARM 上正常的 NumPy 慢 20–50 倍；大概率是**板上 NumPy 没有可用 BLAS**
   （或实现退化），而不是切分造成的。先修这个，收益远大于再切一刀。

---

## 3. 后段实测结构（PC，NumPy，比例优先看）

用 `ps_tail_numpy.json` 在样例特征上逐节点计时（PC 上总时间 77–156 ms 视负载波动，
下表用**占比**）：

| 块 | MACs | 占后段时间 | 含 attention | 能否直接进 DPU |
| --- | ---: | ---: | :---: | :---: |
| io/reshape（unfold/fold） | 0 | 4.4% | 否 | 否 |
| `stack3_block2` | 66.2 M | 8.7% | 是 | 否 |
| `stack3_block3` | 85.0 M | 12.7% | 是 | 否 |
| **`stack4_block1`** | **104.7 M** | **19.4%** | 否 | **是（纯 conv/BN/ReLU/Pad/Add）** |
| `stack4_block2` | 48.5 M | 9.7% | 是 | 否 |
| `stack4_block3` | 37.2 M | 7.1% | 是 | 否 |
| `stack4_block4` | 37.2 M | 8.7% | 是 | 否 |
| `stack4_block5` | 47.8 M | 9.6% | 是 | 否 |
| **`stack5_block1`** | **57.0 M** | **8.9%** | 否 | **是** |
| `stack5_block2` | 21.5 M | 4.0% | 是 | 否 |
| `stack5_block3` | 16.5 M | 3.0% | 是 | 否 |
| `stack5_block4` | 21.2 M | 3.7% | 是 | 否 |
| `HEAD`（1×1 conv + pool + linear） | 0.0 M | 0.07% | 否 | 是，但没意义 |

按算子：Conv2D 占 99.5% MACs、约 70% 时间；BN/BiasAdd/ReLU/Mean/Mul/SquaredDifference/
Reshape 等逐元素与内存操作占剩余约 30%（PC 上；ARM 上这一块比例会更高）。

对照你提出的方案：L=3 的三个 transformer 块（`stack5_block2/3/4`）合计 59.2 M（10.9%），
attention 不能进 DPU；它们之后的 fusion conv + 分类头是上表的 `HEAD`，0.07%。

---

## 4. “最后两部分进 DPU”方案评估

| 维度 | 结论 |
| --- | --- |
| 计算收益 | 0.00% MACs、0.07% 时间 → 端到端 <0.1% |
| 工程成本 | 多切一个边界、多一次量化/编译/链接、runtime 多一次 kernel 调用 |
| 精度风险 | 分类头量化本身没问题（此前“只量化输出特征”10/10），但没有收益对冲风险 |
| 建议 | **不做**。若目标是让 DPU 覆盖更多层，应该选纯卷积块，而不是分类头 |

---

## 5. 可行的再切分：3 个 DPU 段

把两个纯卷积块单独切出来，链路变成：

```text
DPU-1 前段（0.970 GMAC，已有）
  → PS-1  stack3_block2 + block3（151.2 M，attention）
  → DPU-2 stack4_block1（104.7 M，纯卷积）
  → PS-2  stack4_block2…block5（170.7 M，attention）
  → DPU-3 stack5_block1（57.0 M，纯卷积）
  → PS-3  stack5_block2…block4 + HEAD（59.2 M，attention + 分类头）
  → logits
```

- **理论上限**：后段时间减少 28%（19 s → 约 13.6 s），前提是 DPU 段接近“免费”。
- **边界数据量很小**：最大的边界张量约 32×32×192（786 KB），其余在 50–400 KB；
  即使在 100 MB/s 的保守带宽下也只有几毫秒，相对 19 s 可以忽略。
- **技术可行性**：这两个块只有 Conv2D/DepthwiseConv2dNative/BN/ReLU/Pad/AddV2，Vitis AI 1.2
  完全支持；难点在工程量和精度：
  1. 当前 `split_model.py` 只支持两段切分，需要改成“多切点 + 多段导出”；
  2. 每个新段都要独立量化（需要从浮点模型抓该段输入的校准特征，300 张图 × 中间激活）；
  3. 每个新段都要编译、链接、更新 kernel 名与边界张量名；
  4. runtime 要串行调用 3 个 kernel 与 3 段 PS，状态与计时口径要重写；
  5. 每多一段 INT8 边界，就多一次量化误差来源，需要重新验证 29/32 的准确率。
- **工作量估计**：3–5 天（不含精度返工），收益上限 28%。

结论：**可行，但优先级应排在“修 BLAS / 优化 PS 实现”之后**。如果 19 s 里有 10× 是环境问题，
先做切分等于在错误的基数上优化。

---

## 6. 更优先的三个方向

### 6.1 先确认 19 s 花在哪（1 小时）

19 s / 0.543 GMAC ≈ 57 MFLOP/s；而 Cortex-A53 用 NEON + OpenBLAS 单核通常能到 2–4 GFLOP/s。
差了 30–70 倍，最可能是 NumPy 没有 BLAS，`dot` 退化成标量循环。

板上执行（新增脚本，已放在 runtime）：

```bash
cd compile_tools/deploy_in_board
python3 runtime/profile_tail_blocks.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 3
```

输出包含：每块耗时与占比、`np.__config__.show()`、512×512 float32 matmul 的 GFLOP/s。
若 matmul < 1 GFLOP/s，就是 BLAS 问题。

如果手边有 `run_hybrid_batch.py` 的 `summary.json`，也可以先看 `mean_seconds` 里的
`preprocess` / `dpu_with_transfer` / `ps_tail` / `total` 四项，先确认 19 s 是不是都落在
`ps_tail` 上。

### 6.2 换掉没有 BLAS 的 NumPy（半天，预期 5–20×）

装 aarch64 的 OpenBLAS 版 NumPy（pip 的 manylinux wheel 自带 OpenBLAS），或用 `OMP_NUM_THREADS`
控制线程数。修好后重跑上面的 profile，再决定要不要切分。

### 6.3 优化 NumPy 后段实现（1–2 天，预期 2–3×）

即使有 BLAS，当前解释器仍有明显浪费，PC 上也能量化：

| 问题 | 现状 | 改法 |
| --- | --- | --- |
| BN/BiasAdd/ReLU 各占一趟 | 11 个 BN、40 个 BiasAdd、24 个 ReLU | 离线折进卷积权重/偏置，运行时只剩 conv |
| 卷积展开方式 | `as_strided` + `tensordot`（产生大 transpose/copy） | 连续 im2col + `matmul` |
| attention 用逐元素乘加 | 78 个 Mul、18 个 Sum、86 个 Mean | 改成 reshape 到 `[B,H,N,C]` 的批量 matmul |
| 无效拷贝 | 156 个 Reshape、268 个 Identity | 能别名就别名，去掉逐趟拷贝 |

这些改动只动 `ps_tail_numpy.py` 和导出，不动模型与 DPU，风险低，且与再切分不冲突。

### 6.4 备选：ONNX Runtime 跑后段/整网（评估 1 天）

工程里已有 INT8 QDQ ONNX（4485 张 99.22%）。ARM 上 ONNX Runtime 的 CPU EP 有 NEON 优化，
后段可能从秒级降到百毫秒级。代价是板上要装 `onnxruntime`（aarch64 wheel），且不再走 DPU，
属于“如果时间比体系结构更重要”的备选，而不是主路线。

---

## 7. 建议的下一步

| 顺序 | 动作 | 判据 | 预期 |
| --- | --- | --- | --- |
| 1 | 板上跑 `profile_tail_blocks.py` | 拿到每块耗时 + matmul GFLOP/s | 定位 19 s 来源 |
| 2 | 若 matmul < 1 GFLOP/s：换 OpenBLAS NumPy | 重跑 profile | 19 s → 2–4 s |
| 3 | 仍慢：按 6.3 优化解释器 | 后段再降 2–3× | 1 s 量级 |
| 4 | 还需要更快：做 3 段 DPU 切分 | 后段 −28% | 0.7 s 量级 |
| 5 | 分类头单独进 DPU | — | 不做 |

同时并行：CLE 前段（`front_cle`）的量化/编译/上板还没做，它关系到 29/32 的准确率能否回到
浮点水平，和提速是两条独立的工作流。

---

## 8. 若走 3 段切分，需要改的文件

| 类别 | 文件/内容 |
| --- | --- |
| 切分 | `code/dpu/split/split_model.py`（多切点）、新增两个中间段的 PB/JSON/NPZ |
| 量化 | `code/dpu/baseline/1_quantize_front.sh` 的三份变体 + 两套中间特征校准数据 |
| 编译/链接 | `2_compile_front.sh`、`3_link_front.sh` 的三份变体，kernel 名区分 |
| 运行 | `deploy_in_board/runtime/run_hybrid_board.py`、`run_hybrid_batch.py` 改成 3+3 段串联 |
| 验证 | `verify_split.py` 扩展到多段；`check_hybrid_board.py` 加分段计时 |
| 文档 | `docs/model_versions.md`、`deploy_in_board/README.md`、本文件 |

---

## 附录：本次新增的板端脚本

`compile_tools/deploy_in_board/runtime/profile_tail_blocks.py`

- 逐块统计后段耗时、MACs、是否含 attention；
- 打印 `np.__config__.show()` 与 512×512 `matmul` 的 GFLOP/s，用于判断 BLAS；
- 首次运行会自动预热一次，之后按 `--repeat` 取平均。

PC 上已用它验证过输出格式；板端结果请回传，我按实测数字再校准第 5、6 节的估算。
