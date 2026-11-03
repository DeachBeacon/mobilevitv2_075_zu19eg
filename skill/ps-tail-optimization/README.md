# PS 后段加速技能（个人贡献）

本技能来自本队 MobileViTv2-0.75 在 AXU19EG 上的实际优化：把“33 s/张（未切分纯 PS）→ 19 s（切分后仍无 BLAS）”的 NumPy 后段
降到“0.43 s”，并给出纯 PS 基线用于对比 DPU+PS 切分的收益。

> 分工说明：多段切分的工程实现、DPU kernel 编译/偏置校正、`visual_final` 集成由队友完成；
> 本技能只覆盖 CPU 侧（NumPy 后段）的测量、优化与验证。

## 1. 适用场景

- DPU+PS 混合部署，后段（注意力/归一化）跑在 ARM CPU 的 NumPy 上；
- 单张耗时远高于 PC，`numpy.dot` 明显在跑标量循环；
- 需要向评审/队友证明“为什么要切 DPU”，或评估后段还能快多少。

不适用：纯 DPU 部署、量化/编译问题、需要改模型精度的场景。

## 2. 输入清单

- 后段图：`*_tail_numpy.json` + `*_weights.npz`（NHWC，float32）；
- 一张样例特征 `sample_features_fp32.npy` 与参考 logits；
- 板端：Python 3.7、NumPy（可换 wheel）、可选 OpenCV（读图）。

## 3. 脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/profile_tail_blocks.py` | 打印 `np.__config__`、matmul GFLOP/s、分块/分算子耗时 |
| `scripts/bench_tail.py` | baseline / fast / fused 三方 logits 与耗时对比 |
| `scripts/ps_tail_fused.py` | GroupNorm + 可分离注意力融合实现（自检入口） |
| `scripts/run_all_ps.py` | 纯 PS（全 NumPy）流水线，输出同口径 `summary.json` |

```bash
# 1) 测 BLAS 与分块
python3 scripts/profile_tail_blocks.py --graph <tail.json> \
  --features-npy <features.npy> --repeat 3

# 2) 三方对比
python3 scripts/bench_tail.py --graph <tail.json> \
  --features-npy <features.npy> --repeat 5

# 3) 融合自检
python3 scripts/ps_tail_fused.py --graph <tail.json> --weights <weights.npz> \
  --features-npy <features.npy> --reference-logits <logits.npy> --repeat 3

# 4) 纯 PS 基线
python3 scripts/run_all_ps.py --model-dir <model> --image-dir <images> \
  --labels-from-folders --output-dir <out>
```

## 4. 实测数据（本队）

| 阶段 | 后段 | 端到端 | 关键动作 |
| --- | ---: | ---: | --- |
| 纯 PS（未切分、未优化） | — | **33.0 s** | NumPy 1.17，无 BLAS |
| 切一刀（front DPU） | ≈19 s | 19.0 s | 后段仍无 BLAS |
| + OpenBLAS | 0.606 s | 0.745 s | NumPy 1.19.5 aarch64 wheel |
| + fast-tail | ≈0.577 s | ≈0.63 s | BN/Bias 折叠、去 Identity、im2col |
| + 融合 + CLE | 0.434 s | **0.527 s** | 21 处 GroupNorm + 9 处注意力 |
| 纯 PS（FP32） | — | 0.20–0.31 s（PC） | front 也走 NumPy，仅作对照 |

验证：fast 与 baseline 的 logits 差 3.1e-6；fused 差 1.5e-6；纯 PS 整链差 2.9e-6。

## 5. 关键经验

1. **先测 BLAS 再改代码**：19 s 的根因是 `dot` 跑循环，不是解释器写得差；
2. **折叠是“零风险”的第一步**：BN/BiasAdd 是逐通道仿射，只改运算顺序；
3. **融合要按图匹配，不要按名字**：GroupNorm 的 axes/reshape 常量随模型变；
4. **输出节点要重映射**：Identity 被折叠后，`graph["output_tensor"]` 必须 `resolve()`；
5. **PC 与板上收益不同**：PC 上融合 −11%，板上只 −4.8%（卷积仍占大头），结论以板上为准。

## 6. 失效条件

- 板端无 BLAS 或内存带宽极低时，im2col/融合可能变慢；
- 换模型后 GroupNorm 结构不同，模式匹配失败；
- 小图/小张量下 Python 调用开销占比高；
- 纯 PS 与 INT8 的精度不可直接比较。

## 7. 来源

`references/` 下是本队的分阶段记录：

- `01_baseline_and_numpy_blas.md`
- `03_tail_benchmark.md`
- `05_tail_fusion.md`
- `09_board_validation.md`
- `10_pure_ps_guide.md`
