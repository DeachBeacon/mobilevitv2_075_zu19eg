---
name: ps-tail-optimization
description: "在 ARM 板上定位并优化 NumPy 后段（DPU+PS 混合部署的 CPU 侧）：先测 BLAS，再折叠 BN/Bias、做 GroupNorm/注意力融合，并用纯 PS 基线量化切分收益；适用于 PS 后段占端到端大头的情况，不适用于改模型精度或替代 DPU 编译。"
---

# PS 后段加速（ARM + NumPy）

先读 [README.md](README.md) 确认脚本用法。本技能处理 DPU+PS 混合部署中**跑在 CPU 上的
NumPy 后段**；不涉及量化、编译或 DPU 侧，也不承诺任何精度收益。

## 先量再改

1. 测 `numpy.show_config()` 与 512×512 matmul GFLOP/s；<1 GFLOP/s 说明没有 BLAS，
   先解决环境，不要急着改代码。
2. 用 `scripts/profile_tail_blocks.py` 拿后段总时间、分块占比、分算子占比。
3. 确认后段占端到端的比例：若 DPU/预处理才是大头，本技能收益有限。

## 按顺序做三件事

1. **BLAS**：装带 OpenBLAS 的 aarch64 wheel（Python 3.7 用 1.19.5）。板上 pip <19.3
   不认 manylinux2014 时，把 wheel 当 zip 解进 site-packages，或先装 pip 23.3.1；
   装前备份 `numpy`，装后复测 matmul。
2. **fast-tail**：加载时折叠 BN、BiasAdd、Identity，卷积改 im2col+matmul；
   用 `scripts/bench_tail.py` 对比 baseline/fast 的 logits 与耗时。
3. **融合**：把 GroupNorm 链（`Reshape→Mean→Sub→SquaredDifference→Mean→Add→Rsqrt→Mul→Reshape`）
   和可分离注意力（`SplitV(1+96+96)→Softmax→加权 Sum→×ReLU(gate)`）各合成一个算子；
   用 `scripts/ps_tail_fused.py` 的 `--reference-logits` 自检。

## 证明正确与收益

- logits 差 <1e-4（本工程实测：fast 3e-6、fused 1.5e-6）；
- 板端 `summary.json` 的 `mean_seconds` 分段对比；PC 数字只作量级参考；
- 想证明“切 DPU 值得”，用 `scripts/run_all_ps.py` 跑纯 PS 基线，同时报告时间与精度。

## 失效条件

- 板端无 BLAS 时 im2col 可能更慢；
- 换模型后 GroupNorm 的 reshape/axes 常量不同，融合模式匹配会失败；
- 折叠 Identity 后必须重映射 `graph["output_tensor"]`，否则 `KeyError: ...Identity:0`；
- 纯 PS 是 FP32，不能拿它的精度和 INT8 直接比较。

## 交付

可执行命令、`summary.json`、logits 对比、回退方式（去掉 `--fast-tail` / `--fuse` 即可）。
