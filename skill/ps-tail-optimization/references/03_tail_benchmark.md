# 提速实测结果与下一步（2026-10-31）

板端环境：Python 3.7.6、NumPy 1.19.5（OpenBLAS，`/usr/local/lib`）、AXU19EG 板卡。
本轮只换了 NumPy，**模型、切分、DPU、量化产物都没动**。

---

## 1. 本轮实测

| 指标 | 修复前 | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| matmul 512×512 | 未测（推断 < 0.1 GFLOP/s） | **9.37 GFLOP/s** | — |
| 后段（1 张） | ≈ 19 s（端到端推断） | **606.6 ms** | ≈ 31× |
| 后段等效吞吐 | 0.057 GFLOP/s | 1.086 / 0.607 ≈ **1.79 GFLOP/s** | ≈ 31× |

> 后段 0.543 GMAC = 1.086 GFLOP；606.6 ms 对应 1.79 GFLOP/s，而 512×512 matmul 能到 9.37 GFLOP/s，
> 说明卷积已经跑在 BLAS 上，剩下的是小块卷积效率 + 逐元素算子开销。

### 1.1 时间都花在哪（按块）

| 块 | ms | conv_ms | MACs | 占比 | attention |
| --- | ---: | ---: | ---: | ---: | :---: |
| `stack4_block1` | 102.80 | 55.50 | 104.7 M | 16.9% | 否（纯卷积） |
| `stack3_block3` | 89.47 | 26.27 | 85.0 M | 14.7% | 是 |
| `stack5_block1` | 65.17 | 46.02 | 57.0 M | 10.7% | 否（纯卷积） |
| `stack3_block2` | 57.50 | 18.46 | 66.2 M | 9.5% | 是 |
| `io/reshape` | 53.44 | 0.00 | 0 | 8.8% | 是 |
| `stack4_block2` | 50.53 | 26.66 | 48.5 M | 8.3% | 是 |
| `stack4_block5` | 45.77 | 19.07 | 47.8 M | 7.5% | 是 |
| `stack5_block2` | 32.84 | 20.81 | 21.5 M | 5.4% | 是 |
| `stack5_block4` | 30.68 | 17.60 | 21.2 M | 5.1% | 是 |
| `stack4_block3` | 28.50 | 11.97 | 37.2 M | 4.7% | 是 |
| `stack4_block4` | 28.25 | 11.83 | 37.2 M | 4.7% | 是 |
| `stack5_block3` | 20.62 | 12.31 | 16.5 M | 3.4% | 是 |
| `HEAD` | 1.08 | 0.00 | 0 | 0.2% | 是 |

### 1.2 时间都花在哪（按算子）

| 类别 | 算子 | ms | 占比 |
| --- | --- | ---: | ---: |
| 卷积 | Conv2D + Depthwise | 266.5 | **43.9%** |
| **可折叠** | BN 77.8 + BiasAdd 55.5 + Identity 1.7 | **134.9** | **22.2%** |
| 归一化/注意力 | Mul 34.8 + Mean 26.3 + SquaredDifference 31.4 + Sub 16.1 + Sum 8.8 + SplitV 9.1 + Softmax 5.4 + Rsqrt 2.6 | 134.5 | 22.2% |
| 布局 | Reshape 14.0 + Transpose 2.9 | 16.9 | 2.8% |
| 其他 | Relu 28.0 + AddV2 16.9 + Pad 8.5 + StopGradient 0.3 + MatMul 0.1 | 53.8 | 8.9% |

---

## 2. 结论：BLAS 问题已解决，直接进第四点（快速后段）

- **不用做 3.4**（线程数是可选项，5 分钟，见第 5 节）；
- 下一步就是交接包 README 的**第 4 节**：`bench_tail.py` 验证 + `--fast-tail` 端到端；
- 可折叠的 134.9 ms（22.2%）正好是 `ps_tail_fast.py` 处理的部分，预计后段
  **606 → 440–480 ms（1.25–1.38×）**，乐观 420 ms（1.44×）。

---

## 3. 第 4 步：快速后段（照做）

### 3.1 单独验证正确性 + 速度

```bash
cd /root/test4/handover_speedup_20261030/deploy_in_board
python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy \
  --repeat 5
```

判据：

| 字段 | 要求 |
| --- | --- |
| `same_class` | true |
| `max_abs_logit_diff` | < 1e-4 |
| `speedup` | > 1（预计 1.25–1.45） |
| `nodes` | 736 |
| `folded` | bn 11 / bias 39 / identity 155 |

PC 参考：speedup 1.44–1.73、diff 3.3e-6。板端如果 speedup < 1，就不要用 fast，原路径不受影响。

### 3.2 端到端单图（先拿现在的总时间基线）

```bash
python3 runtime/check_hybrid_board.py \
  --kernel mobilevitv2_075_front \
  --input-node stem_conv_Conv2D \
  --output-node stack3_block2_pre_2_conv_Conv2D \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --input-npy baseline/samples/sample_input_fp32.npy \
  --reference-features baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy \
  --warmup 1 --repeat 5 \
  --output-dir board_check_numpy119
```

```bash
python3 -c "import json; r=json.load(open('board_check_numpy119/report.json')); print(r['timing_seconds_mean']); print(r['class_index'], r.get('class_matches_reference'))"
```

记下 `numpy_tail` 和 `sum`（`sum` 是 DPU 段 + 后段，不含预处理）。这是“换 NumPy 后”的端到端基线。

### 3.3 端到端单图（加 --fast-tail）

```bash
python3 runtime/check_hybrid_board.py \
  --kernel mobilevitv2_075_front \
  --input-node stem_conv_Conv2D \
  --output-node stack3_block2_pre_2_conv_Conv2D \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --input-npy baseline/samples/sample_input_fp32.npy \
  --reference-features baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy \
  --warmup 1 --repeat 5 --fast-tail \
  --output-dir board_check_fast
```

对比两个 `report.json`：`timing_seconds_mean.numpy_tail`、`sum`、`feature_comparison`、`logit_comparison`。

### 3.4 批量（有 `test1_p` 时）

```bash
python3 runtime/run_hybrid_batch.py \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --fast-tail --output-dir batch_fast
```

300 张预计 300 × (0.6 + DPU + 预处理) ≈ 4–6 分钟。
看 `batch_fast/summary.json` 的 `accuracy`（应为 29/32 或更好，模型没变）、
`mean_seconds.numpy_tail`、`mean_seconds.total`。

---

## 4. 现在的预期账

| 项目 | 现在 | 第 4 步后（预计） |
| --- | ---: | ---: |
| 后段 | 606.6 ms | 440–480 ms |
| DPU 段 + 传输 | 待测（`report.json` 的 `dpu_with_transfer`） | 不变 |
| 端到端 | 待测 | 后段省 130–170 ms |
| 相对最初 19 s | ≈ 27–30× | ≈ 32–35× |

说明：19 s 是端到端的旧数字，后段现在是 0.6 s；端到端真实值要等 3.2 的 `report.json`
（DPU 段和预处理目前没有实测值）。如果 `dpu_with_transfer` 也是几百毫秒，下一步要查 DPU 侧
（kernel 加载、节点名、驱动），而不是继续压后段。

---

## 5. 3.4（线程数）到底做什么——可选，5 分钟

OpenBLAS 默认会用多线程。A53 四核上线程开太多有时反而慢（缓存争用、降频）。
测法：

```bash
cd /root/test4/handover_speedup_20261030/deploy_in_board
for T in 1 2 4; do
  echo "== OPENBLAS_NUM_THREADS=$T =="
  OPENBLAS_NUM_THREADS=$T python3 runtime/profile_tail_blocks.py \
    --graph baseline/model/ps_tail_numpy.json \
    --features-npy baseline/samples/sample_features_fp32.npy --repeat 3 \
    2>&1 | grep -E "matmul|tail total"
done
```

看两行：`matmul ... GFLOP/s` 和 `tail total: ... ms`。

- 如果 4 线程明显快（>10%）：在跑推理前 `export OPENBLAS_NUM_THREADS=4`；
- 如果差不多或更差：固定 `1`，减少抖动；
- 这个设置只影响当前 shell；要固定可写进 `~/.bashrc` 或启动脚本。

顺带可以看一眼 CPU governor：

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# 如果是 powersave，可临时设 performance（需要 root）
echo performance | tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

---

## 6. 第 4 步之后还能做什么（按性价比）

| 优先级 | 改动 | 目标 | 预期 |
| --- | --- | --- | --- |
| 1 | 注意力 + GroupNorm 融合：现在 Mul/Mean/SquaredDifference/Sub/Sum/SplitV 共 134.5 ms，大多是逐元素整趟读写 | 改成一个函数内完成归一化与 attention | 省 60–90 ms |
| 2 | `io/reshape` 53.4 ms：unfold/fold 的 Reshape/Transpose 有大量拷贝 | 用 `as_strided` 视图或调整布局 | 省 20–40 ms |
| 3 | 纯卷积块 `stack4_block1`（102.8 ms）+ `stack5_block1`（65.2 ms）搬回 DPU | 切分方案，之前分析过 | 后段 −28%（约 170 ms） |
| 4 | ONNX Runtime 跑后段 | 换运行时 | 后段可能 2–4× |

纯 NumPy 的现实下限大约在 **350–400 ms**（卷积 266 ms 已经占大头，小块卷积效率约 43%）。
也就是说第 4 步 + 1、2 两项做完，后段大约 **0.35–0.4 s**，端到端再叠加 DPU 与预处理。

---

## 7. 记录表（已填本轮）

| 配置 | NumPy | matmul GFLOP/s | numpy_tail ms | DPU+传输 ms | 端到端 | 准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| 最初 | 1.17.0 | 未测 | ≈ 19 s（推断） | 未测 | 19 s | 29/32 |
| 换 OpenBLAS NumPy | 1.19.5 | 9.37 | 606.6 | 待测 | 待测 | 待测 |
| + `--fast-tail` | 1.19.5 | 9.37 | 预计 440–480 | 待测 | 待测 | 待测 |

---

## 8. 命令速查（本轮之后）

```bash
cd /root/test4/handover_speedup_20261030/deploy_in_board
export LD_LIBRARY_PATH="$PWD/baseline/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 1) 快速后段 A/B
python3 runtime/bench_tail.py --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5

# 2) 端到端基线（不带 fast）
python3 runtime/check_hybrid_board.py --kernel mobilevitv2_075_front \
  --input-node stem_conv_Conv2D --output-node stack3_block2_pre_2_conv_Conv2D \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --input-npy baseline/samples/sample_input_fp32.npy \
  --reference-features baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy \
  --warmup 1 --repeat 5 --output-dir board_check_numpy119

# 3) 端到端 fast
python3 runtime/check_hybrid_board.py --kernel mobilevitv2_075_front \
  --input-node stem_conv_Conv2D --output-node stack3_block2_pre_2_conv_Conv2D \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --input-npy baseline/samples/sample_input_fp32.npy \
  --reference-features baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy \
  --warmup 1 --repeat 5 --fast-tail --output-dir board_check_fast

# 4) 批量（有 test1_p）
python3 runtime/run_hybrid_batch.py --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json --fast-tail --output-dir batch_fast
```
