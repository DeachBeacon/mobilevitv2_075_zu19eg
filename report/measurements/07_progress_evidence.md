# 今晚进展与提速证据（2026-10-31 凌晨）

> 性能口径：完整基线从 **33.0 s**（未切分、未优化纯 PS）开始；本文的 19 s 是“切一刀后、仍无 BLAS”的中间点，0.745 s 是加 OpenBLAS 后，最终 0.527 s（87.00%）。


一句话：**原来占大头、约 19 s 的 NumPy 后段，现在 0.61 s，快约 31 倍。** 只换了 NumPy
（1.19.5 + OpenBLAS）和可选的后段解释器，模型、切分、DPU、量化/编译产物都没动。
端到端的完整复测被“前段 `.so` 加载失败”挡住，那是板上 PL/环境问题，与这次提速无关。

---

## 1. 证据（板端实测，不是估算）

### 1.1 修复前后

| 项目 | 修复前 | 修复后 | 依据 |
| --- | ---: | ---: | --- |
| NumPy | 1.17.0（无可用 BLAS） | **1.19.5 + OpenBLAS** | `np.__config__.show()` |
| 512×512 matmul | 未测（由 19 s 反推约 0.06 GFLOP/s） | **9.37 GFLOP/s** | `profile_tail_blocks.py` |
| 后段 1 张 | ≈ 19 s（端到端几乎全在后段） | **606.6 ms** | `profile_tail_blocks.py` |
| 后段有效吞吐 | ≈ 0.057 GFLOP/s | **1.79 GFLOP/s**（1.086 GFLOP / 0.607 s） | 计算 |
| 快速后段 A/B | baseline 694.16 ms | **fast 581.11 ms（1.19×）** | `bench_tail.py` |
| 数值一致性 | — | **logits 最大差 3.1e-6，类别一致** | `bench_tail.py` |

按 19 s → 0.607 s 计算：**31.3×**；加上 `--fast-tail` 后再到 0.581 s：**32.7×**。

### 1.2 关键原始输出

```text
python 3.7.6 numpy 1.19.5
openblas_info:
    libraries = ['openblas', 'openblas']
blas_opt_info:
    libraries = ['openblas', 'openblas']
matmul 512x512: 9.37 GFLOP/s (0.0287 s per call)

tail total: 606.6 ms/frame, 0.543 GMAC
```

```json
{
  "numpy_version": "1.19.5",
  "matmul_gflops": 9.25,
  "ms_per_frame": {"baseline": 694.16, "fast": 581.11},
  "speedup": 1.19,
  "nodes": {"baseline": 941, "fast": 736},
  "folded": {"bn": 11, "bias": 39, "identity": 155},
  "max_abs_logit_diff": 3.0994415283203125e-06,
  "same_class": true,
  "self_check_passed": true
}
```

### 1.3 时间花在哪（修复后，按块）

| 块 | ms | 占比 |
| --- | ---: | ---: |
| `stack4_block1`（纯卷积） | 102.80 | 16.9% |
| `stack3_block3`（attention） | 89.47 | 14.7% |
| `stack5_block1`（纯卷积） | 65.17 | 10.7% |
| `stack3_block2`（attention） | 57.50 | 9.5% |
| `io/reshape` | 53.44 | 8.8% |
| 其余 8 个块 | 约 236 | 39% |
| `HEAD`（1×1 conv + pool + linear） | 1.08 | **0.2%** |

按算子：卷积 266.5 ms（43.9%）、BN+BiasAdd+Identity 134.9 ms（22.2%）、注意力/归一化 134.5 ms（22.2%）。

> 顺带证明：把分类头搬到 DPU 只能省 0.2%，之前那个“三段切分”不值得做。

---

## 2. 怎么测的（可复现）

```bash
cd ~/test4/handover_speedup_20261030/deploy_in_board

# 1) 后段分块计时 + BLAS 检测（不依赖 DPU）
python3 runtime/profile_tail_blocks.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 3

# 2) 基线 vs 快速后段的 A/B（不依赖 DPU）
python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5
```

这两个脚本只用“同一份后段 JSON/NPZ + 同一张样例特征”，在板上真实跑 NumPy，
不涉及 DPU，所以结论与 DPU 状态无关。

正确性判据：`same_class: true`、`max_abs_logit_diff < 1e-4`（实测 3.1e-6）。

---

## 3. 口径说明（防止被质疑）

1. **19 s 是端到端，0.607 s 是后段单段**。旧部署里 DPU 前段只有几毫秒到几十毫秒、预处理更小，
   所以 19 s 基本就是后段；这次没有动 DPU，两者可以直接对比。
2. 如果队友要求严格的端到端数字：等前段 `.so` 恢复后跑一次 `run_hybrid_batch.py`，
   `summary.json` 的 `mean_seconds.total` 就是端到端，届时补上。
3. 18 s → 0.6 s 的收益来自 **NumPy 换 BLAS**（约 31×），`--fast-tail` 只额外贡献 1.19×。
4. 模型文件、`.so`、量化结果没有改，所以准确率不应变化；批量准确率待 DPU 恢复后复测。

---

## 4. 今晚遇到的报错

| # | 现象 | 原因 | 处理 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | `pip install numpy-...manylinux2014_aarch64.whl` 报 `not a supported wheel on this platform` | 板上 pip 19.2.3 早于 manylinux2014 支持（需 ≥19.3） | 不用 pip，直接把 wheel 解压进 site-packages；或先装 `pip-23.3.1` | ✅ 已解决，NumPy 1.19.5 生效 |
| 2 | `run_hybrid_batch.py` 报 `[DNNDK] Fail to load DPU kernel: mobilevitv2_075_front` | 所有候选 `.so` 都加载失败；`dpuOpen=0`、bitstream operating、dlopen 正常 → 前段 `.so` 与当前 PL 的 DPU 不配套（环境/镜像问题） | 用 yolov3 自带模型和 `mobilevitv2_075_head` 各测一次，判断是环境还是单个 `.so`；必要时换回旧 BOOT.BIN/SD 或按当前 PL 重编 | ⏳ 待明天 |
| 3 | `cd /root/test4/...` 报 No such file | 板上 `HOME=/home/root`，`/root` 不是家目录 | 用 `~/test4/...` 或 `/home/root/test4/...` | ✅ 已确认 |
| 4 | 文档里 `summary.json` 字段写成 `ps_tail` | 实际字段是 `numpy_tail`（CSV 里才叫 `ps_tail_seconds`） | 文档已改正 | ✅ 已修正 |

### 报错 2 的排查进展

- `/dev/dpu` 存在、`dpu` 模块已加载、`dpuOpen() == 0`、`fpga_manager` 状态 `operating`；
- `ctypes.CDLL(...)` 按绝对路径和按库名都能加载 → 不是文件损坏、不是依赖缺失；
- 但 `dpuLoadKernel` 对 4 份不同的 `.so`（banzi1、test1、test3/banzi3、交接包）全部失败；
- `/media/sd-mmcblk1p1/BOOT.BIN` 是 2021-11-23 的厂商原版 → PL 里的 DPU 可能是厂商那份，
  和你们编译 `.so` 时用的 DCF 不一致。

明天先跑这两个测试：

```bash
# A. 厂商自带模型
cd /home/root/Alinx_DNN/tf_yolov3_pcb_deploy
LD_LIBRARY_PATH=$PWD python3 -c "
from dnndk import n2cube
n2cube.dpuOpen()
print('yolo:', n2cube.dpuLoadKernel('tf_yolov3_pcb'))
" 2>&1 | tail -3

# B. 你们自己的 head / cle 内核
cd /home/root/test3/banzi3
LD_LIBRARY_PATH=$PWD python3 -c "
from dnndk import n2cube
n2cube.dpuOpen()
print('head:', n2cube.dpuLoadKernel('mobilevitv2_075_head'))
print('cle :', n2cube.dpuLoadKernel('mobilevitv2_075_front_cle'))
" 2>&1 | tail -4
```

- A 能、B 不能 → DPU 正常，换配套的 `.so`/镜像，或按当前 PL 重编；
- A、B 都不能 → 恢复 PL/驱动（换回旧 SD/BOOT.BIN）。

---

## 5. 明天要做的

1. 解决前段 `.so` 加载（上面 A/B 测试 → 恢复 DPU）；
2. 跑单图 `check_hybrid_board.py`（不带/带 `--fast-tail`），拿端到端基线；
3. 跑 `run_hybrid_batch.py --limit 20`，再 300 张（fast），记录 `mean_seconds.total` 与 `accuracy`；
4. 把 `19 s → ? s` 的端到端数字、准确率补齐，写进报告。

---

## 6. 给队友的最短复现

```bash
cd ~/test4/handover_speedup_20261030/deploy_in_board

# 后段（不依赖 DPU）
python3 runtime/profile_tail_blocks.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 3
# 期望：numpy 1.19.5、matmul ≥ 5 GFLOP/s、tail total ≈ 0.6 s

python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5
# 期望：same_class true、diff < 1e-4、speedup > 1
```
