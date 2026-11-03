# sim：验证与计时脚本

本目录放“证明切分/融合正确、并量化耗时”的脚本与结果。它们大多只依赖 NumPy，
部分需要 TensorFlow 1.15（PC 端）。

| 文件 | 作用 | 依赖 |
| --- | --- | --- |
| `verify_split.py` | 比对切分前后（PT/TF/NumPy）logits，确认切分无损 | TF 1.15（PC） |
| `review_hybrid_split.py` | 检查 DPU/PS 边界、张量名、shape 是否连通 | TF（PC） |
| `bench_tail.py` | 后段 baseline / fast / fused 三方对比 + 数值校验 | NumPy |
| `profile_tail_blocks.py` | 后段分块/分算子计时 + `np.__config__` + matmul GFLOP/s | NumPy |
| `diagnostics/` | 量化精度诊断结果（FixNeuron 原生结果、精度下降排查、CLE 迁移） | — |

## 1. 后段三方对比（板上/PC 均可）

```bash
python3 bench_tail.py --graph <ps_tail_numpy.json> \
  --features-npy <sample_features_fp32.npy> --repeat 5
```

判据：`same_class: true`、`max_abs_logit_diff < 1e-4`（实测 3e-6），并记录
`ms_per_frame.baseline / fast / fused`。

## 2. 分块计时

```bash
python3 profile_tail_blocks.py --graph <ps_tail_numpy.json> \
  --features-npy <sample_features_fp32.npy> --repeat 3
```

输出：NumPy 版本 / BLAS 信息、512×512 matmul GFLOP/s、每块耗时与占比。
本工程实测：matmul 9.37 GFLOP/s、后段 606.6 ms（修复后）。

## 3. 切分一致性

```bash
python3 verify_split.py --help
```

在 PC 上用 TF 1.15 跑切分前后的 PB，要求 FP32 logits 差 <1e-4。
本工程实测：整链差 2.9e-6。

## 4. 诊断结果

`diagnostics/` 里是量化阶段的记录：

- `fixneuron_native_analysis_20261029.md`：FixNeuron 原生量化结果（10 分类里 1/10 的异常）；
- `quantization_accuracy_investigation_20261029.md`：量化掉点排查；
- `cle_migration_notes.md`：CLE 通道均衡迁移说明。

这些是“为什么最终采用 CLE + bias correction”的依据，详细数据见
`report/measurements/`。
