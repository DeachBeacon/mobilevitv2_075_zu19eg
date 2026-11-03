# visual_final 纯 PS 版（全 NumPy，FP32）

用途：在同一台板上，用**全 PS（CPU + NumPy）**跑完整模型，拿到“不切 DPU”的端到端
时间与准确率，和五段 DPU+PS 版（0.529 s / 84.33%）做对比。

本包 = `visual_final` 的同一套图，但**所有段都用 NumPy 执行**：

```text
front(NumPy) → ps1(NumPy) → mv2_4(NumPy) → ps2(NumPy) → mv2_5(NumPy) → ps3(NumPy) → logits
```

注意：这版是 **FP32**（NumPy 权重），和板端 INT8 的 DPU 版精度不可直接比较；
它回答的是“全 PS 要多久”，准确率只作参考。

## 目录

```text
visual_final_pure_ps/
├── model/
│   ├── front.json / front_weights.npz      ← 由推理 PB 导出的前段（新增）
│   ├── ps1.json / ps1_weights.npz
│   ├── mv2_4.json / mv2_4_weights.npz
│   ├── ps2.json / ps2_weights.npz
│   ├── mv2_5.json / mv2_5_weights.npz
│   └── ps3.json / ps3_weights.npz
├── runtime/
│   ├── run_all_ps.py                       ← 批量入口（本包核心）
│   ├── ps_tail_numpy.py
│   ├── ps_tail_fast.py
│   └── ps_tail_fused.py
├── readme_ps.md
└── manifest.txt
```

## 板上运行

```bash
cd ~
unzip visual_final_pure_ps.zip
cd visual_final_pure_ps

# 20 张冒烟
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders --limit 20 \
  --output-dir batch_ps_20

# 300 张全量
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders \
  --output-dir batch_ps_300

# 可选：融合版（更快一点，数值等价）
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders --fuse \
  --output-dir batch_ps_fused_300
```

> `--image-dir` 换成板上实际的 300 张目录（`test1_p` 若不在包里，用你们的路径）。
> 输出目录必须不存在，重跑换名字。

## 看什么

`summary.json`：

- `mode: "all_ps"`；
- `mean_seconds`：`preprocess / front / ps1 / mv2_4 / ps2 / mv2_5 / ps3 / numpy_tail / total`；
- `numpy_tail` = 六个段的总和，`total` = 含预处理的端到端；
- `accuracy` / `confusion_matrix_*`。

## PC 参考（不是板上数据）

| 项 | 值 |
| --- | --- |
| 300 张准确率（FP32） | **91.33%**（274/300） |
| PC 单张 total | **0.20–0.31 s**（随负载波动） |
| 其中 front | 0.11–0.12 s |
| 其中 ps2 | 0.03–0.07 s |

板上 NumPy/BLAS 比 PC 慢（板上 matmul 约 9.4 GFLOP/s），预计 total 在 **1–2 s** 量级，
以实测为准。

## 对比口径

| 版本 | 精度 | 板端 total |
| --- | --- | ---: |
| 五段 DPU+PS（`deploy_best_fused`） | INT8，84.33% | **0.529 s**（实测） |
| 纯 PS（本包，FP32） | FP32，91.33%（PC） | 待测 |

结论写法建议：纯 PS 的精度更高（FP32），但速度会慢数倍；两者对比要同时说明“精度不同”，
不能只看时间。

## 备注

- 需要板上 NumPy 带 BLAS（之前已装 1.19.5 + OpenBLAS；`python3 -c "import numpy; print(numpy.__version__)"` 可查）；
- 不需要 DPU、不需要 `.so`、不需要 bitstream；
- `--fuse` 走 GroupNorm/注意力融合，数值与 fast 差 1e-6 量级，可作加速项；
- 若某张图失败，`predictions.csv` 里会记录 `error`，批量不会中断。
