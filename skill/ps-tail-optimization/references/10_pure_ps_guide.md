# 纯 PS 版上板教程（2026-11-02）

目的：在同一块板上，用**全 PS（CPU + NumPy）**跑最终版模型，拿到“不切 DPU”的端到端
时间与准确率，和五段 DPU+PS 版（`deploy_best_fused`）做对比，回答
“为什么要切 DPU / 全 PS 到底慢多少”。

对应包：`packages/visual_final_pure_ps.zip`（12.16 MB，含 `test1_p` 300 张）。

---

## 1. 它是什么

`visual_final` 的同一套图，但**所有段都用 NumPy 在 CPU 上执行**：

```text
front(NumPy) → ps1(NumPy) → mv2_4(NumPy) → ps2(NumPy) → mv2_5(NumPy) → ps3(NumPy) → logits
```

- `front.json` 是从推理 PB 新导出的前段（输入 `[1,256,256,3]` → `[1,32,32,96]`），
  和 DPU 版的前段是同一段网络；
- 其余段复用 `visual_final` 的 `ps1/ps2/ps3`、`mv2_4/mv2_5` 图；
- **FP32**，不需要 DPU、`.so`、bitstream。

> 注意：纯 PS 是 FP32，板端五段是 INT8；两者的**精度不可直接比**，
> 时间对比也要注明“精度不同”。

---

## 2. 板上步骤

```bash
cd ~
unzip visual_final_pure_ps.zip
cd visual_final_pure_ps

# 0) 确认 NumPy 有 BLAS（之前装过 1.19.5 + OpenBLAS）
python3 -c "import numpy; print(numpy.__version__)"

# 1) 20 张冒烟（约 10–40 秒，视板子而定）
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ./test1_p --labels-from-folders --limit 20 \
  --output-dir batch_ps_20

# 2) 300 张全量（预计几分钟）
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ./test1_p --labels-from-folders \
  --output-dir batch_ps_300

# 3) 可选：融合版（GroupNorm/注意力融合，快一点，数值等价）
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ./test1_p --labels-from-folders --fuse \
  --output-dir batch_ps_fused_300
```

`--output-dir` 必须是**不存在的新目录**，重跑换名字。

---

## 3. 看什么

```bash
python3 -c "
import json
for name in ('batch_ps_20','batch_ps_300','batch_ps_fused_300'):
    try:
        s=json.load(open(name+'/summary.json'))
        print(name, s['tail_impl'], round(s['mean_seconds']['numpy_tail'],3),
              round(s['mean_seconds']['total'],3), s['accuracy'])
    except FileNotFoundError: pass
"
```

`summary.json` 关键字段：

- `mode: "all_ps"`；
- `mean_seconds.front / ps1 / mv2_4 / ps2 / mv2_5 / ps3 / numpy_tail / total`；
- `accuracy`、`confusion_matrix_true_rows_predicted_columns`。

---

## 4. 预期与对比

PC 参考（同一套图、FP32、300 张）：

| 项 | 值 |
| --- | --- |
| 准确率 | **91.33%**（274/300） |
| 单张 total | 0.20–0.31 s（随负载波动） |
| 其中 front | 0.11–0.12 s |
| 其中 ps2 | 0.03–0.07 s |

板上 NumPy/BLAS 比 PC 慢（板上 matmul 约 9.4 GFLOP/s），预计 `total` 在 **1–2 s** 量级，
以实测为准。

对比表（把实测填进去）：

| 版本 | 精度 | 板端 total | 相对纯 PS |
| --- | --- | ---: | ---: |
| 五段 DPU+PS（`deploy_best_fused`） | INT8，84.33% | 0.529 s | — |
| 纯 PS（本包） | FP32，91.33%（PC） | 待测 | 1.0× |
| 纯 PS + `--fuse` | FP32 | 待测 | 待测 |

写报告的口径：

- 速度：`speedup = 纯 PS total / 0.529`；
- 精度：纯 PS（FP32）更高是正常的，INT8 的 84.33% 不能直接和它比；
- 结论要同时给“时间”和“精度”，否则会被质疑。

---

## 5. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: ps_tail_fast` | 没在包根目录跑 | `cd visual_final_pure_ps` 后执行 `python3 runtime/...` |
| `No images` | `--image-dir` 不对 | 用 `./test1_p` 或板上实际路径 |
| `Choose a new output directory` | 输出目录已存在 | 换目录名 |
| 单张很慢（>5 s） | NumPy 没 BLAS | 装 1.19.5 + OpenBLAS（见提速实践文档） |
| 某张图 failed | 图片损坏/格式不支持 | 看 `predictions.csv` 的 `error`，不影响其余 |

---

## 6. 一句话

纯 PS 包只用来回答“全 CPU 跑要多久、FP32 能到多少”；**它不替代 DPU 版**。
拿到 `batch_ps_300/summary.json` 后，和 0.529 s / 84.33% 放在一起，就是完整的
“切 DPU 的收益 + 精度代价”对比。
