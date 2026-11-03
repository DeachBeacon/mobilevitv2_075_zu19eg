# 最终板端运行（2026-11-03）

来源：`Visual_final(1).zip` 的 `output_visual/`，对应 `deploy_best_fused`
（`fronta_ctlb100` + `mv4a_ctlb100` + `mv2_5_cle`，`tail_impl = ps_tail_fused`）。

## 结果

| 项 | 值 |
| --- | ---: |
| 测试集 | 300 张（`test1_p`，c0–c9） |
| 正确 / 总数 | **261 / 300** |
| 准确率 | **87.00%** |
| 单张总耗时（含预处理） | 均值 **526.43 ms**（min 518.58，max 583.55） |
| 平均预测置信度 | 86.50% |
| DPU kernel | `fronta_ctlb100`、`mv4a_ctlb100`、`mv2_5_cle` |
| 后段实现 | `ps_tail_fused`（GroupNorm/注意力融合） |

## 文件

| 文件 | 说明 |
| --- | --- |
| `visual_summary.json` | 逐图结果：预测类、真实类、是否正确、置信度、单张 total_ms |
| `summary_result.jpg` | 运行结束后的全局指标看板 |
| `summary_result_preview.jpg` | 看板预览图 |
| `c0_1_result.jpg`、`img_1031_result.jpg`、`img_1006_result.jpg`、`img_48363_result.jpg` | 4 张逐图可视化样例 |

> 与 `data/results/five_stage_20261031/` 的区别：那是 CLE 精度修正前的五段版
> （0.529 s / 84.33%，kernel 含 `front_cle`）；本目录是最终版（0.527 s 量级 / 87.00%）。
