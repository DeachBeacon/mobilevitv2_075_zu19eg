# deploy_best + GroupNorm/注意力融合（--fuse）

本包 = 队友最终的 `deploy_best`（六段：front DPU → ps1 NumPy → mv2_4 DPU → ps2 NumPy
→ mv2_5 DPU → ps3 NumPy；mv2_4 用 `mv4a_ctlb100` 带 bias correction）加上后段融合补丁。
模型、`.so`、量化产物都没变，只改了 NumPy 段的执行方式。

## 相对原 deploy_best 的改动

| 文件 | 改动 |
| --- | --- |
| `runtime/ps_tail_fused.py` | 新增：在 `ps_tail_fast` 基础上融合 GroupNorm 与注意力（ps1/ps2/ps3 共 21 + 9 处） |
| `runtime/run_hybrid_board.py` | 加 `--fuse`；`HybridRunner(..., fused_tail=False)`；NumPy 段工厂按 fused > fast > numpy 选择 |
| `runtime/run_hybrid_batch.py` | 加 `--fuse`；summary 的 `tail_impl` 记录 `ps_tail_fused` |

## 在板上运行

```bash
cd /path/to/Z19-Onborad/board/deploy_best_fused
export LD_LIBRARY_PATH="$PWD/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -m py_compile runtime/ps_tail_fused.py runtime/run_hybrid_board.py runtime/run_hybrid_batch.py

# 20 张冒烟（融合）
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders --fuse --limit 20 \
  --output-dir batch_fuse_20

# 300 张全量
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders --fuse \
  --output-dir batch_fuse_300
```

对照：同一批图片分别跑 `--no-fast-tail`（原始 NumPy）、`--fast-tail`（默认）、`--fuse`，
比较 `summary.json` 的 `mean_seconds` 与 `accuracy`。

## 数值与预期

- 融合只改运算顺序：整条六段流水线的 logits 与 fast 版相差 `2.2e-6`（PC 验证），类别一致；
- 每个 NumPy 段的融合规模：ps1 5 个 GroupNorm + 2 个注意力，ps2 9 + 4，ps3 7 + 3；
- PC 上 NumPy 段合计 85.8 → 76.0 ms（约 -11%）；板上逐元素更吃内存，预计收益更大，需实测；
- 若板上 `--fuse` 反而更慢，去掉 `--fuse` 即可，零副作用。

## 详细文档

`docs/deploy_best_fusion_guide.md`：六段图、改了什么、融合规模、数值验证、
板上命令、预期与回退。

回滚：不带 `--fuse` 运行就是原路径。

> 说明：原版 `BUILD_ORIGIN.json` 已从本包移除（它记录的是未打补丁的 runtime 哈希）；
> 原始构建元数据未随本工作区提供；归档文件见根目录 `backup/organization_20261103/`。
