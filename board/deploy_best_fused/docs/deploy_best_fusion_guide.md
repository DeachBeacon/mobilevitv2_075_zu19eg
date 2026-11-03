# deploy_best（六段）融合说明（2026-11-01）

队友的新方案把后段再切了一刀，执行图从“两段”变成“六段”：

```text
front(DPU) → ps1(NumPy) → mv2_4(DPU) → ps2(NumPy) → mv2_5(DPU) → ps3(NumPy) → logits
```

实测端到端 **0.5556688 s**。

> 2026-11-01 晚更新：队友最终发来的是 `deploy_best.zip`（control_bc100）：
> 六段结构不变，但 mv2_4 的 DPU kernel 从 `mobilevitv2_075_mv2_4_cle` 换成了
> `mobilevitv2_075_mv4a_ctlb100`（带 bias correction），native 精度 83.33%、
> 比上一版多对 10 张，板上验收待做。本文件与融合包都已基于 `deploy_best`。因此昨天准备的两段包（`handover_speedup_20261030.zip`）
**不能再直接发给队友**：它的 runtime 只认一个 tail 图。融合算法本身没变，但要按新图接入。

---

## 1. 新包：`packages/deploy_best_fused.zip`（当前未提供）

在队友的 `deploy_control` 上只加了一个开关，模型、`.so`、量化产物都没动：

| 文件 | 改动 |
| --- | --- |
| `runtime/ps_tail_fused.py` | **新增**：在 `ps_tail_fast` 基础上融合 GroupNorm / 注意力 |
| `runtime/run_hybrid_board.py` | 加 `--fuse`；`HybridRunner(..., fused_tail=False)`；NumPy 段工厂按 fused > fast > numpy 选择 |
| `runtime/run_hybrid_batch.py` | 加 `--fuse`；summary 的 `tail_impl` 记录 `ps_tail_fused` |
| `readme_fuse.md` | 包内简短说明 |

```text
deploy_best_fused/
├── model/                 ← 原样（3 个 .so + ps1/ps2/ps3 + mv2_4/5 图 + 权重）
├── runtime/               ← 加 ps_tail_fused.py，改两个 runner
├── runtime_config.json    ← 原样
├── readme_fuse.md
└── manifest.txt
```

## 2. 融合规模（三个 NumPy 段）

| 段 | 节点数（fast → fused） | GroupNorm | 注意力 |
| --- | --- | ---: | ---: |
| ps1 | 162 → 108 | 5 | 2 |
| ps2 | 293 → 193 | 9 | 4 |
| ps3 | 239 → 162 | 7 | 3 |
| 合计 | 694 → 463 | **21** | **9** |

与旧的两段 tail 完全对应（旧 941→505 里也正好是 21 + 9，只是旧图还包含 mv2 卷积段）。

## 3. 数值验证（PC，用原样例特征）

```text
ps1  diff 2.3e-05
ps2  diff 2.7e-05
ps3  diff 2.2e-06
全链路 logits diff 2.2e-06，类别一致
```

> ps1/ps2 的中间特征差 2e-5 是 float32 重排导致的（相对误差 ~1e-6 量级），
> 到 logits 只剩 2.2e-6，远小于 1e-4 的判据。

## 4. PC 实测（每段 warm-up 后取 5 次中位数）

| 段 | fast | fused | 变化 |
| --- | ---: | ---: | ---: |
| ps1 | 18.0 ms | 16.1 ms | −10.4% |
| ps2 | 27.1 ms | 23.0 ms | −15.1% |
| ps3 | 17.7 ms | 15.6 ms | −12.0% |
| NumPy 合计 | **85.8 ms** | **76.0 ms** | **−11.4%** |

板上 NumPy 段是内存瓶颈，预计收益不低于 PC 的这个比例；以板上实测为准。

## 5. 板上怎么跑

```bash
# 解压后
cd ~/deploy_best_fused
export LD_LIBRARY_PATH="$PWD/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 -m py_compile runtime/ps_tail_fused.py runtime/run_hybrid_board.py runtime/run_hybrid_batch.py

# 20 张冒烟（融合）
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ./test1_p --labels-from-folders --fuse --limit 20 \
  --output-dir batch_fuse_20

# 300 张全量
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ./test1_p --labels-from-folders --fuse \
  --output-dir batch_fuse_300
```

对照实验（同一批图，三个目录）：

```bash
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ./test1_p --labels-from-folders --no-fast-tail --limit 20 \
  --output-dir batch_numpy_20
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ./test1_p --labels-from-folders --limit 20 \
  --output-dir batch_fast_20
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ./test1_p --labels-from-folders --fuse --limit 20 \
  --output-dir batch_fuse_20
```

对比：

```bash
python3 -c "
import json
for name in ('batch_numpy_20','batch_fast_20','batch_fuse_20'):
    s=json.load(open(name+'/summary.json'))
    print(name, s['tail_impl'], round(s['mean_seconds']['numpy_tail'],3),
          round(s['mean_seconds']['total'],3), s['accuracy'])
"
```

## 6. 判据与预期

- `tail_impl` = `ps_tail_fused`；
- `accuracy` 与 fast 版一致（同一模型，logits 差 2e-6）；
- 参考：队友 fast 版 **0.5557 s**；融合后 NumPy 段 −10% 左右，端到端预计 **0.50–0.53 s**
  （估算，板上实测为准）；
- 若 `--fuse` 反而更慢，去掉该参数即可，零副作用。

## 7. 和旧包的关系

| 包 | 用途 | 状态 |
| --- | --- | --- |
| `packages/handover_speedup_20261030.zip` | 两段（front + 单 tail）的提速包 | 历史，只用于旧部署 |
| `packages/deploy_control.zip` | 六段上一版（mv2_4 用 `mv2_4_cle`） | 历史 |
| `packages/deploy_best.zip` | 队友最终版（mv2_4 用 `mv4a_ctlb100`） | 基线，原样保留 |
| `packages/deploy_best_fused.zip`（当前未提供） | 六段最终版 + 融合 | 历史压缩包名；当前使用 board/deploy_best_fused 目录 |

旧包里 `docs/后端融合操作教学_20261031.md` 讲的改文件方法仍适用，但**命令和路径是两段版的**；
六段版以本文件为准。
