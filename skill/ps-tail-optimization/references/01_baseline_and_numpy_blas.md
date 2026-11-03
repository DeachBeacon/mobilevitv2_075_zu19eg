# 板上提速实践：测 19 s、修 NumPy/BLAS、换快速后段

日期：2026-10-30　适用：`MobileViTv2_075_ZU19EG` 当前两段部署（DPU 前段 + PS NumPy 后段）
前提：**不做再切分**，先把 19 s/帧的异常耗时定位并压下来。

本文件是可直接照做的上板手册，全部命令都在板端执行；PC 只负责下载 aarch64 wheel。

---

## 0. 结论预期

| 阶段 | 动作 | 预期 |
| --- | --- | --- |
| 测量 | 跑 `profile_tail_blocks.py` | 确认 19 s 落在后段、看 matmul GFLOP/s |
| 修 BLAS | 换 OpenBLAS 版 NumPy（1.19.5 aarch64 wheel） | 后段 5–20×（若原来没有 BLAS） |
| 换快速后段 | `--fast-tail`（折叠 BN/BiasAdd、im2col） | 再 1.4–1.6×（PC 实测） |
| 仍不够 | attention matmul / ONNX Runtime / 再切分 | 见第 5 节 |

PC 参考（x86 + 39 GFLOP/s BLAS）：后段 66→41 ms（1.61×），logits 最大差 3.3e-6，类别一致。

---

## 1. 要拷到板上的文件

把整个 `deploy_in_board` 目录同步过去（或只更新 `runtime/`）：

```text
deploy_in_board/
  runtime/
    ps_tail_numpy.py            # 原解释器（不动）
    ps_tail_fast.py             # 新增：折叠 BN/BiasAdd 的快速解释器
    bench_tail.py               # 新增：两者 A/B 对比
    profile_tail_blocks.py      # 新增：分块计时 + BLAS 检测
    run_hybrid_board.py         # 已改：新增 --fast-tail
    run_hybrid_batch.py         # 已改：新增 --fast-tail
    check_hybrid_board.py       # 已改：新增 --fast-tail
  baseline/model/
    ps_tail_numpy.json
    ps_tail_numpy_weights.npz
    libdpumodelmobilevitv2_075_front.so
  baseline/samples/
    sample_input_fp32.npy
    sample_features_fp32.npy
    sample_reference_logits.npy
```

板上进入部署目录（下面所有命令都假设在这里）：

```bash
cd /home/root/deploy_in_board
export LD_LIBRARY_PATH="$PWD/baseline/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

## 2. 第 0 步：先测量，不改任何东西

### 2.1 如果有 `run_hybrid_batch.py` 的 `summary.json`

```bash
python3 - <<'PY'
import json, glob
for path in glob.glob('**/summary.json', recursive=True):
    s = json.load(open(path))
    print(path, s.get('mean_seconds'), 'total', s.get('mean_seconds', {}).get('total'))
PY
```

看 `mean_seconds` 的 `preprocess / dpu_with_transfer / numpy_tail / total`（CSV 里对应的列名是 `ps_tail_seconds`）：

- `numpy_tail` 占大头 → 继续 2.2；
- `dpu_with_transfer` 很大 → 问题在 DPU 侧（kernel 加载、驱动、节点名），先不要改 NumPy；
- `preprocess` 很大 → 图像解码/预处理问题。

### 2.2 跑分块计时 + BLAS 检测

```bash
python3 runtime/profile_tail_blocks.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy \
  --repeat 3
```

输出三段：

1. `np.__config__.show()` —— 有没有 `openblas` / `blas`；
2. `matmul 512x512: X GFLOP/s` —— **核心判据**；
3. 每块耗时表（`stack4_block1`、`stack5_block1`、`stack5_block4`、`HEAD` 等）。

判据：

| matmul GFLOP/s | 含义 | 下一步 |
| --- | --- | --- |
| < 1 | 没有可用 BLAS，`dot` 在跑标量循环 | 做第 3 节 |
| 1–4 | 有 BLAS，单核水平 | 直接做第 4 节 |
| > 5 | BLAS 正常 | 后段慢就是实现问题，做第 4 节 |

把这张表和 `numpy_tail` / 总时间一起记下来，作为基线。

---

## 3. 第 3 步：换 OpenBLAS 版 NumPy（若 matmul < 1 GFLOP/s）

### 3.1 先备份

```bash
python3 -c "import numpy, os; print(numpy.__version__, os.path.dirname(numpy.__file__))" | tee numpy_before.txt
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "SITE=$SITE"
cp -a "$SITE/numpy" "$SITE/numpy.bak"
python3 -c "import numpy; print(numpy.__version__)"
```

记下 `numpy_before.txt` 的版本（可能是 1.17.0）和 `SITE`。

### 3.2 在 PC 上下载 aarch64 的 OpenBLAS 版 wheel

numpy 1.17.0 没有 aarch64 wheel，选 **1.19.5**（仍支持 Python 3.7，后段只用 1.16+ API）。
已在 PC 上验证可用：

```powershell
python -m pip download numpy==1.19.5 `
  --platform manylinux2014_aarch64 --python-version 37 `
  --implementation cp --abi cp37m --only-binary=:all: --no-deps -d wheels `
  -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com
```

得到 `numpy-1.19.5-cp37-cp37m-manylinux2014_aarch64.whl`（11.8 MB），
里面带 `numpy.libs/libopenblasp-r0-*.so`，即 OpenBLAS。

```powershell
scp wheels\numpy-1.19.5-cp37-cp37m-manylinux2014_aarch64.whl root@<板子IP>:/home/root/
```

### 3.3 板上安装

```bash
python3 -m pip --version || python3 -m ensurepip --upgrade
python3 -m pip install --no-deps --force-reinstall \
  /home/root/numpy-1.19.5-cp37-cp37m-manylinux2014_aarch64.whl
```

- 必须加 `--no-deps`：板上已有依赖，不要让它联网；
- 若报 `... is not a supported wheel on this platform`：板上 pip 19.2.3 早于 manylinux2014 支持
  （需要 pip ≥ 19.3）。两种解法：① 不升级 pip，直接把 wheel 解压到 site-packages；
  ② 先装 `pip-23.3.1-py3-none-any.whl` 再安装。具体命令见交接包 README 的 3.2 节（方法 A / B）；
- 若 pip 不存在且 `ensurepip` 不可用：把 wheel 当 zip 解开到 site-packages

  ```bash
  SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
  rm -rf "$SITE/numpy" "$SITE/numpy.libs"
  cd "$SITE" && python3 -c "import zipfile; zipfile.ZipFile('/home/root/numpy-1.19.5-cp37-cp37m-manylinux2014_aarch64.whl').extractall('.')"
  ```

### 3.4 验证

```bash
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import numpy as np; print(np.__config__.get_info('blas_opt'))"

python3 - <<'PY'
import time, numpy as np
a = np.random.rand(512, 512).astype('float32')
b = np.random.rand(512, 512).astype('float32')
a.dot(b)
t = time.time()
for _ in range(5):
    a.dot(b)
print('GFLOP/s', round(2 * 512 ** 3 * 5 / (time.time() - t) / 1e9, 2))
PY
```

期望 ≥ 1.5 GFLOP/s（A53 单核 2–4 比较正常）。再重跑 2.2 的 profile，记录新的 `numpy_tail` / 尾段总时间。

### 3.5 线程数

```bash
export OPENBLAS_NUM_THREADS=1    # 先单线程测
# 或 4，看是否有改善；A53 四核时通常 2–4 线程略快
```

两种都记一次，选稳定且快的。

### 3.6 回滚（如果装坏了）

```bash
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SITE/numpy" "$SITE/numpy.libs"
mv "$SITE/numpy.bak" "$SITE/numpy"
python3 -c "import numpy; print(numpy.__version__)"
```

---

## 4. 第 4 步：启用快速后段

### 4.1 它做了什么

`ps_tail_fast.py` 与原解释器读同一份 JSON/NPZ，在加载时做等价改写：

| 优化 | 数量 | 说明 |
| --- | --- | --- |
| BN 折进卷积 | 11 个 | `W ← W·γ/√(σ²+ε)`，`b ← (b−μ)·scale+β`，数学等价 |
| BiasAdd 折进卷积 | 39 个 | 直接加到卷积偏置 |
| 去掉 Identity | 155 个 | 纯别名 |
| 卷积实现 | 45 个 | 连续 im2col + `matmul`；1×1 直接 matmul；depthwise 按 kernel tap 累加 |

节点数 941 → 736。不改模型、不改 JSON/NPZ、不碰 DPU。

### 4.2 正确性 + 速度（先单独验证）

```bash
cd /home/root/deploy_in_board
python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy \
  --repeat 5
```

输出关键字段：

| 字段 | 通过标准 | PC 实测 |
| --- | --- | --- |
| `same_class` | true | true |
| `max_abs_logit_diff` | < 1e-4 | 3.3e-6 |
| `speedup` | > 1 | 1.44–1.61 |
| `nodes` | 736 | 736 |
| `folded` | bn 11 / bias 39 / identity 155 | 同 |

不通过（类别变化或 diff > 1e-4）就停在这里，把 JSON 发回来，不要继续端到端。

### 4.3 端到端（单图，带分段计时）

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

对比 `board_check_fast` 与原 `board_check` 的指标：`timing_seconds_mean.numpy_tail`、总时间、特征误差、logits 误差。

### 4.4 端到端（批量，含准确率）

```bash
python3 runtime/run_hybrid_batch.py \
  --image-dir ./test1_p \
  --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --fast-tail \
  --output-dir batch_fast
```

`batch_fast/summary.json` 里：

- `tail_impl` 应为 `ps_tail_fast`；
- `accuracy` 应与原批量一致（同一模型，浮点误差 1e-6 量级）；
- `mean_seconds.numpy_tail` 与 `total` 是提速后的数字。

不加 `--fast-tail` 时行为与原来完全一致，可随时 A/B。

---

## 5. 如果还不够快

按顺序考虑：

1. **attention 改成批量 matmul**：当前后段里 attention 用 `Mul + Sum`（78 个 Mul、18 个 Sum、86 个 Mean），
   在 ARM 上是内存带宽瓶颈。改成 `[B,H,N,C]` 的 `matmul` 可再省 20–40%，但要在解释器里做模式匹配，
   工作量 1–2 天，可以用同样的 A/B 方式验证。
2. **ONNX Runtime**：工程里已有 `int8_best/model_int8_qdq.onnx`（4485 张 99.22%）。板上装
   `onnxruntime`（aarch64 wheel）后跑整网或后段，CPU EP 有 NEON 优化；这是绕开 NumPy 的路线，
   与 DPU 方案并行评估。
3. **再切分**：把 `stack4_block1`、`stack5_block1` 两个纯卷积块搬回 DPU，理论上限后段 −28%，
   但工程量大（多套量化/编译/链接/校准），应放在 1、2 之后。

---

## 6. 记录模板

每次改一项就记一行：

| 配置 | NumPy | matmul GFLOP/s | numpy_tail ms | DPU ms | total s | 准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| 现状 | 1.17.0 | | | | 19 | 29/32 |
| 换 NumPy 后 | 1.19.5 | | | | | |
| + `--fast-tail` | 1.19.5 | | | | | |
| + 线程数调整 | 1.19.5 | | | | | |

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `pip install` 报 `is not a supported wheel on this platform` | 多半是 pip 太老（< 19.3），不认识 manylinux2014 标签 | 用交接包 README 3.2 的方法 A 解压安装，或先装 `pip-23.3.1`；确认 pip ≥ 20 后再查 Python/ABI |
| 装完 `import numpy` 报 `libopenblasp...so: cannot open` | wheel 解压不完整或旧文件残留 | 先 `rm -rf numpy numpy.libs` 再装 |
| 装完 numpy 报 GLIBC 版本错误 | manylinux2014 需要 glibc ≥ 2.17 | `ldd --version` 确认；太低就只能用原厂 NumPy |
| `matmul` 仍然 < 1 GFLOP/s | BLAS 没被加载 | 看 `np.__config__.show()` 是否有 openblas；检查 `numpy.libs` 是否存在 |
| `--fast-tail` 报 `ModuleNotFoundError: ps_tail_fast` | 没在 runtime 目录或 PYTHONPATH 不含 runtime | `cd deploy_in_board` 后跑 `python3 runtime/...` |
| `bench_tail` 的 diff > 1e-4 | 折叠出错 | 停，把 JSON 发回来；不要用 fast 做端到端 |
| fast 反而更慢 | 板端 BLAS 缺失时 im2col copy 更贵 | 不加 `--fast-tail` 即可，零风险 |
| 批量准确率下降 | 用了不同模型/节点名 | 核对 `--tail-graph`、kernel 名、ddump 节点名 |

---

## 8. 附录：PC 实测与原理

PC（x86_64，Python 3.12，NumPy 2.3.5，matmul 23–39 GFLOP/s）：

```json
{
  "ms_per_frame": {"baseline": 66.03, "fast": 41.08},
  "speedup": 1.61,
  "nodes": {"baseline": 941, "fast": 736},
  "folded": {"bn": 11, "bias": 39, "identity": 155},
  "max_abs_logit_diff": 3.34e-06,
  "same_class": true
}
```

等价性依据：BN 折叠与 BiasAdd 折叠都是逐通道仿射变换，只改变运算顺序，不改变语义；
`matmul` 与 `tensordot` 在 float32 下差异在 1e-6 量级，远小于 INT8/参考容差。

板端结果请回传 `bench_tail.py` 与 `profile_tail_blocks.py` 的 JSON/输出，我按实测数字更新第 0 节的预期。
