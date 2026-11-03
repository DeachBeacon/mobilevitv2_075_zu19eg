# 后端融合（`--fuse`）操作教学：改哪个文件、什么时候跑什么

对应《后段融合优化教学_20261031.md》的 3.2。这里只讲**操作**：文件在哪、改了哪几行、
什么时候运行什么命令、改完长什么样。

> **先说结论：不需要你改代码。** 新版交接包里这些文件已经改好，你只要拷贝 + 运行。
> 第 4 节给出“如果你自己改”的精确改动，方便理解或移植。

---

## 1. 涉及哪些文件（5 个）

板上目录：`~/test4/handover_speedup_20261030/deploy_in_board/runtime/`

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| `ps_tail_fused.py` | **新增** | 融合实现（`FusedTail`），下面详细讲 |
| `run_hybrid_board.py` | 改 4 处 | 单图入口加 `--fuse`，构造时选 `FusedTail` |
| `run_hybrid_batch.py` | 改 3 处 | 批量入口加 `--fuse`，summary 记录 `tail_impl` |
| `check_hybrid_board.py` | 改 2 处 | 单图检查加 `--fuse` |
| `bench_tail.py` | 改 2 处 | 同时对比 baseline / fast / fused 三列 |

如果你手上是**旧包**，把这 5 个文件从新包 `handover_speedup_20261030/deploy_in_board/runtime/`
拷过去覆盖即可；新包已经包含全部改动。

```bash
# 从 PC 传到板上（示例）
scp ps_tail_fused.py run_hybrid_board.py run_hybrid_batch.py \
    check_hybrid_board.py bench_tail.py \
    root@<板子IP>:/home/root/test4/handover_speedup_20261030/deploy_in_board/runtime/
```

---

## 2. 融合到底做了什么（一张图看懂）

后段是 NumPy 逐算子执行，每个算子都要把 `[1,4,256,96]`（约 393 KB）整趟读一遍、写一遍。
有两类算子链，模式固定，可以各合成一个函数：

### 2.1 GroupNorm（21 处）

原图（以 `stack3_block2_attn_group_norm` 为例，输入 `[1,4,256,96]`）：

```text
Reshape [1,-1,1,96] → Mean(axes=[1,3]) → Sub → SquaredDifference → Mean → +eps
→ Rsqrt → Mul → Reshape [1,4,256,96]
```

融合后一个函数：

```python
x = X.reshape((1, -1, 1, 96))                     # [1,1024,1,96]
mean = x.mean(axis=(1, 3), keepdims=True)
var = ((x - mean) ** 2).mean(axis=(1, 3), keepdims=True)
Y = ((x - mean) / np.sqrt(var + 1e-5)).reshape((1, 4, 256, 96))
```

### 2.2 可分离注意力（9 处）

qkv 卷积输出 193 通道 = `1(score) + 96(value) + 96(gate)`：

```text
SplitV → score/value/gate
score → Transpose → Softmax → Transpose → × value → Sum → × ReLU(gate) → 输出
```

融合后：

```python
score = qkv[..., :1]; value = qkv[..., 1:97]; gate = np.maximum(qkv[..., 97:], 0)
w = np.exp(score - score.max(axis=2, keepdims=True)); w /= w.sum(axis=2, keepdims=True)
out = gate * (w * value).sum(axis=2, keepdims=True)   # [1,4,256,96]
```

两处都已用样例特征验证：GroupNorm 最大差 `9.5e-7`，注意力 `0.0`；整条后段 logits 差 `1.5e-6`，
节点 941 → 505。

---

## 3. 什么时候运行什么（按顺序）

假设已经在板上：

```bash
cd ~/test4/handover_speedup_20261030/deploy_in_board
```

### 第 0 步：语法检查（10 秒）

```bash
python3 -m py_compile runtime/ps_tail_fused.py runtime/run_hybrid_board.py \
  runtime/run_hybrid_batch.py runtime/check_hybrid_board.py runtime/bench_tail.py
echo $?
```

期望输出 `0`。不通过就是文件没拷全或被截断。

### 第 1 步：融合自检（不依赖 DPU，1 分钟）

```bash
python3 runtime/ps_tail_fused.py \
  --graph baseline/model/ps_tail_numpy.json \
  --weights baseline/model/ps_tail_numpy_weights.npz \
  --features-npy baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy --repeat 3
```

期望 JSON：

```json
{
  "fused_ops": {"group_norm": 21, "attention": 9},
  "nodes": 505,
  "max_absolute_logit_error": 1.5e-06,
  "self_check_passed": true
}
```

不通过（差 > 1e-4 或类别变）就停，别往下走。

### 第 2 步：三方对比（不依赖 DPU，2 分钟）

```bash
python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5
```

期望看到三个数：

```json
"ms_per_frame": {"baseline": 694.16, "fast": 581.11, "fused": 500~530},
"fused_same_class": true,
"fused_max_abs_logit_diff": 1.5e-06,
"self_check_passed": true
```

> 板端数字以实测为准。如果 `fused` 比 `fast` 还慢，就只用 `--fast-tail`，不要 `--fuse`。

### 第 3 步：单图端到端（需要 DPU，1 分钟）

```bash
export LD_LIBRARY_PATH="$PWD/baseline/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 runtime/check_hybrid_board.py \
  --kernel mobilevitv2_075_front_cle \
  --input-node stem_conv_Conv2D --output-node stack3_block2_pre_2_conv_Conv2D \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --input-npy baseline/samples/sample_input_fp32.npy \
  --reference-features baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy \
  --warmup 1 --repeat 5 --fuse --output-dir board_check_fuse

python3 -c "import json; r=json.load(open('board_check_fuse/report.json')); print(r['timing_seconds_mean'], r['class_index'], r.get('class_matches_reference'))"
```

### 第 4 步：20 张冒烟（需要 DPU，约 20 秒）

```bash
python3 runtime/run_hybrid_batch.py \
  --kernel mobilevitv2_075_front_cle \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --fuse --limit 20 --output-dir batch_fuse_20

python3 -c "import json; s=json.load(open('batch_fuse_20/summary.json')); print(s['tail_impl'], s['mean_seconds'], s['accuracy'])"
```

期望：`tail_impl` = `ps_tail_fused`，`accuracy` 与 20 张基线一致。

### 第 5 步：300 张全量（需要 DPU，约 3.5 分钟）

```bash
python3 runtime/run_hybrid_batch.py \
  --kernel mobilevitv2_075_front_cle \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --fuse --output-dir batch_fuse_300
```

对比：

```bash
python3 -c "
import json
for name in ('batch_results_cle', 'batch_fast_300', 'batch_fuse_300'):
    try:
        s = json.load(open(name + '/summary.json'))
        print(name, s['tail_impl'], round(s['mean_seconds']['numpy_tail'], 3),
              round(s['mean_seconds']['total'], 3), s['accuracy'])
    except FileNotFoundError:
        pass
"
```

### 第 6 步：决定保留还是回退

- `fused` 比 `fast` 快、准确率不变 → 以后都加 `--fuse`；
- `fused` 不比 `fast` 快 → 只用 `--fast-tail`（`--fuse` 不加即可，零副作用）。

---

## 4. 如果你自己改：4 个文件的精确改动

### 4.1 `run_hybrid_board.py`

**改动 1**：构造函数多一个参数。

```python
# 原来
def __init__(self, kernel_name, input_node, output_node, tail_graph, fast_tail=False):
# 改成
def __init__(self, kernel_name, input_node, output_node, tail_graph, fast_tail=False, fuse_tail=False):
```

**改动 2**：选择后段实现。

```python
# 原来
if fast_tail:
    from ps_tail_fast import FastTail
    self.tail = FastTail(tail_graph)
else:
    self.tail = NumpyPSTail(tail_graph)

# 改成
if fuse_tail:
    from ps_tail_fused import FusedTail
    self.tail = FusedTail(tail_graph)
elif fast_tail:
    from ps_tail_fast import FastTail
    self.tail = FastTail(tail_graph)
else:
    self.tail = NumpyPSTail(tail_graph)
```

**改动 3**：命令行加开关。

```python
parser.add_argument("--fuse", action="store_true",
                    help="Use ps_tail_fused (fast + GroupNorm/attention fusion)")
```

**改动 4**：把开关传下去。

```python
# 原来
with HybridRunner(..., args.tail_graph, args.fast_tail) as runner:
# 改成
with HybridRunner(..., args.tail_graph, args.fast_tail, args.fuse) as runner:
```

### 4.2 `run_hybrid_batch.py`

**改动 1**：命令行加开关（同 4.1 的改动 3）。

**改动 2**：summary 里记录用了哪种实现。

```python
# 原来
"tail_impl": "ps_tail_fast" if args.fast_tail else "ps_tail_numpy",
# 改成
"tail_impl": "ps_tail_fused" if args.fuse else ("ps_tail_fast" if args.fast_tail else "ps_tail_numpy"),
```

**改动 3**：构造时传参。

```python
with HybridRunner(..., args.tail_graph, args.fast_tail, args.fuse) as runner:
```

### 4.3 `check_hybrid_board.py`

**改动 1**：命令行加开关（同 4.1 的改动 3）。

**改动 2**：选择后段实现。

```python
# 原来
if args.fast_tail:
    from ps_tail_fast import FastTail
    tail = FastTail(args.tail_graph)
else:
    tail = NumpyPSTail(args.tail_graph)

# 改成
if args.fuse:
    from ps_tail_fused import FusedTail
    tail = FusedTail(args.tail_graph)
elif args.fast_tail:
    from ps_tail_fast import FastTail
    tail = FastTail(args.tail_graph)
else:
    tail = NumpyPSTail(args.tail_graph)
```

### 4.4 `bench_tail.py`（可选，但强烈建议）

**改动 1**：import。

```python
from ps_tail_fused import FusedTail
```

**改动 2**：构造并计时第三个模型，结果里加 `fused` 三列。

```python
fused = FusedTail(args.graph, args.weights)
fused_out = fused.predict(features)
fused_diff = float(np.max(np.abs(reference - fused_out)))
fused_ms = bench(fused, features, args.repeat) * 1000
result["ms_per_frame"]["fused"] = round(fused_ms, 2)
result["fused_speedup"] = round(baseline_ms / fused_ms, 2)
result["fused_max_abs_logit_diff"] = fused_diff
```

---

## 5. `ps_tail_fused.py` 逐段讲

### 5.1 类结构

```python
from ps_tail_fast import FastTail

class FusedTail(FastTail):
    def __init__(self, graph_path, weights_path=None,
                 fuse_group_norm=True, fuse_attention=True):
        FastTail.__init__(self, graph_path, weights_path)   # 先做 BN/Bias 折叠
        self.fused_ops = {"group_norm": 0, "attention": 0}
        self._fuse(fuse_group_norm, fuse_attention)          # 再做融合
```

继承 `FastTail` 的原因：BN/BiasAdd 折叠、im2col 卷积这些优化它已经做了，融合是在
**折叠之后的图上**继续替换，两者叠加。

### 5.2 `_fuse()`：模式匹配 + 重连边

做三件事：

1. 遍历节点，按第 2 节的模式找到 GroupNorm / 注意力的子图；
2. 把子图最后一个节点改成 `FusedGroupNorm` / `FusedAttention`，输入改接成
   `X、shape、axes、eps、out_shape`（或 `qkv、sizes、axis`）；
3. 把中间节点记进 `replace`，最后统一删除、重写所有边、重建引用计数 `self.uses`。

关键片段（GroupNorm 的识别）：

```python
for node in list(nodes):
    if node["op"] != "Reshape" or not node["name"].endswith("Reshape_1"):
        continue
    mul = producer(node["inputs"][0])
    sub = producer(mul["inputs"][0]); rsqrt = producer(mul["inputs"][1])
    add = producer(rsqrt["inputs"][0])                  # +eps
    variance = producer(add["inputs"][0])               # Mean(sq)
    squared = producer(variance["inputs"][0])           # SquaredDifference
    reshape = producer(sub["inputs"][0])                # 初始 Reshape
    # 全部匹配后：
    node["op"] = "FusedGroupNorm"
    node["inputs"] = [reshape["inputs"][0], shape_edge, axes_edge, eps_edge, out_edge]
```

注意：`shape_edge` 等常量边是**按节点名拼 `:0`** 得到的，所以匹配时不能只看 op，
还要确认上游是 Const。

### 5.3 `_evaluate()`：执行融合算子

```python
@staticmethod
def _evaluate(op, inputs, attrs):
    if op == "FusedGroupNorm":
        shape = tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1).tolist())
        axes = tuple(int(v) for v in np.asarray(inputs[2]).reshape(-1).tolist())
        eps = float(np.asarray(inputs[3]).reshape(-1)[0])
        out = tuple(int(v) for v in np.asarray(inputs[4]).reshape(-1).tolist())
        x = inputs[0].reshape(shape)
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
        return [((x - mean) / np.sqrt(var + np.float32(eps))).reshape(out)]
    if op == "FusedAttention":
        ...
    return FastTail._evaluate(op, inputs, attrs)   # 其它算子交回父类
```

### 5.4 `main()`：自检入口

就是第 3 节第 1 步用的命令，打印 `fused_ops` / `nodes` / logits 误差 / 耗时。

---

## 6. 常见错误与回退

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: ps_tail_fused` | 文件没拷到 `runtime/`，或不在该目录运行 | 确认文件在 `runtime/`，`cd deploy_in_board` 后跑 `python3 runtime/...` |
| 自检 `max_absolute_logit_error` > 1e-4 | 融合匹配错/形状轴错 | 停用 `--fuse`，把 JSON 发回来 |
| `--fuse` 后更慢 | 板端小张量开销占比高 | 只用 `--fast-tail` |
| `summary.json` 里 `tail_impl` 还是 `ps_tail_numpy` | 参数没传下去 | 检查 4.1/4.2 的改动 4 和改动 3 |
| 想彻底回退 | — | 不加 `--fuse`；代码文件留着不影响 |

---

## 7. 记录表（跑完填）

| 配置 | tail_impl | numpy_tail | total | accuracy |
| --- | --- | ---: | ---: | ---: |
| 现状 | ps_tail_numpy | 0.689 | 0.745 | 85.0% |
| `--fast-tail` | ps_tail_fast | ≈0.577 | ≈0.631 | 85.0% |
| `--fuse` | ps_tail_fused | 待测 | 待测 | 待测 |
