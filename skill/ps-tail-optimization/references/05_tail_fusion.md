# 后段融合优化教学（2026-10-31）

本文对应《提速效果分析与下一步》里的两条优化：

| 编号 | 优化 | 状态 | 预期 |
| --- | --- | --- | --- |
| 3.1 | 开 `--fast-tail`（折叠 BN/BiasAdd、im2col） | **已实现、已验证** | 后段 0.689 → ≈0.577 s，端到端 ≈0.63 s |
| 3.2 | 融合 GroupNorm 与可分离注意力 | **已提供 `ps_tail_fused.py`（`--fuse`），公式与 logits 已验证** | 板上预计再省 50–90 ms，端到端 ≈0.55–0.58 s |

两条都不改模型、不改 DPU，数值等价（1e-6 级），随时可回退。

---

## 1. 3.1 直接用 `--fast-tail`（10 分钟）

### 1.1 它做了什么

`ps_tail_fast.py` 读同一份 `ps_tail_numpy.json` / `.npz`，在**加载时**做等价改写：

| 改写 | 数量 | 说明 |
| --- | --- | --- |
| BN 折进卷积 | 11 | `W ← W·γ/√(σ²+ε)`，`b ← (b−μ)·scale+β` |
| BiasAdd 折进卷积偏置 | 39 | 直接加到 bias |
| 去掉 Identity | 155 | 纯别名 |
| 卷积实现 | 45 | 连续 im2col + matmul；1×1 直接 matmul；depthwise 逐 tap |

节点数 941 → 736，数学完全等价（板上 logits 差 3.1e-6）。

### 1.2 板上运行

```bash
cd ~/test4/handover_speedup_20261030/deploy_in_board
export LD_LIBRARY_PATH="$PWD/baseline/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 runtime/run_hybrid_batch.py \
  --kernel mobilevitv2_075_front_cle \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json \
  --fast-tail --output-dir batch_cle_fast_300
```

### 1.3 怎么确认生效

```bash
python3 -c "import json; s=json.load(open('batch_cle_fast_300/summary.json')); print(s['tail_impl'], s['mean_seconds'], s['accuracy'])"
```

期望：

- `tail_impl` = `ps_tail_fast`；
- `mean_seconds.numpy_tail` ≈ **0.577**（原来 0.689）；
- `mean_seconds.total` ≈ **0.631**；
- `accuracy` 应与 0.85 完全一致（同一模型，只改运算顺序）。

### 1.4 如果反而更慢

不加 `--fast-tail` 即可，原路径不受影响。板上实测 fast 是 1.19× 快，正常不会更慢。

---

## 2. 3.2 融合 GroupNorm 与注意力（已提供实现，可直接用）

### 2.1 为什么要融合

板上 profile（后段 606.6 ms）里，除卷积外最大的一块是逐元素/小算子：

| 算子 | 时间 | 说明 |
| --- | ---: | --- |
| Mean | 26.3 ms | group norm 的均值/方差 |
| SquaredDifference | 31.4 ms | 方差 |
| Sub | 16.1 ms | 去均值 |
| Mul | 34.8 ms | 归一化 + 注意力加权 |
| Sum | 8.8 ms | 注意力按 token 求和 |
| SplitV | 9.1 ms | qkv 拆分 |
| Softmax | 5.4 ms | 注意力权重 |
| Rsqrt / Transpose | 5.5 ms | 归一化/转置 |

这些算子每个都要把 `[1,4,256,96]`（约 393 KB）整趟读一遍、写一遍。把它们合成一个函数，
只遍历 1–2 趟，就能省下 50–90 ms。

### 2.2 GroupNorm 的图结构（实测）

每个 block 有两个 group norm（attention 前、MLP 前），图中是一条固定链（以
`stack3_block2_attn_group_norm` 为例）：

```text
X ──Reshape(shape=[1,-1,1,96])──► R
R ──Mean(axes=[1,3], keepdims)──► mean
R, mean ──Sub──► sub
R, mean ──SquaredDifference──► sq
sq ──Mean(axes=[1,3], keepdims)──► var
var, eps(1e-5) ──AddV2──► add
add ──Rsqrt──► rs
sub, rs ──Mul──► norm
norm ──Reshape(out=[1,4,256,96])──► Y
```

也就是：把 `[1,4,256,96]` reshape 成 `[1,1024,1,96]`，在 `(1,3)` 两个轴上求均值/方差，
归一化后再 reshape 回原形状。**逐样本一个标量均值和方差**。

**融合公式**（已用样例特征验证，最大差 `9.5e-7`）：

```python
x = X.reshape((1, -1, 1, 96))
mean = x.mean(axis=(1, 3), keepdims=True)
var = ((x - mean) ** 2).mean(axis=(1, 3), keepdims=True)
Y = ((x - mean) / np.sqrt(var + 1e-5)).reshape((1, 4, 256, 96))
```

### 2.3 可分离注意力的图结构（实测）

这个模型用的是 MobileViTv2 的 **separable attention**：qkv 卷积一次输出 193 通道，
拆成 `1 + 96 + 96`：

```text
qkv [1,4,256,193]
  ├─ 前 1 通道     : score  [1,4,256,1]
  ├─ 中间 96 通道  : value  [1,4,256,96]
  └─ 后 96 通道    : gate   [1,4,256,96]  （先 ReLU）

w       = softmax(score, axis=2)                 # 对 256 个 token 归一化
context = sum(w * value, axis=2, keepdims=True)  # [1,4,1,96]
out     = ReLU(gate) * context                   # 广播回 [1,4,256,96]
```

图中的节点：`SplitV → [transpose → Softmax → transpose_1] → Mul(value) → Sum → Mul(Relu(gate))`。

**融合公式**（已验证，最大差 `0.0`）：

```python
score = qkv[..., :1]
value = qkv[..., 1:97]
gate = np.maximum(qkv[..., 97:], 0)
scores = score - score.max(axis=2, keepdims=True)
w = np.exp(scores)
w /= w.sum(axis=2, keepdims=True)
out = gate * (w * value).sum(axis=2, keepdims=True)
```

### 2.3.1 现成实现与验证（已经写好）

`runtime/ps_tail_fused.py` 在 `FastTail` 基础上自动识别并替换（`bench_tail.py` 已能同时对比三者）：

| 项 | 结果 |
| --- | --- |
| 融合的 GroupNorm | **21 处** |
| 融合的注意力 | **9 处** |
| 节点数 | 941 → **505** |
| 样例 logits 与参考 | 最大差 **1.5e-6**，类别一致 |
| PC 实测（bench_tail，3 次） | baseline 80.8 ms / fast 44.6 ms / **fused 38.0 ms** |
| 相对 baseline | fast 1.4–1.8×、**fused 1.4–2.1×**（比 fast 再快 0–18%，随负载波动） |
| 板上用法 | `--fuse`（已隐含 `--fast-tail`） |

说明：PC 上卷积占主导，fused 相对 fast 只再快 0–18%；板上逐元素是内存瓶颈，
融合的收益预计更大，必须上板用 `bench_tail.py` 实测确认。

### 2.4 自己实现的话：模式匹配与融合算子（教学部分）

思路：在 `FastTail._optimize()` 里再增加两个“模式替换”步骤，把上述子图替换成单个节点，
执行时走一个融合函数。**建议一次只加一个融合，先跑通再加第二个。**

#### 步骤 1：GroupNorm 模式匹配

按键名/结构找到每个 `..._group_norm/Reshape_1`，验证它的上游是
`Mul(Sub, Rsqrt)`，再向上验证 `Mean/SquaredDifference/Reshape` 链，然后：

```python
def fuse_group_norm(nodes, by_name, replace):
    for node in list(nodes):
        if node["op"] != "Reshape" or not node["name"].endswith("Reshape_1"):
            continue
        mul = by_name[node["inputs"][0].split(":")[0]]
        if mul["op"] != "Mul":
            continue
        sub, rsqrt = (by_name[e.split(":")[0]] for e in mul["inputs"])
        if sub["op"] != "Sub" or rsqrt["op"] != "Rsqrt":
            continue
        add = by_name[rsqrt["inputs"][0].split(":")[0]]
        var_mean = by_name[add["inputs"][0].split(":")[0]]
        eps = by_name[add["inputs"][1].split(":")[0]]
        if var_mean["op"] != "Mean" or eps["op"] != "Const":
            continue
        sq = by_name[var_mean["inputs"][0].split(":")[0]]
        reshape = by_name[sub["inputs"][0].split(":")[0]]
        if sq["op"] != "SquaredDifference" or reshape["op"] != "Reshape":
            continue
        # 收集常量：reshape shape、mean axes、eps、输出 shape
        shape_const = by_name[reshape["inputs"][1].split(":")[0]]
        axes_const = by_name[var_mean["inputs"][1].split(":")[0]]
        out_const = by_name[node["inputs"][1].split(":")[0]]
        node = dict(node)
        node["op"] = "FusedGroupNorm"
        node["inputs"] = [reshape["inputs"][0], shape_const["name"] + ":0",
                          axes_const["name"] + ":0", eps["name"] + ":0",
                          out_const["name"] + ":0"]
        by_name[node["name"]] = node
        for victim in (mul, sub, rsqrt, add, var_mean, sq, reshape):
            replace[victim["name"]] = node["name"]
```

> 要点：把中间 7 个节点标记为“被替换”，最后统一从图里删掉、重写边（和 `_optimize` 里
> 处理 BiasAdd/BN 的方式一致）。`Reshape_1` 节点本身保留，只是 op 改成 `FusedGroupNorm`。
> 注意 `Mean` 的 `keep_dims` 是 True、axes 来自 const（本模型是 `[1,3]`）。

#### 步骤 2：注意力模式匹配

```python
def fuse_attention(nodes, by_name, replace):
    for node in list(nodes):
        if node["op"] != "Mul":
            continue
        left, right = (by_name[e.split(":")[0]] for e in node["inputs"])
        if {left["op"], right["op"]} != {"Relu", "Sum"}:
            continue
        gate, total = (left, right) if left["op"] == "Relu" else (right, left)
        if total["op"] != "Sum":
            continue
        weighted = by_name[total["inputs"][0].split(":")[0]]     # Mul(value, weights)
        if weighted["op"] != "Mul":
            continue
        split = by_name[gate["inputs"][0].split(":")[0]]          # SplitV
        if split["op"] != "SplitV":
            continue
        node = dict(node)
        node["op"] = "FusedAttention"
        node["inputs"] = [split["inputs"][0], split["inputs"][1],    # qkv, sizes const
                          split["inputs"][2]]                          # split axis const
        by_name[node["name"]] = node
        for victim in (split, gate, total, weighted, ...):            # 连 transpose/softmax 一起
            replace[victim["name"]] = node["name"]
```

> 这个模式里还要把两条 `Transpose`、`Softmax`、`split` 一起标记为被替换；`FusedAttention`
> 直接从 qkv 算出 `[1,4,256,96]`，跳过中间 8 个节点。

#### 步骤 3：执行时加两个分支

```python
def _fused_group_norm(inputs):
    x = inputs[0].reshape(tuple(np.asarray(inputs[1]).reshape(-1).tolist()))
    axes = tuple(np.asarray(inputs[2]).reshape(-1).tolist())
    eps = float(np.asarray(inputs[3]).reshape(-1)[0])
    out = tuple(np.asarray(inputs[4]).reshape(-1).tolist())
    mean = x.mean(axis=axes, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
    return [((x - mean) / np.sqrt(var + eps)).reshape(out)]

def _fused_attention(inputs):
    qkv = inputs[0]
    sizes = np.asarray(inputs[1]).reshape(-1)
    score = qkv[..., :sizes[0]]
    value = qkv[..., sizes[0]:sizes[0] + sizes[1]]
    gate = np.maximum(qkv[..., sizes[0] + sizes[1]:], 0)
    scores = score - score.max(axis=2, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=2, keepdims=True)
    return [gate * (w * value).sum(axis=2, keepdims=True)]
```

> 注意：`SplitV` 的 sizes 本模型是 `[1, 96, 96]`（从 const 读，别写死）；softmax 沿 axis=2；
> 融合后的输出形状必须和原 `Mul` 节点一致（`[1,4,256,96]`）。

### 2.5 验证与计时（必须做）

```bash
cd ~/test4/handover_speedup_20261030/deploy_in_board

# 1) 数值等价：同一份样例特征，比较 logits
python3 runtime/ps_tail_fast.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy \
  --reference-logits baseline/samples/sample_reference_logits.npy --repeat 5

# 2) A/B：基线 vs 融合后
python3 runtime/bench_tail.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5

# 3) 分块/分算子耗时对比
python3 runtime/profile_tail_blocks.py \
  --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 3
```

通过标准：`max_abs_logit_diff < 1e-4`（预期 1e-5 以内）、类别不变、
`Mean/SquaredDifference/Sub/Mul/Sum/SplitV/Softmax` 的时间明显下降。

### 2.6 预期收益与风险

| 融合项 | 预计省 | 依据 |
| --- | ---: | --- |
| GroupNorm（约 21 处） | 30–55 ms | 每处少 4–5 趟 393 KB 读写 |
| 注意力（9 处） | 20–35 ms | 省掉 split/transpose/softmax/mul/sum 的中间数组 |
| 合计 | **50–90 ms** | 后段时间的 8–15% |

叠加后：`numpy_tail 0.577 → 0.50–0.53 s`，`total ≈0.55–0.59 s`，相对 19 s 约 **33–35×**。

风险：

- 形状/轴写错会“能跑但结果不对”，必须用 logits 对比兜底；
- `-1` 的 reshape 要在运行时按实际形状展开；
- 融合只适用于这个模型的固定模式，换模型要重新确认；
- 建议用 `--fuse` 开关控制，默认关闭，方便回退。

---

## 3. 上板流程（两条一起）

```bash
# 1) 先跑 3.1（已可用），确认 0.63 s
python3 runtime/run_hybrid_batch.py --kernel mobilevitv2_075_front_cle \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json --fast-tail \
  --output-dir batch_fast_300

# 2) 实现 3.2 后，先在板上做数值 + 计时
python3 runtime/bench_tail.py --graph baseline/model/ps_tail_numpy.json \
  --features-npy baseline/samples/sample_features_fp32.npy --repeat 5

# 3) 再跑 300 张，确认 accuracy 不变、total 再降
python3 runtime/run_hybrid_batch.py --kernel mobilevitv2_075_front_cle \
  --image-dir ./test1_p --labels-from-folders \
  --tail-graph baseline/model/ps_tail_numpy.json --fuse \
  --output-dir batch_fuse_300
```

> `--fuse` 已经在 `run_hybrid_batch.py` / `run_hybrid_board.py` / `check_hybrid_board.py`
> 里加好，默认关闭；`--fuse` 会走 `FusedTail`（已包含 fast 的 BN/Bias 折叠）。

---

## 4. 记录表

| 配置 | numpy_tail | total | 准确率 | 相对 19 s |
| --- | ---: | ---: | ---: | ---: |
| 现状（ps_tail_numpy） | 0.689 | 0.745 | 85.0% | 25.5× |
| + `--fast-tail` | ≈0.577 | ≈0.631 | 85.0% | ≈30× |
| + GroupNorm 融合 | ≈0.53 | ≈0.59 | 待测 | ≈32× |
| + 注意力融合 | ≈0.50 | ≈0.56 | 待测 | ≈34× |

---

## 5. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 融合后 logits 差 > 1e-3 | 轴/keepdims/reshape 写错 | 用样例逐层对比：先比 groupnorm 输出，再比 attention 输出 |
| 融合后更慢 | 小张量下 Python/拷贝开销占比高 | 只保留 groupnorm 融合，attention 回退 |
| `-1` reshape 报错 | 用 `tuple(int(v) for v in shape)` 后含 -1 | 先算 `np.prod` 推断，或运行时 `x.reshape(shape)` 让 NumPy 处理 |
| 准确率下降 | 通常不会是融合造成的 | 对比 `same_class`，确认只是数值 1e-6 差异 |

---

## 6. 一句话

- **3.1 现在就能用**：`--fast-tail`，10 分钟，0.745 → 0.631 s（30×）。
- **3.2 已经写好**：`ps_tail_fused.py` + `--fuse`，数值已验证；板上先 `bench_tail`
  确认正确性，再跑 300 张看 `numpy_tail` 是否从 0.577 降到 ~0.50。
