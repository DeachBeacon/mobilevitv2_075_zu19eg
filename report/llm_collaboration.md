# 大模型协作记录

> 历史资料：保留原测量口径与过程描述。当前提交版的目录、模型选择和操作命令以根目录 README 为准；旧 ONNX 分支和旧 MV2_4 对照包不随本次提交。
本文件记录本工程中使用大模型（DeepSeek Harness 智能体）的方式、关键提示词类别、
自我纠错轨迹，以及哪些经验被提炼成了 `skill/` 条目。

## 1. 工作方式

- **工具形态**：带工具调用的编码智能体（可读写文件、执行 PowerShell/Python、检索文档），
  在本地工作区直接操作工程文件；
- **协作边界**：模型负责分析、写脚本、跑验证、整理文档；所有结论必须落到
  “命令 + 输出 + 文件”，不接受只有文字没有数据的结论；
- **验证闭环**：每个改动都要求有对应的校验（logits 对比、`bench_tail`、板端
  `summary.json`），否则标记为“预期/未验证”。

## 2. 提示词类别（可复用）

| 类别 | 典型提示词 | 产出 |
| --- | --- | --- |
| 消歧/定位 | “为什么 19 s 这么慢？先测 BLAS，再决定改哪里” | 分块计时脚本、GFLOP/s 判据 |
| 代码生成 | “写一个把 GroupNorm 链替换成单算子的 Tail 子类，并自检 logits” | `ps_tail_fused.py` |
| 交叉验证 | “用同一份样例对比 baseline/fast/fused 的 logits 和耗时” | `bench_tail.py` 三方输出 |
| 排错 | “dpuOpen=0 但 dpuLoadKernel 全失败，按层排查” | 分层排查清单（skill S3） |
| 文档化 | “把这段分析整理成队友能照做的上板教程” | `report/measurements/10_pure_ps_guide.md` |
| 交付 | “按竞赛推荐结构整理目录，文件名全英文” | 本提交树 |

## 3. 自我纠错轨迹（真实发生）

1. **pip 装 NumPy 失败**
   - 现象：`not a supported wheel on this platform`；
   - 模型先建议升级 pip，但板上 pip 19.2.3 升级也可能受网络限制；
   - 纠正：改为“把 wheel 当 zip 解进 site-packages”，并在文档里同时给两条路。

2. **FastTail 折叠 Identity 后 KeyError**
   - 现象：`KeyError: 'stack3_block3_output/.../Identity:0'`；
   - 原因：折叠后输出节点名变了，但没有重映射 `graph["output_tensor"]`；
   - 纠正：切分版 `ps_tail_fast.py` 在优化后 `resolve(output_name)`，并在 `skill` S4 记录失效条件。

3. **DPU 全部 kernel 加载失败**
   - 现象：`[DNNDK] Fail to load DPU kernel`，所有 `.so` 都一样；
   - 模型最初怀疑 `.so`/路径，后来用 `dlopen` + 厂商模型对照，定位为 PL/bitstream
     与编译 DCF 不匹配；
   - 纠正：形成“先分层、再对照”的排查法（skill S3），避免继续在 `.so` 上打转。

4. **融合收益被高估**
   - PC 上融合后段 −11%，但板上只 −4.8%（0.5557→0.529，早期五段）；
   - 纠正：文档改成“PC 参考 + 板上实测”，并解释 ps1/ps2 里卷积仍占大头。

5. **纯 PS 版精度口径**
   - 纯 PS 是 FP32（91.33%），不能拿来和 INT8（84.33%）直接比；
   - 纠正：在 README/教程里固定写“时间+精度一起给，精度不同不可直接比较”。

## 4. 智能体工作流（可复用）

```text
需求 → 读现场（文件/日志/数据）
     → 先测量（不改代码，拿基线）
     → 最小改动 + 数值校验（logits/耗时）
     → 上板复测（summary.json）
     → 把结论写成“命令 + 判据 + 回退”
     → 把可复用的部分提炼进 skill/
```

## 5. 哪些经验进了技能包

| 经验 | 技能条目 |
| --- | --- |
| 按节点切 PB、导出 NumPy 图 | S1 |
| 板端 OpenBLAS NumPy | S2 |
| DPU kernel 加载失败分层排查 | S3 |
| GroupNorm/注意力融合 | S4 |
| 纯 PS 基线 | S5 |
| 可复现交付包 | S6 |

## 6. 未采纳/失败的方向（留档）

- 把分类头搬进 DPU：实测 head 仅占 0.2% 时间，收益可忽略；
- 继续无脑切分：ps1+ps2 的卷积占比高，切分收益递减且增加量化边界；
- 直接用 ONNX Runtime 替代 NumPy：可行，但会换运行时，未作为主线。
