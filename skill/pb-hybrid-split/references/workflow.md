# 分段工作流与数据契约

## 输入清单

记录源 PB、完整输入/输出 tensor:port、输入预处理、样本、CPU 库/版本、DPU/FPGA 目标和运行库。板上只有 Python 不等于有 TensorFlow；有 NumPy 也不代表任意 PB 都有可运行的 NumPy 实现。优先读现有脚本/日志，缺少的事实再问用户。

工具仅接收 frozen GraphDef。SavedModel、checkpoint、H5 需先在原框架正确导出推理图；不是把文件改后缀。量化 eval/deploy PB 已有特殊算子或嵌入属性，不能当成黄金 FP32 源图。

## 选边界

- 用实际版本的支持表、图形状和编译日志判断候选。量化器与编译器的支持范围分别确认。
- 按实际边界列出所有数据依赖。残差分支、多输入、多输出和 `SplitV:1` 等非零端口不能只保留一个默认张量。
- 首版可以少量大段建立可运行基线，再根据卷积耗时与传输开销增加 DPU 段；不要把“更多在 PL”当成必然更快。
- GroupNorm 的运行时均值/方差不能按推理 BN 固定统计量折叠；文件叫 `gnfold` 也不能据此认定整网已消除 GN。

## 导出与验证

`partition.example.json` 使用原始 source tensor 作为段间逻辑键，`placeholder` 是导出 PB 的局部输入名。除非复用原 Placeholder，否则局部别名要避开源节点名。输出保留端口；段的输入必须来自原始输入或前面某段输出。

NPZ 样本只存实际数值输入，以完整原始输入 tensor 名为键。预处理由调用者按模型定义完成；工具不替用户猜通道顺序、归一化或 resize。每份 NPZ 是一次调用的输入，不能不加说明把样本维度当成模型 batch。

`split` 先裁剪并重新导入验证每段，再写 PB；任何未声明的存活 Placeholder 都使计划失败。保留控制边，不能用删边伪造闭包。有状态操作、变量、动态执行依赖应先导出正确纯推理图或实现专用切分。

`verify` 重新读取每个 PB 和哈希，同一输入跑黄金原图和整个段链，在每个输出边界比较。浮点采用声明的误差容限，整数/布尔输出做精确比较。失败先定位第一个偏差边界，不只看最后的分类结果。

## 可选 CPU NumPy 后端

本包适配器覆盖 Conv/DWConv、推理 FusedBatchNormV3、BiasAdd、ReLU、若干逐元素/归约、Reshape/Transpose/SplitV/Softmax/MatMul/Pad。具体 whitelist 与属性检查以代码为准。仅一个 FP32 输入/输出；卷积限 NHWC、SAME/VALID，不支持的 layout/属性/BN 辅助输出要失败退出。

导出后检查 JSON/NPZ 与 PB 的来源哈希，再用原图产生的同一段输入比对。只有通过这一层，才把 `numpy_runtime.py` 交给无 TF 的板端。新算子由 Agent 补实现并增加黄金参考用例，不能 silently skip。

## 校准与运行

图像输入段用正确图像预处理；中间 DPU 段用真实前缀量化图/板端捕获产生的特征。后续段的缓存依赖前缀模型、量化位置、预处理和数据哈希，任意变化后失效。

交付每段模型/图/权重、kernel、实际 runtime 节点、shape/dtype/layout/scale、执行顺序和完整包。FP32 传输 API 已完成的反量化不重复执行。分别测预处理、传输/DPU、CPU、总体，并说明预热、模型加载是否计入。

验证分层：主机 FP32 → 主机 CPU 后端 → Docker 原生量化 → 编译/链接 → 板端相同输入 → 完整链路精度/速度 → 独立集。未执行层级保持“待验证”。
