# Legacy Vitis AI 1.x / DNNDK：量化和运行时踩坑

本页描述本案例旧工具的处理经验，不把其 PB 属性格式强加给 VART、XIR、ONNX 或新工具版本。

| 现象 | 具体检查 | 正确恢复方式 |
|---|---|---|
| `Fail to load DPU kernel` | 配置 kernel、SO 内嵌 `.deephi.metadata.*`、文件名/哈希、AArch64 架构、进程启动时库路径 | 完成编译/链接，整包同步；在 Python 启动前令 model/ 位于 LD_LIBRARY_PATH |
| `.so` 改名仍无法加载 | 文件名与嵌入 kernel 是两层信息 | 改文件名不改变内嵌 kernel；用匹配 net_name 编译链接或更新真实配置 |
| `Invalid pixel value ... -129` | 实际输入乘运行时 scale 后是否越出 [-128,127]；校准前缀与板端是否一致 | 先定位实际范围和量化元数据，不通过误改 JSON 或强制截断掩盖 |
| eval 已改但板端仍显示旧 scale | 输入 FixNeuron、deploy Placeholder.opos、输入 Pad 的 ipos/opos、首层卷积 ipos | 同步全部路径标签，重原生评估/重编译/重链接；新 SO 与新配置一起拷贝 |
| 首层卷积前有 Pad | 追踪 Placeholder→FixNeuron/Pad→首卷积路径 | 输入不一定直接连 Conv；Pad 自身的量化器和部署标签也需一致 |
| 精度低但没有越界 | 查看逐层误差、参数/激活网格、偏置步长和同输入板端对齐 | 没越界只说明范围可提交，不证明舍入/量化误差可接受 |
| 权重均衡后精度更差 | 检查偏置/激活放大，BN-only 与限幅对照 | 用完整任务指标选候选，不能以权重极差改善作为验收 |
| Runtime Python 3.5 报输出目录不存在 | 对尚不存在目录调用了 Path.resolve() | 允许新路径的规范化，再 mkdir；保留已有目录保护，不简单预建目录绕过来源校验 |
| Docker 找不到 manifest | 在出错容器确认真实目录名和拷贝层级 | `capture_a_300` 与 `board_capture_300` 不是同一路径；Windows 看见文件不能证明容器可见 |
| 新 PB 搭配旧 ELF/SO 仍能运行 | 核对模型、报告、编译与链接哈希 | 可运行不等于同一模型；重建失效产物并核对整包来源 |

输入固定小数位置 `p` 对应 real→integer multiplier `2**p`。保守实数端点区间为 `[-128/scale,127/scale]`；检查端点不等于模拟实际舍入。案例中前端 p=5、内部 MV2 p=0 只是其数据/工具下的选择，不作为陌生模型默认值。

输出 `dpuGetOutputTensorInHWCFP32` 已提供反量化浮点，不能再除一次输出 scale。中间特征不要再做 mean/std 图像归一化。输入范围有意采取 clip 时，要按同一策略重测并记录截断数及精度，不能混入默认 error 策略的结果比较。

Vitis AI 1.2 的 method=0/1 分别为 non-overflow/min-diffs；用安装版本帮助确认。来源：[官方量化参数](https://docs.amd.com/r/1.2-English/ug1414-vitis-ai/vai_q_tensorflow-Usage?contentId=KtxdhOqAQvkW3cz0Vwa2Sw)。CLE/偏置校正方法背景：[Nagel 等，ICCV 2019](https://arxiv.org/abs/1906.04721)。这些资料不构成该模型一定恢复精度的证明。
