# 核对依据

> 历史资料：保留原测量口径与过程描述。当前提交版的目录、模型选择和操作命令以根目录 README 为准；旧 ONNX 分支和旧 MV2_4 对照包不随本次提交。
## 指定对话

1. [拆分中间MV2并部署到DPU](codex://threads/01a19204-10ca-7fb0-bf71-ba6e78ae509d)：六段方案、PS后端提速、早期板端耗时和精度、BN-only对照、输入范围修复、MV2及前端偏置补偿。
2. [检查 hybrid_split 部署工程](codex://threads/01a1878e-3246-7ee2-b264-7b60e377a68f)：原两段部署、N2Cube调用、真实节点名、NumPy后端、CLE与板端验证。
3. [排查 VAI 编译 Transpose 报错](codex://threads/01a18251-2796-75e2-a6d8-53d060327cfa)：Const/Transpose兼容性、PS/PL分工、独立PB切分与无TensorFlow的板端方案。

## 当前代码与记录

| 报告内容 | 核对文件 |
|---|---|
| PC、Vitis AI Docker、Runtime Docker及板端分工 | `mobilevitv2_axu19eg_release/README.md` |
| 六段顺序、真实接口形状、kernel、实际输入尺度期望值 | `mobilevitv2_axu19eg_release/code/deploy_in_board/mobilevitv2_075_deploy/runtime_config.json` |
| 单图入口、初始化、按段调度、logits校验和资源释放 | 同目录`runtime/run_hybrid_board.py` |
| 批量处理、标签、预热、计时和报告输出 | 同目录`runtime/run_hybrid_batch.py` |
| DPU实际边界与尺度查询、范围诊断、三步N2Cube调用 | 同目录`runtime/dpu_runtime.py` |
| 图像标准化及NumPy参考实现 | 同目录`runtime/ps_tail_numpy.py` |
| 算子融合、卷积矩阵乘法、引用计数与算子实现 | 同目录`runtime/ps_tail_fast.py` |
| 共享库路径、AArch64 ELF、嵌入kernel和哈希检查 | 同目录`runtime/model_libraries.py` |
| 板端真实中间特征采集及对照版本限制 | 同目录`runtime/capture_features.py` |
| PT→H5→PB及图数值校验 | `mobilevitv2_axu19eg_release/compile_tools/scripts/upstream/` |
| PC六段导出 | `mobilevitv2_axu19eg_release/compile_tools/scripts/prepare_pc.py`、`models.py`、`split_model.py`、`tail_split.py`、`export_numpy.py` |
| BN及CLE尺度约束 | 同目录`equalize.py` |
| 原生校准、偏置拟合、完整混合链路评价 | 同目录`calibrate.py`、`native.py`；项目`docs/calibration.md` |
| 编译产物身份检查、SO链接、部署包组装 | 同目录`finish.py` |
| 最终候选完整链路原生258/300（86%） | `hybrid_mv2_dpu_20261031/front_accuracy_3dpu_docker_20261101/front_accuracy/runs/hybrid1/comparison.json`及`selection.json` |
| 整理版PC验证记录 | `mobilevitv2_axu19eg_release/local_validation.json` |

以上相对路径均以工作区`C:/WorkSpace/Z19-Onborad/MobileViT/compile_tools/`为起点。

## 指标口径

- 历史板端两段方案：255/300（85%），平均单图约0.6378285秒；数据来自第一段指定对话中用户提供的`summary.json`。
- 历史板端两个MV2迁移后的早期方案：237/300（79%），单图约0.556秒；来自同一对话中对用户板端文件的核对。约13%的改善按`1 - 0.556 / 0.6378285`计算，仅用于说明该次结构调整。
- 后续最终前端候选比较：完整三段DPU原生对照252/300（84%），最佳258/300（86%），增加6张正确预测、提高2个百分点；两个MV2固定。
- 86%来自原生候选筛选，不是本次新测结果，也不是最终板端或独立测试集结论。

## 易混淆事项

- 六段边界以最终`runtime_config.json`为准；早期概念图的通道数不能直接用于最终工程。
- 三个kernel按序运行，不等于部署三个物理DPU。
- Transformer所在PS段仍有卷积；当前分类后端在PS。
- DPU的FP32 API输出已反量化，不能再次手动缩放。
- 浮点等价性针对切分和BN/CLE；偏置补偿本身会改变输出。
- 候选名、SO后缀以及已有浮点验证不能证明INT8精度，需要相应原生和板端报告。
