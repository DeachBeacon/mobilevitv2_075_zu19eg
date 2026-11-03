# 设计源码

`pytorch/` 保留训练与评估 notebook、cvnets 依赖；已去掉 ONNX 导出/评估单元和 ONNX PTQ 分支。训练数据、预训练初始化和中间 checkpoint 不随包提供，训练需自行配置。

`model_code/` 提供正式构建使用的 TF1 最小模型源码与自定义 PyTorch block 对照文件。`tensorflow/` 保留历史转换源码、H5/PB 和数值证据；`dpu/` 保留历史分段源码和产物。最终六段复现只使用根目录 build/mobilevitv2_axu19eg_release 的入口。
