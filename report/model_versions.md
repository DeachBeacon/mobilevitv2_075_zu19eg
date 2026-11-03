# 当前模型对应关系

| 内容 | 位置 |
|---|---|
| 原始 PT | `data/weights/bestmodel_v1_only_weights.pt` |
| 正式转换与编译入口 | `build/mobilevitv2_axu19eg_release/compile_tools/` |
| 最终前端 SO | `board/deploy_best_fused/model/libdpumodelmobilevitv2_075_fronta_ctlb100.so` |
| 最终 MV2_4 SO | `board/deploy_best_fused/model/libdpumodelmobilevitv2_075_mv4a_ctlb100.so` |
| 最终 MV2_5 SO | `board/deploy_best_fused/model/libdpumodelmobilevitv2_075_mv2_5_cle.so` |
| 六段配置 | `board/deploy_best_fused/runtime_config.json` |

旧 mv2_4_cle 对照 SO 包与 ONNX 分支不随本提交提供。src/tensorflow 与 src/dpu/models 中保留的其他不同阶段模型属于历史验证资料。正式版本以三项 SO 的配置内哈希为准。
