# MobileViTv2 前段 INT8 精度下降排查

日期：2026-10-29。用户板端截图显示 c0 图片大量被预测为 c5/c7，单图约 19 秒。用户确认板端 NumPy 后段 `self_check_passed: true`。

## 当前结论

已有很强证据指向前段 INT8 量化造成主要精度损失，尤其是 BN 折叠后的深度卷积大权重与自动 shift-cut 调整。尚未取得原生 Vitis FixNeuron 或真实 DPU 的逐层误差，不能排除板端额外的接口、尺度、加载产物问题。

此前浮点切分和 NumPy 后段一致性验证不覆盖 INT8 精度。此次新增量化路径的定位，原始模型、量化文件、ELF/SO 和板端推理代码未修改。

## 300 张图片的主机实验

数据为 `modle/driver/Images`；已在此前复核中确认与用户 `test1_p` 的 300 张图片内容相同。预处理继续使用校准时的 TensorFlow 解码、RGB、除以 255、mean/std。类别映射为 cN→N。

| 路径 | 正确数 | 准确率 |
| --- | ---: | ---: |
| 原始 FP32 整网 | 275/300 | 91.67% |
| quantize_eval 图关闭 FixNeuron、保留其权重/BN 折叠，再接 FP32 后段 | 275/300 | 91.67% |
| 按现有固定量化位置近似舍入/截断的前段，再接 FP32 后段 | 27/300 | 9.00% |

关闭 FixNeuron 的折叠图与原模型最大 logits 误差为 `1.811981201171875e-05`，这批图片类别完全一致。近似量化输出中，91 张预测为 c5，209 张预测为 c7，其他类别没有预测。

第三行是主机近似实验：固定量化位置、`floor(x*scale+0.5)`、有符号 8 位截断、反量化，以浮点执行其余前段运算。它没有模拟 DPU 的全部累加器/移位/舍入规则，也未调用 Vitis 原生 FixNeuron，因此不能声称是 DPU 的实际 9% 精度。它复现了与板端相似的类别集中退化，提供了量化路径损伤的证据。

报告：`quant_front_emulated_all300.json`。另外保留均匀抽样 10 张及截图前 13 张的实验，结果分别为 FP32 10/10、11/13，近似量化 1/10、0/13。

## 深度卷积权重的具体问题

量化图中两个警告对应的权重数据：

| 层 | 折叠前卷积权重范围 | BN 通道乘数最大值 | 折叠后最大权重 | 当前量化位置 | 8 位实值范围 |
| --- | --- | ---: | ---: | ---: | --- |
| stack1_block1_deep_2_dw_conv | -3.132～1.204 | 514.30 | 368.05 | 0 | -128～127 |
| stack2_block2_deep_2_dw_conv | -2.650～1.867 | 1064.56 | 1473.46 | -2 | -512～508 |

BN 乘数为 `gamma/sqrt(variance+epsilon)`；上述两层乘数绝对值中位数分别约 4.71、6.49，少量通道明显更大。浮点折叠仍保持输出，但会使每层权重的动态范围变大。

量化日志对第一层权重量化位置由 -2 调整为 0，对第二层由 -4 调整为 -2，以满足 shift-cut 范围。调整后的有符号 8 位范围无法表示上述大权重，会发生饱和截断。对整个量化权重数组的主机近似结果，两层权重归一化 RMSE 分别约 43.4%、54.8%。这不是“生成文件即可忽略”的告警。

完整深度卷积统计：`depthwise_quantization_stats.json`。前段最终特征也使用量化位置 -2，即步长 4，会继续带来边界特征误差。

## 下一步：原生量化评估

将新增 `diagnostics/compare_front_quantization.py` 同步到 Vitis AI Docker 的工程对应位置，保留工程已有的 `hybrid_split/common.py`、`ps_tail.py`。进入工程根目录执行：

```bash
conda activate vitis-ai-tensorflow
python diagnostics/compare_front_quantization.py \
  --backend native \
  --source-pb modle/driver/pb_file/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb \
  --tail-pb modle/driver/pb_file/ps_tail_fp32.pb \
  --quantized-pb quantize_output_front/quantize_eval_model.pb \
  --image-dir modle/driver/Images \
  --samples 10 \
  --output diagnostics/native_quant_accuracy_10.json
```

脚本原生模式会导入 `from tensorflow.contrib import decent_q` 注册 FixNeuron，读取真正的量化输出 `stack3_block2_pre_2_conv/Conv2D/aquant:0`，再连接未量化后段。用 `--samples 0` 和新的输出文件名可评估全部图片。

这一步符合 [Vitis AI 1.2 官方量化评估说明](https://docs.amd.com/r/1.2-English/ug1414-vitis-ai/Evaluate-Quantized-Model-Optional?contentId=T4e4UHsclWP7f9KpoQ_FJQ)。本机普通 Windows TF 1.15 没有 decent_q，原生模式尚未执行；不要把主机 emulated 模式的结果当成它的结果。

## 板端单样例对照

先结束长批量任务，再在板端部署目录执行以下检查（所需脚本已在之前批量部署包内）：

```bash
python3 check_hybrid_board.py \
  --input-npy sample_input_fp32.npy \
  --reference-features sample_features_fp32.npy \
  --reference-logits sample_reference_logits.npy \
  --warmup 0 --repeat 1 \
  --output-dir board_quant_check
```

它记录实际 Runtime 输入/输出尺度、特征 RMSE/余弦相似度、混合 logits 和分段耗时。当前量化参数预期输入尺度为 32、输出反量化尺度为 4，仍应以实际 Runtime 和原生量化图对照为准。使用已经预处理的 NPY 可以排除图片解码差异。

原生评估还会保存第一张图的输入、FP32/量化特征和 logits。将原生量化特征与板端 DPU 特征逐元素比较，可以区分量化本身和编译/运行时额外误差。量化结果与 DPU 不要求未经分析就套用 FP32 的 1e-5 阈值。

## 修正方向

如果原生评估也明显掉精度，应优先针对上述深度卷积与相邻层做保持浮点函数的通道尺度平衡，减少 BN 折叠后的离群权重；重新做浮点一致性、量化精度、编译和板端对照。若简单平衡仍不够，再评估与 Vitis AI 1.2 兼容的量化感知微调或调整切分。

不要直接改最终输出的反量化尺度来补偿内部权重截断；丢失的信息无法靠整体乘数恢复。也不要仅手改权重量化位置绕过 shift-cut 约束后复用旧 ELF/SO。任何量化参数或前段模型改动都应重新量化/编译，并检查原生精度及真实板端结果。

如果原生量化精度正常而 DPU 明显差，下一步改查实际加载的 SO、DPU 配置、接口尺度及布局，不应继续盲目调整量化参数。
