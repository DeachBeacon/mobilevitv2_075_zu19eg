# 昨晚 CLE 前段候选

`dpu_front_cle_fp32.pb` 包含前段 14 个 BN 折叠与通道尺度均衡，已经完成 FP32 等价验证。

本目录没有 CLE 的量化 PB、ELF 或 SO；用户确认尚未生成。下一步使用上级目录的四个编号 Shell。原前段及既有二进制在 `compile_tools/modle/driver/pb_file`、`compile_tools/quantize_output_front`、`compile_tools/compile_output_front`，不得与此候选混用。
