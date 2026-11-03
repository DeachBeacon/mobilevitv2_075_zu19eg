# 板端程序

正式包是 `deploy_best_fused/`，内含配套 runtime、JSON/NPZ、配置和三个 SO；它可以独立复制到已有匹配 DPU 系统的板子。`ps_reference/` 的纯 PS 对照与它共用模型/运行代码，使用对照时复制整个 board。

`run_visual_board.py` 使用 qs 的新版实现：默认定位同级正式包，含总结看板、自动显示环境设置、异常降级处理。操作命令见根目录 README。图片从根目录 `data/calibration/jpegimages/` 读取，新的结果请指定新目录。

正式 MV2_4 仅为 `libdpumodelmobilevitv2_075_mv4a_ctlb100.so`。历史结果在根目录 `data/results/`，不是重新运行产物。图形显示及真实板端性能仍需实机验证。
