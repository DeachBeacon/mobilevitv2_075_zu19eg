# 工程包提交检查

本次只交付新工程目录，不自动提交赛事网站。原有两份工程保留。

| 赛事工程包要求 | 本包材料 | 当前状态 |
|---|---|---|
| 完整工程与源码 | src、board、正式构建脚本及固定 PT 输入 | 可从固定权重重建部署；完整训练集、预训练初始化和启动镜像另需准备 |
| 仿真或验证结果 | sim、主机重建记录、纯 PS 冒烟、最终产物核对 | 证明范围见 release_validation.md；不是 RTL/HLS 仿真或新板端测量 |
| 综合与实现报告 | build/implementation_report.md、厂商 HWH、最终三段编译日志和 ELF | 整设计 LUT/FF/BRAM/DSP 实现利用率和时序报告待补，未取得替代提交许可 |
| 上板工程、脚本、实测与参考比对 | board、data/results/final_20261103 的逐图标签/预测 | 历史结果保留；新副本尚未重新上板 |
| 可复现构建脚本 | build/mobilevitv2_axu19eg_release/compile_tools | 本次 PC 验收结果见 local_validation.json；新 Vitis/Runtime 编译执行待完成 |
| 名称与目录说明 | 根目录 README.md、命名自动检查 | 按确认口径保留 README.md/SKILL.md，其余小写英文 |
| 技能与协作记录 | skill、双方协作记录、release_changes.json | 保留并同步引用及哈希 |

从工程根目录运行 `python -B build/verify_submission.py --read-only` 校验搬移后文件。全包未压缩大小由验收脚本检查小于 200,000,000 字节；该阈值沿用团队此前整理约束，不解释为本次所引条款的官方限制。

设计报告 PDF、视频、海报、照片和授权文件等另外按赛事完整要求准备，本次工程目录不替代这些独立材料。正式压缩格式和上传限额以实际赛方入口为准，本次未生成 ZIP/RAR。
