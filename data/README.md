# 输入数据与结果

`weights/` 仅保存一份原始 PT。`calibration/jpegimages/` 保存唯一一份 300 张 c0～c9 图片；校准与测试共用，不能作为独立泛化测试集。原来的 test1_p 目录不再重复保存。

`samples/` 保存数值验证输入与特征。`sample_aliases.json` 记录已合并的重复 NPY 与当前路径。`results/` 保留历史板端实测 JSON、逐图结果、汇总图和展示样例，内容不作为本次新测量。
