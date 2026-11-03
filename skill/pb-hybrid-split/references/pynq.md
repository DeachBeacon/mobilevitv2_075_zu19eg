# 通用 PYNQ PS/PL 边界工作流

本流程以“张量接口与黄金参考”组织，不写死题目、板卡、DMA 实例名或寄存器偏移。适用于用户实际使用 PYNQ Overlay、自定义 AXI IP/DMA 的情形。历史 MobileViT 工程使用 DNNDK，不能把它的运行结果标作 PYNQ 实板验证。

## 先确定适配路径

PYNQ Overlay 加载对应硬件的 `.bit/.hwh`；DPU 模型 SO 交给其对应运行库。先确认板卡、PYNQ/驱动版本、bit/hwh 配对、IP 元数据、数据/控制端口和地址，不把模型 SO 当 bitstream 使用。换板卡通常需重新生成硬件和编译产物；可以复用的是接口契约和验证方法。

对每个边界记录：源/目标 tensor、shape、layout、host dtype、device dtype、量化公式与零点、打包顺序、连续性、字节序、物理传输字节数。`scripts/tensor_contract.py` 可在任意有 NumPy 的主机检查形状/dtype/范围，不会编程 PL。它无法仅凭一个未标注数组判断 NHWC/NCHW 语义，仍要逐边界黄金参考。

```bash
python scripts/tensor_contract.py --contract interface.json --tensor features.npy --report interface_audit.json
```

## Agent 生成板端适配器时的清单

1. 先只读检查已有 Overlay 元数据与 `ip_dict`，再按用户授权的上板任务加载匹配硬件；不猜 IP 名或自动沿用另一块板的 bitstream。
2. 用 `pynq.allocate` 分配与 IP 契约一致的连续缓冲区。区分 host FP32 与设备 INT8 的字节数；普通 NumPy 数组的虚拟地址不能直接当作硬件物理地址。
3. 对非一致缓存，发送前 flush/sync_to_device，完成后 invalidate/sync_from_device。读取与写回的顺序、地址字段和 IP 启动协议以实际 HWH/驱动为准。
4. DMA 是否具备发送/接收通道、simple/SG 模式、对齐和最大传输长度，都按当前设计确认。不要假设教程 DMA 的参数适用于所有设计。
5. 流式输出正确产生 TLAST，并核对字节长度、TKEEP/对齐和批次边界。缺 TLAST 会导致接收无法完成；不要让无界 wait 掩盖问题，应按当前驱动实现超时、错误状态读取及受控停止。
6. 缓冲区保持存活至 DMA/IP 确认完成。超时后先停止相关传输并确认硬件不再访问内存，再释放；不在运行中复用/释放缓冲区。
7. 首先对一份已知输入比较 PL 输出与 CPU 黄金参考，再测完整流水线。分别记录硬件执行、数据打包、缓存同步和传输开销。

只有 float-to-device 的量化/舍入协议已知且测试过，才生成量化缓冲区；本包的范围检查不是硬件舍入模拟。若采用 DPU 的 FP32 传输 API，转换工作由该 API 完成，不重复处理。

## 已验证与未验证

参数化 shape/dtype/范围检查在主机用例中验证；本包没有自动硬件执行脚本，也未在未知 PYNQ 板卡验证 DMA/控制时序。用户需补充实际 bit/hwh 与 IP 协议后，Agent 才能完成该板适配并形成实板证明。

参考：[PYNQ Allocate/Buffer](https://pynq.readthedocs.io/en/latest/pynq_libraries/allocate.html)、[PYNQ DMA](https://pynq.readthedocs.io/en/latest/pynq_libraries/dma.html)。安装版本可能有不同驱动 API，应查对应版本文档。
