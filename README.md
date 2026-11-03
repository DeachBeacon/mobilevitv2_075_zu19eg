# MobileViTv2-0.75 on AXU19EG / Z19

在Alinx Z19（ZU19EG）开发板上部署 MobileViTv2-0.75，实现驾驶员行为的 10 类图像分类。工程采用 **DPU + ARM 混合推理**：卷积段在 PL 侧 DPU 上运行，GroupNorm、注意力及分类后段在 PS 侧通过 NumPy 执行，并提供批量评估、外接显示器演示和纯 PS 对照程序。

仓库已提供可直接上板的模型库和运行程序。只想运行演示，请从下面的“快速上手”开始；需要从原始权重重新生成部署包，再阅读“完整构建流程”。

> **启动镜像需单独获取。** 本工程使用 **Alinx 提供的 bin 文件（启动镜像）**。该镜像不以 MIT 协议开源，因此**未放入本仓库，也不属于本仓库的 MIT 授权范围**。请从 Alinx 配套资料获取与本工程 DPU 平台匹配的版本，并遵守其单独的许可条款。

- [快速上手：直接上板](#quick-start)
- [文件结构](#project-layout)
- [完整构建流程](#full-workflow)
- [验证与实测结果](#validation)
- [常见问题](#faq)
- [文档与参考资料](#references)
- [许可证](#license)

<a id="quick-start"></a>

## 快速上手：直接上板

本节使用现成的 `board/deploy_best_fused/`，无需先训练、转换或编译模型。

### 1. 准备开发板系统

准备 Alinx Z19 开发板、启动介质、Alinx 配套启动镜像和网络连接；需要图形演示时，再连接显示器、键盘和鼠标。

1. 按 Alinx 配套说明把镜像写入SD卡、设置启动方式并启动系统。手册示例使用 `sd_card.img` 整卡镜像，实际写入方式以取得的 bin / 系统镜像及其配套说明为准。
2. 确认系统具备匹配的 DPU 硬件、驱动及 DNNDK / N2Cube 运行库。模型编译目标为仓库中的 `axu19eg_ko` 配置。
3. 在板端终端运行 `ifconfig` 查看 IP，使 PC 与开发板能够通过 SSH 连接。

在**开发板终端**检查 Python 依赖：

```bash
python3 -c "from dnndk import n2cube; import numpy as np; import cv2; print('NumPy:', np.__version__); print('OpenCV:', cv2.__version__)"
python3 -c "import numpy as np; np.show_config()"
```

板端需要 Python 3、NumPy 和 OpenCV，建议使用带 OpenBLAS / BLAS 的 NumPy。板端推理不需要安装 PyTorch 或 TensorFlow。

若镜像未安装 DNNDK，从 Alinx 配套资料取得 `vitis-ai_v1.2_dnndk.tar.gz`，传到板端后按手册第 18 页安装；已有可用 DNNDK 时跳过此步。以下命令在安装包所在目录执行：

```bash
tar -xzvf vitis-ai_v1.2_dnndk.tar.gz
cd vitis-ai_v1.2_dnndk
./install.sh
```

### 2. 将程序和图片复制到开发板

在 **PC 的工程根目录**打开 PowerShell，替换下面的示例 IP。账号和密码以所用 Alinx 镜像的配置为准，示例使用 `root`：

```powershell
$boardIp = '192.168.1.100'
ssh "root@$boardIp" 'mkdir -p ~/mobilevitv2_075_zu19eg/data/calibration'
scp -r .\board "root@${boardIp}:~/mobilevitv2_075_zu19eg/"
scp -r .\data\calibration\jpegimages "root@${boardIp}:~/mobilevitv2_075_zu19eg/data/calibration/"
```

也可以通过读卡器或其他文件传输工具复制。请保留 `board/` 与 `data/calibration/jpegimages/` 的相对位置；复制整个 `board/` 还能直接使用后面的纯 PS 对照。

### 3. 检查模型库并运行批量推理

以下命令全部在**开发板终端**执行：

```bash
cd ~/mobilevitv2_075_zu19eg/board/deploy_best_fused
export LD_LIBRARY_PATH="$PWD/model${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 检查三个 SO 的哈希、内嵌 kernel 名称及动态加载依赖
python3 runtime/model_libraries.py --config runtime_config.json --load

# 先运行 20 张图片，确认整条链路可用
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders \
  --mv2-backend dpu --fuse --limit 20 --output-dir results_smoke_20

# 再运行仓库提供的全部 300 张图片
python3 runtime/run_hybrid_batch.py --config runtime_config.json \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders \
  --mv2-backend dpu --fuse --output-dir results_batch_300
```

模型库检查通过时会输出 `"passed": true`；它检查文件和动态加载，真正的 DPU 执行由后续推理命令验证。`--fuse` 启用 PS 侧 GroupNorm / 注意力融合。

批量运行结果保存在指定的输出目录：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 成功 / 失败图片数、准确率、混淆矩阵和各阶段平均耗时 |
| `predictions.csv` | 每张图片的标签、预测、logits 和分阶段耗时 |
| `dpu_input_ranges.json` | DPU 输入数值范围与越界统计 |

`--labels-from-folders` 要求图片根目录下为 `c0/` 至 `c9/`，文件夹编号即真实标签。使用自己的无标签图片时去掉该参数；输入图片须为 **256 × 256**，程序不会自动缩放。每次重跑请使用新的输出目录，保留原始实测记录。

### 4. 可选：外接显示器演示

在开发板本地连接显示器，并在**板端终端**执行：

```bash
cd ~/mobilevitv2_075_zu19eg/board
bash setup_dp_display.sh
python3 run_visual_board.py \
  --image-dir ../data/calibration/jpegimages --labels-from-folders \
  --display --output-dir results_visual_300
```

显示程序默认使用 `deploy_best_fused/` 并启用融合，逐图展示分类结果，结束后显示总结看板。输出包含结果图片、`summary_result.jpg` 和 `visual_summary.json`。只保存图片时去掉 `--display`；需要慢速演示时增加 `--display-delay 5000`，每张停留 5 秒。

<a id="project-layout"></a>

## 文件结构

下面列出主要目录和入口，省略重复的模型、图片及历史记录：

```text
mobilevitv2_075_zu19eg/
├── README.md
├── license                              # 项目 MIT 许可证及第三方许可说明
├── board/                               # 板端程序
│   ├── deploy_best_fused/                # 可直接上板的最终混合部署包
│   │   ├── model/                       # 三个 DPU SO、PS 图描述和 NPZ 权重
│   │   ├── runtime/                     # N2Cube 调度、NumPy 执行器和批量入口
│   │   └── runtime_config.json           # 六段执行配置与模型库校验信息
│   ├── ps_reference/                    # 纯 PS / FP32 对照入口及前段模型
│   ├── run_visual_board.py              # 可视化推理与总结看板
│   └── setup_dp_display.sh              # 板端显示环境检查与配置辅助
├── build/
│   ├── mobilevitv2_axu19eg_release/      # 正式构建入口
│   │   ├── compile_tools/
│   │   │   ├── 0a_pt_to_h5_pc.ps1       # PC：PT → H5 权重
│   │   │   ├── 0b_h5_to_pb_pc.ps1       # PC：H5 → 部署 PB
│   │   │   ├── 0_prepare_pc.ps1         # PC：六段拆分与模型 / 校准输入导出
│   │   │   ├── 1_tf_quantize.sh         # Vitis AI：量化及固定偏置修正
│   │   │   ├── 2_vai_compile.sh         # Vitis AI：PB → DPU ELF
│   │   │   ├── 3_lib_compiler_runtime.sh # Runtime：ELF → SO，组装部署包
│   │   │   ├── 4_evaluate_native.sh     # 可选：原生量化模型评估
│   │   │   ├── hardware/axu19eg_ko/     # HWH、DCF 与架构 JSON
│   │   │   ├── modle/driver/           # 固定参数、父量化模型及生成目录
│   │   │   └── scripts/                # 转换、拆分、校准和打包实现
│   │   ├── environment/                # Docker 镜像配置、启动及检查脚本
│   │   ├── docs/                       # PT / H5 / PB、Docker 与校准说明
│   │   └── audit/                      # 主机重建验证脚本
│   ├── historical/                     # 历史两段方案与实验脚本
│   ├── reports/final_compile/          # 三个最终 kernel 的编译日志与 ELF
│   ├── implementation_report.md        # 硬件配置、模型编译与证据范围
│   ├── verify_submission.py            # 工程完整性检查
│   └── sha256sums.json                 # 文件哈希清单
├── data/
│   ├── weights/                        # 原始 bestmodel_v1_only_weights.pt
│   ├── calibration/jpegimages/         # 300 张 c0～c9 图片
│   ├── samples/                        # 数值验证样本
│   └── results/                        # 历史板端结果和展示图片
├── src/
│   ├── pytorch/                        # 训练 / 评估 notebook 与 cvnets 源码
│   ├── model_code/                     # 自定义 block、TF1 模型代码及许可
│   ├── tensorflow/                     # 历史转换、校验脚本与 H5 / PB 参考
│   └── dpu/                            # 历史切分及 baseline / CLE 材料
├── sim/                                # 数值验证、性能分析与诊断
├── skill/                              # 切分、CLE 和 PS 优化经验
└── report/                             # 设计报告、协作与验证记录
```

正式重建使用 `build/mobilevitv2_axu19eg_release/`；`src/dpu/` 和 `build/historical/` 保留历史实验。路径中的 `modle` 沿用工程原有拼写。`sim/` 提供软件数值验证，不包含独立 RTL / HLS 仿真工程。

<a id="full-workflow"></a>

## 完整构建流程

本流程参考 Alinx《Vitis-AI基础 V1.2.4》（2021-07-15）的 Docker 准备及“量化 → 编译 → 动态链接库 → 板端运行”流程，将教程中的 YOLOv3 示例替换为本工程的 MobileViTv2 混合模型。教程的 Docker 部分见第 4–7 页，量化 / 编译 / 链接见第 11–16 页，上板见第 16–19 页。

### 1. 了解执行环境与模型链路

| 执行位置 | 环境 / 版本 | 负责步骤 |
| --- | --- | --- |
| Windows PC | 独立 PyTorch 环境：Python 3.10、torch 1.13.0、NumPy 1.23.0 | 读取 PT、映射权重 |
| Windows PC | 独立 TF1 环境：Python 3.6.10、TensorFlow 1.15.2、NumPy 1.19.5、h5py 3.1.0 | H5、PB、六段拆分和导出 |
| x86_64 Ubuntu 宿主机 / 虚拟机 | Docker | 下载镜像、挂载工程、启动容器 |
| Vitis AI Docker | `xilinx/vitis-ai:1.2.82`，`vitis-ai-tensorflow` | INT8 量化、原生评估、ELF 编译 |
| Runtime Docker | `xilinx/vitis-ai:runtime-1.0.0-cpu` | AArch64 交叉链接、部署包组装 |
| AXU19EG / Z19 | Alinx 匹配镜像、DNNDK / N2Cube、NumPy、OpenCV | 实际推理与显示 |

表中 PC 版本对应工程记录及配套环境。PyTorch 与 TF1 使用独立环境，转换可以在 CPU 上完成。训练 notebook 应使用能够运行其 PyTorch / torchvision 代码的独立训练环境。

```text
原始 PT 权重
  → H5 权重 → 固定 batch 的推理 PB → 六段拆分
  → 三个 DPU 段量化 / 偏置修正 → ELF → AArch64 SO
  → 与 PS 的 JSON / NPZ、runtime_config.json 和运行代码组装
  → 开发板推理

图像 → front[DPU] → ps1[NumPy] → mv2_4[DPU]
     → ps2[NumPy] → mv2_5[DPU] → ps3[NumPy] → 10 类 logits
```

这里的三个 DPU 段是三个模型 kernel，不代表三个硬件计算核。仓库 HWH 记录的硬件配置为 2 × B4096；实际启动镜像仍需与该配置核对。

### 2. 准备权重与数据，或自行训练

复现随仓库发布的模型，直接使用：

- 原始权重：`data/weights/bestmodel_v1_only_weights.pt`。
- 校准图片：`data/calibration/jpegimages/`，共 300 张。
- 模型配置：宽度系数 0.75、10 类、ReLU、自定义 `my_MobileViTBlockv2`、固定部署输入 `[1, 256, 256, 3]`。
- 预处理：RGB / 255，减去 `mean=[0.485, 0.456, 0.406]`，再除以 `std=[0.229, 0.224, 0.225]`。

如需自行训练，参考 [训练 notebook](src/pytorch/scripts/train.ipynb) 和 [评估 notebook](src/pytorch/scripts/inference.ipynb)：先准备 `c0/` 至 `c9/` 分类目录和 ImageNet 初始化权重，在 notebook 中设置数据路径、`PRETRAINED_WEIGHTS`、输出路径及训练参数，再运行训练、评估和纯 `state_dict` 提取单元。训练数据可通过 `MOBILEVIT_TRAIN_DIR` 指定，评估数据可通过 `MOBILEVIT_TEST_DIR` 指定，工程根目录可通过 `MOBILEVIT_PROJECT_ROOT` 指定。

仓库不提供完整训练集、ImageNet 初始化权重或 notebook 默认引用的续训检查点。首次训练须将 `RESUME` 设为 `False`，或提供自己的有效检查点；输入尺寸保持 256 × 256。

**下面的正式脚本针对随仓库发布的固定 PT、图指纹和校正参数。** 自行训练、更换类别或修改结构后，需要重新建立转换校验、量化与偏置修正方案，不能直接套用当前固定参数。

### 3. Windows PC：PT → H5 → PB → 六段导出

准备上表中的两个 PC 环境。PT 读取环境依赖见 [pc-torch-requirements.txt](build/mobilevitv2_axu19eg_release/environment/pc-torch-requirements.txt)；TF1 环境需要表中的 TensorFlow、NumPy 和 h5py。

在 **PC 工程根目录的 PowerShell** 中执行，并将解释器路径改为本机实际路径：

```powershell
Set-Location .\build\mobilevitv2_axu19eg_release\compile_tools
$torchPython = 'C:\ProgramData\anaconda3\envs\cvnets\python.exe'
$tfPython = 'C:\ProgramData\anaconda3\envs\tf115\python.exe'

.\0a_pt_to_h5_pc.ps1 -TorchPython $torchPython -TfPython $tfPython
.\0b_h5_to_pb_pc.ps1 -PythonExe $tfPython
.\0_prepare_pc.ps1 -PythonExe $tfPython
```

三个步骤依次生成以下内容，路径均相对于 `compile_tools/`：

| 步骤 | 主要输出 | 处理内容 |
| --- | --- | --- |
| `0a` | `modle/driver/build/upstream/model_weights.h5` | 读取 PT、转换权重布局、核对张量；输出为 H5 权重文件 |
| `0b` | `modle/driver/pb_file/source.pb` | 构建推理图、固定 batch=1、形状常量化、折叠 GN 通道仿射参数 |
| `0` | `modle/driver/prepared/fixed/` | 六段 PB、JSON / NPZ、固定参数与前端校准输入 |

最终整网 PB 的输入为 `input_1:0`（`float32 [1,256,256,3]`），输出为 `predictions/BiasAdd:0`（`float32 [1,10]`）。GN 的动态均值 / 方差计算仍保留。转换细节和校验规则见 [PT → H5 → PB 说明](build/mobilevitv2_axu19eg_release/docs/pt_h5_pb.md)。

### 4. Ubuntu 宿主机：准备 Docker 并复制工程

将工程放入 x86_64 Ubuntu 宿主机或虚拟机，例如 `~/work/mobilevitv2_075_zu19eg/`。至少保留 `src/`、`build/`、`board/`、`data/` 及上一步生成的 `compile_tools/modle/driver/prepared/fixed/`，保持原有相对结构。

在 **Ubuntu 宿主机**检查 Docker：

```bash
docker version
docker info
docker run hello-world
```

如果尚未安装 Docker，按 [Docker 环境说明](build/mobilevitv2_axu19eg_release/docs/docker_setup.md) 准备。原 PDF 使用 Ubuntu 18.04 的历史安装步骤，安装命令应与宿主机版本相匹配。后续启动脚本要求使用已配置 Docker 权限的普通用户；按教程配置用户组后重新登录即可：

```bash
sudo usermod -aG docker "$USER"
```

仍在 **Ubuntu 宿主机**执行：

```bash
cd ~/work/mobilevitv2_075_zu19eg/build/mobilevitv2_axu19eg_release
bash environment/pull_images.sh all
bash environment/check_environment.sh host
```

脚本使用 `environment/images.env` 中固定的两个镜像标签。不要自行替换为 `latest`。这两个 Docker 镜像也需要单独下载，它们与板端 Alinx 启动镜像是不同的依赖。

### 5. Vitis AI Docker：量化、评估与编译

在 **Ubuntu 宿主机**的 `build/mobilevitv2_axu19eg_release/` 中启动容器，按提示阅读许可：

```bash
bash environment/run_vitis.sh
```

进入 **Vitis AI 容器**后执行：

```bash
cd /workspace/build/mobilevitv2_axu19eg_release/compile_tools
conda activate vitis-ai-tensorflow
bash ../environment/check_environment.sh vitis
python3 -B scripts/finish.py check-prepared

bash 1_tf_quantize.sh
# 可选：检查原生量化模型的完整混合链路
bash 4_evaluate_native.sh
bash 2_vai_compile.sh
exit
```

量化步骤处理 `front`、`mv2_4`、`mv2_5` 三个 DPU 段，并应用固定方案中的偏置修正；编译步骤使用 `hardware/axu19eg_ko/` 下的架构配置生成三个 ELF。输出位于 `modle/driver/build/fixed/quantized/` 和 `modle/driver/build/fixed/compiled/`。

`modle/driver/params/` 中的 CLE 和偏置参数，以及 `modle/driver/reference/quantized/` 中的父量化 PB，都是当前固定方案的必要输入。父量化 PB 用于生成后续段的校准特征，请保留。更多说明见 [校准流程](build/mobilevitv2_axu19eg_release/docs/calibration.md)。

启动脚本将整个工程挂载到 `/workspace`，结果保存在宿主机工程目录中。容器内使用 Linux 路径。

### 6. Runtime Docker：生成 SO 并组装部署包

回到 **Ubuntu 宿主机**，仍在同一工程的 `build/mobilevitv2_axu19eg_release/` 中执行：

```bash
bash environment/run_runtime.sh
```

进入 **Runtime 容器**后执行：

```bash
cd /workspace/build/mobilevitv2_axu19eg_release/compile_tools
bash ../environment/check_environment.sh runtime
bash 3_lib_compiler_runtime.sh
exit
```

此步骤使用 `aarch64-xilinx-linux-gcc` 交叉链接三个 ELF，并把 SO、PC 导出的 JSON / NPZ、运行代码及配置组装到：

```text
build/mobilevitv2_axu19eg_release/compile_tools/modle/driver/build/fixed/deploy/
```

Vitis AI 与 Runtime 容器必须挂载同一份工程，才能共享 `prepared/` 和 `build/`。默认 sysroot 为 `/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux`；SDK 环境初始化与其他路径的设置方法见 [Docker 环境说明](build/mobilevitv2_axu19eg_release/docs/docker_setup.md)。

仅需把新导出的 PC 模型与仓库现成 SO 组装时，也可以在 **PC 的 `compile_tools/`** 执行下面的命令；它使用第 3 步设置的 `$tfPython`，输出 `modle/driver/build/fixed/deploy_prebuilt/`，不重新量化或编译：

```powershell
& $tfPython -B scripts/finish.py bundle-prebuilt
```

### 7. 部署新生成的模型包

把 `deploy/` 整个目录复制到开发板，保留其中的 `model/`、`runtime/` 和 `runtime_config.json`。以板端目录 `~/mobilevitv2_075_zu19eg/board/deploy_rebuilt/` 为例，进入该目录后设置 `LD_LIBRARY_PATH`，再使用“快速上手”第 3 步的模型库检查和批量命令；此位置下图片相对路径仍为 `../../data/calibration/jpegimages`。

需要使用新包进行可视化时，在**板端的 `board/` 目录**指定部署位置：

```bash
python3 run_visual_board.py --deploy-dir ./deploy_rebuilt \
  --image-dir ../data/calibration/jpegimages --labels-from-folders \
  --display --output-dir results_visual_rebuilt
```

同一构建产生的 SO、JSON / NPZ 和配置应一起部署。新构建结果需要重新上板核验，不能直接沿用仓库中的历史精度和耗时。

<a id="validation"></a>

## 验证与实测结果

### 纯 PS 对照

已复制整个 `board/` 时，可在**开发板终端**运行：

```bash
cd ~/mobilevitv2_075_zu19eg/board/ps_reference
python3 runtime/run_all_ps.py --model-dir model \
  --image-dir ../../data/calibration/jpegimages --labels-from-folders \
  --fuse --output-dir results_ps_300
```

该入口以 NumPy / FP32 执行全部六段，与 `deploy_best_fused/` 共用部分代码和模型文件。它适合核对数值和 CPU 执行耗时；与 DPU / INT8 方案比较时需同时说明数值精度。

### 历史板端结果

最终结果见 [原始逐图记录](data/results/final_20261103/visual_summary.json)：

| 指标 | 结果 |
| --- | --- |
| 推理方式 | 三段 DPU / INT8 + 三段 PS / NumPy，启用融合 |
| 图片数 / 正确数 | 300 / 261 |
| 准确率 | **87.00%** |
| 逐图总耗时平均值 | **526.4251 ms** |

这里的 300 张图片同时用于校准和测试，结果不代表独立测试集上的泛化精度。总耗时包含图像解码 / 预处理和模型推理，不能等同于完整显示界面的帧率。此次 README 更新未重新执行 Vitis 编译或板端测试。

### 工程与主机验证

在 **PC 工程根目录的 PowerShell** 中执行：

```powershell
# 工程完整性检查：只读，不刷新报告和文件哈希
python -B build/verify_submission.py --read-only

# 主机重建验证：需要前文的 TF1 和 PyTorch 环境
& 'C:\ProgramData\anaconda3\envs\tf115\python.exe' -B build/mobilevitv2_axu19eg_release/audit/validate_local.py
```

完整性检查适合在未修改、未生成新文件的仓库副本中执行。主机重建验证在临时目录生成 H5、PB、六段模型并检查与预存 SO 的组装，同时更新 `build/mobilevitv2_axu19eg_release/local_validation.json`；它不执行真实 Vitis 编译或 DPU 推理。解释器位置不同可给验证脚本传入 `--tf-python` 和 `--torch-python`。

完整性检查要求文件与保存的哈希清单一致；自行生成或修改文件后，差异需要单独审查。检查脚本不带 `--read-only` 时会刷新报告及哈希清单。

验证范围见 [验证记录](report/release_validation.md)。本项目使用厂商固定 DPU 平台，没有自建 RTL / HLS 工程；现存 HWH 配置与模型编译日志的证据范围，以及尚缺的整设计资源 / 时序报告，见 [实现说明](build/implementation_report.md)。

<a id="faq"></a>

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 仓库中找不到启动 bin / 系统镜像 | 从 Alinx 配套资料单独取得；镜像不以 MIT 协议开源，未随仓库分发 |
| `No module named dnndk` 或缺少 `libn2cube.so` | 检查板端镜像、DNNDK 安装及当前 Python 环境，参照教程第 18 页 |
| SO 加载失败、kernel 或哈希不匹配 | 在板端运行 `model_libraries.py --load`，核对动态库依赖，并复制同一构建的完整模型包 |
| DPU 输入 scale / 形状不匹配 | 核对镜像的 DPU 配置，以及配套 SO、模型和 `runtime_config.json` |
| 图片尺寸错误或无法计算准确率 | 输入须为 256 × 256；使用 `--labels-from-folders` 时保留 `c0/` 至 `c9/` 层级 |
| 输出目录已存在 | 为本次运行换一个新目录名 |
| 容器中找不到量化器 / 编译器 | 确认使用固定版本的 Vitis AI 镜像，并激活 `vitis-ai-tensorflow` |
| Runtime 中找不到编译器或 ELF | 检查 SDK 环境，并确认两个容器共享同一份工程和构建输出 |
| 外接屏无显示 | 先运行 `setup_dp_display.sh` 检查连接与图形环境；去掉 `--display` 可先保存结果图片 |
| PS 阶段耗时较长 | 检查 NumPy 的 BLAS 配置，并使用 `--fuse` 对比相同输入下的实际耗时 |

<a id="references"></a>

## 文档与参考资料

- Alinx《Vitis-AI基础 V1.2.4》（2021-07-15）：第 1 章 Docker 环境、第 2 章部署流程、第 5 章调试工具。教程及配套镜像通过 Alinx 资料获取。
- [正式构建说明](build/mobilevitv2_axu19eg_release/README.md)
- [PT → H5 → PB 转换详解](build/mobilevitv2_axu19eg_release/docs/pt_h5_pb.md)
- [Docker 环境与故障定位](build/mobilevitv2_axu19eg_release/docs/docker_setup.md)
- [固定方案校准说明](build/mobilevitv2_axu19eg_release/docs/calibration.md)
- [中间产物与重新生成](build/mobilevitv2_axu19eg_release/docs/generated_files.md)
- [板端程序](board/README.md)、[设计报告](report/design_report.md)、[验证范围](report/release_validation.md)

<a id="license"></a>

## 许可证

本项目自有代码按 [MIT License](license) 发布。第三方模型源码、框架、工具及随附组件保留各自许可，参见 [cvnets 许可](src/pytorch/model_source/license.txt)、[TF1 模型代码许可](src/model_code/tf115/license) 和对应源码中的版权声明。

Alinx 提供的启动 bin / 系统镜像不随仓库分发，其许可独立于本项目的 MIT 许可证；请按 Alinx 配套条款获取和使用。
