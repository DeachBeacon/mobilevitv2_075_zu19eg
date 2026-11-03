# cle 位于 modle/driver/pb_file/cle 时的路径

工程根目录通过 `input_fn.py` 和 `hardware/` 共同定位。修改集中在 `cle/common.sh`，四个编号 Shell 共用它。

| 变量 | 对应位置（相对于 compile_tools） |
| --- | --- |
| ROOT_DIR | compile_tools 根目录 |
| CLE_SCRIPT_DIR | modle/driver/pb_file/cle |
| CLE_DIR | modle/driver/pb_file/cle/artifacts |
| SOURCE_PB | modle/driver/pb_file/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb |
| TAIL_PB | modle/driver/pb_file/ps_tail_fp32.pb |
| MOBILEVIT_CALIB_DIR | modle/driver/Images |
| GATE | modle/driver/pb_file/cle/check_candidate.py |
| 原生评估脚本 | diagnostics/compare_front_quantization.py |
| ARCH / DCF | hardware/axu19eg_ko/ 下原有文件 |

原 `common.sh` 中 `CLE_SCRIPT_DIR/../..` 对这个结构只会上到 driver，若手工固定层数应改成 `CLE_SCRIPT_DIR/../../../..`。本次提供的版本自动向上定位工程根目录，无需再数层数。

路径修正包按以下层级解压到 Docker/WSL 的 compile_tools 根目录：

```text
compile_tools/
  input_fn.py                   已有
  hardware/                     已有
  diagnostics/
    compare_front_quantization.py
  hybrid_split/
    common.py                   Python 共用函数，不是 common.sh
    ps_tail.py                  原生评估使用的 TF 后段执行器
  modle/driver/
    Images/                     已有
    pb_file/
      原始整网.pb                已有，实际文件名按脚本配置
      ps_tail_fp32.pb            已有
      cle/
        common.sh
        check_candidate.py
        1_quantize_cle.sh
        2_evaluate_cle.sh
        3_compile_cle.sh
        4_link_cle.sh
        artifacts/              已有，修正包不覆盖
          dpu_front_cle_fp32.pb
          manifest.json
          fp32_verification.json
```

在实际 Docker 的 compile_tools 根目录执行：

```bash
conda activate vitis-ai-tensorflow
bash modle/driver/pb_file/cle/1_quantize_cle.sh
bash modle/driver/pb_file/cle/2_evaluate_cle.sh
```

原生评估通过后，再在 Vitis Docker 运行第 3 步，在教程 Runtime Docker 运行第 4 步。

此目录调整不要求修改 `input_nodes`、`output_nodes`、输入形状、NET_NAME、sysroot 或 DCF 配置内容。图片与模型实际在哪里，就通过上述路径变量定位；WSL 的 `/mnt/c/...` 和 Docker 的 `/workspace/...` 挂载路径无需写死，但运行容器必须能访问这套工程。

如果之前手动 export 过 `CLE_DIR`、`CLE_SOURCE_PB`、`CLE_TAIL_PB` 或 `MOBILEVIT_CALIB_DIR`，这些环境变量会优先覆盖默认值，请确认它们没有继续指向旧目录。
