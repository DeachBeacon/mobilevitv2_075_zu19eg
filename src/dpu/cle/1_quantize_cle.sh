#!/bin/bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
command -v vai_q_tensorflow >/dev/null 2>&1 || { echo "Activate vitis-ai-tensorflow in Vitis AI 1.2." >&2; exit 1; }
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage float
test -d "${MOBILEVIT_CALIB_DIR}" || { echo "Missing calibration images." >&2; exit 1; }
test ! -e "${CLE_DIR}/int8" || { echo "Quantization directory already exists. Use a new candidate directory." >&2; exit 1; }
mkdir "${CLE_DIR}/int8"
vai_q_tensorflow quantize \
  --input_frozen_graph "${CLE_DIR}/fp32/dpu_front_cle_fp32.pb" \
  --input_shapes "1,256,256,3" --input_nodes "input_1" \
  --output_nodes "stack3_block2_pre_2_conv/Conv2D" \
  --input_fn input_fn.calib_input --calib_iter 300 \
  --output_dir "${CLE_DIR}/int8" 2>&1 | tee "${CLE_DIR}/int8/quantize.log"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage record-quantized
