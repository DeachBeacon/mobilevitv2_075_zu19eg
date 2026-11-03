#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

cd "${ROOT_DIR}"

MODEL_PATH="${ROOT_DIR}/models/front_baseline/fp32/dpu_front_fp32.pb"
OUTPUT_DIR="${BASELINE_QUANT_DIR:-${ROOT_DIR}/models/front_baseline/int8}"



command -v vai_q_tensorflow >/dev/null 2>&1 || {
    echo "Activate vitis-ai-tensorflow inside the Vitis AI 1.2 Docker first." >&2
    exit 1
}
test -s "${MODEL_PATH}" || { echo "Run code/dpu/split/split_model.py first: ${MODEL_PATH}" >&2; exit 1; }
test -d "${MOBILEVIT_CALIB_DIR}" || { echo "Missing calibration images: ${MOBILEVIT_CALIB_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

vai_q_tensorflow quantize \
    --input_frozen_graph "${MODEL_PATH}" \
    --input_shapes "1,256,256,3" \
    --input_nodes "input_1" \
    --output_nodes "stack3_block2_pre_2_conv/Conv2D" \
    --input_fn input_fn.calib_input \
    --calib_iter 300 \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/quantize.log"
