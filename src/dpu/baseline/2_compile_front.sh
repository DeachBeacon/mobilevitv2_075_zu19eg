#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
cd "${ROOT_DIR}"

TARGET="axu19eg_ko"
NET_NAME="mobilevitv2_075_front"
MODEL_PATH="${BASELINE_QUANT_DIR:-${ROOT_DIR}/models/front_baseline/int8}/deploy_model.pb"
OUTPUT_DIR="${BASELINE_COMPILE_DIR:-${ROOT_DIR}/models/front_baseline/compiled}"
ARCH="${ROOT_DIR}/hardware/${TARGET}/${TARGET}.json"
DCF_PATH="${ROOT_DIR}/hardware/${TARGET}/${TARGET}.dcf"

command -v vai_c_tensorflow >/dev/null 2>&1 || {
    echo "Activate vitis-ai-tensorflow inside the Vitis AI 1.2 Docker first." >&2
    exit 1
}
test -s "${MODEL_PATH}" || { echo "Quantize the front first: ${MODEL_PATH}" >&2; exit 1; }
test -s "${ARCH}" || { echo "Missing architecture JSON: ${ARCH}" >&2; exit 1; }
test -s "${DCF_PATH}" || { echo "Missing DCF: ${DCF_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

vai_c_tensorflow \
    --frozen_pb "${MODEL_PATH}" \
    --arch "${ARCH}" \
    --output_dir "${OUTPUT_DIR}" \
    --net_name "${NET_NAME}" \
    --options "{'dcf':'${DCF_PATH}'}" \
    --quant_info \
    2>&1 | tee "${OUTPUT_DIR}/compile.log"
