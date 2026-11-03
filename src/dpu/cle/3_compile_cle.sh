#!/bin/bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage native --max-drop "${MAX_DROP}" --min-samples "${MIN_SAMPLES}"
command -v vai_c_tensorflow >/dev/null 2>&1 || { echo "Activate vitis-ai-tensorflow in Vitis AI 1.2." >&2; exit 1; }
test ! -e "${CLE_DIR}/compiled" || { echo "Compile directory already exists; inspect the existing run." >&2; exit 1; }
mkdir "${CLE_DIR}/compiled"
vai_c_tensorflow \
  --frozen_pb "${CLE_DIR}/int8/deploy_model.pb" \
  --arch "${ROOT_DIR}/hardware/axu19eg_ko/axu19eg_ko.json" \
  --output_dir "${CLE_DIR}/compiled" --net_name "${NET_NAME}" \
  --options "{'dcf':'${ROOT_DIR}/hardware/axu19eg_ko/axu19eg_ko.dcf'}" \
  --quant_info 2>&1 | tee "${CLE_DIR}/compiled/compile.log"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage record-compiled --max-drop "${MAX_DROP}" --min-samples "${MIN_SAMPLES}"
