#!/bin/bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage quantized
"${CLE_PYTHON_BIN}" "${PROJECT_DIR}/code/dpu/diagnostics/compare_front_quantization.py" \
  --backend native --source-pb "${SOURCE_PB}" --tail-pb "${TAIL_PB}" \
  --quantized-pb "${CLE_DIR}/int8/quantize_eval_model.pb" \
  --image-dir "${CLE_EVAL_DIR:-${MOBILEVIT_CALIB_DIR}}" --samples 0 \
  --output "${CLE_DIR}/records/native_quant_accuracy.json"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage native --max-drop "${MAX_DROP}" --min-samples "${MIN_SAMPLES}"
