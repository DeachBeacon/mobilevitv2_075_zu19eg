#!/bin/bash
set -euo pipefail
CLE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${CLE_SCRIPT_DIR}/../common.sh"
cd "${ROOT_DIR}"
CLE_DIR="${CLE_DIR:-${ROOT_DIR}/models/front_cle}"
SOURCE_PB="${CLE_SOURCE_PB:-${PROJECT_DIR}/code/tensorflow/pb/deploy/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb}"
TAIL_PB="${CLE_TAIL_PB:-${ROOT_DIR}/models/tail_shared/ps_tail_fp32.pb}"
NET_NAME="mobilevitv2_075_front_cle"
GATE="${CLE_SCRIPT_DIR}/check_candidate.py"
MAX_DROP="${CLE_MAX_ACCURACY_DROP:-0.06}"
MIN_SAMPLES="${CLE_MIN_EVAL_SAMPLES:-300}"
CLE_PYTHON_BIN="${CLE_PYTHON_BIN:-python3}"
