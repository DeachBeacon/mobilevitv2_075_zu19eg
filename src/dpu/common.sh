#!/bin/bash
# Locate the complete project from this file, without requiring a Docker mount path.
set -euo pipefail
DPU_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${DPU_CODE_DIR}/../.." && pwd)"
ROOT_DIR="${PROJECT_DIR}/compile_tools"
export MOBILEVIT_CALIB_DIR="${MOBILEVIT_CALIB_DIR:-${ROOT_DIR}/calibration/images}"
export PYTHONPATH="${DPU_CODE_DIR}/calibration:${DPU_CODE_DIR}/split:${ROOT_DIR}/deploy_in_board/runtime${PYTHONPATH:+:${PYTHONPATH}}"
