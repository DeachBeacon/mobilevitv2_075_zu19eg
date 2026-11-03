#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
cd "${ROOT_DIR}"
NET_NAME="mobilevitv2_075_front"
MODEL_DIR="${BASELINE_COMPILE_DIR:-${ROOT_DIR}/models/front_baseline/compiled}"
ELF_PATH="${MODEL_DIR}/dpu_${NET_NAME}.elf"
SYSROOT="${SYSROOT:-/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux}"

command -v aarch64-xilinx-linux-gcc >/dev/null 2>&1 || {
    echo "Run inside the tutorial's Runtime Docker with its ARM cross compiler." >&2
    exit 1
}
test -s "${ELF_PATH}" || {
    echo "Expected ELF not found: ${ELF_PATH}. Check compile.log for actual kernel names." >&2
    exit 1
}
test -d "${SYSROOT}" || { echo "Missing Runtime Docker sysroot: ${SYSROOT}" >&2; exit 1; }

aarch64-xilinx-linux-gcc --sysroot="${SYSROOT}" \
    -fPIC -shared "${ELF_PATH}" -o "${MODEL_DIR}/libdpumodel${NET_NAME}.so"
echo "Generated ${MODEL_DIR}/libdpumodel${NET_NAME}.so"
