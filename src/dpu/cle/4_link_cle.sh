#!/bin/bash
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage compiled --max-drop "${MAX_DROP}" --min-samples "${MIN_SAMPLES}"
command -v aarch64-xilinx-linux-gcc >/dev/null 2>&1 || { echo "Use the tutorial Runtime Docker cross compiler." >&2; exit 1; }
SYSROOT="${SYSROOT:-/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux}"
test -d "${SYSROOT}" || { echo "Missing sysroot: ${SYSROOT}" >&2; exit 1; }
test ! -e "${CLE_DIR}/compiled/libdpumodel${NET_NAME}.so" || { echo "Shared library already exists." >&2; exit 1; }
aarch64-xilinx-linux-gcc --sysroot="${SYSROOT}" -fPIC -shared \
  "${CLE_DIR}/compiled/dpu_${NET_NAME}.elf" \
  -o "${CLE_DIR}/compiled/libdpumodel${NET_NAME}.so"
"${CLE_PYTHON_BIN}" "${GATE}" --artifact-dir "${CLE_DIR}" --stage record-linked --max-drop "${MAX_DROP}" --min-samples "${MIN_SAMPLES}"
