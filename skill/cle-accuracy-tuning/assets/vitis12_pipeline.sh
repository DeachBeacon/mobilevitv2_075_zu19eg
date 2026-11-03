#!/bin/bash
# Parameterized Vitis AI 1.2 / DNNDK / AArch64 CNN adapter. Read workflow first.
set -euo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${1:?Use quantize, compile or link}"
: "${BUILD_DIR:?Set a new build directory}"
: "${KERNEL:?Set the actual compiler kernel name}"
case "${ACTION}" in
quantize)
    : "${MODEL_PB:?Set the verified float_model.pb}"
    : "${FLOAT_MANIFEST:?Set the matching float transform manifest}"
    : "${INPUT_NODES:?Set quantizer input OP names}"
    : "${OUTPUT_NODES:?Set quantizer output OP names}"
    : "${INPUT_SHAPES:?Set the real shapes}"
    : "${INPUT_FN:?Set a real module.function calibration callback}"
    : "${CALIB_MODULE_DIR:?Set its Python module directory}"
    : "${CALIB_ITER:?Set an explicit positive number of calibration batches}"
    [[ "${CALIB_ITER}" =~ ^[1-9][0-9]*$ ]]
    METHOD="${METHOD:-1}"; [[ "${METHOD}" == 0 || "${METHOD}" == 1 ]]
    "${PYTHON_BIN}" "${SKILL_DIR}/scripts/artifact_gate.py" float --model "${MODEL_PB}" --float-manifest "${FLOAT_MANIFEST}"
    command -v vai_q_tensorflow >/dev/null
    test ! -e "${BUILD_DIR}/quantize"
    mkdir -p "${BUILD_DIR}/quantize"
    export PYTHONPATH="${CALIB_MODULE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    vai_q_tensorflow quantize --input_frozen_graph "${MODEL_PB}" --input_nodes "${INPUT_NODES}" \
        --output_nodes "${OUTPUT_NODES}" --input_shapes "${INPUT_SHAPES}" --input_fn "${INPUT_FN}" \
        --calib_iter "${CALIB_ITER}" --method "${METHOD}" --output_dir "${BUILD_DIR}/quantize" \
        2>&1 | tee "${BUILD_DIR}/quantize/quantize.log"
    echo 'Next: audit boundary metadata, run genuine native complete-chain evaluation, then rank reports.'
    ;;
compile)
    : "${SELECTION_JSON:?Set the checked selection file}"
    : "${ARCH_JSON:?Set the real target arch.json}"
    : "${DCF_PATH:?Set the real target DCF}"
    DEPLOY_PB="${DEPLOY_PB:-${BUILD_DIR}/quantize/deploy_model.pb}"
    "${PYTHON_BIN}" "${SKILL_DIR}/scripts/artifact_gate.py" selection --selection "${SELECTION_JSON}" --deploy "${DEPLOY_PB}"
    command -v vai_c_tensorflow >/dev/null
    test ! -e "${BUILD_DIR}/compile"
    mkdir -p "${BUILD_DIR}/compile"
    vai_c_tensorflow --frozen_pb "${DEPLOY_PB}" --arch "${ARCH_JSON}" --output_dir "${BUILD_DIR}/compile" \
        --net_name "${KERNEL}" --options "{'dcf':'${DCF_PATH}'}" --quant_info \
        2>&1 | tee "${BUILD_DIR}/compile/compile.log"
    "${PYTHON_BIN}" "${SKILL_DIR}/scripts/artifact_gate.py" compiled --selection "${SELECTION_JSON}" --deploy "${DEPLOY_PB}" --build "${BUILD_DIR}" --kernel "${KERNEL}"
    ;;
link)
    : "${SELECTION_JSON:?Set the same selection file}"
    : "${DPU_SYSROOT:?Set the real Runtime cross-compiler sysroot}"
    DEPLOY_PB="${DEPLOY_PB:-${BUILD_DIR}/quantize/deploy_model.pb}"
    "${PYTHON_BIN}" "${SKILL_DIR}/scripts/artifact_gate.py" compiled --selection "${SELECTION_JSON}" --deploy "${DEPLOY_PB}" --build "${BUILD_DIR}" --kernel "${KERNEL}"
    command -v aarch64-xilinx-linux-gcc >/dev/null
    test -d "${DPU_SYSROOT}"
    SO="${BUILD_DIR}/compile/libdpumodel${KERNEL}.so"
    if [[ ! -f "${SO}" ]]; then
        aarch64-xilinx-linux-gcc --sysroot="${DPU_SYSROOT}" -fPIC -shared "${BUILD_DIR}/compile/dpu_${KERNEL}.elf" -o "${SO}"
    fi
    "${PYTHON_BIN}" "${SKILL_DIR}/scripts/artifact_gate.py" linked --selection "${SELECTION_JSON}" --deploy "${DEPLOY_PB}" --build "${BUILD_DIR}" --kernel "${KERNEL}"
    ;;
*) echo 'Choose quantize, compile or link.' >&2; exit 2;;
esac
