# SPDX-License-Identifier: Apache-2.0

add_executable(CustomFnTest CustomFnTest.cpp)

target_link_libraries(CustomFnTest
        ${OMLibs}
        ${MLIRLibs}
        onnx)

target_include_directories(CustomFnTest
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/third_party/onnx
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(CustomFnTest ${OMLibs})