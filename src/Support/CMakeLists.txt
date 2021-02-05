# SPDX-License-Identifier: Apache-2.0

add_library(OMSupport
        KrnlSupport.cpp)
target_include_directories(OMSupport
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMSupport
        OMKrnlOps
        OMONNXOps)

install(TARGETS OMSupport DESTINATION lib)
