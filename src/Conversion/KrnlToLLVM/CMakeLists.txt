# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToLLVM
        KrnlToLLVM.cpp)

# Header dependencies
add_dependencies(OMKrnlToLLVM OMKrnlOpsInc OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToLLVM OMKrnlOps OMONNXOps)

target_link_libraries(OMKrnlToLLVM
        onnx)
