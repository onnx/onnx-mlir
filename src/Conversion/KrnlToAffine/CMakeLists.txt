# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToAffine
        KrnlToAffine.cpp)
# Header dependencies
add_dependencies(OMKrnlToAffine OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToAffine OMKrnlOps)
