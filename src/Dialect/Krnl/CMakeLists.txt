# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(KrnlOps)

# Header dependencies target for KrnOps.hpp
add_custom_target(OMKrnlOpsInc
        DEPENDS OMKrnlOpsIncGen)

add_onnx_mlir_library(OMKrnlOps
        KrnlOps.cpp
        KrnlTypes.cpp
        KrnlHelper.cpp)
add_dependencies(OMKrnlOps OMKrnlOpsInc)

# Linking dependencies:
add_dependencies(OMKrnlOps
        OMSpecializedKernelOpInterface
        OMSimplifyKrnl)
