# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(KrnlOps)

# Header dependencies target for KrnOps.hpp
add_custom_target(OMKrnlOpsInc
        DEPENDS OMKrnlOpsIncGen)

add_library(OMKrnlOps
        KrnlOps.hpp
        KrnlOps.cpp
        KrnlTypes.hpp
        KrnlTypes.cpp
        KrnlHelper.hpp
        KrnlHelper.cpp)
target_include_directories(OMKrnlOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMKrnlOps OMKrnlOpsInc)

# Linking dependencies:
add_dependencies(OMKrnlOps
        OMSpecializedKernelOpInterface)

install(TARGETS OMKrnlOps DESTINATION lib)
