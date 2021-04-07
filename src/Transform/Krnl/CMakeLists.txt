# SPDX-License-Identifier: Apache-2.0

add_library(OMSimplifyKrnl
        SimplifyKrnl.cpp)
target_include_directories(OMSimplifyKrnl PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})

# Linking dependency:
add_dependencies(OMKrnlOps
        OMSimplifyKrnl)

# Header dependency:
add_dependencies(OMSimplifyKrnl
        OMSpecializedKernelOpInterfaceIncGen
        OMKrnlOpsInc)

install(TARGETS OMSimplifyKrnl DESTINATION lib)
