# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSimplifyKrnl
        SimplifyKrnl.cpp)

# Linking dependency:
add_dependencies(OMKrnlOps
        OMSimplifyKrnl)

# Header dependency:
add_dependencies(OMSimplifyKrnl
        OMSpecializedKernelOpInterfaceIncGen
        OMKrnlOpsInc)
