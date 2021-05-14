# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSimplifyKrnl
        SimplifyKrnl.cpp)

# Header dependency:
add_dependencies(OMSimplifyKrnl
        OMSpecializedKernelOpInterfaceIncGen
        OMKrnlOpsInc)
