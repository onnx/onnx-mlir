# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
   # currently needed for a bug on Z for the KrnlOps.cpp.inc file
   llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
   llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

set(LLVM_TARGET_DEFINITIONS KrnlOps.td)
onnx_mlir_tablegen(KrnlOps.hpp.inc -gen-op-decls)
onnx_mlir_tablegen(KrnlOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(OMKrnlOpsIncGen)

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
