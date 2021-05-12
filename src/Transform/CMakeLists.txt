# SPDX-License-Identifier: Apache-2.0

add_library(OMElideKrnlGlobalConstants
        ElideKrnlGlobalConstants.cpp
        ElideKrnlGlobalConstants.hpp)
target_include_directories(OMElideKrnlGlobalConstants
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMElideKrnlGlobalConstants OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMElideKrnlGlobalConstants OMKrnlOps)

add_library(OMEnableMemoryPool
        EnableMemoryPool.cpp)
target_include_directories(OMEnableMemoryPool
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMEnableMemoryPool
        OMSupport)
add_dependencies(OMEnableMemoryPool
        OMSupport
        OMKrnlOps)

add_library(OMOptimizeMemoryPools
        OptimizeMemoryPools.cpp)
target_include_directories(OMOptimizeMemoryPools
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMOptimizeMemoryPools
        OMSupport)
add_dependencies(OMOptimizeMemoryPools
        OMSupport
        OMKrnlOps)

add_library(OMBundleMemoryPools
        BundleMemoryPools.cpp)
target_include_directories(OMBundleMemoryPools
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMBundleMemoryPools
        OMSupport)
add_dependencies(OMBundleMemoryPools
        OMSupport
        OMKrnlOps)

add_library(OMDisconnectKrnlDimFromAlloc
        DisconnectKrnlDimFromAlloc.cpp)
target_include_directories(OMDisconnectKrnlDimFromAlloc
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMDisconnectKrnlDimFromAlloc
        OMSupport)
add_dependencies(OMDisconnectKrnlDimFromAlloc
        OMSupport
        OMKrnlOps)

add_library(OMLowerKrnlShape
        LowerKrnlShape.cpp)
target_include_directories(OMLowerKrnlShape
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMLowerKrnlShape)
add_dependencies(OMLowerKrnlShape
        OMKrnlOps)

add_subdirectory(ONNX)
add_subdirectory(Krnl)

install(TARGETS OMElideKrnlGlobalConstants   DESTINATION lib)
install(TARGETS OMEnableMemoryPool           DESTINATION lib)
install(TARGETS OMOptimizeMemoryPools        DESTINATION lib)
install(TARGETS OMBundleMemoryPools          DESTINATION lib)
install(TARGETS OMDisconnectKrnlDimFromAlloc DESTINATION lib)
install(TARGETS OMLowerKrnlShape             DESTINATION lib)
