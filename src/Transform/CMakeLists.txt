# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMElideKrnlGlobalConstants
        ElideKrnlGlobalConstants.cpp)
# Header dependencies
add_dependencies(OMElideKrnlGlobalConstants OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMElideKrnlGlobalConstants OMKrnlOps)

add_onnx_mlir_library(OMEnableMemoryPool
        EnableMemoryPool.cpp)
target_link_libraries(OMEnableMemoryPool
        OMSupport)
add_dependencies(OMEnableMemoryPool
        OMSupport
        OMKrnlOps)

add_onnx_mlir_library(OMOptimizeMemoryPools
        OptimizeMemoryPools.cpp)
target_link_libraries(OMOptimizeMemoryPools
        OMSupport)
add_dependencies(OMOptimizeMemoryPools
        OMSupport
        OMKrnlOps)

add_onnx_mlir_library(OMBundleMemoryPools
        BundleMemoryPools.cpp)
target_link_libraries(OMBundleMemoryPools
        OMSupport)
add_dependencies(OMBundleMemoryPools
        OMSupport
        OMKrnlOps)

add_onnx_mlir_library(OMDisconnectKrnlDimFromAlloc
        DisconnectKrnlDimFromAlloc.cpp)
target_link_libraries(OMDisconnectKrnlDimFromAlloc
        OMSupport)
add_dependencies(OMDisconnectKrnlDimFromAlloc
        OMSupport
        OMKrnlOps)

add_onnx_mlir_library(OMLowerKrnlShape
        LowerKrnlShape.cpp)
target_link_libraries(OMLowerKrnlShape)
add_dependencies(OMLowerKrnlShape
        OMKrnlOps)

add_subdirectory(ONNX)
add_subdirectory(Krnl)
