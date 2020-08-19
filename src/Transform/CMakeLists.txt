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

add_library(OMPackKrnlGlobalConstants
        PackKrnlGlobalConstants.cpp)
target_include_directories(OMPackKrnlGlobalConstants
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMPackKrnlGlobalConstants
        OMKrnlOps)

add_library(OMEnableMemoryPool
        EnableMemoryPool.cpp)
target_include_directories(OMEnableMemoryPool
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMEnableMemoryPool
        onnx)
add_dependencies(OMEnableMemoryPool
        OMKrnlOps
        OMONNXOps)

add_library(OMBundleMemoryPools
        BundleMemoryPools.cpp)
target_include_directories(OMBundleMemoryPools
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMBundleMemoryPools
        onnx)
add_dependencies(OMBundleMemoryPools
        OMKrnlOps
        OMONNXOps)

add_library(OMDisconnectKrnlDimFromAlloc
        DisconnectKrnlDimFromAlloc.cpp)
target_include_directories(OMDisconnectKrnlDimFromAlloc
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMDisconnectKrnlDimFromAlloc
        onnx)
add_dependencies(OMDisconnectKrnlDimFromAlloc
        OMKrnlOps
        OMONNXOps)

add_subdirectory(ONNX)
