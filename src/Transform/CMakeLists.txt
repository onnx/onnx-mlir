add_library(OMKrnlToAffine
        LowerKrnl.cpp)
target_include_directories(OMKrnlToAffine
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# Linking dependencies:
add_dependencies(OMKrnlToAffine
        OMKrnlOps
        OMONNXOps)

add_library(OMKrnlToLLVM
        LowerToLLVM.cpp)
target_include_directories(OMKrnlToLLVM
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMKrnlToLLVM
        onnx)

#Linking dependencies:
add_dependencies(OMKrnlToLLVM
        OMKrnlOps
        OMONNXOps)

add_library(OMElideKrnlGlobalConstants
        ElideKrnlGlobalConstants.cpp)
target_include_directories(OMElideKrnlGlobalConstants
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMElideKrnlGlobalConstants
        OMKrnlOps
        OMONNXOps)

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

add_subdirectory(ONNX)
