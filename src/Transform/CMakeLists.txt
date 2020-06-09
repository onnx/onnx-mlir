add_library(OMKrnlToAffine
        LowerKrnl.cpp)
target_include_directories(OMKrnlToAffine
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMKrnlToAffine OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToAffine OMKrnlOps)

add_library(OMKrnlToLLVM
        LowerToLLVM.cpp)
target_include_directories(OMKrnlToLLVM
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# Header dependencies
add_dependencies(OMKrnlToLLVM OMKrnlOpsInc OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToLLVM OMKrnlOps OMONNXOps)

target_link_libraries(OMKrnlToLLVM
        onnx)

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

add_subdirectory(ONNX)
