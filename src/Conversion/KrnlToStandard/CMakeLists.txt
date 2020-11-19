add_library(OMKrnlToStandard
        KrnlToStandard.cpp)
target_include_directories(OMKrnlToStandard
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMKrnlToStandard OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToStandard OMKrnlOps)