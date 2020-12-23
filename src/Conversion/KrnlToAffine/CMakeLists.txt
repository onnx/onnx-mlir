add_library(OMKrnlToAffine
        KrnlToAffine.cpp)
target_include_directories(OMKrnlToAffine
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMKrnlToAffine OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToAffine OMKrnlOps)

install(TARGETS OMKrnlToAffine DESTINATION lib)
