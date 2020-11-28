add_library(OMSimplifyKrnl
        SimplifyKrnl.cpp)
target_include_directories(OMSimplifyKrnl PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
add_dependencies(OMSimplifyKrnl OMKrnlOps)