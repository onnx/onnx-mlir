add_library(OMSupport
        KrnlSupport.cpp)
target_include_directories(OMSupport
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMSupport
        onnx)
add_dependencies(OMSupport
        OMKrnlOps
        OMONNXOps)

