add_library(OMKrnlToLLVM
        KrnlToLLVM.cpp)
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

install(TARGETS OMKrnlToLLVM DESTINATION lib)
