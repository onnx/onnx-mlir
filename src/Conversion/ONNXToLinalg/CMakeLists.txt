add_library(OMONNXToLinalg
        ConvertONNXToLinalg.cpp)

target_include_directories(OMONNXToLinalg
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

target_link_libraries(OMONNXToLinalg
        onnx)

# Header dependencies
add_dependencies(OMONNXToLinalg OMONNXOpsInc)

# Linking dependencies
add_dependencies(OMONNXToLinalg OMONNXOps OMONNXToKrnl)
