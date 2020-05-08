add_executable(onnx-mlir-opt ONNXMLIROpt.cpp)
add_dependencies(onnx-mlir-opt OMKrnlOpsInc OMONNXOpsInc)

target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(onnx-mlir-opt
        ${OMLibs}
        ${MLIRLibs}
        onnx)
