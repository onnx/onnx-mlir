add_executable(onnx-mlir-opt ONNXMLIROpt.cpp)
add_dependencies(onnx-mlir-opt OMKrnlOpsIncGen)

target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(onnx-mlir-opt
        OMBuilder
        ${MLIRLibs}
        OMKrnlOps
        onnx_mlir_transform
        OMShapeInference
        OMONNXToKrnl
        OMPromotableConstOperandsOpInterface
        curses)
whole_archive_link_mlir(onnx-mlir-opt
        ${MLIRWholeArchiveLibs})
whole_archive_link_onnx_mlir(onnx-mlir-opt
        onnx_mlir_transform
        OMONNXToKrnl
        OMONNXRewrite
        OMShapeInference
        onnx_mlir_attribute_promotion)