add_executable(onnx-mlir-opt ONNXMLIROpt.cpp)
add_dependencies(onnx-mlir-opt OMKrnlOpsIncGen)

target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(onnx-mlir-opt
        ${MLIRLibs}
        OMBuilder
        OMKrnlOps
        OMKrnlToAffine
        OMKrnlToLLVM
        OMShapeInference
        OMONNXToKrnl
        OMPromotableConstOperandsOpInterface
        curses)
whole_archive_link_mlir(onnx-mlir-opt
        ${MLIRWholeArchiveLibs})
whole_archive_link_onnx_mlir(onnx-mlir-opt
        OMKrnlToAffine
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXRewrite
        OMShapeInference
        OMAttributePromotion
        OMElideConstants)