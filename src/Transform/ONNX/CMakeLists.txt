add_library(onnx_mlir_attribute_promotion
        AttributePromotion.cpp)
target_include_directories(onnx_mlir_attribute_promotion
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_attribute_promotion
        OMPromotableConstOperandsOpInterface)

set(LLVM_TARGET_DEFINITIONS ONNXRewrite.td)
onnx_mlir_tablegen(onnx_rewrite.inc -gen-rewriters)
add_public_tablegen_target(OMONNXRewriteIncGen)

set(LLVM_TARGET_DEFINITIONS ONNXCombine.td)
onnx_mlir_tablegen(onnx_combine.inc -gen-rewriters)
add_public_tablegen_target(OMONNXCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ONNXDecompose.td)
onnx_mlir_tablegen(onnx_decompose.inc -gen-rewriters)
add_public_tablegen_target(OMONNXDecomposeIncGen)

add_library(OMONNXRewrite
        ONNXRewrite.cpp
        ONNXCombine.cpp
        ONNXDecompose.cpp)
target_include_directories(OMONNXRewrite
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
add_dependencies(OMONNXRewrite
        OMONNXRewriteIncGen
        OMONNXDecomposeIncGen
        OMONNXCombineIncGen)

add_library(OMShapeInference ShapeInferencePass.cpp)
target_include_directories(OMShapeInference
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(OMShapeInference
        ${MLIRLibs}
        OMShapeInferenceOpInterface)
add_dependencies(OMShapeInference ShapeInferenceOpInterfaceIncGen)