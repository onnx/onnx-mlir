add_library(OMAttributePromotion
        AttributePromotion.cpp)
target_include_directories(OMAttributePromotion
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})

# Linking dependencies:
add_dependencies(OMAttributePromotion
        OMPromotableConstOperandsOpInterface)

add_library(OMElideConstants
        ElideConstants.cpp)
target_include_directories(OMElideConstants
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})

add_dependencies(OMElideConstants
        OMONNXOps)

set(LLVM_TARGET_DEFINITIONS ONNXRewrite.td)
onnx_mlir_tablegen(ONNXRewrite.inc -gen-rewriters)
add_public_tablegen_target(OMONNXRewriteIncGen)

set(LLVM_TARGET_DEFINITIONS ONNXCombine.td)
onnx_mlir_tablegen(ONNXCombine.inc -gen-rewriters)
add_public_tablegen_target(OMONNXCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ONNXDecompose.td)
onnx_mlir_tablegen(ONNXDecompose.inc -gen-rewriters)
add_public_tablegen_target(OMONNXDecomposeIncGen)

set(LLVM_TARGET_DEFINITIONS ONNXConstProp.td)
onnx_mlir_tablegen(ONNXConstProp.inc -gen-rewriters)
add_public_tablegen_target(OMONNXConstPropIncGen)

add_library(OMONNXRewrite
        ONNXRewrite.cpp
        ONNXCombine.cpp
        ONNXDecompose.cpp
        ONNXConstProp.cpp)
target_include_directories(OMONNXRewrite
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
add_dependencies(OMONNXRewrite
        OMONNXRewriteIncGen
        OMONNXDecomposeIncGen
        OMONNXCombineIncGen
        OMONNXConstPropIncGen)
# Linking dependencies:
add_dependencies(OMONNXRewrite
        OMONNXOps)

add_library(OMShapeInference ShapeInferencePass.cpp)
target_include_directories(OMShapeInference
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMShapeInference ShapeInferenceOpInterfaceIncGen)
# Linking dependencies:
add_dependencies(OMShapeInference
        OMShapeInferenceOpInterface)
