# SPDX-License-Identifier: Apache-2.0

add_library(OMElideConstants
        ElideConstants.cpp)
target_include_directories(OMElideConstants
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})

# Header dependencies
add_dependencies(OMElideConstants OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMElideConstants OMONNXOps)

add_onnx_mlir_rewriter(Rewrite)
add_onnx_mlir_rewriter(Combine)
add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)

add_library(OMONNXRewrite
        Rewrite.cpp
        Combine.cpp
        Decompose.cpp
        ConstProp.cpp)
target_include_directories(OMONNXRewrite
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})

# Header dependencies
add_dependencies(OMONNXRewrite
        OMONNXOpsInc
        OMONNXRewriteIncGen
        OMONNXDecomposeIncGen
        OMONNXCombineIncGen
        OMONNXConstPropIncGen)

# Linking dependencies
add_dependencies(OMONNXRewrite OMONNXOps)

add_library(OMShapeInference ShapeInferencePass.cpp)
target_include_directories(OMShapeInference
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterfaceIncGen)
# Linking dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterface)

install(TARGETS OMElideConstants     DESTINATION lib)
install(TARGETS OMONNXRewrite        DESTINATION lib)
install(TARGETS OMShapeInference     DESTINATION lib)
