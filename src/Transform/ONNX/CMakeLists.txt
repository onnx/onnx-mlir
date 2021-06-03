# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMElideConstants
        ElideConstants.cpp)

# Header dependencies
add_dependencies(OMElideConstants OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMElideConstants OMONNXOps)

add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)

add_onnx_mlir_library(OMONNXRewrite
        Decompose.cpp
        ConstProp.cpp)

# Header dependencies
add_dependencies(OMONNXRewrite
        OMONNXOpsInc
        OMONNXDecomposeIncGen
        OMONNXConstPropIncGen)

# Linking dependencies
add_dependencies(OMONNXRewrite OMONNXOps)

add_onnx_mlir_library(OMShapeInference ShapeInferencePass.cpp)

# Header dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterfaceIncGen)
# Linking dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterface)
