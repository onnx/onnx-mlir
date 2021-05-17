# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMElideConstants
        ElideConstants.cpp)

# Header dependencies
add_dependencies(OMElideConstants OMONNXOpsInc)
# Linking dependencies
add_dependencies(OMElideConstants OMONNXOps)

add_onnx_mlir_library(OMConstPropHelper
  ConstPropHelper.hpp
  ConstPropHelper.cpp)
# Header dependencies
add_dependencies(OMConstPropHelper
        OMONNXOpsIncGen)
# Linking dependencies
add_dependencies(OMConstPropHelper
        OMONNXOps)

add_onnx_mlir_rewriter(Rewrite)
add_onnx_mlir_rewriter(Combine)
add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)

add_onnx_mlir_library(OMONNXRewrite
        Rewrite.cpp
        Combine.cpp
        ConstProp.cpp
        Decompose.cpp)

# Header dependencies
add_dependencies(OMONNXRewrite
        OMONNXOpsInc
        OMONNXRewriteIncGen
        OMONNXDecomposeIncGen
        OMONNXConstPropIncGen
        OMONNXCombineIncGen)

# Linking dependencies
add_dependencies(OMONNXRewrite OMConstPropHelper OMONNXOps)

add_onnx_mlir_library(OMShapeInference ShapeInferencePass.cpp)

# Header dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterfaceIncGen)
# Linking dependencies
add_dependencies(OMShapeInference OMShapeInferenceOpInterface)

