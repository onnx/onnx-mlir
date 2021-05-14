# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(ONNXOps)
add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

# Header dependencies target for ONNXOps.hpp
add_custom_target(OMONNXOpsInc
        DEPENDS OMONNXOpsIncGen
                OMResultTypeInferenceOpInterfaceIncGen
                OMShapeInferenceOpInterfaceIncGen
                OMHasOnnxSubgraphOpInterfaceIncGen)

add_onnx_mlir_library(OMONNXOps
        ONNXOps.cpp
        ONNXOpsHelper.cpp
        ONNXShapeHelper.cpp
        IndexExpr.cpp
        IndexExprDetail.cpp)

add_dependencies(OMONNXOps OMONNXOpsInc)

# Linking dependencies:
add_dependencies(OMONNXOps
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface)
