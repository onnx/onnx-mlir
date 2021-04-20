# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(ONNXOps)

# Header dependencies target for ONNXOps.hpp
add_custom_target(OMONNXOpsInc
        DEPENDS OMONNXOpsIncGen
                OMResultTypeInferenceOpInterfaceIncGen
                OMShapeInferenceOpInterfaceIncGen
                OMHasOnnxSubgraphOpInterfaceIncGen)

add_library(OMONNXOps
        ONNXOps.cpp
        ONNXOps.hpp
        ONNXOpsHelper.cpp
        ONNXOpsHelper.hpp
        ONNXShapeHelper.cpp
        ONNXShapeHelper.hpp
        IndexExpr.hpp
        IndexExpr.cpp
        IndexExprDetail.hpp
        IndexExprDetail.cpp)

target_include_directories(OMONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(OMONNXOps OMONNXOpsInc OMKrnlOpsInc)
## TODO: remove dependency of ONNXOps on KrnlOps
# OMONNXOps depends on symbols in OMKrnlOps, we do not specify
# linking dependency explicitly here because OMKrnlOps also depends
# on OMONNXOps, which is circular.

# Linking dependencies:
add_dependencies(OMONNXOps
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface
        # TODO: remove unnecessary dependency due to
        # ONNX Krnl Helper having a dependence on KrnlOps.
        OMSpecializedKernelOpInterface)

add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

install(TARGETS OMONNXOps DESTINATION lib)
