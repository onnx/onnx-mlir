# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS ONNXOps.td)
onnx_mlir_tablegen(ONNXOps.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
onnx_mlir_tablegen(ONNXOps.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(OMONNXOpsIncGen)

# Header dependencies target for ONNXOps.hpp
add_custom_target(OMONNXOpsInc
        DEPENDS OMONNXOpsIncGen
                OMPromotableConstOperandsOpInterfaceIncGen
                OMResultTypeInferenceOpInterfaceIncGen
                ShapeInferenceOpInterfaceIncGen
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
        OMPromotableConstOperandsOpInterface
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface)

add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

install(TARGETS OMONNXOps DESTINATION lib)
