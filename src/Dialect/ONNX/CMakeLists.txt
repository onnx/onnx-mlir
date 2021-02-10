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
                FoldOpInterfaceIncGen)

add_library(OMONNXOps
        ONNXOps.cpp
        ONNXOps.hpp
        ONNXOpsHelper.cpp
        ONNXOpsHelper.hpp
        ONNXShapeHelper.cpp
        ONNXShapeHelper.hpp
        ONNXFoldHelper.cpp
        ONNXFoldHelper.hpp
        IndexExpr.hpp
        IndexExpr.cpp)

target_include_directories(OMONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(OMONNXOps OMONNXOpsIncGen OMKrnlOpsInc)
target_link_libraries(OMONNXOps
        onnx)
# Linking dependencies:
add_dependencies(OMONNXOps
        OMPromotableConstOperandsOpInterface
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface
        OMFoldOpInterface)

add_onnx_mlir_dialect_doc(onnx ONNXOps.td)
