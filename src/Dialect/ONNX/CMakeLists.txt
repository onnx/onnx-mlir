set(LLVM_TARGET_DEFINITIONS ONNXOps.td)
onnx_mlir_tablegen(ONNXOps.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
onnx_mlir_tablegen(ONNXOps.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(OMONNXOpsIncGen)
add_dependencies(OMONNXOpsIncGen OMONNXOpsTableGenIncGen)

add_library(OMONNXOps
        ONNXOps.cpp
        ONNXOps.hpp)
target_include_directories(OMONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMONNXOps OMONNXOpsIncGen)
target_link_libraries(OMONNXOps
        OMPromotableConstOperandsOpInterface
        OMShapeInferenceOpInterface)

add_onnx_mlir_dialect_doc(onnx ONNXOps.td)
