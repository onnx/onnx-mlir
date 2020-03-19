set(LLVM_TARGET_DEFINITIONS ONNXOps.td)
onnx_mlir_tablegen(onnx.hpp.inc -gen-op-decls "-I${CMAKE_SOURCE_DIR}/compiler/pass")
onnx_mlir_tablegen(onnx.cpp.inc -gen-op-defs "-I${CMAKE_SOURCE_DIR}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(OMONNXOpsIncGen)

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

add_onnx_mlir_dialect_doc(onnx Dialect/ONNX/ONNXOps.td)