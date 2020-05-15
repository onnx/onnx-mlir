set(LLVM_TARGET_DEFINITIONS MLONNXOps.td)
onnx_mlir_tablegen(MLONNXOps.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
onnx_mlir_tablegen(MLONNXOps.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/mlonnx.md)
add_public_tablegen_target(OMMLONNXOpsIncGen)

add_library(OMMLONNXOps
        MLONNXOps.cpp
        MLONNXOps.hpp)
target_include_directories(OMMLONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMMLONNXOps OMMLONNXOpsIncGen)
# Linking dependencies:
add_dependencies(OMMLONNXOps
        OMPromotableConstOperandsOpInterface
        OMShapeInferenceOpInterface)

add_onnx_mlir_dialect_doc(mlonnx MLONNXOps.td)
