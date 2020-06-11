set(LLVM_TARGET_DEFINITIONS MLONNXOps.td)
onnx_mlir_tablegen(MLONNXOps.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")
onnx_mlir_tablegen(MLONNXOps.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}/compiler/pass")

set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/mlonnx.md)
add_public_tablegen_target(OMMLONNXOpsIncGen)

# Header dependencies target for MLONNXOps.hpp
add_custom_target(OMMLONNXOpsInc
        DEPENDS OMMLONNXOpsIncGen
                OMPromotableConstOperandsOpInterfaceIncGen
                OMResultTypeInferenceOpInterfaceIncGen
                ShapeInferenceOpInterfaceIncGen)

add_library(OMMLONNXOps
        MLONNXOps.cpp
        MLONNXOps.hpp)
target_include_directories(OMMLONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
# Header dependencies
add_dependencies(OMMLONNXOps OMMLONNXOpsInc)
# Linking dependencies
add_dependencies(OMMLONNXOps
        OMPromotableConstOperandsOpInterface
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface)

add_onnx_mlir_dialect_doc(mlonnx MLONNXOps.td)
