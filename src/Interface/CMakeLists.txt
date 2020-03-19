set(LLVM_TARGET_DEFINITIONS PromotableConstOperandsOpInterface.td)
onnx_mlir_tablegen(promotable_const_operands_op_interface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(promotable_const_operands_op_interface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(onnx_mlir_gen_promotable_const_operands_op_interface)

add_library(OMPromotableConstOperandsOpInterface
        PromotableConstOperandsOpInterface.hpp
        PromotableConstOperandsOpInterface.cpp)
target_include_directories(OMPromotableConstOperandsOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMPromotableConstOperandsOpInterface
        onnx_mlir_gen_promotable_const_operands_op_interface)


set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
onnx_mlir_tablegen(shape_inference.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(shape_inference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceOpInterfaceIncGen)

add_library(OMShapeInferenceOpInterface
        ShapeInferenceInterface.hpp
        ShapeInferenceInterface.cpp)
target_include_directories(OMShapeInferenceOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMShapeInferenceOpInterface ShapeInferenceOpInterfaceIncGen)